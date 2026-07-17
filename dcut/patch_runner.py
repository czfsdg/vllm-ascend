# SPDX-License-Identifier: Apache-2.0
"""Patch NPUModelRunner for D-Cut adaptive verify."""
from __future__ import annotations

import os

import torch

from vllm.config import CUDAGraphMode

from .globals import logger, ENABLE_GDN_MAIN_PIECEWISE_GRAPH
from .gdn_buffers import _dcut_update_gdn_static
from .controller import _dcut_init_controller, _dcut_enable_drafter_probs
from .truncate import _dcut_truncate
from .probs import _dcut_queue_probs, _maybe_process_adaptive_probs, profile_adaptive_cost
from .adaptive_profile import _adaptive_profile_run

def _patch_runner() -> None:
    import vllm_ascend.worker.model_runner_v1 as m

    # Class-level removal: modify CompilationConfig.splitting_ops at import
    # time so the config parser uses the modified list. The __init__ patch
    # alone is too late if the compiler reads splitting_ops before __init__.
    if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
        try:
            from vllm.config import CompilationConfig as _CC
            _gdn_op = "vllm::qwen_gdn_attention_core"
            if _CC.splitting_ops and _gdn_op in _CC.splitting_ops:
                _CC.splitting_ops = [op for op in _CC.splitting_ops if op != _gdn_op]
                logger.warning("D-Cut: removed %s from CompilationConfig.splitting_ops (class-level)", _gdn_op)
            if hasattr(_CC, '_attention_ops') and _gdn_op in _CC._attention_ops:
                _CC._attention_ops = [op for op in _CC._attention_ops if op != _gdn_op]
                logger.warning("D-Cut: removed %s from CompilationConfig._attention_ops (class-level)", _gdn_op)
        except Exception as e:
            logger.warning("D-Cut: class-level removal failed: %s", e)

    R = m.NPUModelRunner
    if getattr(R, "_dcut_patched", False):
        return

    _orig_init = R.__init__

    def __init__(self, *a, **k):
        # Re-enabled: Remove GDN from splitting_ops so attention core is
        # captured in the PIECEWISE graph. The attention core's metadata
        # (actual_seq_lengths, ssm_state_indices, num_accepted_tokens) is
        # updated at replay time via graph_task_update (see
        # update_gdn_attn_graph_params in gdn.py).
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            try:
                _vc = k.get("vllm_config") or (a[0] if a else None)
                if _vc is not None:
                    _cc = _vc.compilation_config
                    _gdn_op = "vllm::qwen_gdn_attention_core"
                    if _cc.splitting_ops and _gdn_op in _cc.splitting_ops:
                        _cc.splitting_ops.remove(_gdn_op)
                        logger.warning("D-Cut: REMOVED %s from instance splitting_ops, remaining=%d ops", _gdn_op, len(_cc.splitting_ops))
                    else:
                        logger.warning("D-Cut: %s NOT in splitting_ops (len=%d, type=%s)", _gdn_op, len(_cc.splitting_ops) if _cc.splitting_ops else -1, type(_cc.splitting_ops))
                    _cls = type(_cc)
                    if hasattr(_cls, '_attention_ops') and _gdn_op in _cls._attention_ops:
                        _cls._attention_ops = [op for op in _cls._attention_ops if op != _gdn_op]
                        logger.warning("D-Cut: REMOVED %s from class _attention_ops", _gdn_op)
                    logger.warning("D-Cut: GDN removed from splitting_ops and _attention_ops (Phase 2: true入图)")
            except Exception as e:
                logger.warning("D-Cut: failed to remove GDN in __init__: %s", e)
        _orig_init(self, *a, **k)
        try:
            _dcut_init_controller(self)
        except Exception as e:
            logger.error("D-Cut init failed; running vanilla: %s", e)
            self._verify_adaptive_controller = None

    _orig_exec = R.execute_model

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        # DEBUG: print to stdout to confirm wrapper is called
        _ctrl = getattr(self, "_verify_adaptive_controller", None)
        _has_spec = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
        if _ctrl is not None:
            _dcut_enable_drafter_probs(self)
            import os as _os_dcut
            if not _os_dcut.environ.get("VLLM_DCUT_DISABLE"):
                scheduler_output = _dcut_truncate(self, scheduler_output)
        # DEBUG: log every N calls to avoid flooding
        if not hasattr(self, "_dcut_exec_count"):
            self._dcut_exec_count = 0
        self._dcut_exec_count += 1
        if self._dcut_exec_count % 50 == 1:
            logger.info("DCUT_EXEC[%d]: ctrl=%s, has_spec=%s",
                        self._dcut_exec_count,
                        "None" if _ctrl is None else "set",
                        _has_spec)
        return _orig_exec(self, scheduler_output, intermediate_tensors)

    _orig_sample_tokens = R.sample_tokens

    def sample_tokens(self, *a, **k):
        out = _orig_sample_tokens(self, *a, **k)
        if getattr(self, "_adaptive_probs_pending", False):
            try:
                _maybe_process_adaptive_probs(self)
            except Exception as e:
                logger.warning("D-Cut: process probs failed: %s", e)
                self._adaptive_probs_pending = False
        return out

    _orig_copy = R._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        _orig_copy(self, scheduler_output, zeros_only)
        if getattr(self, "_verify_adaptive_controller", None) is not None:
            try:
                _dcut_queue_probs(self, zeros_only)
            except Exception as e:
                logger.warning("D-Cut: queue probs failed: %s", e)

    _orig_update = R._update_states

    def _update_states(self, scheduler_output):
        ret = _orig_update(self, scheduler_output)
        ctrl = getattr(self, "_verify_adaptive_controller", None)
        if ctrl is not None:
            for rid in scheduler_output.finished_req_ids:
                ctrl.invalidate(rid)
        return ret

    R.__init__ = __init__
    R.execute_model = execute_model
    R.sample_tokens = sample_tokens
    R._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    R._update_states = _update_states
    R._adaptive_profile_run = _adaptive_profile_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_enable_drafter_probs = _dcut_enable_drafter_probs
    R._dcut_patched = True

    # --- Force attn_metadata build during PIECEWISE graph capture ---
    # By default, _dummy_run only builds attn_metadata when
    # force_attention=True or cudagraph_runtime_mode==FULL.  In PIECEWISE
    # mode, force_attention defaults to False, so attn_metadata is None.
    # GDN's _forward_core returns early when attn_metadata is None, making
    # GDN a no-op in the captured graph -> garbled output.  Force
    # force_attention=True for PIECEWISE so attn_metadata is built during
    # capture, allowing GDN's recurrent attention capturing branch to run.
    _orig_dummy_run = R._dummy_run

    def _dummy_run(self, num_tokens, cudagraph_runtime_mode=None,
                   force_attention=False, **kwargs):
        # Force attention metadata build for PIECEWISE mode (both warmup AND capture).
        # During warmup, cudagraph_runtime_mode=NONE but we still need attn_metadata
        # so the GDN recurrent path is reached and v8b instance buffers are
        # pre-allocated in the regular memory pool (not the graph private pool).
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH and (
                cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE
                    and self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE)):
            # Only force attention during REAL warmup (not profile_cudagraph_memory,
            # which uses a minimal KV cache and hits GDN's build_for_cudagraph_capture
            # assertion when num_tokens > decode_cudagraph_max_bs).
            if getattr(self, '_dcut_in_real_warmup', False):
                force_attention = True
        self._dcut_in_dummy_run = True
        try:
            return _orig_dummy_run(
                self, num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention, **kwargs)
        finally:
            self._dcut_in_dummy_run = False

    R._dummy_run = _dummy_run
    logger.warning(
        "D-Cut: patched _dummy_run to force force_attention=True "
        "for PIECEWISE mode (GDN needs attn_metadata during capture)."
    )

    # --- Force use_spec_decode=True during _dummy_run for GDN recurrent path ---
    # During _dummy_run (warmup + capture), use_spec_decode defaults to False.
    # During capture, build_for_cudagraph_capture derives num_accepted_tokens
    # from query_start_loc, so spec_sequence_masks IS set and the recurrent
    # path is reached.  But during WARMUP, for_cudagraph_capture=False and
    # use_spec_decode=False, so extra_attn_metadata_args is empty,
    # num_accepted_tokens=None, spec_sequence_masks=None, and the recurrent
    # path is SKIPPED.  This means v8b instance buffers are never
    # pre-allocated during warmup -> during capture they fall back to the
    # graph's private pool -> overwritten during replay -> garbled output.
    #
    # Fix: when _dcut_in_dummy_run is True and speculative_config is set,
    # force use_spec_decode=True and set up num_decode_draft_tokens /
    # num_accepted_tokens so the GDN builder creates spec_sequence_masks
    # during warmup.  Buffers allocated during warmup (eager, not capturing)
    # go to the REGULAR memory pool, surviving into capture.
#     _orig_build_attn = R._build_attention_metadata
# 
#     def _build_attention_metadata(self, *args, **kwargs):
#         if (getattr(self, '_dcut_in_real_warmup', False)
#                 and self.speculative_config is not None
#                 and not kwargs.get('use_spec_decode', False)):
#             _nst_np = kwargs.get('num_scheduled_tokens_np')
#             _nr = kwargs.get('num_reqs')
#             if _nst_np is not None and _nr is not None and _nr > 0:
#                 num_spec = self.num_spec_tokens
#                 _nrp = kwargs.get('num_reqs_padded') or _nr
#                 # Set up num_decode_draft_tokens (>= 0 marks spec-decode reqs)
#                 if hasattr(self, 'num_decode_draft_tokens'):
#                     for i in range(_nr):
#                         self.num_decode_draft_tokens.np[i] = min(
#                             int(_nst_np[i]) - 1, num_spec
#                         )
#                     self.num_decode_draft_tokens.np[_nr:_nrp].fill(-1)
#                     self.num_decode_draft_tokens.copy_to_gpu()
#                 # Set up num_accepted_tokens
#                 if hasattr(self, 'num_accepted_tokens'):
#                     for i in range(_nr):
#                         self.num_accepted_tokens.gpu[i] = min(
#                             int(_nst_np[i]), num_spec + 1
#                         )
#                     self.num_accepted_tokens.gpu[_nr:_nrp].fill_(1)
#                 kwargs['use_spec_decode'] = True
#         return _orig_build_attn(self, *args, **kwargs)
# 
#     R._build_attention_metadata = _build_attention_metadata
#     logger.warning(
#         "D-Cut: patched _build_attention_metadata to force use_spec_decode=True "
#         "during _dummy_run (GDN recurrent path needs spec_sequence_masks)."
#     )

    # --- PIECEWISE conv1d subtask update ---
    # update_full_graph_params (which calls update_conv1d_graph_params) is
    # only invoked for FULL mode in the upstream model_runner.  Now that GDN
    # is inside the PIECEWISE graph, conv1d subtask host args must be updated
    # before each PIECEWISE replay too.  Without this, the events created
    # during capture are never re-recorded, and the graph replay hangs
    # waiting for them.
    # NOTE: Only apply when ENABLE_GDN_MAIN_PIECEWISE_GRAPH=True. When False,
    # GDN is a splitting op (eager), so conv1d is also eager and does not
    # need graph param updates. Calling update_conv1d_graph_params in that
    # case causes silent EngineCore crashes under high concurrency.
    _orig_update_full = R._update_full_graph_params_if_needed

    def _update_full_graph_params_if_needed(
        self, forward_context, num_tokens_padded, positions
    ):
        # When GDN is a splitting op (flag=False), conv1d is eager and does
        # not need graph param updates. Calling update_conv1d_graph_params
        # in that case causes silent EngineCore crashes under high concurrency.
        if not ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            _orig_update_full(self, forward_context, num_tokens_padded, positions)
            return
        # For prefill steps, force eager mode. The GDN graph only captured
        # the spec (decode) branch, which can't handle prefill (needs
        # chunk_gated_delta_rule).
        _am = getattr(forward_context, 'attn_metadata', None)
        # Detect spec decode: check if any metadata in the dict has
        # spec_sequence_masks is not None. If none do, this is a
        # prefill or regular decode step — force eager to avoid
        # graph replay hang (events never re-recorded when
        # spec_sequence_masks is None).
        # Also check for mixed batches (prefill + spec decode): the conv1d
        # host arg padding can't handle mixed batches (EZ9999 tiling failure).
        # Force eager for any batch with prefills.
        _has_spec = False
        _has_prefill = False
        if isinstance(_am, dict):
            for _v in _am.values():
                if getattr(_v, 'spec_sequence_masks', None) is not None:
                    _has_spec = True
                if getattr(_v, 'num_prefills', 0) > 0:
                    _has_prefill = True
        else:
            if getattr(_am, 'spec_sequence_masks', None) is not None:
                _has_spec = True
            if getattr(_am, 'num_prefills', 0) > 0:
                _has_prefill = True
        if not _has_spec or _has_prefill:
            if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    and not getattr(self, '_dcut_in_dummy_run', False)):
                forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        _orig_update_full(self, forward_context, num_tokens_padded, positions)
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            and not forward_context.capturing
            and not self.use_sparse
            and not self.use_compress
        ):
            if not hasattr(self, 'update_stream'):
                import torch_npu as _tn
                self.update_stream = _tn.npu.Stream()
            from vllm_ascend.ops.gdn import update_conv1d_graph_params
            from vllm_ascend.compilation.acl_graph import _EXTRA_CTX as _ec
            update_conv1d_graph_params(
                self.update_stream,
                forward_context,
                num_tokens_padded,
                self.vllm_config,
                _ec.is_draft_model,
                None,
            )

    R._update_full_graph_params_if_needed = _update_full_graph_params_if_needed
    logger.warning(
        'D-Cut: patched _update_full_graph_params_if_needed for PIECEWISE conv1d update.'
    )

    # Patch _model_forward to call correct_conv1d_state() after graph replay.
    # With ENPU enabled, the flow is:
    #   _update_full_graph_params_if_needed() -> saves correction data (state before replay)
    #   run_model() -> graph replay (corrupts last segment state via padding)
    #   correct_conv1d_state() -> recomputes last segment state from saved state + real data
    _orig_model_forward = R._model_forward

    def _model_forward(self, *args, **kwargs):
        # Fill GDN static buffers before graph replay (outside captured graph).
        # At capture time: fills buffers so _forward_core's capture path can
        # pass them to the GDN op (stable data_ptr baked into graph).
        # At replay time: fills buffers with runtime values so the GDN op
        # inside the replayed graph reads updated ASL/SSI/NAT.
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            try:
                from vllm.forward_context import get_forward_context
                from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
                fc = get_forward_context()
                if fc is not None and fc.attn_metadata is not None:
                    # num_tokens is NOT passed to _model_forward as an arg.
                    # It's stored in fc.batch_descriptor.num_tokens by
                    # set_forward_context(num_tokens=num_tokens_padded).
                    _nt = fc.batch_descriptor.num_tokens if fc.batch_descriptor else None
                    if _nt is not None:
                        _dcut_update_gdn_static(fc, _nt, GDNAttentionMetadata)
            except Exception as _e:
                logger.debug("D-Cut: GDN static buf update skipped: %s", _e)
        result = _orig_model_forward(self, *args, **kwargs)
        # DISABLED: state correction causes accuracy regression
        # if getattr(self, 'enable_enpu', False):
        #     from vllm_ascend.ops.gdn import correct_conv1d_state
        #     correct_conv1d_state()
        return result

    R._model_forward = _model_forward
    logger.warning(
        'D-Cut: patched _model_forward for post-replay conv1d state correction.'
    )


    logger.warning("D-Cut: patched NPUModelRunner.")



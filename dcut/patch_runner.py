# SPDX-License-Identifier: Apache-2.0
"""Patch NPUModelRunner for D-Cut adaptive verify."""
from __future__ import annotations

import os

from .controller import _dcut_init_controller, _dcut_enable_drafter_probs
from .dcut_profile import _adaptive_profile_run
from .gdn_buffers import _dcut_prepare_gdn_piecewise_replay
from .globals import ENV_CONFIG, ENV_FULL_DECODE_ONLY, logger
from .patch_gdn_v023 import (
    _enable_gdn_piecewise_graph,
    _gdn_piecewise_graph_enabled,
)
from .probs import (
    _dcut_queue_probs,
    _maybe_process_adaptive_probs,
    profile_adaptive_cost,
)
from .truncate import _dcut_truncate

ENV_DEBUG_STATS = "VLLM_DCUT_DEBUG_STATS"


def _dcut_piecewise_capture_dummy_enabled(
    runner,
    cudagraph_runtime_mode,
    is_profile: bool = False,
    is_graph_capturing: bool = False,
) -> bool:
    """Return whether this dummy run must capture the spec GDN branch."""
    from vllm.config import CUDAGraphMode

    compilation_config = getattr(runner, "compilation_config", None)
    return (
        _gdn_piecewise_graph_enabled()
        and not is_profile
        and is_graph_capturing
        and getattr(runner, "_dcut_in_real_warmup", False)
        and getattr(compilation_config, "cudagraph_mode", None)
        == CUDAGraphMode.PIECEWISE
        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


def _patch_runner() -> None:
    import vllm_ascend.worker.model_runner_v1 as m

    R = m.NPUModelRunner
    if getattr(R, "_dcut_patched", False):
        return

    _orig_init = R.__init__

    def __init__(self, *a, **k):
        if os.environ.get(ENV_CONFIG):
            vllm_config = k.get("vllm_config")
            if vllm_config is None and a:
                vllm_config = a[0]
            if not _enable_gdn_piecewise_graph(vllm_config):
                raise RuntimeError(
                    "D-Cut could not enable GDN capture in PIECEWISE ACLGraph"
                )
        _orig_init(self, *a, **k)
        self._dcut_gdn_piecewise_capture_sizes = set()
        try:
            _dcut_init_controller(self)
        except Exception as e:
            logger.error("D-Cut init failed; running vanilla: %s", e)
            self._verify_adaptive_controller = None

    _orig_exec = R.execute_model
    _orig_model_forward = R._model_forward
    _orig_dummy_run = R._dummy_run
    _orig_should_build_dummy = R._should_build_dummy_attn_metadata

    def _dummy_run(self, num_tokens, *args, **kwargs):
        cudagraph_runtime_mode = kwargs.get("cudagraph_runtime_mode")
        if cudagraph_runtime_mode is None and len(args) > 1:
            cudagraph_runtime_mode = args[1]
        is_profile = kwargs.get("is_profile", False)
        if len(args) > 4:
            is_profile = args[4]
        is_graph_capturing = kwargs.get("is_graph_capturing", False)
        if len(args) > 9:
            is_graph_capturing = args[9]
        capture_dummy = _dcut_piecewise_capture_dummy_enabled(
            self,
            cudagraph_runtime_mode,
            is_profile=bool(is_profile),
            is_graph_capturing=bool(is_graph_capturing),
        )
        if is_graph_capturing:
            from vllm.config import CUDAGraphMode

            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                logger.warning(
                    "D-Cut: PIECEWISE capture dummy token_bucket=%d "
                    "enabled=%s real_warmup=%s.",
                    num_tokens,
                    capture_dummy,
                    getattr(self, "_dcut_in_real_warmup", False),
                )
        previous = getattr(
            self,
            "_dcut_piecewise_capture_dummy",
            False,
        )
        self._dcut_piecewise_capture_dummy = capture_dummy
        try:
            return _orig_dummy_run(self, num_tokens, *args, **kwargs)
        finally:
            self._dcut_piecewise_capture_dummy = previous

    def _should_build_dummy_attn_metadata(
        self,
        force_attention=False,
        is_profile=False,
        cudagraph_runtime_mode=None,
    ):
        return (
            _orig_should_build_dummy(
                self,
                force_attention,
                is_profile,
                cudagraph_runtime_mode,
            )
            or getattr(
                self,
                "_dcut_piecewise_capture_dummy",
                False,
            )
        )

    def _model_forward(self, num_tokens_padded, *args, **kwargs):
        capture_dummy = getattr(
            self,
            "_dcut_piecewise_capture_dummy",
            False,
        )
        graph_safe = False
        if _gdn_piecewise_graph_enabled():
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context
            from vllm.v1.attention.backends.gdn_attn import (
                GDNAttentionMetadata,
            )

            forward_context = get_forward_context()
            if (
                forward_context is not None
                and getattr(
                    forward_context, "cudagraph_runtime_mode", None
                )
                == CUDAGraphMode.PIECEWISE
            ):
                max_num_seqs = (
                    self.vllm_config.scheduler_config.max_num_seqs
                )
                try:
                    graph_safe = _dcut_prepare_gdn_piecewise_replay(
                        forward_context,
                        num_tokens_padded,
                        GDNAttentionMetadata,
                        max_num_seqs,
                    )
                except Exception as exc:
                    logger.warning(
                        "D-Cut: PIECEWISE GDN metadata preparation failed; "
                        "falling back to eager for this batch: %s",
                        exc,
                    )
                    graph_safe = False
                capture_sizes = (
                    self._dcut_gdn_piecewise_capture_sizes
                )
                if (
                    graph_safe
                    and not capture_dummy
                    and num_tokens_padded not in capture_sizes
                ):
                    logger.warning(
                        "D-Cut: PIECEWISE GDN token bucket %d was not "
                        "captured during startup; using eager for this batch.",
                        num_tokens_padded,
                    )
                    graph_safe = False
                if not graph_safe:
                    # qwen_gdn_attention_core selects its execution branch
                    # from Python forward-context metadata. A graph captured
                    # for pure spec decode cannot be replayed for prefill,
                    # regular decode, mixed batches, or a metadata-free dummy
                    # run.
                    forward_context.cudagraph_runtime_mode = (
                        CUDAGraphMode.NONE
                    )
        if capture_dummy and not graph_safe:
            raise RuntimeError(
                "D-Cut could not build pure speculative GDN metadata "
                f"while capturing PIECEWISE token bucket "
                f"{num_tokens_padded}"
            )
        result = _orig_model_forward(
            self, num_tokens_padded, *args, **kwargs
        )
        if capture_dummy:
            self._dcut_gdn_piecewise_capture_sizes.add(
                num_tokens_padded
            )
            logger.warning(
                "D-Cut: captured PIECEWISE GDN token bucket %d",
                num_tokens_padded,
            )
        return result

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return _orig_exec(self, scheduler_output, intermediate_tensors)

        _ctrl = getattr(self, "_verify_adaptive_controller", None)
        _has_spec = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
        debug_stats = bool(os.environ.get(ENV_DEBUG_STATS))
        # Capture trim info before truncation only when optional debug timing is
        # enabled.  The regular D-Cut trim logger already records verify-token
        # reduction inside _dcut_truncate; keeping a second unconditional stats
        # path here adds Python work to every decode iteration.
        _full_draft = 0
        if debug_stats and _ctrl is not None and _has_spec:
            _orig_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
            _full_draft = sum(len(t) for t in _orig_spec.values())
        dcut_enabled = _ctrl is not None and not os.environ.get("VLLM_DCUT_DISABLE")
        if _ctrl is not None:
            _dcut_enable_drafter_probs(self)
            if dcut_enabled:
                scheduler_output = _dcut_truncate(self, scheduler_output)

        if not debug_stats:
            return _orig_exec(self, scheduler_output, intermediate_tensors)

        # Optional slow-path debug timing.  Keep it behind an env gate because
        # perf_counter plus per-step Python aggregation is visible at high ITL.
        _kept_draft = _full_draft
        if dcut_enabled and _has_spec:
            _new_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
            _kept_draft = sum(len(t) for t in _new_spec.values())
        import time as _time
        _t0 = _time.perf_counter()
        result = _orig_exec(self, scheduler_output, intermediate_tensors)
        _fwd_ms = (_time.perf_counter() - _t0) * 1000
        if not hasattr(self, "_dcut_fwd_accum"):
            self._dcut_fwd_accum = {"full": 0, "kept": 0, "cut": 0, "fwd_ms": 0.0, "steps": 0, "spec_steps": 0}
        acc = self._dcut_fwd_accum
        acc["steps"] += 1
        acc["fwd_ms"] += _fwd_ms
        if _has_spec:
            acc["spec_steps"] += 1
            acc["full"] += _full_draft
            acc["kept"] += _kept_draft
            acc["cut"] += (_full_draft - _kept_draft)
        if acc["steps"] % 50 == 0:
            _avg_fwd = acc["fwd_ms"] / acc["steps"]
            if acc["full"] > 0:
                _cut_pct = 100.0 * acc["cut"] / acc["full"]
                logger.warning(
                    "D-Cut step %d: full_draft=%d cut=%d (%.1f%%) kept=%d avg_fwd=%.1fms",
                    acc["steps"], acc["full"], acc["cut"], _cut_pct, acc["kept"], _avg_fwd)
            else:
                logger.warning(
                    "D-Cut step %d: no spec reqs, avg_fwd=%.1fms", acc["steps"], _avg_fwd)
        return result

    _orig_sample_tokens = R.sample_tokens

    def sample_tokens(self, *a, **k):
        out = _orig_sample_tokens(self, *a, **k)
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return out
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
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return
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
    R._model_forward = _model_forward
    R._dummy_run = _dummy_run
    R._should_build_dummy_attn_metadata = _should_build_dummy_attn_metadata
    R.execute_model = execute_model
    R.sample_tokens = sample_tokens
    R._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    R._update_states = _update_states
    R._adaptive_profile_run = _adaptive_profile_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_enable_drafter_probs = _dcut_enable_drafter_probs
    R._dcut_patched = True

    logger.info(
        "D-Cut: using graph-captured GDN in the vLLM 0.23 PIECEWISE path."
    )

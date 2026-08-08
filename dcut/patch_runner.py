# SPDX-License-Identifier: Apache-2.0
"""Patch NPUModelRunner for D-Cut adaptive verify."""
from __future__ import annotations

import os

from vllm.distributed import get_pp_group, get_tp_group, get_world_group

from .controller import _dcut_init_controller, _dcut_enable_drafter_probs
from .dcut_profile import _adaptive_profile_run
from .draft_profile import _adaptive_profile_draft_run
from .gdn_buffers import (
    _dcut_prepare_gdn_eager_state,
    _dcut_prepare_gdn_piecewise_replay,
)
from .globals import ENV_FULL_DECODE_ONLY, logger
from .patch_gdn_v023 import _dcut_gdn_has_prefill
from .patch_piecewise import _is_enabled as _gdn_piecewise_graph_enabled
from .probs import (
    _dcut_bypass_prob_capture_for_prefill,
    _dcut_prepare_prob_capture,
    _dcut_queue_probs,
    _maybe_process_adaptive_probs,
    profile_adaptive_cost,
)
from .truncate import _dcut_has_prefill, _dcut_truncate

ENV_DEBUG_STATS = "VLLM_DCUT_DEBUG_STATS"


def _dcut_debug_rank_info() -> dict[str, int | bool]:
    """Return stable distributed identifiers for single-writer debug logs."""
    world_group = get_world_group()
    world_rank = int(world_group.rank)
    return {
        "world_rank": world_rank,
        "tp_rank": int(get_tp_group().rank_in_group),
        "pp_rank": int(get_pp_group().rank_in_group),
        "is_writer": world_rank == 0,
    }


def _dcut_execute_with_gdn_prefill_route(
    runner,
    execute_model,
    scheduler_output,
    intermediate_tensors,
    has_prefill: bool,
):
    """Expose real prefill to GDN without overriding the outer graph mode."""
    attr = "_dcut_gdn_scheduler_has_prefill"
    had_previous = hasattr(runner, attr)
    previous = getattr(runner, attr, None)
    setattr(runner, attr, bool(has_prefill))
    try:
        return execute_model(
            runner,
            scheduler_output,
            intermediate_tensors,
        )
    finally:
        if had_previous:
            setattr(runner, attr, previous)
        elif hasattr(runner, attr):
            delattr(runner, attr)


def _dcut_piecewise_capture_dummy_enabled(
    runner,
    cudagraph_runtime_mode,
    is_profile: bool = False,
    is_graph_capturing: bool = False,
) -> bool:
    """Return whether this dummy run must capture the spec GDN branch."""
    from vllm.config import CUDAGraphMode

    compilation_config = getattr(runner, "compilation_config", None)
    graph_enabled = getattr(
        runner, "_dcut_gdn_piecewise_enabled", None
    )
    if graph_enabled is None:
        graph_enabled = _gdn_piecewise_graph_enabled()
    return (
        graph_enabled
        and not is_profile
        and is_graph_capturing
        and getattr(runner, "_dcut_in_real_warmup", False)
        and getattr(runner, "pcp_size", 1) == 1
        and getattr(runner, "dcp_size", 1) == 1
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
        _orig_init(self, *a, **k)
        self._dcut_gdn_piecewise_enabled = (
            _gdn_piecewise_graph_enabled()
        )
        self._dcut_gdn_piecewise_capture_sizes = set()
        self._dcut_gdn_piecewise_missing_sizes_logged = set()
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
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadata,
        )

        capture_dummy = getattr(
            self,
            "_dcut_piecewise_capture_dummy",
            False,
        )
        graph_safe = False
        forward_context = get_forward_context()
        scheduler_has_prefill = getattr(
            self,
            "_dcut_gdn_scheduler_has_prefill",
            None,
        )
        native_gdn_batch = (
            _dcut_gdn_has_prefill(forward_context)
            if scheduler_has_prefill is None
            else bool(scheduler_has_prefill)
        )
        if forward_context is not None:
            # The context may be reused across forwards. Never let prepared
            # eager state outlive the metadata values it was derived from.
            forward_context._dcut_gdn_eager_spec_state = None
            forward_context._dcut_gdn_native_batch = native_gdn_batch
        if self._dcut_gdn_piecewise_enabled:
            from vllm.config import CUDAGraphMode
            if forward_context is not None:
                # Forward contexts may be reused. Reset local routing before
                # inspecting this batch so no graph-safe state leaks across
                # PIECEWISE/eager or decode/prefill transitions.
                forward_context._dcut_gdn_local_graph_safe = False
                forward_context._dcut_gdn_local_graph_capture_requested = (
                    False
                )
                forward_context._dcut_gdn_local_graph_captured_prefixes = set()
            if (
                forward_context is not None
                and not native_gdn_batch
                and getattr(
                    forward_context, "cudagraph_runtime_mode", None
                )
                == CUDAGraphMode.PIECEWISE
            ):
                parallel_safe = (
                    getattr(self, "pcp_size", 1) == 1
                    and getattr(self, "dcp_size", 1) == 1
                )
                if parallel_safe:
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
                            "D-Cut: local GDN graph metadata preparation "
                            "failed; only GDN boundaries use eager: %s",
                            exc,
                        )
                elif not getattr(
                    self, "_dcut_gdn_parallel_fallback_logged", False
                ):
                    logger.warning(
                        "D-Cut: local GDN graphs are disabled for PCP/DCP "
                        "(pcp=%d, dcp=%d); only GDN boundaries use eager.",
                        getattr(self, "pcp_size", 1),
                        getattr(self, "dcp_size", 1),
                    )
                    self._dcut_gdn_parallel_fallback_logged = True

                capture_sizes = self._dcut_gdn_piecewise_capture_sizes
                if (
                    graph_safe
                    and not capture_dummy
                    and num_tokens_padded not in capture_sizes
                ):
                    missing_sizes = (
                        self._dcut_gdn_piecewise_missing_sizes_logged
                    )
                    if num_tokens_padded not in missing_sizes:
                        logger.warning(
                            "D-Cut: local GDN token bucket %d was not "
                            "captured during startup; only its GDN "
                            "boundaries use eager.",
                            num_tokens_padded,
                        )
                        missing_sizes.add(num_tokens_padded)
                    graph_safe = False

                # These attributes are consumed only at the eager GDN splitting
                # boundary. They never disable the surrounding PIECEWISE graph.
                forward_context._dcut_gdn_local_graph_safe = graph_safe
                forward_context._dcut_gdn_local_graph_capture_requested = (
                    capture_dummy
                )

        if (
            forward_context is not None
            and not graph_safe
            and not native_gdn_batch
        ):
            try:
                _dcut_prepare_gdn_eager_state(
                    forward_context,
                    GDNAttentionMetadata,
                )
            except Exception as exc:
                forward_context._dcut_gdn_eager_spec_state = None
                if not getattr(
                    self, "_dcut_gdn_eager_prepare_fallback_logged", False
                ):
                    logger.warning(
                        "D-Cut: eager GDN shared-state preparation failed; "
                        "falling back to per-layer preparation: %s",
                        exc,
                    )
                    self._dcut_gdn_eager_prepare_fallback_logged = True

        if capture_dummy and not graph_safe:
            raise RuntimeError(
                "D-Cut could not build pure speculative GDN metadata "
                f"while capturing PIECEWISE token bucket "
                f"{num_tokens_padded}"
            )

        self._dcut_last_num_tokens_padded = num_tokens_padded
        self._dcut_last_graph_safe = graph_safe
        runtime_mode = (
            getattr(forward_context, "cudagraph_runtime_mode", None)
            if forward_context is not None
            else None
        )
        self._dcut_last_runtime_mode = (
            getattr(runtime_mode, "name", None)
            or (str(runtime_mode) if runtime_mode is not None else "UNKNOWN")
        )
        result = _orig_model_forward(
            self, num_tokens_padded, *args, **kwargs
        )
        if capture_dummy:
            expected_prefixes = getattr(
                forward_context,
                "_dcut_gdn_local_graph_expected_prefixes",
                frozenset(),
            )
            captured_prefixes = getattr(
                forward_context,
                "_dcut_gdn_local_graph_captured_prefixes",
                set(),
            )
            missing_prefixes = expected_prefixes - captured_prefixes
            if missing_prefixes:
                raise RuntimeError(
                    "D-Cut did not capture every local GDN graph for "
                    f"token bucket {num_tokens_padded}: "
                    f"missing={sorted(missing_prefixes)}"
                )
            self._dcut_gdn_piecewise_capture_sizes.add(
                num_tokens_padded
            )
            logger.warning(
                "D-Cut: captured local GDN graphs for token bucket %d",
                num_tokens_padded,
            )
        return result

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        debug_stats = bool(os.environ.get(ENV_DEBUG_STATS))
        if debug_stats:
            import time as _time

            _t_entry = _time.perf_counter()
            _last_end = getattr(self, "_dcut_last_debug_end", None)
            _has_gap_sample = _last_end is not None
            _gap_after_prev_ms = (
                (_t_entry - _last_end) * 1000
                if _has_gap_sample
                else 0.0
            )
            _prev_step = getattr(self, "_dcut_last_debug_step", None)
            _stats_io_after_prev_ms = getattr(
                self,
                "_dcut_last_stats_io_ms",
                0.0,
            )
            _rank_info = getattr(self, "_dcut_debug_rank_info", None)
            if _rank_info is None:
                _rank_info = _dcut_debug_rank_info()
                self._dcut_debug_rank_info = _rank_info
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return _orig_exec(self, scheduler_output, intermediate_tensors)

        _has_prefill = _dcut_has_prefill(self, scheduler_output)
        _ctrl = getattr(self, "_verify_adaptive_controller", None)
        _has_spec = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
        # Capture trim info before truncation only when optional debug timing is
        # enabled.  The regular D-Cut trim logger already records verify-token
        # reduction inside _dcut_truncate; keeping a second unconditional stats
        # path here adds Python work to every decode iteration.
        _full_draft = 0
        _batch_size = 0
        _spec_batch_size = 0
        _full_num_tokens = 0
        if debug_stats:
            _num_scheduled = getattr(
                scheduler_output,
                "num_scheduled_tokens",
                {},
            )
            _orig_spec = getattr(
                scheduler_output,
                "scheduled_spec_decode_tokens",
                {},
            )
            _full_draft = sum(len(t) for t in _orig_spec.values())
            _batch_size = len(_num_scheduled)
            _spec_batch_size = len(_orig_spec)
            _full_num_tokens = int(
                getattr(scheduler_output, "total_num_scheduled_tokens", 0)
            )
        dcut_enabled = _ctrl is not None and not os.environ.get("VLLM_DCUT_DISABLE")
        self._dcut_skip_current_prob_capture = bool(
            _ctrl is not None and _has_prefill
        )
        if debug_stats:
            _classify_ms = (_time.perf_counter() - _t_entry) * 1000
            _adaptive_probs_process_ms = 0.0
            _drafter_enable_ms = 0.0
            _truncate_ms = 0.0
            _prob_capture_bypass_ms = 0.0
            _prob_capture_reset_ms = 0.0
        if _ctrl is not None and _has_prefill:
            if debug_stats:
                _t_component = _time.perf_counter()
            _dcut_bypass_prob_capture_for_prefill(self)
            if debug_stats:
                _prob_capture_bypass_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000

        if _ctrl is not None and not _has_prefill:
            if getattr(self, "_adaptive_probs_pending", False):
                if debug_stats:
                    _t_component = _time.perf_counter()
                try:
                    _maybe_process_adaptive_probs(
                        self,
                        stage="pre_truncate",
                    )
                except Exception as e:
                    logger.warning("D-Cut: process probs failed: %s", e)
                    self._adaptive_probs_pending = False
                finally:
                    if debug_stats:
                        _adaptive_probs_process_ms = (
                            _time.perf_counter() - _t_component
                        ) * 1000
            if debug_stats:
                _t_component = _time.perf_counter()
            _dcut_enable_drafter_probs(self)
            if debug_stats:
                _drafter_enable_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000
            if dcut_enabled:
                if debug_stats:
                    _t_component = _time.perf_counter()
                scheduler_output = _dcut_truncate(
                    self,
                    scheduler_output,
                    has_prefill=_has_prefill,
                )
                if debug_stats:
                    _truncate_ms = (
                        _time.perf_counter() - _t_component
                    ) * 1000
            if debug_stats:
                _t_component = _time.perf_counter()
            _dcut_prepare_prob_capture(self, scheduler_output)
            if debug_stats:
                _prob_capture_reset_ms = (
                    _time.perf_counter() - _t_component
                ) * 1000

        if not debug_stats:
            import torch as _torch_fast
            _torch_fast.npu.synchronize()
            result = _dcut_execute_with_gdn_prefill_route(
                self,
                _orig_exec,
                scheduler_output,
                intermediate_tensors,
                _has_prefill,
            )
            _torch_fast.npu.synchronize()
            return result

        # Optional slow-path debug timing.  Keep it behind an env gate because
        # perf_counter plus per-step Python aggregation is visible at high ITL.
        _kept_draft = _full_draft
        if dcut_enabled and _has_spec:
            _new_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
            _kept_draft = sum(len(t) for t in _new_spec.values())
        _num_tokens_actual = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        )
        import torch as _torch

        _t_pre_cpu_end = _time.perf_counter()
        _pre_cpu_total_ms = (_t_pre_cpu_end - _t_entry) * 1000
        _pre_cpu_other_ms = max(
            0.0,
            _pre_cpu_total_ms
            - _classify_ms
            - _adaptive_probs_process_ms
            - _drafter_enable_ms
            - _truncate_ms
            - _prob_capture_bypass_ms
            - _prob_capture_reset_ms,
        )
        _torch.npu.synchronize()
        _t_execute_start = _time.perf_counter()
        _pre_sync_ms = (_t_execute_start - _t_pre_cpu_end) * 1000
        result = _dcut_execute_with_gdn_prefill_route(
            self,
            _orig_exec,
            scheduler_output,
            intermediate_tensors,
            _has_prefill,
        )
        _t_execute_return = _time.perf_counter()
        _execute_call_ms = (
            _t_execute_return - _t_execute_start
        ) * 1000
        _torch.npu.synchronize()
        _t_fwd_end = _time.perf_counter()
        _post_sync_ms = (_t_fwd_end - _t_execute_return) * 1000
        _fwd_ms = (_t_fwd_end - _t_execute_start) * 1000
        if not hasattr(self, "_dcut_fwd_accum"):
            self._dcut_fwd_accum = {
                "full": 0,
                "kept": 0,
                "cut": 0,
                "fwd_ms": 0.0,
                "gap_after_prev_ms": 0.0,
                "gap_samples": 0,
                "steps": 0,
                "spec_steps": 0,
            }
        acc = self._dcut_fwd_accum
        acc["steps"] += 1
        acc["fwd_ms"] += _fwd_ms
        if _has_gap_sample:
            acc["gap_after_prev_ms"] += _gap_after_prev_ms
            acc["gap_samples"] += 1
        if _has_spec:
            acc["spec_steps"] += 1
            acc["full"] += _full_draft
            acc["kept"] += _kept_draft
            acc["cut"] += (_full_draft - _kept_draft)
        if acc["steps"] % 50 == 0 and _rank_info["is_writer"]:
            _avg_fwd = acc["fwd_ms"] / acc["steps"]
            _avg_gap = (
                acc["gap_after_prev_ms"] / acc["gap_samples"]
                if acc["gap_samples"]
                else 0.0
            )
            if acc["full"] > 0:
                _cut_pct = 100.0 * acc["cut"] / acc["full"]
                logger.warning(
                    "D-Cut step %d: full_draft=%d cut=%d (%.1f%%) "
                    "kept=%d avg_fwd=%.1fms avg_inter_call_gap=%.1fms",
                    acc["steps"],
                    acc["full"],
                    acc["cut"],
                    _cut_pct,
                    acc["kept"],
                    _avg_fwd,
                    _avg_gap,
                )
            else:
                logger.warning(
                    "D-Cut step %d: no spec reqs, avg_fwd=%.1fms "
                    "avg_inter_call_gap=%.1fms",
                    acc["steps"],
                    _avg_fwd,
                    _avg_gap,
                )

        _t_post_cpu_end = _time.perf_counter()
        _post_cpu_ms = (_t_post_cpu_end - _t_fwd_end) * 1000
        _fwd_stats_out = os.environ.get("VLLM_DCUT_FWD_STATS_OUT")
        _stats_io_ms = 0.0
        if _fwd_stats_out and _rank_info["is_writer"]:
            import json as _json

            _num_padded = getattr(self, "_dcut_last_num_tokens_padded", 0)
            _graph_safe = getattr(self, "_dcut_last_graph_safe", False)
            _runtime_mode = getattr(
                self,
                "_dcut_last_runtime_mode",
                "UNKNOWN",
            )
            _is_eager = _runtime_mode in {
                "NONE",
                "EAGER",
            }
            _t_stats_io_start = _time.perf_counter()
            _entry = {
                "step": acc["steps"],
                "pid": os.getpid(),
                "world_rank": _rank_info["world_rank"],
                "tp_rank": _rank_info["tp_rank"],
                "pp_rank": _rank_info["pp_rank"],
                "bs": _batch_size,
                "spec_bs": _spec_batch_size,
                "has_prefill": _has_prefill,
                "has_spec": _has_spec,
                "mixed_batch": _has_prefill and _has_spec,
                "pure_prefill": _has_prefill and not _has_spec,
                "decode_only": not _has_prefill,
                "dcut_enabled": dcut_enabled,
                "prob_capture_enabled": (
                    _ctrl is not None
                    and not self._dcut_skip_current_prob_capture
                ),
                "drafter_needs_draft_probs": bool(
                    getattr(
                        getattr(self, "drafter", None),
                        "needs_draft_probs",
                        False,
                    )
                ),
                "draft_ran_python": bool(
                    getattr(
                        getattr(self, "drafter", None),
                        "_dcut_last_draft_ran_python",
                        False,
                    )
                ),
                "adaptive_probs_pending_after_step": bool(
                    getattr(self, "_adaptive_probs_pending", False)
                ),
                "prob_capture_skipped_for_prefill": (
                    self._dcut_skip_current_prob_capture
                ),
                "full_draft": _full_draft,
                "kept_draft": _kept_draft,
                "trimmed": _full_draft - _kept_draft,
                "cut_applied": _full_draft != _kept_draft,
                "full_num_tokens": _full_num_tokens,
                "num_tokens_actual": _num_tokens_actual,
                "num_tokens_padded": _num_padded,
                "runtime_mode": _runtime_mode,
                "is_eager": _is_eager,
                "gdn_native_path": _has_prefill,
                "gdn_graph_safe": _graph_safe,
                "prev_step": _prev_step,
                "gap_sample_valid": _has_gap_sample,
                "inter_call_gap_after_prev_step_ms": round(
                    _gap_after_prev_ms,
                    3,
                ),
                "stats_io_after_prev_step_ms": round(
                    _stats_io_after_prev_ms,
                    3,
                ),
                "classify_ms": round(_classify_ms, 3),
                "adaptive_probs_process_ms": round(
                    _adaptive_probs_process_ms,
                    3,
                ),
                "drafter_enable_ms": round(_drafter_enable_ms, 3),
                "truncate_ms": round(_truncate_ms, 3),
                "prob_capture_bypass_ms": round(
                    _prob_capture_bypass_ms,
                    3,
                ),
                "prob_capture_reset_ms": round(
                    _prob_capture_reset_ms,
                    3,
                ),
                "pre_cpu_other_ms": round(_pre_cpu_other_ms, 3),
                "pre_cpu_total_ms": round(_pre_cpu_total_ms, 3),
                "pre_sync_ms": round(_pre_sync_ms, 3),
                "pre_total_ms": round(
                    _pre_cpu_total_ms + _pre_sync_ms,
                    3,
                ),
                "execute_call_ms": round(_execute_call_ms, 3),
                "post_sync_ms": round(_post_sync_ms, 3),
                "fwd_ms": round(_fwd_ms, 2),
                "post_cpu_ms": round(_post_cpu_ms, 3),
            }
            try:
                with open(_fwd_stats_out, "a") as _f:
                    _f.write(_json.dumps(_entry) + chr(10))
            except Exception:
                pass
            _stats_io_ms = (
                _time.perf_counter() - _t_stats_io_start
            ) * 1000
        self._dcut_last_stats_io_ms = _stats_io_ms
        self._dcut_last_debug_step = acc["steps"]
        self._dcut_last_debug_end = _time.perf_counter()
        return result

    _orig_sample_tokens = R.sample_tokens

    def sample_tokens(self, *a, **k):
        out = _orig_sample_tokens(self, *a, **k)
        if (
            os.environ.get(ENV_FULL_DECODE_ONLY)
            or getattr(self, "_dcut_skip_current_prob_capture", False)
        ):
            return out
        if (
            getattr(self, "_adaptive_probs_pending", False)
            and not getattr(self, "_dcut_skip_unready_probs", False)
            and getattr(
                self,
                "_dcut_process_probs_stage",
                "pre_truncate",
            )
            == "post_sample"
        ):
            try:
                _maybe_process_adaptive_probs(self, stage="post_sample")
            except Exception as e:
                logger.warning("D-Cut: process probs failed: %s", e)
                self._adaptive_probs_pending = False
        return out

    _orig_copy = R._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        _orig_copy(self, scheduler_output, zeros_only)
        if (
            os.environ.get(ENV_FULL_DECODE_ONLY)
            or getattr(self, "_dcut_skip_current_prob_capture", False)
        ):
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
    R._adaptive_profile_draft_run = _adaptive_profile_draft_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_enable_drafter_probs = _dcut_enable_drafter_probs
    R._dcut_patched = True

    logger.info(
        "D-Cut: using graph-captured GDN in the vLLM 0.23 PIECEWISE path."
    )

# SPDX-License-Identifier: Apache-2.0
"""Patch NPUModelRunner for D-Cut adaptive verify."""
from __future__ import annotations

import os
import time

from .controller import _dcut_init_controller, _dcut_enable_drafter_probs
from .dcut_profile import _adaptive_profile_run
from .draft_profile import _adaptive_profile_draft_run
from .gdn_buffers import _dcut_prepare_gdn_piecewise_replay
from .globals import ENV_FULL_DECODE_ONLY, logger
from .patch_piecewise import _is_enabled as _gdn_piecewise_graph_enabled
from .probs import (
    _dcut_queue_probs,
    _maybe_process_adaptive_probs,
    profile_adaptive_cost,
)
from .truncate import _dcut_truncate

ENV_DEBUG_STATS = "VLLM_DCUT_DEBUG_STATS"


def _dcut_runtime_mode(
    runner,
    *,
    batch_size: int,
    spec_request_count: int,
) -> str:
    if spec_request_count <= 0:
        return "non_spec"
    if spec_request_count != batch_size:
        return "mixed"
    if getattr(runner, "_dcut_step_gdn_graph_safe", False):
        return "pure_spec_graph"
    return "pure_spec_eager_gdn"


def _dcut_finish_execute_timing(
    runner,
    scheduler_output,
    execute_start_s: float,
) -> None:
    timing = getattr(runner, "_dcut_runtime_timing", None)
    execute_end_s = (
        time.perf_counter() if timing is not None else 0.0
    )
    batch_size = len(getattr(scheduler_output, "num_scheduled_tokens", {}))
    spec_request_count = len(
        getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {}
    )
    mode = _dcut_runtime_mode(
        runner,
        batch_size=batch_size,
        spec_request_count=spec_request_count,
    )
    runner._dcut_last_step_cost_mode = mode
    if timing is None:
        return
    timing.update(
        {
            "batch_size": batch_size,
            "sum_query_len": int(scheduler_output.total_num_scheduled_tokens),
            "mode": mode,
            "execute_start_s": execute_start_s,
            "execute_end_s": execute_end_s,
            "execute_wall_ms": (execute_end_s - execute_start_s) * 1e3,
        }
    )


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
        capture_dummy = getattr(
            self,
            "_dcut_piecewise_capture_dummy",
            False,
        )
        graph_safe = False
        forward_context = None
        if self._dcut_gdn_piecewise_enabled:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context
            from vllm.v1.attention.backends.gdn_attn import (
                GDNAttentionMetadata,
            )

            forward_context = get_forward_context()
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

        self._dcut_step_gdn_graph_safe = bool(
            getattr(self, "_dcut_step_gdn_graph_safe", False)
            or graph_safe
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
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return _orig_exec(self, scheduler_output, intermediate_tensors)

        _ctrl = getattr(self, "_verify_adaptive_controller", None)
        _has_spec = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
        debug_stats = bool(os.environ.get(ENV_DEBUG_STATS))
        _batch_size = len(
            getattr(scheduler_output, "num_scheduled_tokens", {})
        )
        _spec_request_count = len(
            getattr(
                scheduler_output,
                "scheduled_spec_decode_tokens",
                {},
            )
            or {}
        )
        self._dcut_step_gdn_graph_safe = False
        _last_runtime_mode = getattr(
            self, "_dcut_last_step_cost_mode", "all"
        )
        if _spec_request_count != _batch_size:
            _provisional_mode = "mixed"
        elif _last_runtime_mode in (
            "pure_spec_graph",
            "pure_spec_eager_gdn",
        ):
            _provisional_mode = _last_runtime_mode
        elif self._dcut_gdn_piecewise_enabled:
            _provisional_mode = "pure_spec_graph"
        else:
            _provisional_mode = "pure_spec_eager_gdn"
        _measure_runtime = bool(
            _ctrl is not None
            and _has_spec
            and _ctrl.should_measure_runtime_step(
                batch_size=_batch_size,
                mode=_provisional_mode,
            )
        )
        if _measure_runtime:
            self._sync_device()
            _step_start_s = time.perf_counter()
            _previous_end_s = getattr(
                self, "_dcut_prev_sample_end", None
            )
            self._dcut_runtime_timing = {
                "step_start_s": _step_start_s,
                "inter_step_gap_ms": (
                    (_step_start_s - _previous_end_s) * 1e3
                    if _previous_end_s is not None
                    else 0.0
                ),
            }
            self._dcut_step_copy_ms = 0.0
            self._dcut_step_prob_queue_ms = 0.0
            self._dcut_step_sample_core_ms = 0.0
            self._dcut_step_bookkeeping_ms = 0.0
            self._dcut_step_draft_model_ms = 0.0
        else:
            self._dcut_runtime_timing = None
        _prepare_start_s = (
            time.perf_counter()
            if self._dcut_runtime_timing is not None
            else 0.0
        )
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
        if self._dcut_runtime_timing is not None:
            self._dcut_runtime_timing["dcut_prepare_ms"] = (
                time.perf_counter() - _prepare_start_s
            ) * 1e3

        if not debug_stats:
            _execute_start_s = (
                time.perf_counter() if _measure_runtime else 0.0
            )
            result = _orig_exec(self, scheduler_output, intermediate_tensors)
            _dcut_finish_execute_timing(
                self, scheduler_output, _execute_start_s
            )
            return result

        # Optional slow-path debug timing.  Keep it behind an env gate because
        # perf_counter plus per-step Python aggregation is visible at high ITL.
        _kept_draft = _full_draft
        if dcut_enabled and _has_spec:
            _new_spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
            _kept_draft = sum(len(t) for t in _new_spec.values())
        import time as _time
        _t0 = _time.perf_counter()
        result = _orig_exec(self, scheduler_output, intermediate_tensors)
        _dcut_finish_execute_timing(self, scheduler_output, _t0)
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
    _orig_sample_core = R._sample
    _orig_bookkeeping_sync = R._bookkeeping_sync
    _orig_propose_draft_token_ids = R.propose_draft_token_ids

    def _sample(self, *a, **k):
        timing = getattr(self, "_dcut_runtime_timing", None)
        start_s = time.perf_counter() if timing is not None else 0.0
        result = _orig_sample_core(self, *a, **k)
        if timing is not None:
            self._dcut_step_sample_core_ms += (
                time.perf_counter() - start_s
            ) * 1e3
        return result

    def _bookkeeping_sync(self, *a, **k):
        timing = getattr(self, "_dcut_runtime_timing", None)
        start_s = time.perf_counter() if timing is not None else 0.0
        result = _orig_bookkeeping_sync(self, *a, **k)
        if timing is not None:
            self._dcut_step_bookkeeping_ms += (
                time.perf_counter() - start_s
            ) * 1e3
        return result

    def propose_draft_token_ids(self, *a, **k):
        timing = getattr(self, "_dcut_runtime_timing", None)
        start_s = time.perf_counter() if timing is not None else 0.0
        result = _orig_propose_draft_token_ids(self, *a, **k)
        if timing is not None:
            self._dcut_step_draft_model_ms += (
                time.perf_counter() - start_s
            ) * 1e3
        return result

    def sample_tokens(self, *a, **k):
        timing = getattr(self, "_dcut_runtime_timing", None)
        sample_start_s = (
            time.perf_counter() if timing is not None else 0.0
        )
        out = _orig_sample_tokens(self, *a, **k)
        sample_end_s = (
            time.perf_counter() if timing is not None else 0.0
        )
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return out

        decision_start_s = (
            time.perf_counter() if timing is not None else 0.0
        )
        if getattr(self, "_adaptive_probs_pending", False):
            try:
                _maybe_process_adaptive_probs(self)
            except Exception as e:
                logger.warning("D-Cut: process probs failed: %s", e)
                self._adaptive_probs_pending = False
        decision_end_s = (
            time.perf_counter() if timing is not None else 0.0
        )

        if timing is not None:
            drain_start_s = time.perf_counter()
            self._sync_device()
            step_end_s = time.perf_counter()
            execute_end_s = timing.get(
                "execute_end_s", sample_start_s
            )
            sample_tokens_ms = (
                sample_end_s - sample_start_s
            ) * 1e3
            sample_core_ms = getattr(
                self, "_dcut_step_sample_core_ms", 0.0
            )
            bookkeeping_ms = getattr(
                self, "_dcut_step_bookkeeping_ms", 0.0
            )
            draft_model_ms = getattr(
                self, "_dcut_step_draft_model_ms", 0.0
            )
            draft_copy_ms = getattr(
                self, "_dcut_step_copy_ms", 0.0
            )
            prob_queue_ms = getattr(
                self, "_dcut_step_prob_queue_ms", 0.0
            )
            sample_other_ms = max(
                sample_tokens_ms
                - sample_core_ms
                - bookkeeping_ms
                - draft_model_ms
                - draft_copy_ms
                - prob_queue_ms,
                0.0,
            )
            runner_step_s = step_end_s - timing["step_start_s"]
            scheduler_gap_s = max(
                timing.get("inter_step_gap_ms", 0.0) / 1e3,
                0.0,
            )
            full_iteration_s = runner_step_s + scheduler_gap_s
            components_ms = {
                "full_iteration_total": full_iteration_s * 1e3,
                "scheduler_and_ipc_gap": scheduler_gap_s * 1e3,
                "runner_step_total": runner_step_s * 1e3,
                "dcut_prepare": timing.get("dcut_prepare_ms", 0.0),
                "target_execute_total": timing.get(
                    "execute_wall_ms", 0.0
                ),
                "execute_to_sample_gap": (
                    sample_start_s - execute_end_s
                ) * 1e3,
                "sample_tokens_total": sample_tokens_ms,
                "sampling": sample_core_ms,
                "bookkeeping": bookkeeping_ms,
                "draft_model": draft_model_ms,
                "draft_id_copy": draft_copy_ms,
                "selected_probs_queue": prob_queue_ms,
                "sample_tokens_other": sample_other_ms,
                "adaptive_decision": (
                    decision_end_s - decision_start_s
                ) * 1e3,
                "device_drain": (step_end_s - drain_start_s) * 1e3,
            }
            ctrl = getattr(self, "_verify_adaptive_controller", None)
            if ctrl is not None:
                ctrl.observe_runtime_step(
                    batch_size=timing["batch_size"],
                    sum_query_len=timing["sum_query_len"],
                    mode=timing["mode"],
                    full_step_s=full_iteration_s,
                    components_ms=components_ms,
                )
            self._dcut_runtime_timing = None
            self._dcut_prev_sample_end = time.perf_counter()
        else:
            self._dcut_prev_sample_end = None
        return out

    _orig_copy = R._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        timing = getattr(self, "_dcut_runtime_timing", None)
        copy_start_s = time.perf_counter() if timing is not None else 0.0
        _orig_copy(self, scheduler_output, zeros_only)
        if timing is not None:
            self._dcut_step_copy_ms = (
                time.perf_counter() - copy_start_s
            ) * 1e3
        if os.environ.get(ENV_FULL_DECODE_ONLY):
            return
        if getattr(self, "_verify_adaptive_controller", None) is not None:
            queue_start_s = (
                time.perf_counter() if timing is not None else 0.0
            )
            try:
                _dcut_queue_probs(self, zeros_only)
            except Exception as e:
                logger.warning("D-Cut: queue probs failed: %s", e)
            finally:
                if timing is not None:
                    self._dcut_step_prob_queue_ms = (
                        time.perf_counter() - queue_start_s
                    ) * 1e3

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
    R._sample = _sample
    R._bookkeeping_sync = _bookkeeping_sync
    R.propose_draft_token_ids = propose_draft_token_ids
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

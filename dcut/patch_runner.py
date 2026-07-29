# SPDX-License-Identifier: Apache-2.0
"""Patch NPUModelRunner for D-Cut adaptive verify."""
from __future__ import annotations

import os

from .controller import _dcut_init_controller, _dcut_enable_drafter_probs
from .dcut_profile import _adaptive_profile_run
from .globals import ENV_FULL_DECODE_ONLY, logger
from .probs import (
    _dcut_queue_probs,
    _maybe_process_adaptive_probs,
    profile_adaptive_cost,
)
from .truncate import _dcut_truncate

ENV_DEBUG_STATS = "VLLM_DCUT_DEBUG_STATS"


def _patch_runner() -> None:
    import vllm_ascend.worker.model_runner_v1 as m

    R = m.NPUModelRunner
    if getattr(R, "_dcut_patched", False):
        return

    _orig_init = R.__init__

    def __init__(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            _dcut_init_controller(self)
        except Exception as e:
            logger.error("D-Cut init failed; running vanilla: %s", e)
            self._verify_adaptive_controller = None

    _orig_exec = R.execute_model

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

    # Patch _model_forward to update GDN static buffers before graph replay
    _orig_model_forward = R._model_forward

    def _model_forward(self, *args, **kwargs):
        # Update GDN static buffers before graph replay (if in PIECEWISE mode)
        if os.environ.get("VLLM_DCUT_GDN_PIECEWISE") == "1":
            try:
                from vllm.forward_context import get_forward_context
                from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
                from .gdn_buffers import _dcut_update_gdn_static

                forward_context = get_forward_context()
                # Extract num_tokens from args or kwargs
                if args:
                    num_tokens = args[0] if args else kwargs.get('num_tokens_padded', 0)
                else:
                    num_tokens = kwargs.get('num_tokens_padded', 0)

                _dcut_update_gdn_static(forward_context, num_tokens, GDNAttentionMetadata)
            except Exception as e:
                logger.warning(f"D-Cut: failed to update GDN static buffers: {e}")

        return _orig_model_forward(self, *args, **kwargs)

    R.__init__ = __init__
    R.execute_model = execute_model
    R.sample_tokens = sample_tokens
    R._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    R._update_states = _update_states
    R._model_forward = _model_forward
    R._adaptive_profile_run = _adaptive_profile_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_enable_drafter_probs = _dcut_enable_drafter_probs
    R._dcut_patched = True

    logger.info(
        "D-Cut: using the native vLLM 0.23 PIECEWISE GDN splitting path."
    )

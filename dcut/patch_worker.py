# SPDX-License-Identifier: Apache-2.0
"""Patch NPUWorker: warmup hook + cost profiling trigger."""
from __future__ import annotations

from .globals import logger

def _patch_worker() -> None:
    import vllm_ascend.worker.worker as m

    W = m.NPUWorker
    if getattr(W, "_dcut_patched", False):
        return

    _orig = W.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *a, **k):
        runner = getattr(self, "model_runner", None)
        if runner is not None and hasattr(runner, "_dcut_enable_drafter_probs"):
            try:
                runner._dcut_enable_drafter_probs()
            except Exception as e:
                logger.warning("D-Cut: enabling draft probs before warmup failed: %s", e)
        # Mark that we're in the REAL warmup (not profile_cudagraph_memory).
        # _build_attention_metadata patch checks this to force use_spec_decode=True
        # only during real warmup, not during profile_cudagraph_memory (which uses
        # a minimal KV cache that can't support spec-decode conv1d).
        if runner is not None:
            runner._dcut_in_real_warmup = True
        try:
            ret = _orig(self, *a, **k)
            if runner is not None and hasattr(runner, "profile_adaptive_cost"):
                try:
                    runner.profile_adaptive_cost()
                except Exception as e:
                    # Empty cost table => controller no-ops => graceful fall back to
                    # vanilla DFlash (full-length verify).
                    import traceback
                    logger.error("D-Cut: cost profiling failed; falling back: %s", e)
                    logger.error("D-Cut: full traceback: %s", traceback.format_exc())
                    ctrl = getattr(runner, "_verify_adaptive_controller", None)
                    if ctrl is not None:
                        ctrl._cost_table.clear()
                        ctrl._sorted_bs.clear()
                        ctrl._sorted_sql_per_bs.clear()
        finally:
            if runner is not None:
                runner._dcut_in_real_warmup = False
        return ret

    W.compile_or_warm_up_model = compile_or_warm_up_model
    W._dcut_patched = True
    logger.info("D-Cut: patched NPUWorker.")



# SPDX-License-Identifier: Apache-2.0
"""D-Cut patch application + vLLM general-plugin entrypoint."""
from __future__ import annotations
import os

import torch

from .globals import logger, ENABLE_GDN_MAIN_PIECEWISE_GRAPH, ENV_CONFIG
from .patch_proposer import _patch_proposer
from .patch_runner import _patch_runner
from .patch_worker import _patch_worker
from .patch_attention import _patch_attention
from .patch_gdn import _patch_gdn_dcut

def _apply_patches_once() -> None:
    """Apply the real monkey patches.  Runs once per process, deferred to the
    first worker construction (see ``install``) so that importing the NPU
    worker/runner/proposer modules is safe."""
    from . import globals as _g
    if _g._PATCHED:
        return
    # Mark done up-front so a failure (e.g. non-Ascend platform) is not retried
    # on every subsequent worker construction and does not spam the log.
    _g._PATCHED = True
    try:
        import sys as _dbg
        _patch_proposer()
        _patch_runner()
        _patch_worker()
        _patch_attention()
        # Always inject _conv1d_spec_varlen_eager into vllm_ascend.ops.gdn
        # so it's available as a module-level function (needed by other code
        # that imports from vllm_ascend.ops.gdn).
        try:
            from .gdn_eager import _conv1d_spec_varlen_eager
            import vllm_ascend.ops.gdn as _gdn_mod
            _gdn_mod._conv1d_spec_varlen_eager = _conv1d_spec_varlen_eager
            logger.info('D-Cut: injected _conv1d_spec_varlen_eager into vllm_ascend.ops.gdn')
        except Exception as e:
            logger.warning('D-Cut: failed to inject _conv1d_spec_varlen_eager: %s', e)
        if os.environ.get(ENV_CONFIG):
            _patch_gdn_dcut()
        logger.info(
            "D-Cut adaptive-verify patches applied for NPU "
            "(active only if VLLM_DCUT_CONFIG is set + method is dflash/PARD)."
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut patching failed (vLLM continues normally): %s", e)


def install(*args, **kwargs) -> None:
    """vLLM general-plugin entrypoint.  Idempotent; safe to call per process.

    IMPORTANT — deferred by design.  ``install`` runs during *general-plugin
    load*, which happens BEFORE vllm-ascend has finished importing its own
    ``ops/fused_moe`` / ``device`` graph.  Eagerly importing the NPU
    worker/runner/proposer modules here re-enters that partially-initialised
    graph and raises a circular ``ImportError`` that poisons ``sys.modules`` —
    which then breaks vllm-ascend's *own* later imports (e.g.
    ``pre_register_and_update`` -> ``select_experts``), taking down even vanilla
    serving.  So here we only *arm* a deferred trigger on the vLLM-core
    ``WorkerBase`` (safe to import at this point) and apply the real patches on
    the first worker construction, by which time vllm-ascend is fully imported.
    """
    from . import globals as _g
    if _g._INSTALLED:
        return
    _g._INSTALLED = True
    try:
        # GDN removal from _attention_ops — putting GDN inside the PIECEWISE
        # graph so its compute cost is captured in the cost table.
        # This runs in install() which is BEFORE set_splitting_ops_for_v1()
        # copies _attention_ops into splitting_ops. Without this,
        # set_splitting_ops_for_v1() re-adds GDN to splitting_ops even
        # after the __init__ patch removes it.
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            try:
                from vllm.config import CompilationConfig
                ops = CompilationConfig._attention_ops
                if isinstance(ops, list) and "vllm::qwen_gdn_attention_core" in ops:
                    CompilationConfig._attention_ops = [
                        op for op in ops if op != "vllm::qwen_gdn_attention_core"
                    ]
                    logger.warning(
                        "D-Cut: removed qwen_gdn_attention_core from _attention_ops "
                        "(class-level, in install())"
                    )
            except Exception as e:
                logger.warning(
                    "D-Cut: failed to remove GDN from _attention_ops in install(): %s", e
                )

        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            # Patch GDNAttentionMetadataBuilder.build_for_cudagraph_capture to
            # skip the decode-only assertion.  The assertion checks
            # num_actual_tokens <= decode_cudagraph_max_bs (256), which is only
            # valid for FULL cudagraph mode (decode-only).  In PIECEWISE mode,
            # GDN is inside the graph and needs attn_metadata built during
            # capture with larger token counts (up to max capture size).
            # The method body after the assertion is just self.build(...), so
            # skipping the assertion is safe.
            try:
                from vllm.v1.attention.backends.gdn_attn import (
                    GDNAttentionMetadataBuilder,
                )

                _orig_build_cg = GDNAttentionMetadataBuilder.build_for_cudagraph_capture

                def _build_for_cudagraph_capture(self, common_attn_metadata):
                    m = common_attn_metadata
                    num_accepted_tokens = torch.diff(m.query_start_loc)
                    num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
                    return self.build(
                        0, m, num_accepted_tokens, num_decode_draft_tokens_cpu
                    )

                GDNAttentionMetadataBuilder.build_for_cudagraph_capture = (
                    _build_for_cudagraph_capture
                )
                logger.warning(
                    "D-Cut: patched GDNAttentionMetadataBuilder."
                    "build_for_cudagraph_capture to skip decode-only assertion "
                    "(needed for PIECEWISE mode with GDN in graph)."
                )
            except Exception as e:
                logger.warning(
                    "D-Cut: failed to patch GDN build_for_cudagraph_capture: %s", e
                )

        else:
            logger.info("D-Cut: skipped GDN build_for_cudagraph_capture patch (ENABLE_GDN_MAIN_PIECEWISE_GRAPH=False).")
        from vllm.v1.worker.worker_base import WorkerBase

        if getattr(WorkerBase, "_dcut_defer_armed", False):
            return
        _orig_wb_init = WorkerBase.__init__

        def __init__(self, *a, **k):
            # NPUWorker.__init__ calls super().__init__() (this) early, before it
            # builds the model runner — so patching here lands before any
            # NPUModelRunner / proposer instance exists.
            _apply_patches_once()
            return _orig_wb_init(self, *a, **k)

        WorkerBase.__init__ = __init__
        WorkerBase._dcut_defer_armed = True
        logger.info(
            "D-Cut deferred installer armed on WorkerBase "
            "(patches apply on first worker init to avoid a vllm-ascend "
            "circular import)."
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut install (arm) failed (vLLM continues normally): %s", e)

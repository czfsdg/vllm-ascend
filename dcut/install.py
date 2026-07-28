# SPDX-License-Identifier: Apache-2.0
"""D-Cut patch application + vLLM general-plugin entrypoint."""

from __future__ import annotations

import os

from .globals import ENV_CONFIG, logger
from .patch_gdn_v023 import _enable_gdn_piecewise_graph, _patch_gdn_dcut
from .patch_proposer import _patch_proposer
from .patch_runner import _patch_runner
from .patch_worker import _patch_worker


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
        if os.environ.get(ENV_CONFIG) and not _patch_gdn_dcut():
            raise RuntimeError(
                "D-Cut GDN state operators are unavailable; run `bash dcut/kernel/build.sh` first"
            )
        _patch_proposer()
        _patch_runner()
        _patch_worker()
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
        if os.environ.get(ENV_CONFIG) and not _enable_gdn_piecewise_graph():
            raise RuntimeError(
                "D-Cut could not configure GDN capture for PIECEWISE ACLGraph"
            )

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

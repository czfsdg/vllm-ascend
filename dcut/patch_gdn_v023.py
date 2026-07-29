# SPDX-License-Identifier: Apache-2.0
"""Install graph-capturable D-Cut GDN operators for vLLM 0.23."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .globals import logger

ENV_TORCH_OP_LIBRARY = "VLLM_DCUT_TORCH_OP_LIBRARY"
_REQUIRED_OPS = (
    "npu_dcut_causal_conv1d",
    "npu_dcut_recurrent_gated_delta_rule",
)
GDN_PIECEWISE_SPLITTING_OP = "vllm::qwen_gdn_attention_core"


def _gdn_piecewise_graph_enabled() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE


def _without_gdn_piecewise_split(ops: list[str]) -> list[str]:
    """Keep all PIECEWISE boundaries except the graph-safe GDN core."""
    return [op for op in ops if op != GDN_PIECEWISE_SPLITTING_OP]


def _enable_gdn_piecewise_graph(vllm_config=None) -> bool:
    """Conditionally make the GDN core part of its PIECEWISE ACLGraph.

    vLLM adds ``qwen_gdn_attention_core`` to the default attention splitting
    ops. A splitting op is deliberately executed between graph pieces, so
    patching only its Python implementation can never put GDN in the graph.

    Update both the class default (before ``VllmConfig`` finalization) and an
    optional live config (worker-side fallback). The D-Cut kernels have Meta
    implementations and explicit mutation aliases, and vLLM 0.23 owns the
    stable graph inputs through ``GDNSpecDecodeMetadata``.
    """
    if not _gdn_piecewise_graph_enabled():
        if vllm_config is None:
            logger.info(
                "D-Cut: GDN PIECEWISE graph capture is disabled "
                "(VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE=0); "
                "preserving %s in splitting_ops.",
                GDN_PIECEWISE_SPLITTING_OP,
            )
        return True
    try:
        from vllm.config.compilation import CompilationConfig
    except Exception as exc:  # pragma: no cover - depends on runtime imports
        logger.warning("D-Cut: cannot configure PIECEWISE GDN capture: %s", exc)
        return False

    default_ops = getattr(CompilationConfig, "_attention_ops", None)
    if default_ops is not None:
        CompilationConfig._attention_ops = _without_gdn_piecewise_split(
            list(default_ops)
        )

    if vllm_config is None:
        return True

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        logger.warning(
            "D-Cut: VllmConfig has no compilation_config; "
            "PIECEWISE GDN capture was not enabled."
        )
        return False

    splitting_ops = getattr(compilation_config, "splitting_ops", None)
    if splitting_ops is not None:
        compilation_config.splitting_ops = _without_gdn_piecewise_split(
            list(splitting_ops)
        )

    configured_ops = getattr(compilation_config, "splitting_ops", None)
    if configured_ops is not None and GDN_PIECEWISE_SPLITTING_OP in configured_ops:
        logger.error(
            "D-Cut: failed to remove %s from PIECEWISE splitting_ops.",
            GDN_PIECEWISE_SPLITTING_OP,
        )
        return False

    logger.info(
        "D-Cut: GDN core is graph-capturable in PIECEWISE mode "
        "(VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE=1; "
        "removed %s from splitting_ops).",
        GDN_PIECEWISE_SPLITTING_OP,
    )
    return True


def _ops_registered() -> bool:
    return all(hasattr(torch.ops._C_ascend, name) for name in _REQUIRED_OPS)


def _load_dcut_torch_ops() -> bool:
    """Load the D-Cut-only Torch registration library before GDN execution."""
    if _ops_registered():
        return True

    configured_path = os.environ.get(ENV_TORCH_OP_LIBRARY)
    if configured_path:
        candidates = (Path(configured_path).expanduser(),)
    else:
        candidates = (
            Path(__file__).resolve().parent
            / "kernel"
            / "build"
            / "torch_extension"
            / "dcut_torch_ops.so",
        )

    load_errors: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            load_errors.append(f"{candidate} (not found)")
            continue
        try:
            torch.ops.load_library(str(candidate))
        except (OSError, RuntimeError) as exc:
            load_errors.append(f"{candidate} ({exc})")
            continue
        if _ops_registered():
            return True
        load_errors.append(f"{candidate} (loaded, but schemas are missing)")

    logger.error(
        "D-Cut: custom GDN Torch operators are unavailable: %s. "
        "Build them with `bash dcut/kernel/build.sh` or set %s.",
        "; ".join(load_errors),
        ENV_TORCH_OP_LIBRARY,
    )
    return False


def _patch_gdn_dcut() -> bool:
    """Replace ``_forward_core`` used by the vLLM GDN custom op."""
    try:
        from vllm_ascend.ops import gdn as ascend_gdn
        from vllm_ascend.patch.worker import patch_qwen3_5 as qwen_patch
        from vllm_ascend.utils import is_310p
    except Exception as exc:  # pragma: no cover - depends on runtime imports
        logger.warning("D-Cut: cannot import vLLM 0.23 GDN symbols: %s", exc)
        return False

    if is_310p():
        logger.warning("D-Cut: variable-length GDN verification is not enabled on 310P.")
        return False

    target_class = qwen_patch._GDN_PATCH_TARGET
    if getattr(target_class._forward_core, "_dcut_patched", False):
        return True

    if not _load_dcut_torch_ops():
        return False

    from .gdn_forward_v023 import AscendGatedDeltaNetAttention as DcutGatedDeltaNetAttention

    dcut_forward_core = DcutGatedDeltaNetAttention._forward_core
    dcut_forward_core._dcut_patched = True  # type: ignore[attr-defined]

    # ``forward`` remains the vllm-ascend implementation and still enters
    # torch.ops.vllm.qwen_gdn_attention_core. The environment switch controls
    # whether the installer removes this op from PIECEWISE splitting_ops; the
    # default preserves the boundary and executes the GDN custom op eagerly.
    ascend_gdn.AscendGatedDeltaNetAttention._forward_core = dcut_forward_core
    target_class._forward_core = dcut_forward_core
    logger.info(
        "D-Cut: enabled independent recurrent/conv state selection for "
        "the vLLM 0.23 GDN core."
    )
    return True

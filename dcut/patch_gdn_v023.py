# SPDX-License-Identifier: Apache-2.0
"""Install the graph-capturable D-Cut GDN core for vLLM 0.23."""

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


def _ops_registered() -> bool:
    return all(hasattr(torch.ops._C_ascend, name) for name in _REQUIRED_OPS)


def _load_dcut_torch_ops() -> bool:
    """Load the D-Cut-only Torch registration library before graph capture."""
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
    """Replace ``_forward_core`` while preserving the native custom-op API."""
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
    # torch.ops.vllm.qwen_gdn_attention_core. The early config patch decides
    # whether that custom op is an eager boundary or part of a PIECEWISE graph.
    ascend_gdn.AscendGatedDeltaNetAttention._forward_core = dcut_forward_core
    target_class._forward_core = dcut_forward_core
    logger.info(
        "D-Cut: enabled independent recurrent/conv state selection for "
        "the vLLM 0.23 GDN core."
    )
    return True

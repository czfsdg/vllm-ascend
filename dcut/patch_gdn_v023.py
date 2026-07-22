# SPDX-License-Identifier: Apache-2.0
"""Adapt the vLLM 0.23 GDN speculative path for variable draft lengths."""
from __future__ import annotations

import torch

from .globals import logger


def _compact_spec_state_indices(
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact padded GDN state indices to the scheduled tokens.

    vLLM 0.23 keeps one num_speculative_tokens + 1 row per speculative
    request. D-Cut can schedule fewer tokens for an individual request, so
    flattening those rows would pass stale trailing state indices to the
    recurrent GDN operator. Build the flattened indices from the live query
    lengths and clamp the accepted-token count to the same lengths.

    This helper intentionally stays on device; using item() here would add a
    CPU/NPU synchronization to every speculative decode step.
    """
    active_state_indices = state_indices[:num_spec_decodes]
    query_lens = (
        query_start_loc[1 : num_spec_decodes + 1]
        - query_start_loc[:num_spec_decodes]
    )

    if active_state_indices.ndim == 1:
        compact_state_indices = active_state_indices
    else:
        token_offsets = torch.arange(
            active_state_indices.shape[1],
            device=active_state_indices.device,
        )
        valid_tokens = token_offsets.unsqueeze(0) < query_lens.unsqueeze(1)
        compact_state_indices = active_state_indices[valid_tokens].contiguous()

    clamped_num_accepted_tokens = torch.minimum(
        num_accepted_tokens[:num_spec_decodes].to(torch.int32),
        query_lens.to(torch.int32),
    )
    return compact_state_indices, clamped_num_accepted_tokens


def _patch_gdn_dcut() -> None:
    """Patch the vLLM 0.23 GDN implementation at its eager splitting op.

    The 0.22 implementation used graph-task host arguments and several static
    buffers. Those interfaces were removed in 0.23, which now owns graph input
    lifetime through GDNSpecDecodeMetadata. The only D-Cut-specific fix still
    required is compacting per-request state indices after verifier truncation.
    """
    try:
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm_ascend.utils import is_310p
    except Exception as exc:  # pragma: no cover - depends on runtime imports
        logger.warning("D-Cut: cannot import vLLM 0.23 GDN symbols: %s", exc)
        return

    if is_310p():
        logger.warning(
            "D-Cut: variable-length GDN verification is not enabled on 310P."
        )
        return

    target = QwenGatedDeltaNetAttention
    if getattr(target, "_dcut_gdn_patched", False):
        return

    original_forward_core = target._forward_core

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        forward_context = get_forward_context()
        per_layer_metadata = getattr(forward_context, "attn_metadata", None)
        if not isinstance(per_layer_metadata, dict):
            return original_forward_core(self, mixed_qkv, b, a, core_attn_out)

        attn_metadata = per_layer_metadata.get(self.prefix)
        if not isinstance(attn_metadata, GDNAttentionMetadata):
            return original_forward_core(self, mixed_qkv, b, a, core_attn_out)

        spec_metadata = getattr(attn_metadata, "spec_decode_metadata", None)
        state_indices = getattr(attn_metadata, "spec_state_indices_tensor", None)
        num_spec_decodes = int(getattr(attn_metadata, "num_spec_decodes", 0))
        if spec_metadata is None or state_indices is None or num_spec_decodes <= 0:
            return original_forward_core(self, mixed_qkv, b, a, core_attn_out)

        conv_metadata = spec_metadata.spec_causal_conv1d
        original_state_indices = attn_metadata.spec_state_indices_tensor
        original_num_accepted_tokens = conv_metadata.num_accepted_tokens
        try:
            (
                attn_metadata.spec_state_indices_tensor,
                conv_metadata.num_accepted_tokens,
            ) = _compact_spec_state_indices(
                original_state_indices,
                conv_metadata.query_start_loc,
                original_num_accepted_tokens,
                num_spec_decodes,
            )
            return original_forward_core(self, mixed_qkv, b, a, core_attn_out)
        finally:
            # Metadata is shared by all GDN layers in a cache group. Restore it
            # after this layer so later consumers still see the canonical 0.23
            # representation.
            attn_metadata.spec_state_indices_tensor = original_state_indices
            conv_metadata.num_accepted_tokens = original_num_accepted_tokens

    target._forward_core = _forward_core
    target._dcut_gdn_patched = True
    logger.info(
        "D-Cut: patched vLLM 0.23 GDN state-index compaction for variable "
        "verifier lengths."
    )

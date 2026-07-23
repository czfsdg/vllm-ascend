# SPDX-License-Identifier: Apache-2.0
"""Adapt the vLLM 0.23 GDN speculative path for variable draft lengths."""
from __future__ import annotations

import torch

from .globals import logger


def _rebase_spec_gdn_states(
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
) -> torch.Tensor:
    """Commit each request's accepted speculative state to slot zero.

    vLLM normally carries the previous acceptance count into the next GDN
    forward and lets both state kernels select that offset. A later D-Cut can
    be shorter than that count, so the selector is no longer part of the live
    variable-length row. Move the selected recurrent state and convolution
    window to slot zero before running the new row, then use an acceptance
    selector of one. This preserves the exact accepted state without imposing
    a lower bound on the next cut.
    """
    active_state_indices = state_indices[:num_spec_decodes]
    active_num_accepted = num_accepted_tokens[:num_spec_decodes].to(torch.int64)
    if active_state_indices.ndim != 2 or active_state_indices.shape[1] == 0:
        return torch.ones_like(active_num_accepted, dtype=torch.int32)

    row_width = active_state_indices.shape[1]
    accepted_offsets = active_num_accepted.clamp(1, row_width) - 1
    row_indices = torch.arange(
        num_spec_decodes,
        device=active_state_indices.device,
    )
    destination_indices = active_state_indices[:, 0].to(torch.int64)
    source_indices = active_state_indices[
        row_indices,
        accepted_offsets,
    ].to(torch.int64)

    accepted_recurrent_state = recurrent_state.index_select(
        0,
        source_indices,
    ).clone()
    recurrent_state.index_copy_(
        0,
        destination_indices,
        accepted_recurrent_state,
    )

    active_conv_state = conv_state.index_select(
        0,
        destination_indices,
    ).clone()
    state_length = active_conv_state.shape[1]
    state_positions = torch.arange(
        state_length,
        device=conv_state.device,
    ).unsqueeze(0)
    source_positions = state_positions + accepted_offsets.unsqueeze(1)
    valid_positions = source_positions < state_length
    source_positions = source_positions.clamp(max=state_length - 1)
    shifted_conv_state = torch.gather(
        active_conv_state,
        1,
        source_positions.unsqueeze(-1).expand_as(active_conv_state),
    )
    rebased_conv_state = torch.where(
        valid_positions.unsqueeze(-1),
        shifted_conv_state,
        active_conv_state,
    )
    conv_state.index_copy_(
        0,
        destination_indices,
        rebased_conv_state,
    )

    return torch.ones_like(active_num_accepted, dtype=torch.int32)


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
    lengths. The accepted state has already been committed to slot zero by
    ``_rebase_spec_gdn_states``, so the normalized selector is always part of
    the live row.

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

    active_num_accepted_tokens = torch.minimum(
        num_accepted_tokens[:num_spec_decodes].to(torch.int32),
        query_lens.to(torch.int32),
    )
    return compact_state_indices, active_num_accepted_tokens


def _run_padded_spec_causal_conv1d(
    original_run,
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    activation_mode: int,
) -> None:
    """Pad variable D-Cut rows to the fixed CANN speculative stride.

    ``aclnnCausalConv1d`` run mode 1 stores a fixed speculative state row per
    request. Keep one batched CANN invocation by packing live tokens into that
    layout, then gather only live outputs. All mapping stays on device.
    """
    if cache_indices.ndim != 2:
        original_run(
            output,
            x,
            weight,
            conv_state,
            bias,
            query_start_loc,
            cache_indices,
            num_accepted_tokens,
            activation_mode,
        )
        return

    num_requests, row_width = cache_indices.shape
    padded_token_count = num_requests * row_width
    if x.shape[0] == padded_token_count:
        original_run(
            output,
            x,
            weight,
            conv_state,
            bias,
            query_start_loc,
            cache_indices,
            num_accepted_tokens,
            activation_mode,
        )
        return

    query_lens = query_start_loc[1 : num_requests + 1] - query_start_loc[:num_requests]
    request_indices = torch.repeat_interleave(
        torch.arange(
            num_requests,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        ),
        query_lens,
        output_size=x.shape[0],
    )
    packed_positions = torch.arange(
        x.shape[0],
        dtype=query_start_loc.dtype,
        device=query_start_loc.device,
    )
    token_offsets = packed_positions - query_start_loc.index_select(
        0,
        request_indices.to(torch.int64),
    )
    padded_positions = (request_indices * row_width + token_offsets).to(torch.int64)

    padded_x = x.new_zeros((padded_token_count, x.shape[1]))
    padded_x.index_copy_(0, padded_positions, x)
    padded_output = torch.empty_like(padded_x)
    padded_query_start_loc = torch.arange(
        0,
        padded_token_count + 1,
        row_width,
        dtype=query_start_loc.dtype,
        device=query_start_loc.device,
    )
    original_run(
        padded_output,
        padded_x,
        weight,
        conv_state,
        bias,
        padded_query_start_loc,
        cache_indices,
        num_accepted_tokens,
        activation_mode,
    )
    output.copy_(padded_output.index_select(0, padded_positions))


def _patch_gdn_dcut() -> None:
    """Patch the vLLM 0.23 GDN implementation at its eager splitting op.

    The 0.22 implementation used graph-task host arguments and several static
    buffers. Those interfaces were removed in 0.23, which now owns graph input
    lifetime through GDNSpecDecodeMetadata. D-Cut must commit the previous
    accepted state, compact recurrent state indices, and pad the causal-conv
    input back to its fixed speculative stride.
    """
    try:
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm_ascend.utils import is_310p
        import vllm_ascend.ops.gdn as ascend_gdn
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
    original_spec_causal_conv1d = ascend_gdn._run_spec_causal_conv1d

    def _dcut_spec_causal_conv1d(*args, **kwargs):
        return _run_padded_spec_causal_conv1d(
            original_spec_causal_conv1d,
            *args,
            **kwargs,
        )

    ascend_gdn._run_spec_causal_conv1d = _dcut_spec_causal_conv1d

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
            normalized_num_accepted_tokens = _rebase_spec_gdn_states(
                self.kv_cache[0],
                self.kv_cache[1],
                original_state_indices,
                original_num_accepted_tokens,
                num_spec_decodes,
            )
            (
                attn_metadata.spec_state_indices_tensor,
                conv_metadata.num_accepted_tokens,
            ) = _compact_spec_state_indices(
                original_state_indices,
                conv_metadata.query_start_loc,
                normalized_num_accepted_tokens,
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
        "D-Cut: patched vLLM 0.23 GDN state commit, recurrent compaction, "
        "and fixed-stride causal-conv packing."
    )

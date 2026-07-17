# SPDX-License-Identifier: Apache-2.0
"""GDN static buffer alloc/fill/update for PIECEWISE graph replay."""
from __future__ import annotations

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .globals import logger, _dcut_gdn_static


def _dcut_alloc_gdn_spec_bufs(prefix, num_tokens, spec_state_indices_tensor, device):
    """Allocate per-layer ASL/SSI/NAT buffers for the spec decode path.

    Do not build SSI from conv1d host tuples: conv1d cache_indices are not the
    recurrent GDN per-token SSM state indices.  The correct SSI source is the
    per-layer ``spec_state_indices_tensor`` in GDN metadata, matching the
    original full-decode path.
    """
    key = (prefix, num_tokens, "spec")
    if key not in _dcut_gdn_static:
        b_cap = spec_state_indices_tensor.size(0)
        nsp1 = spec_state_indices_tensor.size(1)  # num_spec + 1
        t_cap = b_cap * nsp1
        _dcut_gdn_static[key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "ssi": torch.full((t_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "nat": torch.zeros(b_cap, dtype=torch.int32, device=device),
            "col_idx": torch.arange(nsp1, device=device),
            "b_cap": b_cap,
            "nsp1": nsp1,
            "t_cap": t_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN spec static bufs prefix=%s num_tokens=%d "
            "b_cap=%d t_cap=%d", prefix, num_tokens, b_cap, t_cap)
    return _dcut_gdn_static[key]


def _dcut_fill_gdn_spec_bufs(prefix, num_tokens, spec_query_start_loc,
                              spec_state_indices_tensor, num_accepted_tokens,
                              num_spec_decodes, device):
    """Fill ASL/SSI/NAT buffers with runtime values before graph replay."""
    bufs = _dcut_alloc_gdn_spec_bufs(
        prefix, num_tokens, spec_state_indices_tensor, device)
    asl, ssi, nat = bufs["asl"], bufs["ssi"], bufs["nat"]

    # Keep the conservative, correctness-first path: all three tensors are
    # prefix-local and rebuilt from GDN metadata.  This costs NPU fill kernels,
    # but avoids the invalid-state-index crash caused by treating conv1d host
    # cache indices as recurrent SSI.
    asl.zero_()
    ssi.fill_(PAD_SLOT_ID)
    nat.zero_()

    if num_spec_decodes > 0:
        cu = spec_query_start_loc[:num_spec_decodes + 1]
        per_seq_lens = cu[1:] - cu[:-1]

        asl[1:num_spec_decodes + 1].copy_(per_seq_lens)

        col_idx = bufs["col_idx"]
        mask = col_idx.unsqueeze(0) < per_seq_lens.unsqueeze(1)
        real = spec_state_indices_tensor[:num_spec_decodes][mask]
        ssi[:real.size(0)].copy_(real)

        clamped = torch.minimum(
            num_accepted_tokens[:num_spec_decodes].to(torch.int32),
            per_seq_lens.to(torch.int32)
        )
        nat[:num_spec_decodes].copy_(clamped)
    return bufs


def _dcut_alloc_gdn_nonspec_bufs(prefix, num_tokens,
                                  non_spec_state_indices_tensor, device):
    """Allocate per-layer ASL/SSI buffers for the non-spec decode path."""
    key = (prefix, num_tokens, "nonspec")
    if key not in _dcut_gdn_static:
        b_cap = non_spec_state_indices_tensor.size(0)
        _dcut_gdn_static[key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "ssi": torch.full((b_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN nonspec static bufs prefix=%s num_tokens=%d "
            "b_cap=%d", prefix, num_tokens, b_cap)
    return _dcut_gdn_static[key]


def _dcut_fill_gdn_nonspec_bufs(prefix, num_tokens, non_spec_query_start_loc,
                                  non_spec_state_indices_tensor, num_decodes,
                                  device):
    """Fill ASL/SSI buffers for non-spec decode before graph replay."""
    bufs = _dcut_alloc_gdn_nonspec_bufs(
        prefix, num_tokens, non_spec_state_indices_tensor, device)
    asl, ssi = bufs["asl"], bufs["ssi"]

    asl.zero_()
    if num_decodes > 0:
        cu = non_spec_query_start_loc[:num_decodes + 1]
        asl[1:num_decodes + 1].copy_(cu[1:] - cu[:-1])

    ssi.fill_(PAD_SLOT_ID)
    if num_decodes > 0:
        ssi[:num_decodes].copy_(non_spec_state_indices_tensor[:num_decodes])
    return bufs


def _dcut_update_gdn_static(forward_context, num_tokens, GDNAttentionMetadata):
    """Update GDN static buffers from forward context's attn_metadata.

    Called from patched _model_forward before _orig_model_forward, i.e. before
    graph replay.  The recurrent GDN operator currently takes tensor metadata,
    not host tuple optional args like conv1d, so correctness requires preserving
    per-layer tensors here.
    """
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return
    for prefix, meta in attn_metadata.items():
        if not isinstance(meta, GDNAttentionMetadata):
            continue
        if meta.spec_sequence_masks is not None and meta.num_spec_decodes > 0:
            _dcut_fill_gdn_spec_bufs(
                prefix, num_tokens,
                meta.spec_query_start_loc,
                meta.spec_state_indices_tensor,
                meta.num_accepted_tokens,
                meta.num_spec_decodes,
                meta.spec_query_start_loc.device,
            )
        elif meta.num_decodes > 0:
            _dcut_fill_gdn_nonspec_bufs(
                prefix, num_tokens,
                meta.non_spec_query_start_loc,
                meta.non_spec_state_indices_tensor,
                meta.num_decodes,
                meta.non_spec_query_start_loc.device,
            )

# SPDX-License-Identifier: Apache-2.0
"""GDN static buffer alloc/fill/update for PIECEWISE graph replay."""
from __future__ import annotations

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .globals import logger, _dcut_gdn_static

def _dcut_alloc_gdn_spec_bufs(prefix, num_tokens, spec_state_indices_tensor, device):
    """Allocate pre-allocated ASL/SSI/NAT buffers for spec decode path.
    Called once at capture time (inside _forward_core, _EXTRA_CTX.capturing)."""
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
    """Fill ASL/SSI/NAT buffers in-place with runtime values + b_cap padding.
    Called from _model_forward (outside captured graph) before each replay."""
    _key = (prefix, num_tokens, "spec")
    _was_cached = _key in _dcut_gdn_static
    bufs = _dcut_alloc_gdn_spec_bufs(
        prefix, num_tokens, spec_state_indices_tensor, device)
    asl, ssi, nat = bufs["asl"], bufs["ssi"], bufs["nat"]
    import os
    if False:  # DBG disabled
        print(f"DBG_KEY prefix={prefix} nt={num_tokens} cached={_was_cached} b_cap={bufs["b_cap"]} nsd={num_spec_decodes}", flush=True)

    # Zero/fill all buffers unconditionally (graph expects clean state when nsd=0)
    asl.zero_()
    ssi.fill_(PAD_SLOT_ID)
    nat.zero_()

    if num_spec_decodes > 0:
        # Compute per_seq_lens ONCE (reused for ASL, SSI mask, NAT clamping)
        cu = spec_query_start_loc[:num_spec_decodes + 1]
        per_seq_lens = cu[1:] - cu[:-1]

        # ASL: [0, per_seq_lens..., 0, 0, ...]
        asl[1:num_spec_decodes + 1].copy_(per_seq_lens)

        # SSI: compact via pre-allocated col_idx (no torch.arange per call)
        col_idx = bufs["col_idx"]
        mask = col_idx.unsqueeze(0) < per_seq_lens.unsqueeze(1)
        real = spec_state_indices_tensor[:num_spec_decodes][mask]
        ssi[:real.size(0)].copy_(real)

        # NAT: clamped to per_seq_lens (reuse, no second subtraction)
        clamped = torch.minimum(
            num_accepted_tokens[:num_spec_decodes].to(torch.int32),
            per_seq_lens.to(torch.int32)
        )
        nat[:num_spec_decodes].copy_(clamped)

    import os
    if False:  # DBG disabled
        # Sample SSI at request boundaries to detect if all requests use same slot
        _psl = int(per_seq_lens[0].item()) if num_spec_decodes > 0 and len(per_seq_lens) > 0 else 0
        _ssi_sample = [int(ssi[i * _psl].item()) for i in range(min(num_spec_decodes, 8))] if _psl > 0 and num_spec_decodes > 0 else []
        _nat_sample = nat[:min(num_spec_decodes, 8)].tolist() if num_spec_decodes > 0 else []
        _ssi_unique = ssi[:num_spec_decodes * _psl].unique().tolist() if num_spec_decodes > 0 and _psl > 0 else []
        print(f"DBG_DETAIL prefix={prefix} nt={num_tokens} nsd={num_spec_decodes} psl={_psl} nat={_nat_sample} ssi_per_req={_ssi_sample} ssi_unique_count={len(_ssi_unique)} ssi_unique_sample={_ssi_unique[:10]}", flush=True)
    return bufs


def _dcut_alloc_gdn_nonspec_bufs(prefix, num_tokens,
                                  non_spec_state_indices_tensor, device):
    """Allocate pre-allocated ASL/SSI buffers for non-spec decode path."""
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
    """Fill ASL/SSI buffers in-place for non-spec decode path."""
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

    import os; False and print(f"DBG_FILL prefix={prefix} nt={num_tokens} nsd={num_spec_decodes} asl[:5]={asl[:5].tolist()} ssi[:5]={ssi[:5].tolist()} nat[:5]={nat[:5].tolist()}")
    return bufs


def _dcut_update_gdn_static(forward_context, num_tokens, GDNAttentionMetadata):
    """Update GDN static buffers from forward context's attn_metadata.
    Called from patched _model_forward before _orig_model_forward (i.e. before
    graph replay).  Runs eagerly — NOT inside the captured graph piece."""
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



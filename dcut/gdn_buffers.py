# SPDX-License-Identifier: Apache-2.0
"""GDN static buffer alloc/fill/update for PIECEWISE graph replay."""
from __future__ import annotations

import os

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .globals import logger, _dcut_gdn_static, ENV_GDN_SHARED_STATIC


def _dcut_use_shared_gdn_static() -> bool:
    return os.environ.get(ENV_GDN_SHARED_STATIC, "1").lower() not in (
        "0", "false", "no"
    )


def _dcut_gdn_static_key(prefix, num_tokens, kind):
    # Only ASL/NAT are batch-level values shared by all Qwen3.5 GDN layers.
    # SSI comes from the per-layer metadata's state-index tuple and must stay
    # prefix-local, matching the original full-decode path.
    if kind in ("spec_asl_nat", "nonspec_asl") and _dcut_use_shared_gdn_static():
        owner = "__shared__"
    else:
        owner = prefix
    return (owner, num_tokens, kind)


def _to_int_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if hasattr(value, "tolist"):
        return tuple(int(v) for v in value.tolist())
    return (int(value),)


def _spec_host_args(meta) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    fallback_meta = getattr(meta, "spec_decode_fallback_meta", None)
    if fallback_meta is not None:
        conv_meta = fallback_meta.spec_causal_conv1d
        return (
            _to_int_tuple(conv_meta.query_start_loc_cpu),
            _to_int_tuple(conv_meta.cache_indices_cpu),
            _to_int_tuple(conv_meta.num_accepted_tokens_cpu),
        )
    # Fallback for non-standard metadata; should not be used on the replay hot
    # path because .tolist() on device tensors would synchronize.
    return (
        _to_int_tuple(meta.spec_query_start_loc),
        _to_int_tuple(meta.spec_state_indices_tensor.reshape(-1)),
        _to_int_tuple(meta.num_accepted_tokens),
    )


def _nonspec_host_args(meta) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fallback_meta = getattr(meta, "non_spec_decode_fallback_meta", None)
    if fallback_meta is not None:
        conv_meta = fallback_meta.causal_conv1d
        return (
            _to_int_tuple(conv_meta.query_start_loc_cpu),
            _to_int_tuple(conv_meta.cache_indices_cpu),
        )
    return (
        _to_int_tuple(meta.non_spec_query_start_loc),
        _to_int_tuple(meta.non_spec_state_indices_tensor),
    )


def _copy_cpu_to_gpu(gpu_tensor, cpu_tensor, length: int | None = None) -> None:
    if length is None:
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    elif length > 0:
        gpu_tensor[:length].copy_(cpu_tensor[:length], non_blocking=True)




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
        _dcut_gdn_static[shared_key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "nat": torch.zeros(b_cap, dtype=torch.int32, device=device),
            "asl_cpu": torch.zeros(b_cap + 1, dtype=torch.int32, device="cpu"),
            "nat_cpu": torch.zeros(b_cap, dtype=torch.int32, device="cpu"),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN shared spec ASL/NAT prefix=%s num_tokens=%d "
            "b_cap=%d", prefix, num_tokens, b_cap)

    ssi_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_ssi")
    if ssi_key not in _dcut_gdn_static:
        b_cap = spec_state_indices_tensor.size(0)
        nsp1 = spec_state_indices_tensor.size(1)  # num_spec + 1
        t_cap = b_cap * nsp1
        _dcut_gdn_static[ssi_key] = {
            "ssi": torch.full((t_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "ssi_cpu": torch.full((t_cap,), PAD_SLOT_ID, dtype=torch.int32, device="cpu"),
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
        _dcut_gdn_static[shared_key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "asl_cpu": torch.zeros(b_cap + 1, dtype=torch.int32, device="cpu"),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN shared nonspec ASL prefix=%s num_tokens=%d "
            "b_cap=%d", prefix, num_tokens, b_cap)

    ssi_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_ssi")
    if ssi_key not in _dcut_gdn_static:
        b_cap = non_spec_state_indices_tensor.size(0)
        _dcut_gdn_static[ssi_key] = {
            "ssi": torch.full((b_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "ssi_cpu": torch.full((b_cap,), PAD_SLOT_ID, dtype=torch.int32, device="cpu"),
        }
        logger.debug(
            "D-Cut: alloc GDN per-layer nonspec SSI prefix=%s num_tokens=%d "
            "b_cap=%d", prefix, num_tokens, b_cap)

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
    filled_shared_keys = set()
    for prefix, meta in attn_metadata.items():
        if not isinstance(meta, GDNAttentionMetadata):
            continue
        if meta.spec_sequence_masks is not None and meta.num_spec_decodes > 0:
            shared_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_asl_nat")
            fill_shared = shared_key not in filled_shared_keys
            filled_shared_keys.add(shared_key)
            _dcut_fill_gdn_spec_bufs(
                prefix,
                num_tokens,
                meta,
                meta.spec_query_start_loc.device,
                fill_shared_asl_nat=fill_shared,
            )
        elif meta.num_decodes > 0:
            shared_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_asl")
            fill_shared = shared_key not in filled_shared_keys
            filled_shared_keys.add(shared_key)
            _dcut_fill_gdn_nonspec_bufs(
                prefix,
                num_tokens,
                meta,
                meta.non_spec_query_start_loc.device,
                fill_shared_asl=fill_shared,
            )

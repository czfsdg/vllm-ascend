# SPDX-License-Identifier: Apache-2.0
"""GDN static buffer alloc/fill/update for PIECEWISE graph replay.

Key optimizations vs old versions:
- The legacy static path shares batch-level ASL/NAT while keeping SSI local to
  the owning cache layer.
- The v0.23 PIECEWISE path follows vLLM attention grouping: layer prefixes that
  reference the same GDN metadata share QSL/SSI/NAT/ASL/mask buffers and refresh
  them once per metadata group. Different cache groups remain isolated.
"""
from __future__ import annotations

import os

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .globals import logger, _dcut_gdn_static

ENV_GDN_SHARED_STATIC = "VLLM_DCUT_GDN_SHARED_STATIC"


def _dcut_use_shared_gdn_static() -> bool:
    return os.environ.get(ENV_GDN_SHARED_STATIC, "1").lower() not in (
        "0", "false", "no"
    )


def _dcut_gdn_static_key(prefix, num_tokens, kind):
    # ASL/NAT are batch-level values shared by all Qwen3.5 GDN layers.
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
    """Read spec conv1d host args from CPU metadata (no NPU sync)."""
    fallback_meta = getattr(meta, "spec_decode_fallback_meta", None)
    if fallback_meta is not None:
        conv_meta = fallback_meta.spec_causal_conv1d
        return (
            _to_int_tuple(conv_meta.query_start_loc_cpu),
            _to_int_tuple(conv_meta.cache_indices_cpu),
            _to_int_tuple(conv_meta.num_accepted_tokens_cpu),
        )
    # Fallback: .tolist() on device tensors causes NPU sync — avoid on hot path.
    return (
        _to_int_tuple(meta.spec_query_start_loc),
        _to_int_tuple(meta.spec_state_indices_tensor.reshape(-1)),
        _to_int_tuple(meta.num_accepted_tokens),
    )


def _nonspec_host_args(meta) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read non-spec conv1d host args from CPU metadata (no NPU sync)."""
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


def _dcut_alloc_gdn_spec_bufs(prefix, num_tokens, spec_state_indices_tensor, device):
    """Allocate shared ASL/NAT + per-layer SSI buffers for spec decode path."""
    b_cap = spec_state_indices_tensor.size(0)
    nsp1 = spec_state_indices_tensor.size(1)  # num_spec + 1
    t_cap = b_cap * nsp1

    # Shared ASL/NAT (all layers share same batch composition)
    shared_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_asl_nat")
    if shared_key not in _dcut_gdn_static:
        _dcut_gdn_static[shared_key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "nat": torch.zeros(b_cap, dtype=torch.int32, device=device),
            "asl_cpu": torch.zeros(b_cap + 1, dtype=torch.int32, device="cpu"),
            "nat_cpu": torch.zeros(b_cap, dtype=torch.int32, device="cpu"),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN shared spec ASL/NAT num_tokens=%d b_cap=%d",
            num_tokens, b_cap)

    # Per-layer SSI
    ssi_key = _dcut_gdn_static_key(prefix, num_tokens, "spec_ssi")
    if ssi_key not in _dcut_gdn_static:
        _dcut_gdn_static[ssi_key] = {
            "ssi": torch.full((t_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "col_idx": torch.arange(nsp1, device=device),
            "b_cap": b_cap,
            "nsp1": nsp1,
            "t_cap": t_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN spec SSI prefix=%s num_tokens=%d b_cap=%d t_cap=%d",
            prefix, num_tokens, b_cap, t_cap)

    # Return combined dict for convenience
    return {**_dcut_gdn_static[shared_key], **_dcut_gdn_static[ssi_key]}


def _dcut_fill_gdn_spec_bufs(prefix, num_tokens, meta, device,
                              fill_shared_asl_nat=True):
    """Fill ASL/SSI/NAT buffers before graph replay.

    Optimization: ASL/NAT built on CPU from conv1d host args, one copy to GPU.
    SSI still uses NPU ops (source is device tensor, unavoidable).
    """
    spec_state_indices_tensor = meta.spec_state_indices_tensor
    num_spec_decodes = meta.num_spec_decodes
    spec_query_start_loc = meta.spec_query_start_loc
    num_accepted_tokens = meta.num_accepted_tokens

    bufs = _dcut_alloc_gdn_spec_bufs(
        prefix, num_tokens, spec_state_indices_tensor, device)

    # --- Shared ASL/NAT: CPU build + one GPU copy ---
    if fill_shared_asl_nat:
        asl_cpu = bufs["asl_cpu"]
        nat_cpu = bufs["nat_cpu"]
        asl_cpu.zero_()
        nat_cpu.zero_()

        if num_spec_decodes > 0:
            # Try CPU host args first (no NPU sync)
            qsl_host, _, nat_host = _spec_host_args(meta)

            if qsl_host and nat_host:
                # ASL: diff of cumulative query_start_loc
                n = min(num_spec_decodes, len(qsl_host) - 1)
                for i in range(n):
                    asl_cpu[i + 1] = qsl_host[i + 1] - qsl_host[i]
                # NAT: from host args, clamped to segment length
                n_nat = min(num_spec_decodes, len(nat_host))
                for i in range(n_nat):
                    val = nat_host[i]
                    if i + 1 < len(qsl_host):
                        seg_len = qsl_host[i + 1] - qsl_host[i]
                        val = min(val, seg_len)
                    nat_cpu[i] = val
            else:
                # Fallback: build from NPU tensors (causes sync, but correct)
                cu = spec_query_start_loc[:num_spec_decodes + 1]
                per_seq_lens = cu[1:] - cu[:-1]
                asl_cpu[1:num_spec_decodes + 1].copy_(per_seq_lens.cpu())
                clamped = torch.minimum(
                    num_accepted_tokens[:num_spec_decodes].to(torch.int32),
                    per_seq_lens.to(torch.int32)
                )
                nat_cpu[:num_spec_decodes].copy_(clamped.cpu())

        # One copy CPU -> GPU
        bufs["asl"].copy_(asl_cpu, non_blocking=True)
        bufs["nat"].copy_(nat_cpu, non_blocking=True)

    # --- Per-layer SSI: NPU path (unavoidable, source is device tensor) ---
    ssi = bufs["ssi"]
    ssi.fill_(PAD_SLOT_ID)

    if num_spec_decodes > 0:
        cu = spec_query_start_loc[:num_spec_decodes + 1]
        per_seq_lens = cu[1:] - cu[:-1]
        col_idx = bufs["col_idx"]
        mask = col_idx.unsqueeze(0) < per_seq_lens.unsqueeze(1)
        real = spec_state_indices_tensor[:num_spec_decodes][mask]
        ssi[:real.size(0)].copy_(real)

    return bufs


def _dcut_alloc_gdn_nonspec_bufs(prefix, num_tokens,
                                  non_spec_state_indices_tensor, device):
    """Allocate shared ASL + per-layer SSI buffers for non-spec decode path."""
    b_cap = non_spec_state_indices_tensor.size(0)

    # Shared ASL
    shared_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_asl")
    if shared_key not in _dcut_gdn_static:
        _dcut_gdn_static[shared_key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "asl_cpu": torch.zeros(b_cap + 1, dtype=torch.int32, device="cpu"),
            "b_cap": b_cap,
        }
        logger.debug(
            "D-Cut: alloc GDN shared nonspec ASL num_tokens=%d b_cap=%d",
            num_tokens, b_cap)

    # Per-layer SSI
    ssi_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_ssi")
    if ssi_key not in _dcut_gdn_static:
        _dcut_gdn_static[ssi_key] = {
            "ssi": torch.full((b_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        }
        logger.debug(
            "D-Cut: alloc GDN nonspec SSI prefix=%s num_tokens=%d b_cap=%d",
            prefix, num_tokens, b_cap)

    return {**_dcut_gdn_static[shared_key], **_dcut_gdn_static[ssi_key]}


def _dcut_fill_gdn_nonspec_bufs(prefix, num_tokens, meta, device,
                                 fill_shared_asl=True):
    """Fill ASL/SSI buffers for non-spec decode before graph replay."""
    non_spec_query_start_loc = meta.non_spec_query_start_loc
    non_spec_state_indices_tensor = meta.non_spec_state_indices_tensor
    num_decodes = meta.num_decodes

    bufs = _dcut_alloc_gdn_nonspec_bufs(
        prefix, num_tokens, non_spec_state_indices_tensor, device)

    # --- Shared ASL: CPU build + one GPU copy ---
    if fill_shared_asl:
        asl_cpu = bufs["asl_cpu"]
        asl_cpu.zero_()

        if num_decodes > 0:
            qsl_host, _ = _nonspec_host_args(meta)
            if qsl_host:
                n = min(num_decodes, len(qsl_host) - 1)
                for i in range(n):
                    asl_cpu[i + 1] = qsl_host[i + 1] - qsl_host[i]
            else:
                cu = non_spec_query_start_loc[:num_decodes + 1]
                asl_cpu[1:num_decodes + 1].copy_((cu[1:] - cu[:-1]).cpu())

        bufs["asl"].copy_(asl_cpu, non_blocking=True)

    # --- Per-layer SSI: NPU path ---
    ssi = bufs["ssi"]
    ssi.fill_(PAD_SLOT_ID)
    if num_decodes > 0:
        ssi[:num_decodes].copy_(non_spec_state_indices_tensor[:num_decodes])

    return bufs


def _dcut_update_gdn_static(forward_context, num_tokens, GDNAttentionMetadata):
    """Update GDN static buffers from forward context's attn_metadata.

    Called from patched _model_forward before _orig_model_forward (i.e. before
    graph replay). ASL/NAT are filled once (shared), SSI per-layer.
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
                prefix, num_tokens, meta, meta.spec_query_start_loc.device,
                fill_shared_asl_nat=fill_shared,
            )
        elif meta.num_decodes > 0:
            shared_key = _dcut_gdn_static_key(prefix, num_tokens, "nonspec_asl")
            fill_shared = shared_key not in filled_shared_keys
            filled_shared_keys.add(shared_key)
            _dcut_fill_gdn_nonspec_bufs(
                prefix, num_tokens, meta, meta.non_spec_query_start_loc.device,
                fill_shared_asl=fill_shared,
            )


def _dcut_gdn_piecewise_spec_key(forward_context, prefix, num_tokens):
    """Return a key that does not alias target and draft model buffers."""
    model_instance = getattr(forward_context, "model_instance", None)
    return (id(model_instance), prefix, num_tokens, "v023_piecewise_spec")


def _dcut_alloc_gdn_piecewise_spec_bufs(
    forward_context,
    prefixes,
    num_tokens,
    state_indices,
    max_num_seqs,
):
    """Allocate fixed-shape inputs consumed by a PIECEWISE GDN graph.

    PIECEWISE ACLGraph keys contain the padded token count, but not the live
    number of speculative requests. Use the scheduler request capacity for
    every graph of a given token size so that different request compositions
    can safely replay the same graph.
    """
    if state_indices.ndim != 2:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN requires 2-D per-request state indices, "
            f"got shape={tuple(state_indices.shape)}"
        )

    if not prefixes:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN buffer group has no layer prefixes"
        )

    keys = [
        _dcut_gdn_piecewise_spec_key(
            forward_context, prefix, num_tokens
        )
        for prefix in prefixes
    ]
    state_index_stride = state_indices.shape[1]
    expected_shape = (max_num_seqs, state_index_stride)
    existing_bufs = [
        _dcut_gdn_static[key]
        for key in keys
        if key in _dcut_gdn_static
    ]
    bufs = existing_bufs[0] if existing_bufs else None
    if bufs is not None:
        if any(existing is not bufs for existing in existing_bufs[1:]):
            raise RuntimeError(
                "D-Cut PIECEWISE GDN layer aliases resolved to different "
                f"buffer groups: prefixes={tuple(prefixes)}"
            )
        if tuple(bufs["ssi"].shape) != expected_shape:
            raise RuntimeError(
                "D-Cut PIECEWISE GDN buffer shape changed for an existing "
                f"graph key: expected={expected_shape}, "
                f"actual={tuple(bufs['ssi'].shape)}"
            )
        for key in keys:
            _dcut_gdn_static[key] = bufs
        return bufs

    device = state_indices.device
    bufs = {
        "qsl": torch.zeros(
            max_num_seqs + 1, dtype=torch.int32, device=device
        ),
        "ssi": torch.full(
            expected_shape,
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=device,
        ),
        "nat": torch.zeros(
            max_num_seqs, dtype=torch.int32, device=device
        ),
        "asl": torch.zeros(
            max_num_seqs + 1, dtype=torch.int32, device=device
        ),
        "token_index": torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ),
        "token_mask": torch.zeros(
            num_tokens, dtype=torch.bool, device=device
        ),
    }
    for key in keys:
        _dcut_gdn_static[key] = bufs
    logger.info(
        "D-Cut: allocated v0.23 PIECEWISE GDN buffers "
        "group_prefix=%s group_layers=%d num_tokens=%d "
        "max_num_seqs=%d stride=%d",
        prefixes[0],
        len(prefixes),
        num_tokens,
        max_num_seqs,
        state_index_stride,
    )
    return bufs


def _dcut_fill_gdn_piecewise_spec_bufs(
    forward_context,
    prefixes,
    num_tokens,
    meta,
    max_num_seqs,
):
    """Refresh fixed-address v0.23 GDN inputs before capture or replay."""
    num_spec_decodes = int(meta.num_spec_decodes)
    if num_spec_decodes <= 0 or num_spec_decodes > max_num_seqs:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN speculative request count is outside "
            f"the configured capacity: requests={num_spec_decodes}, "
            f"capacity={max_num_seqs}"
        )

    spec_decode_metadata = meta.spec_decode_metadata
    conv_meta = spec_decode_metadata.spec_causal_conv1d
    state_indices = meta.spec_state_indices_tensor
    if state_indices is None:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN is missing spec_state_indices_tensor"
        )

    bufs = _dcut_alloc_gdn_piecewise_spec_bufs(
        forward_context,
        prefixes,
        num_tokens,
        state_indices,
        max_num_seqs,
    )

    qsl = bufs["qsl"]
    qsl.zero_()
    qsl[: num_spec_decodes + 1].copy_(
        conv_meta.query_start_loc[: num_spec_decodes + 1],
        non_blocking=True,
    )
    qsl_tail = qsl[num_spec_decodes + 1 :]
    if qsl_tail.numel() > 0:
        qsl_tail.copy_(
            qsl[num_spec_decodes].expand_as(qsl_tail),
            non_blocking=True,
        )

    ssi = bufs["ssi"]
    ssi.fill_(PAD_SLOT_ID)
    ssi[:num_spec_decodes].copy_(
        state_indices[:num_spec_decodes],
        non_blocking=True,
    )

    asl = bufs["asl"]
    asl.zero_()
    asl[:1].copy_(qsl[:1], non_blocking=True)
    torch.sub(
        qsl[1 : num_spec_decodes + 1],
        qsl[:num_spec_decodes],
        out=asl[1 : num_spec_decodes + 1],
    )

    nat = bufs["nat"]
    nat.zero_()
    accepted_tokens = conv_meta.num_accepted_tokens[
        :num_spec_decodes
    ].to(torch.int32)
    torch.minimum(
        accepted_tokens,
        asl[1 : num_spec_decodes + 1],
        out=nat[:num_spec_decodes],
    )

    torch.lt(
        bufs["token_index"],
        qsl[num_spec_decodes],
        out=bufs["token_mask"],
    )
    return bufs


def _dcut_prepare_gdn_piecewise_replay(
    forward_context,
    num_tokens,
    GDNAttentionMetadata,
    max_num_seqs,
):
    """Prepare pure speculative GDN replay, or reject the graph safely.

    The GDN custom op chooses prefill/decode/spec branches from Python
    metadata that is not part of the compiled graph signature. Only a pure
    speculative batch can therefore reuse the speculative PIECEWISE graph.
    Other compositions must execute the custom op eagerly.
    """
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        return False

    gdn_items = [
        (prefix, meta)
        for prefix, meta in attn_metadata.items()
        if isinstance(meta, GDNAttentionMetadata)
    ]
    if not gdn_items:
        return False

    # model_runner assigns the same metadata object to every layer in one
    # attention group. Identity grouping preserves separate hybrid-cache groups.
    metadata_groups = {}
    for prefix, meta in gdn_items:
        group = metadata_groups.setdefault(
            id(meta), {"meta": meta, "prefixes": []}
        )
        group["prefixes"].append(prefix)

    for group in metadata_groups.values():
        meta = group["meta"]
        if (
            meta.spec_sequence_masks is None
            or int(meta.num_spec_decodes) <= 0
            or int(meta.num_prefills) != 0
            or int(meta.num_decodes) != 0
            or meta.spec_decode_metadata is None
        ):
            return False
        state_indices = meta.spec_state_indices_tensor
        if (
            state_indices is None
            or state_indices.ndim != 2
            or state_indices.shape[0] > max_num_seqs
        ):
            return False

    # A captured prefix must keep the same buffer alias topology. If vLLM ever
    # changes the attention grouping at runtime, reject PIECEWISE safely rather
    # than letting two now-distinct groups overwrite one captured buffer.
    claimed_buffer_ids = set()
    for group in metadata_groups.values():
        existing_buffer_ids = set()
        for prefix in group["prefixes"]:
            key = _dcut_gdn_piecewise_spec_key(
                forward_context, prefix, num_tokens
            )
            if key in _dcut_gdn_static:
                existing_buffer_ids.add(id(_dcut_gdn_static[key]))
        if (
            len(existing_buffer_ids) > 1
            or not claimed_buffer_ids.isdisjoint(existing_buffer_ids)
        ):
            return False
        claimed_buffer_ids.update(existing_buffer_ids)

    for group in metadata_groups.values():
        _dcut_fill_gdn_piecewise_spec_bufs(
            forward_context,
            tuple(group["prefixes"]),
            num_tokens,
            group["meta"],
            max_num_seqs,
        )
    return True


def _dcut_get_gdn_piecewise_spec_bufs(
    forward_context,
    prefix,
    num_tokens,
):
    """Get buffers already prepared by the graph-external runner hook."""
    key = _dcut_gdn_piecewise_spec_key(
        forward_context, prefix, num_tokens
    )
    try:
        return _dcut_gdn_static[key]
    except KeyError as exc:
        raise RuntimeError(
            "D-Cut PIECEWISE GDN buffers were not prepared before capture: "
            f"prefix={prefix}, num_tokens={num_tokens}"
        ) from exc

# SPDX-License-Identifier: Apache-2.0
"""Small utility functions."""
from __future__ import annotations

import os

import torch

from vllm.distributed import get_tp_group

from .globals import ENV_PROCESS_PROBS_STAGE, ENV_REUSE_ARGMAX


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _dcut_process_probs_stage() -> str:
    raw = (
        os.environ.get(ENV_PROCESS_PROBS_STAGE, "pre_truncate")
        or "pre_truncate"
    ).strip().lower()
    stage = raw.replace("-", "_")
    if stage in {"post", "post_sample", "sync", "synchronous"}:
        return "post_sample"
    return "pre_truncate"


def _dcut_reuse_argmax_enabled() -> bool:
    return _env_flag(ENV_REUSE_ARGMAX, True)


def _dcut_in_graph_capture() -> bool:
    """Return whether the current Ascend forward is recording an ACLGraph."""
    try:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        if bool(getattr(forward_context, "capturing", False)):
            return True
    except Exception:
        pass

    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        return bool(getattr(_EXTRA_CTX, "capturing", False))
    except Exception:
        return False


def _dcut_selected_token_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute probabilities for token IDs already selected from *logits*.

    This avoids repeating argmax over the full vocabulary. The caller only
    uses this helper on paths where the original sampler selected IDs directly
    from the same logits tensor, so their vocab indexing is identical.
    """
    idx = token_ids.long().unsqueeze(-1)
    chosen = logits.gather(-1, idx).squeeze(-1)
    return (chosen - logits.logsumexp(dim=-1)).exp()


def _dcut_can_reuse_argmax_for_probs(drafter) -> bool:
    return (
        _dcut_reuse_argmax_enabled()
        and getattr(drafter, "method", None) == "dflash"
        and getattr(type(drafter), "_dcut_run_merged_patched", False)
    )


def _dcut_selected_probs_from_reused_logits(
    drafter,
    draft_token_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Derive selected probabilities after eager execution or graph replay."""
    if (
        draft_token_ids is None
        or not torch.is_tensor(draft_token_ids)
        or not _dcut_can_reuse_argmax_for_probs(drafter)
    ):
        return None

    logits = getattr(drafter, "_dcut_last_logits_for_probs", None)
    if logits is None:
        # A Python execution should have populated the per-step logits. Only a
        # graph replay is allowed to fall back to the fixed-address capture
        # tensors retained below.
        if getattr(drafter, "_dcut_last_draft_ran_python", False):
            return None
        if getattr(drafter, "_dcut_graph_logits_for_probs_ready", False):
            shape_key = tuple(draft_token_ids.shape)
            by_shape = getattr(
                drafter,
                "_dcut_graph_logits_for_probs_by_shape",
                {},
            )
            by_numel = getattr(
                drafter,
                "_dcut_graph_logits_for_probs_by_numel",
                {},
            )
            logits = by_shape.get(shape_key)
            if logits is None:
                logits = by_numel.get(int(draft_token_ids.numel()))
            if logits is None:
                logits = getattr(
                    drafter,
                    "_dcut_graph_logits_for_probs",
                    None,
                )
    if logits is None:
        return None

    flat_token_ids = draft_token_ids.reshape(-1)
    n_tokens = min(int(logits.shape[0]), int(flat_token_ids.numel()))
    if n_tokens <= 0:
        return None
    selected_probs = _dcut_selected_token_probs(
        logits[:n_tokens],
        flat_token_ids[:n_tokens],
    )
    if selected_probs.numel() == draft_token_ids.numel():
        selected_probs = selected_probs.view(draft_token_ids.shape)
    return selected_probs.float().contiguous()


def _dcut_selected_probs_from_graph(
    drafter,
    draft_token_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the selected-prob tensor produced by the replayed draft graph.

    The merged draft function is captured once per output bucket. Python
    instance attributes are not reassigned during graph replay, so a single
    last-selected-probs pointer can silently refer to the final bucket captured
    at startup. Keep fixed-address graph outputs by shape and select the
    matching tensor for the current draft output instead.
    """
    if (
        draft_token_ids is None
        or not torch.is_tensor(draft_token_ids)
        or getattr(drafter, "_dcut_last_draft_ran_python", False)
        or not getattr(
            drafter,
            "_dcut_graph_selected_probs_ready",
            False,
        )
    ):
        return None

    shape_key = tuple(draft_token_ids.shape)
    by_output_ptr = getattr(
        drafter,
        "_dcut_graph_selected_probs_by_output_ptr",
        {},
    )
    by_shape = getattr(
        drafter,
        "_dcut_graph_selected_probs_by_shape",
        {},
    )
    by_numel = getattr(
        drafter,
        "_dcut_graph_selected_probs_by_numel",
        {},
    )
    selected_probs = by_output_ptr.get(int(draft_token_ids.data_ptr()))
    if selected_probs is None:
        selected_probs = by_shape.get(shape_key)
    if selected_probs is None:
        selected_probs = by_numel.get(int(draft_token_ids.numel()))
    if selected_probs is None:
        return None

    flat_probs = selected_probs.reshape(-1)
    num_tokens = int(draft_token_ids.numel())
    if num_tokens <= 0 or flat_probs.numel() < num_tokens:
        return None
    return flat_probs[:num_tokens].view(draft_token_ids.shape)


def _npu_event(enable_timing: bool = False):
    """torch.npu.Event, mirroring torch.cuda.Event on the CUDA plugin."""
    return torch.npu.Event(enable_timing=enable_timing)


def _supports_adaptive_verify(spec_cfg) -> bool:
    """Mirror of SpeculativeConfig.supports_adaptive_verify (which 0.22.x lacks)."""
    if spec_cfg is None:
        return False
    method = getattr(spec_cfg, "method", None)
    parallel = getattr(spec_cfg, "parallel_drafting", False)
    return method == "dflash" or (method == "draft_model" and parallel)


def _dcut_greedy_sample_with_selected_probs(logits):
    tp_group = get_tp_group()
    _, v_local = logits.shape
    rank = tp_group.rank_in_group

    local_max_logits, local_max_indices = logits.max(dim=-1)
    local_global_idx = local_max_indices + rank * v_local

    gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
    gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)
    global_max_rank = gathered_logits.argmax(dim=-1)
    next_token = gathered_global_idx.gather(
        dim=-1, index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1)
    selected_logits = gathered_logits.gather(
        dim=-1, index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1)

    local_lse = logits.logsumexp(dim=-1)
    gathered_lse = tp_group.all_gather(local_lse.unsqueeze(-1), dim=-1)
    global_lse = gathered_lse.logsumexp(dim=-1)
    selected_probs = (selected_logits - global_lse).exp()
    return next_token, selected_probs



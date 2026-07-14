# SPDX-License-Identifier: Apache-2.0
"""Small utility functions."""
from __future__ import annotations

import torch

from vllm.distributed import get_tp_group

from .globals import logger

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



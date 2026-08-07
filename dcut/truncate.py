# SPDX-License-Identifier: Apache-2.0
"""D-Cut draft truncation + verify-reduction stats."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

from vllm.distributed import get_pp_group, get_tp_group

from .globals import logger


def _dcut_has_prefill(self, scheduler_output) -> bool:
    """Return whether the current scheduler batch contains prefill work.

    D-Cut's cost model assumes that every request contributes exactly one
    anchor token. That assumption does not hold for mixed prefill/decode
    batches, so leave all speculative drafts intact whenever prefill work is
    present.
    """
    if getattr(scheduler_output, "scheduled_new_reqs", None):
        return True

    scheduled_spec = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        None,
    ) or {}
    num_scheduled_tokens = getattr(
        scheduler_output,
        "num_scheduled_tokens",
        None,
    ) or {}

    input_batch = getattr(self, "input_batch", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", {})
    num_computed_tokens = getattr(
        input_batch,
        "num_computed_tokens_cpu",
        None,
    )
    num_prompt_tokens = getattr(input_batch, "num_prompt_tokens", None)

    for req_id, num_scheduled in num_scheduled_tokens.items():
        if req_id in scheduled_spec:
            continue

        # Ordinary decode contributes one token. Any larger non-spec query is
        # prefill (including a continuation chunk).
        if int(num_scheduled) != 1:
            return True

        # The final chunk of a prefill can contain exactly one token. Use the
        # runner's request state to distinguish it from ordinary decode.
        req_idx = req_id_to_index.get(req_id)
        if (
            req_idx is not None
            and num_computed_tokens is not None
            and num_prompt_tokens is not None
            and num_computed_tokens[req_idx] < num_prompt_tokens[req_idx]
        ):
            return True

    return False


def _dcut_get_target_draft_lens(ctrl, original_spec) -> list[int]:
    """Return rank-0's draft-length decisions in scheduler request order.

    The async selected-probability D2H event can become ready on different
    scheduler ticks on different TP ranks. Letting every rank consume its
    local controller cache would then produce different verifier token counts
    and eventually deadlock a later TP collective. Broadcast one small Python
    integer list because truncation immediately consumes the result on CPU;
    using a device tensor here would add an avoidable NPU-to-CPU sync.
    """
    tp_group = get_tp_group()
    target_draft_lens = None
    if tp_group.rank_in_group == 0:
        target_draft_lens = []
        for req_id, draft_tokens in original_spec.items():
            max_draft_len = len(draft_tokens)
            adaptive_len = ctrl.get_adaptive_draft_len(req_id)
            if adaptive_len is None:
                adaptive_len = max_draft_len
            target_draft_lens.append(
                min(max(int(adaptive_len), 0), max_draft_len)
            )

    if tp_group.world_size > 1:
        target_draft_lens = tp_group.broadcast_object(
            target_draft_lens,
            src=0,
        )

    if target_draft_lens is None or len(target_draft_lens) != len(original_spec):
        raise RuntimeError(
            "D-Cut received an invalid TP decision broadcast: "
            f"expected {len(original_spec)} draft lengths, got "
            f"{target_draft_lens!r}"
        )
    return target_draft_lens


def _dcut_truncate(
    self,
    scheduler_output,
    has_prefill: bool | None = None,
):
    """Apply cached per-request draft caps without a GDN state floor.

    The D-Cut recurrent and conv1d kernels read the previous accepted state
    through independent device-side indices, so the current verifier length no
    longer needs to preserve the previous accepted position.
    """
    ctrl = self._verify_adaptive_controller
    scheduled_spec = getattr(
        scheduler_output,
        "scheduled_spec_decode_tokens",
        None,
    )
    if ctrl is None or not scheduled_spec:
        return scheduler_output
    if has_prefill is None:
        has_prefill = _dcut_has_prefill(self, scheduler_output)
    if has_prefill:
        return scheduler_output

    original_spec = scheduler_output.scheduled_spec_decode_tokens
    full_draft = sum(len(tokens) for tokens in original_spec.values())
    num_spec_requests = len(original_spec)
    new_spec = original_spec.copy()
    new_num_scheduled = scheduler_output.num_scheduled_tokens.copy()

    target_draft_lens = _dcut_get_target_draft_lens(ctrl, original_spec)

    trimmed_tokens = 0
    for (req_id, draft_tokens), target_len in zip(
        list(new_spec.items()),
        target_draft_lens,
    ):
        max_draft_len = len(draft_tokens)
        if target_len >= max_draft_len:
            continue

        trimmed_tokens += max_draft_len - target_len
        new_num_scheduled[req_id] -= max_draft_len - target_len
        # Keep zero-length decisions as empty speculative entries. The model
        # runner represents a speculative request with no draft tokens as
        # num_decode_draft_tokens == 0 (ordinary decode/prefill uses -1).
        # Dropping the key would therefore turn a pure-spec batch into a mixed
        # spec/non-spec batch, making GDN take its prefill fallback and miss the
        # local PIECEWISE graph. Retaining the key keeps the one anchor token
        # on the speculative path without adding verifier tokens.
        new_spec[req_id] = draft_tokens[:target_len]

    _dcut_record_trim(
        self,
        full_draft,
        trimmed_tokens,
        num_spec_requests,
    )
    if trimmed_tokens == 0:
        return scheduler_output

    return replace(
        scheduler_output,
        scheduled_spec_decode_tokens=new_spec,
        num_scheduled_tokens=new_num_scheduled,
        total_num_scheduled_tokens=(scheduler_output.total_num_scheduled_tokens - trimmed_tokens),
    )


def _dcut_record_trim(self, full_draft: int, trimmed: int, n_spec_reqs: int) -> None:
    """Accumulate verify-reduction stats and log them every N steps (rank 0).

    Answers "how much verify did D-Cut save": trimmed vs full draft positions
    (the verifier checks one position per draft token).  Cadence is controlled
    by ``VLLM_DCUT_STAT_EVERY`` (steps; 0 disables).  Cumulative totals, so the
    running percentage is stable.
    """
    self._dcut_stat_full += full_draft
    self._dcut_stat_trimmed += trimmed
    self._dcut_stat_reqs += n_spec_reqs
    self._dcut_stat_steps += 1
    every = self._dcut_stat_log_every
    if not every or (self._dcut_stat_steps % every) != 0:
        return
    if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
        return
    full = self._dcut_stat_full
    trimmed_tot = self._dcut_stat_trimmed
    pct = 100.0 * trimmed_tot / full if full else 0.0
    kept = full - trimmed_tot
    reqs = max(self._dcut_stat_reqs, 1)
    _dcut_dump_trim_stats(
        self,
        full=full,
        trimmed_tot=trimmed_tot,
        kept=kept,
        pct=pct,
        reqs=reqs,
        last_full=full_draft,
        last_trimmed=trimmed,
        last_reqs=n_spec_reqs,
    )
    logger.info(
        "D-Cut verify trim: cut %d/%d draft positions (%.1f%% fewer verifies) "
        "over %d steps; avg %.2f->%.2f verified tok/spec-req",
        trimmed_tot,
        full,
        pct,
        self._dcut_stat_steps,
        full / reqs,
        kept / reqs,
    )


def _dcut_dump_trim_stats(
    self,
    *,
    full: int,
    trimmed_tot: int,
    kept: int,
    pct: float,
    reqs: int,
    last_full: int,
    last_trimmed: int,
    last_reqs: int,
) -> None:
    """Append rank-0 trim stats to JSONL for scripts that cannot see worker logs."""
    path = getattr(self, "_dcut_trim_stats_out", None)
    if not path:
        return
    row = {
        "time_unix": time.time(),
        "steps": self._dcut_stat_steps,
        "spec_reqs": self._dcut_stat_reqs,
        "full_draft_positions": full,
        "trimmed_draft_positions": trimmed_tot,
        "kept_draft_positions": kept,
        "trim_pct": pct,
        "avg_full_per_spec_req": full / reqs,
        "avg_kept_per_spec_req": kept / reqs,
        "last_full_draft_positions": last_full,
        "last_trimmed_draft_positions": last_trimmed,
        "last_spec_reqs": last_reqs,
    }
    try:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover - observability must not affect serving
        logger.debug("D-Cut: failed to write trim stats: %s", e)

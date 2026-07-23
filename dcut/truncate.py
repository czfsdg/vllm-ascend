# SPDX-License-Identifier: Apache-2.0
"""D-Cut draft truncation + verify-reduction stats."""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace

from vllm.distributed import get_pp_group, get_tp_group

from .globals import ENV_TRIM_STATS_OUT, logger


def _get_gdn_min_draft_lens(self, req_ids) -> dict[str, int]:
    """Return the draft-length floor required by live GDN state.

    ``num_accepted_tokens`` identifies the state slot produced by the previous
    speculative step. The recurrent GDN kernel selects that slot from the
    current request row, so shortening the current query below the accepted
    count would either make the kernel reject the row or, if the count were
    clamped, resume from an older state. Wait for the same D2H event that the
    runner consumes later and keep enough draft positions to retain the real
    accepted-state slot.
    """
    if not getattr(self, "_has_gdn", False):
        return {}

    accepted_event = getattr(self, "num_accepted_tokens_event", None)
    if accepted_event is not None:
        accepted_event.synchronize()

    input_batch = getattr(self, "input_batch", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    accepted_tokens = getattr(input_batch, "num_accepted_tokens_cpu", None)
    if not req_id_to_index or accepted_tokens is None:
        return {}

    min_draft_lens = {}
    for req_id in req_ids:
        req_index = req_id_to_index.get(req_id)
        if req_index is not None:
            min_draft_lens[req_id] = max(int(accepted_tokens[req_index]) - 1, 0)
    return min_draft_lens


def _dcut_truncate(self, scheduler_output):
    """Apply per-request draft_len caps cached by the previous step.

    Device-agnostic (pure scheduler-output bookkeeping); identical to CUDA.
    """
    ctrl = self._verify_adaptive_controller
    _spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)

    if ctrl is None or not _spec:
        return scheduler_output

    orig_spec = scheduler_output.scheduled_spec_decode_tokens
    # Draft positions that WOULD be verified this step without D-Cut, and the
    # number of spec-decode requests — captured before we mutate anything.
    full_draft = sum(len(t) for t in orig_spec.values())
    n_spec_reqs = len(orig_spec)

    new_spec = orig_spec.copy()
    new_num_sched = scheduler_output.num_scheduled_tokens.copy()
    gdn_min_draft_lens = _get_gdn_min_draft_lens(self, orig_spec)
    target_draft_lens = {}
    for req_id, draft_toks in orig_spec.items():
        max_dl = len(draft_toks)
        adaptive_len = ctrl.get_adaptive_draft_len(req_id)
        if adaptive_len is None:
            adaptive_len = max_dl  # no cached decision -> full spec
        adaptive_len = min(max(adaptive_len, 0), max_dl)
        # Preserve the state slot selected by the previous speculative step.
        # The extra query token means a previous accepted count of N requires
        # at least N - 1 draft tokens in this step.
        target_draft_lens[req_id] = max(
            adaptive_len,
            min(gdn_min_draft_lens.get(req_id, 0), max_dl),
        )

    tokens_delta = 0
    for req_id, draft_toks in list(new_spec.items()):
        max_dl = len(draft_toks)
        adaptive_len = target_draft_lens[req_id]
        if adaptive_len < max_dl:
            diff = max_dl - adaptive_len
            tokens_delta += diff
            new_num_sched[req_id] -= diff
            if adaptive_len:
                new_spec[req_id] = draft_toks[:adaptive_len]
            else:
                # An empty entry still makes model_runner treat the step as
                # speculative. Remove it so a full cut becomes normal decode.
                new_spec.pop(req_id)

    _dcut_record_trim(self, full_draft, tokens_delta, n_spec_reqs)

    if tokens_delta > 0:
        scheduler_output = replace(
            scheduler_output,
            scheduled_spec_decode_tokens=new_spec,
            num_scheduled_tokens=new_num_sched,
            total_num_scheduled_tokens=(
                scheduler_output.total_num_scheduled_tokens - tokens_delta
            ),
        )
    return scheduler_output


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
        trimmed_tot, full, pct, self._dcut_stat_steps,
        full / reqs, kept / reqs,
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

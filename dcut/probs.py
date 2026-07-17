# SPDX-License-Identifier: Apache-2.0
"""Async D2H selected-probs queue + controller cache update."""
from __future__ import annotations

import torch

from .globals import logger
from .controller import _dcut_enable_drafter_probs

def _dcut_queue_probs(self, zeros_only: bool) -> None:
    """Queue this step's selected_probs D2H (non-blocking) for next-step use.

    Device-agnostic apart from the async D2H copy + event record, which work
    the same on NPU (torch_npu supports non_blocking copies + npu.Event).
    """
    if (
        zeros_only
        or self._adaptive_probs_pending
        or self._adaptive_probs_pinned is None
        or self._adaptive_probs_event is None
    ):
        return
    _dcut_enable_drafter_probs(self)
    drafter = getattr(self, "drafter", None)
    if drafter is None or not hasattr(drafter, "take_last_selected_probs"):
        cnt = getattr(self, "_dcut_missing_probs_steps", 0) + 1
        self._dcut_missing_probs_steps = cnt
        if cnt <= 3 or cnt % 200 == 0:
            logger.warning(
                "D-Cut: drafter has no selected-probs hook; decision stats "
                "will not update (count=%s).",
                cnt,
            )
        return
    probs = drafter.take_last_selected_probs()
    if probs is None:
        cnt = getattr(self, "_dcut_missing_probs_steps", 0) + 1
        self._dcut_missing_probs_steps = cnt
        if cnt <= 3 or cnt % 200 == 0:
            logger.warning(
                "D-Cut: drafter did not expose selected draft probs; decision "
                "stats will not update (count=%s).",
                cnt,
            )
        return
    num_reqs = self.input_batch.num_reqs
    num_spec = self.num_spec_tokens
    if probs.dim() == 1:
        needed = num_reqs * num_spec
        if probs.numel() < needed:
            logger.warning(
                "D-Cut: selected draft probs too short: got=%s need=%s",
                probs.numel(),
                needed,
            )
            return
        probs = probs[:needed].view(num_reqs, num_spec)
    else:
        probs = probs[:num_reqs]
        if probs.shape[-1] != num_spec:
            logger.warning(
                "D-Cut: selected draft probs shape mismatch: shape=%s num_spec=%s",
                tuple(probs.shape),
                num_spec,
            )
            return
    self._adaptive_probs_pending = True
    self._adaptive_num_reqs = num_reqs
    self._adaptive_req_ids = self.input_batch.req_ids.copy()
    self._adaptive_active = {
        self.input_batch.req_ids[i]
        for i in range(num_reqs)
        if (
            self.input_batch.num_computed_tokens_cpu[i]
            >= self.input_batch.num_prompt_tokens[i]
        )
    }
    # Non-blocking D2H on the default stream (the drafter runs there too); the
    # event lets _maybe_process_adaptive_probs verify completion cheaply.
    self._adaptive_probs_pinned[:num_reqs].copy_(probs.contiguous(), non_blocking=True)
    self._adaptive_probs_event.record()


def _maybe_process_adaptive_probs(self) -> None:
    """Consume step-N probs and update the controller's draft_len cache.

    Device-agnostic; npu.Event exposes the same query()/synchronize() API.
    """
    if not self._adaptive_probs_pending:
        return
    assert self._adaptive_probs_event is not None
    # Do not force a host-side wait here.  D-Cut can reuse the previous cached
    # decision for one more scheduler tick, while synchronizing every tick adds
    # a CPU/NPU barrier on the serving hot path and can make adaptive verify
    # slower than vanilla speculative decoding.
    if not self._adaptive_probs_event.query():
        return
    self._adaptive_probs_pending = False

    num_reqs = self._adaptive_num_reqs
    active = self._adaptive_active
    if active and self._verify_adaptive_controller is not None:
        assert self._adaptive_probs_pinned is not None
        self._verify_adaptive_controller.process_draft_output(
            selected_probs=self._adaptive_probs_pinned[:num_reqs],
            req_ids=self._adaptive_req_ids,
            active_draft_req_ids=active,
            batch_size=num_reqs,
        )


def profile_adaptive_cost(self) -> None:
    """Profile verifier ITL after warmup (called from NPUWorker)."""
    if getattr(self, "_verify_adaptive_controller", None) is not None:
        self._verify_adaptive_controller.profile_cost_table(self)

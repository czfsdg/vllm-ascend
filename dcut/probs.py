# SPDX-License-Identifier: Apache-2.0
"""Async D2H selected-probs queue + controller cache update."""
from __future__ import annotations


from .controller import _dcut_enable_drafter_probs
from .globals import logger
from .utils import (
    _dcut_selected_probs_from_graph,
    _dcut_selected_probs_from_reused_logits,
)


def _dcut_prepare_prob_capture(self, scheduler_output) -> None:
    """Reset per-step execution state before the drafter runs or replays."""
    drafter = getattr(self, "drafter", None)
    if drafter is not None:
        drafter._dcut_last_draft_ran_python = False
        # Graph replay updates the fixed-address tensor retained per bucket;
        # it does not reassign this Python attribute. Clear it so a replay can
        # never consume the final bucket captured during startup by accident.
        drafter._last_selected_probs = None
        # Force graph replay to select its retained logits by current output
        # shape instead of accidentally reusing the final startup-capture bucket.
        drafter._dcut_last_logits_for_probs = None

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
    draft_token_ids = getattr(self, "_draft_token_ids", None)
    ran_python = getattr(drafter, "_dcut_last_draft_ran_python", False)
    if ran_python:
        probs = drafter.take_last_selected_probs()
    else:
        probs = _dcut_selected_probs_from_graph(
            drafter,
            draft_token_ids,
        )
        if (
            probs is not None
            and not getattr(
                self,
                "_dcut_logged_graph_selected_probs",
                False,
            )
        ):
            logger.info(
                "D-Cut: using selected probabilities produced by the "
                "replayed draft graph."
            )
            self._dcut_logged_graph_selected_probs = True
    if probs is None:
        if (
            not ran_python
            and not getattr(
                self,
                "_dcut_logged_graph_probs_fallback",
                False,
            )
        ):
            logger.warning(
                "D-Cut: replayed draft graph has no matching selected-prob "
                "buffer; falling back to a graph-logits reduction."
            )
            self._dcut_logged_graph_probs_fallback = True
        try:
            probs = _dcut_selected_probs_from_reused_logits(
                drafter,
                draft_token_ids,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "D-Cut: deriving selected probs from reused logits failed: %s",
                e,
            )
            probs = None
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
    # Non-blocking D2H on the default stream (the drafter runs there too). The
    # next execute_model consumes it immediately before truncation, allowing
    # this copy to overlap the remainder of the current scheduler step.
    self._adaptive_probs_pinned[:num_reqs].copy_(
        probs.contiguous(),
        non_blocking=True,
    )
    self._adaptive_probs_event.record()


def _maybe_process_adaptive_probs(
    self,
    stage: str = "pre_truncate",
) -> None:
    """Consume queued probs before truncating the next verifier batch."""
    if not self._adaptive_probs_pending:
        return
    assert self._adaptive_probs_event is not None
    if not self._adaptive_probs_event.query():
        if getattr(self, "_dcut_skip_unready_probs", False):
            return
        # In the default pre_truncate path the copy has had the rest of the
        # previous iteration to complete. Synchronize only if it is still late,
        # so this step uses fresh probabilities and the next D2H queue is free.
        self._adaptive_probs_event.synchronize()
    self._adaptive_probs_pending = False

    num_reqs = self._adaptive_num_reqs
    active = self._adaptive_active
    if active:
        current_req_ids = set(
            self.input_batch.req_ids[
                : getattr(self.input_batch, "num_reqs", 0)
            ]
        )
        active = active & current_req_ids
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

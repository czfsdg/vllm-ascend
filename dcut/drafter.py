# SPDX-License-Identifier: Apache-2.0
"""Patch live Ascend drafter instance for selected draft probs."""
from __future__ import annotations

from types import MethodType

import torch

from .globals import logger
from .utils import _dcut_greedy_sample_with_selected_probs

def _dcut_patch_drafter_instance(drafter) -> None:
    """Patch the live Ascend drafter instance; robust to MRO/load order quirks."""
    if not hasattr(drafter, "take_last_selected_probs"):
        drafter.take_last_selected_probs = lambda: getattr(
            drafter, "_last_selected_probs", None
        )

    model = getattr(drafter, "model", None)
    if (
        model is not None
        and hasattr(model, "compute_logits")
        and not getattr(model, "_dcut_compute_logits_patched", False)
    ):
        orig_compute_logits = model.compute_logits

        def compute_logits(self_model, hidden_states, *args, **kwargs):
            logits = orig_compute_logits(hidden_states, *args, **kwargs)
            if getattr(drafter, "needs_draft_probs", False) and logits is not None:
                try:
                    token_ids = logits.argmax(dim=-1)
                    chosen = logits.gather(-1, token_ids.long().unsqueeze(-1))
                    selected_probs = (
                        chosen.squeeze(-1) - logits.logsumexp(dim=-1)
                    ).exp()
                    drafter._last_selected_probs = (
                        selected_probs.float().contiguous()
                    )
                    if not getattr(
                        drafter, "_dcut_logged_compute_logits_probs", False
                    ):
                        logger.warning(
                            "D-Cut: captured selected draft probs from "
                            "compute_logits on %s (logits_shape=%s).",
                            type(drafter).__name__,
                            tuple(logits.shape),
                        )
                        drafter._dcut_logged_compute_logits_probs = True
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "D-Cut: gather selected probs from compute_logits "
                        "failed: %s",
                        e,
                    )
                    drafter._last_selected_probs = None
            return logits

        model.compute_logits = MethodType(compute_logits, model)
        model._dcut_compute_logits_patched = True

    if (
        not hasattr(drafter, "compute_draft_token_ids")
        or getattr(drafter, "_dcut_instance_compute_patched", False)
    ):
        return

    orig_compute = drafter.compute_draft_token_ids

    def compute_draft_token_ids(self, hidden_states):
        self._last_selected_probs = None
        if not getattr(self, "needs_draft_probs", False):
            return orig_compute(hidden_states)
        try:
            logits = self.model.logits_processor(self.model.lm_head, hidden_states)
            logits = logits.contiguous()
            next_token, selected_probs = _dcut_greedy_sample_with_selected_probs(
                logits
            )
            self._last_selected_probs = selected_probs.float().contiguous()

            draft_map = getattr(self.model, "draft_id_to_target_id", None)
            if draft_map is None:
                return next_token
            bias = torch.index_select(
                draft_map, dim=0, index=next_token.view(-1)
            ).view(next_token.shape)
            return next_token + bias
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "D-Cut: gather selected probs in live drafter failed: %s", e
            )
            self._last_selected_probs = None
            return orig_compute(hidden_states)

    drafter.compute_draft_token_ids = MethodType(compute_draft_token_ids, drafter)
    drafter._dcut_instance_compute_patched = True



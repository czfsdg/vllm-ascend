# SPDX-License-Identifier: Apache-2.0
"""Patch Ascend spec-decode proposer for selected draft probs."""
from __future__ import annotations

import torch

from .globals import logger
from .utils import _dcut_greedy_sample_with_selected_probs

# ---------------------------------------------------------------------------
# Patch installers (idempotent, per class).  Targets are the *NPU* classes.
# ---------------------------------------------------------------------------

def _patch_proposer() -> None:
    """Patch the Ascend spec-decode proposer to expose selected draft probs.

    The Ascend dflash/PARD proposers inherit ``_sample_draft_tokens`` through a
    multi-level MRO (AscendDflashProposer -> AscendEagleProposer ->
    EagleProposer / AscendSpecDecodeBaseProposer -> SpecDecodeBaseProposer).
    We resolve, per proposer, the class in the MRO that actually *defines*
    ``_sample_draft_tokens`` and patch that class, so the wrapper is not
    shadowed by a subclass override.
    """
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    # Collect the concrete Ascend proposers that can run D-Cut (dflash / PARD).
    proposer_classes = []
    try:
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
        proposer_classes.append(AscendDflashProposer)
    except Exception as e:  # pragma: no cover
        logger.warning("D-Cut: could not import AscendDflashProposer: %s", e)
    try:
        from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
        proposer_classes.append(AscendDraftModelProposer)
    except Exception:  # PARD path is optional
        pass

    # Helper functions live on the shared base so every proposer can call them.
    if not getattr(SpecDecodeBaseProposer, "_dcut_helpers", False):
        @staticmethod
        def _should_collect_draft_probs(self):
            return getattr(self, "needs_draft_probs", False) and (
                getattr(self, "parallel_drafting", False)
                or getattr(self, "method", None) == "dflash"
            )

        @staticmethod
        def _gather_selected_probs(logits, token_ids, full_probs):
            idx = token_ids.long().unsqueeze(-1)
            if full_probs is not None:
                return full_probs.gather(-1, idx).squeeze(-1)
            chosen = logits.gather(-1, idx).squeeze(-1)
            return (chosen - logits.logsumexp(dim=-1)).exp()

        @staticmethod
        def _greedy_sample_with_selected_probs(logits):
            return _dcut_greedy_sample_with_selected_probs(logits)

        def take_last_selected_probs(self):
            return getattr(self, "_last_selected_probs", None)

        SpecDecodeBaseProposer.needs_draft_probs = False
        SpecDecodeBaseProposer._last_selected_probs = None
        SpecDecodeBaseProposer._should_collect_draft_probs = (
            _should_collect_draft_probs
        )
        SpecDecodeBaseProposer._gather_selected_probs = _gather_selected_probs
        SpecDecodeBaseProposer._greedy_sample_with_selected_probs = (
            _greedy_sample_with_selected_probs
        )
        SpecDecodeBaseProposer.take_last_selected_probs = take_last_selected_probs
        SpecDecodeBaseProposer._dcut_helpers = True

    compute_owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "compute_draft_token_ids" in klass.__dict__:
                if klass not in compute_owners:
                    compute_owners.append(klass)
                break

    for owner in compute_owners:
        if getattr(owner, "_dcut_compute_patched", False):
            continue
        _orig_compute = owner.compute_draft_token_ids

        def _make_compute_wrapper(orig):
            def compute_draft_token_ids(self, hidden_states):
                self._last_selected_probs = None
                if not type(self)._should_collect_draft_probs(self):
                    return orig(self, hidden_states)
                try:
                    logits = self.model.logits_processor(
                        self.model.lm_head, hidden_states
                    )
                    logits = logits.contiguous()
                    next_token, selected_probs = (
                        type(self)._greedy_sample_with_selected_probs(logits)
                    )
                    # Keep this flat here. Ascend may pad sample_hidden_states for
                    # lmhead TP; the runner slices and reshapes using real batch size.
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
                        "D-Cut: gather selected probs in compute_draft_token_ids "
                        "failed: %s",
                        e,
                    )
                    self._last_selected_probs = None
                    return orig(self, hidden_states)

            return compute_draft_token_ids

        owner.compute_draft_token_ids = _make_compute_wrapper(_orig_compute)
        owner._dcut_compute_patched = True
        logger.info(
            "D-Cut: patched compute_draft_token_ids on %s.", owner.__name__
        )

    # Find the distinct owner classes of _sample_draft_tokens across our
    # concrete proposers (usually a single class) and wrap each once.
    owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "_sample_draft_tokens" in klass.__dict__:
                if klass not in owners:
                    owners.append(klass)
                break

    for owner in owners:
        if getattr(owner, "_dcut_patched", False):
            continue
        _orig_sample = owner._sample_draft_tokens

        def _make_wrapper(orig):
            def _sample_draft_tokens(self, hidden_states, sampling_metadata):
                self._last_selected_probs = None
                out = orig(self, hidden_states, sampling_metadata)
                # D-Cut only targets parallel drafting (DFlash / PARD), where the
                # whole block is sampled in this single call -> selected_probs is
                # [B*T] which reshapes to [B, T].
                if type(self)._should_collect_draft_probs(self):
                    if isinstance(out, tuple):
                        token_ids = out[0]
                        full_probs = out[1] if len(out) > 1 else None
                    else:
                        token_ids = out
                        full_probs = None
                    try:
                        logits = (
                            None
                            if full_probs is not None
                            else self.model.compute_logits(hidden_states)
                        )
                        sel = type(self)._gather_selected_probs(
                            logits, token_ids, full_probs
                        )
                        self._last_selected_probs = sel.view(
                            -1, self.num_speculative_tokens
                        ).contiguous()
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning(
                            "D-Cut: gather selected probs failed: %s", e
                        )
                        self._last_selected_probs = None
                return out
            return _sample_draft_tokens

        owner._sample_draft_tokens = _make_wrapper(_orig_sample)
        owner._dcut_patched = True
        logger.info(
            "D-Cut: patched _sample_draft_tokens on %s.", owner.__name__
        )



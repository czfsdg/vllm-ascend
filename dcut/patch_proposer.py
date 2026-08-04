# SPDX-License-Identifier: Apache-2.0
"""Patch Ascend spec-decode proposer for selected draft probs."""
from __future__ import annotations

import torch

from .globals import logger
from .utils import (
    _dcut_greedy_sample_with_selected_probs,
    _dcut_in_graph_capture,
    _dcut_selected_token_probs,
)

# ---------------------------------------------------------------------------
# Patch installers (idempotent, per class).  Targets are the *NPU* classes.
# ---------------------------------------------------------------------------


def _patch_proposer() -> None:
    """Patch the Ascend spec-decode proposer to expose selected draft probs.

    vLLM 0.23 Ascend proposers sample through compute_draft_token_ids.
    Patch the concrete method owner so DFlash and parallel draft-model
    proposers expose the selected-token probabilities used by D-Cut.
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
            return _dcut_selected_token_probs(logits, token_ids)

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

    run_merged_owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "_run_merged_draft" in klass.__dict__:
                if klass not in run_merged_owners:
                    run_merged_owners.append(klass)
                break

    for owner in run_merged_owners:
        if getattr(owner, "_dcut_run_merged_patched", False):
            continue
        original_run_merged = owner._run_merged_draft

        def _make_run_merged_wrapper(orig):
            def _run_merged_draft(self, *args, **kwargs):
                in_graph_capture = _dcut_in_graph_capture()
                self._dcut_last_draft_ran_python = True
                self._dcut_last_logits_for_probs = None
                self._last_selected_probs = None
                out = orig(self, *args, **kwargs)
                if not type(self)._should_collect_draft_probs(self):
                    return out

                logits = getattr(self, "_dcut_last_logits_for_probs", None)
                if logits is None:
                    return out
                if in_graph_capture:
                    # Keep fixed-address graph tensors by output shape. Replay
                    # updates these tensors even though this Python wrapper does
                    # not execute again.
                    self._dcut_graph_logits_for_probs_by_shape = getattr(
                        self,
                        "_dcut_graph_logits_for_probs_by_shape",
                        {},
                    )
                    self._dcut_graph_logits_for_probs_by_numel = getattr(
                        self,
                        "_dcut_graph_logits_for_probs_by_numel",
                        {},
                    )
                    self._dcut_graph_logits_for_probs_by_shape[
                        tuple(out.shape)
                    ] = logits
                    self._dcut_graph_logits_for_probs_by_numel[
                        int(out.numel())
                    ] = logits
                    self._dcut_graph_logits_for_probs = logits
                    self._dcut_graph_logits_for_probs_ready = True
                    return out

                try:
                    flat_token_ids = out.reshape(-1)
                    n_tokens = min(
                        int(logits.shape[0]),
                        int(flat_token_ids.numel()),
                    )
                    selected_probs = _dcut_selected_token_probs(
                        logits[:n_tokens],
                        flat_token_ids[:n_tokens],
                    )
                    if selected_probs.numel() == out.numel():
                        selected_probs = selected_probs.view(out.shape)
                    self._last_selected_probs = (
                        selected_probs.float().contiguous()
                    )
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "D-Cut: reused-argmax selected-prob capture failed: %s",
                        e,
                    )
                    self._last_selected_probs = None
                finally:
                    self._dcut_last_logits_for_probs = None
                return out

            return _run_merged_draft

        owner._run_merged_draft = _make_run_merged_wrapper(
            original_run_merged
        )
        owner._dcut_run_merged_patched = True
        logger.info(
            "D-Cut: patched _run_merged_draft on %s.",
            owner.__name__,
        )

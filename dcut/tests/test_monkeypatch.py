# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest


def test_patch_proposer_captures_module_logits_processor_with_forward_hook():
    torch = pytest.importorskip("torch")
    monkeypatch_module = pytest.importorskip("dcut.monkeypatch")
    nn = torch.nn

    class FakeLogitsProcessor(nn.Module):
        def forward(self, lm_head, hidden_states):
            return hidden_states

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = None
            self.logits_processor = FakeLogitsProcessor()

    class FakeProposer:
        method = "dflash"
        parallel_drafting = False
        num_speculative_tokens = 1

        def __init__(self):
            self.model = FakeModel()
            self.runner = SimpleNamespace(_dcut_state_initialized=True, dcut_adaptive_enabled=True)

        def _run_merged_draft(self):
            logits = self.model.logits_processor(self.model.lm_head, torch.tensor([[0.1, 0.9]]))
            return logits.argmax(dim=-1).view(-1, 1)

    module = SimpleNamespace(AscendSpecDecodeBaseProposer=FakeProposer)

    assert monkeypatch_module._patch_proposer_module(module)

    proposer = FakeProposer()
    draft_token_ids = proposer._run_merged_draft()

    expected_prob = torch.softmax(torch.tensor([0.1, 0.9]), 0)[1]
    assert draft_token_ids.tolist() == [[1]]
    assert proposer.latest_draft_token_probs.shape == (1, 1)
    assert torch.allclose(proposer.latest_draft_token_probs, torch.tensor([[expected_prob]]))

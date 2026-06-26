# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest


def import_monkeypatch_with_fake_vllm(monkeypatch):
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    vllm_module = ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    sys.modules.pop("dcut.monkeypatch", None)
    return importlib.import_module("dcut.monkeypatch")


def test_apply_dcut_draft_lens_updates_scheduler_token_counts(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)

    @dataclass(frozen=True)
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12], "r1": [20, 21]},
        num_scheduled_tokens={"r0": 4, "r1": 3},
        total_num_scheduled_tokens=7,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 1, "r1": 0},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(apply_adaptive_lengths=True, min_adaptive_draft_len=0),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10]}
    assert updated.num_scheduled_tokens == {"r0": 2, "r1": 1}
    assert updated.total_num_scheduled_tokens == 3
    assert runner.dcut_next_draft_lens == {}


def test_apply_dcut_draft_lens_uses_configured_minimum_draft_len(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)

    @dataclass(frozen=True)
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12, 13]},
        num_scheduled_tokens={"r0": 5},
        total_num_scheduled_tokens=5,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 0},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(apply_adaptive_lengths=True, min_adaptive_draft_len=2),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10, 11]}
    assert updated.num_scheduled_tokens == {"r0": 3}
    assert updated.total_num_scheduled_tokens == 3


def test_apply_dcut_draft_lens_respects_accepted_token_floor(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)

    @dataclass(frozen=True)
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12, 13, 14, 15, 16]},
        num_scheduled_tokens={"r0": 8},
        total_num_scheduled_tokens=8,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 1},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(apply_adaptive_lengths=True, min_adaptive_draft_len=0),
        input_batch=SimpleNamespace(
            req_id_to_index={"r0": 0},
            num_accepted_tokens_cpu=[8],
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10, 11, 12, 13, 14, 15, 16]}
    assert updated.num_scheduled_tokens == {"r0": 8}
    assert updated.total_num_scheduled_tokens == 8


def test_runtime_event_logging_can_be_disabled(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    emitted: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        monkeypatch_module,
        "_emit_dcut_log",
        lambda message, *args: emitted.append((message, args)),
    )

    @dataclass(frozen=True)
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12]},
        num_scheduled_tokens={"r0": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 1},
        dcut_logged_first_truncation=False,
        dcut_last_concurrency_log_ts=0.0,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True,
            log_runtime_events=False,
            log_concurrency_interval_s=0.0,
            min_adaptive_draft_len=0,
        ),
        input_batch=SimpleNamespace(num_reqs=1),
    )

    monkeypatch_module._log_concurrency(runner, scheduler_output)
    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10]}
    assert emitted == []


def test_apply_dcut_draft_lens_safe_mode_does_not_rewrite_scheduler_output(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)

    @dataclass(frozen=True)
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12]},
        num_scheduled_tokens={"r0": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 1},
        dcut_logged_safe_mode=False,
        dcut_config=SimpleNamespace(apply_adaptive_lengths=False),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated is scheduler_output
    assert runner.dcut_next_draft_lens == {"r0": 1}
    assert not runner.dcut_logged_safe_mode


def test_patch_proposer_captures_module_logits_processor_with_forward_hook(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
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

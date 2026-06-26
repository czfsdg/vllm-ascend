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
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10]}
    assert updated.num_scheduled_tokens == {"r0": 2, "r1": 1}
    assert updated.total_num_scheduled_tokens == 3
    assert runner.dcut_next_draft_lens == {}


def test_apply_dcut_draft_lens_truncates_draft_token_list_without_mutating_source(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    draft_token_ids = [10, 11, 12, 13]
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": draft_token_ids},
        num_scheduled_tokens={"r0": 5},
        total_num_scheduled_tokens=5,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 2},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens["r0"] == [10, 11]
    assert updated.scheduled_spec_decode_tokens["r0"] is not draft_token_ids
    assert draft_token_ids == [10, 11, 12, 13]
    assert updated.num_scheduled_tokens == {"r0": 3}
    assert updated.total_num_scheduled_tokens == 3


def test_apply_dcut_draft_lens_mutates_mutable_scheduler_output(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12]},
        num_scheduled_tokens={"r0": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 1},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated is scheduler_output
    assert scheduler_output.scheduled_spec_decode_tokens == {"r0": [10]}
    assert scheduler_output.num_scheduled_tokens == {"r0": 2}
    assert scheduler_output.total_num_scheduled_tokens == 2


def test_apply_dcut_draft_lens_preserves_native_count_without_removed_tokens(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12, 13, 14, 15, 16]},
        # Simulate a native scheduler count that is lower than len(spec)+1.
        num_scheduled_tokens={"r0": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 7},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated is scheduler_output
    assert scheduler_output.scheduled_spec_decode_tokens == {"r0": [10, 11, 12, 13, 14, 15, 16]}
    assert scheduler_output.num_scheduled_tokens == {"r0": 4}
    assert scheduler_output.total_num_scheduled_tokens == 4


def test_apply_dcut_draft_lens_count_floor_uses_target_len(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12, 13, 14, 15, 16]},
        num_scheduled_tokens={"r0": 4},
        total_num_scheduled_tokens=4,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 3},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10, 11, 12]}
    assert updated.num_scheduled_tokens == {"r0": 4}
    assert updated.total_num_scheduled_tokens == 4


def test_align_scheduled_spec_decode_tokens_with_counts(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "r0": [10, 11, 12, 13, 14, 15, 16],
            "r1": [20, 21, 22, 23, 24, 25, 26],
        },
        num_scheduled_tokens={"r0": 6, "r1": 8},
    )

    changed = monkeypatch_module._align_scheduled_spec_decode_tokens_with_counts(scheduler_output)

    assert changed is True
    assert scheduler_output.scheduled_spec_decode_tokens == {
        "r0": [10, 11, 12, 13, 14],
        "r1": [20, 21, 22, 23, 24, 25, 26],
    }


def test_apply_dcut_draft_lens_updates_scheduled_dict_in_place(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduled = {"r0": [10, 11, 12, 13, 14, 15, 16]}
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens=scheduled,
        num_scheduled_tokens={"r0": 8},
        total_num_scheduled_tokens=8,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 5},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens is scheduled
    assert scheduled == {"r0": [10, 11, 12, 13, 14]}
    assert updated.num_scheduled_tokens == {"r0": 6}


def test_apply_dcut_draft_lens_caps_count_by_retained_decode_width(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12, 13, 14, 15, 16]},
        num_scheduled_tokens={"r0": 63},
        total_num_scheduled_tokens=63,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 5},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10, 11, 12, 13, 14]}
    assert updated.num_scheduled_tokens == {"r0": 6}
    assert updated.total_num_scheduled_tokens == 6


def test_apply_dcut_draft_lens_recomputes_total_from_updated_counts(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12], "r1": [20, 21, 22]},
        num_scheduled_tokens={"r0": 4, "r1": 4},
        # Simulate a stale/runner-local total that should not be decremented.
        total_num_scheduled_tokens=3,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 1, "r1": 2},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated is scheduler_output
    assert scheduler_output.num_scheduled_tokens == {"r0": 2, "r1": 3}
    assert scheduler_output.total_num_scheduled_tokens == 5


def test_apply_dcut_draft_lens_normalizes_negative_scheduled_counts(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r0", "r1"]),
        scheduled_spec_decode_tokens={"r0": [10, 11, 12]},
        num_scheduled_tokens={"r0": -4, "r1": -1},
        total_num_scheduled_tokens=-5,
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_next_draft_lens={"r0": 2},
        dcut_logged_first_truncation=False,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated is scheduler_output
    assert scheduler_output.scheduled_spec_decode_tokens == {"r0": [10, 11]}
    assert scheduler_output.num_scheduled_tokens == {"r0": 3, "r1": 1}
    assert scheduler_output.total_num_scheduled_tokens == 4


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
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=2
        ),
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
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True, mutate_scheduler_output=True, min_adaptive_draft_len=0
        ),
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
            mutate_scheduler_output=True,
            min_adaptive_draft_len=0,
        ),
        input_batch=SimpleNamespace(num_reqs=1),
    )

    monkeypatch_module._log_concurrency(runner, scheduler_output)
    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated.scheduled_spec_decode_tokens == {"r0": [10]}
    assert emitted == []


def test_debug_scheduler_state_logs_compact_summary(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    emitted: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        monkeypatch_module,
        "_emit_dcut_log",
        lambda message, *args: emitted.append((message, args)),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [10, 11], "r1": [20]},
        num_scheduled_tokens={"r0": 3, "r1": 4},
        total_num_scheduled_tokens=7,
    )
    runner = SimpleNamespace(dcut_config=SimpleNamespace(debug_scheduler_state=True))

    monkeypatch_module._debug_scheduler_state(runner, scheduler_output, "unit")

    assert emitted
    message, args = emitted[0]
    assert message.startswith("D-Cut debug %s:")
    assert args[0] == "unit"
    assert args[-1] == 1


def test_update_dcut_next_draft_lens_defaults_to_uniform_batch_length(monkeypatch):
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def to(self, device):
            return self

        def tolist(self):
            return self.values

    draft_token_ids = SimpleNamespace(shape=(3, 7))
    drafter = SimpleNamespace(
        latest_draft_token_probs=FakeTensor(
            [
                [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ]
        )
    )
    runner = SimpleNamespace(
        _dcut_state_initialized=True,
        dcut_adaptive_enabled=True,
        dcut_config=SimpleNamespace(
            apply_adaptive_lengths=True,
            query_len_levels=lambda max_draft_len: [1, 4],
            cost_table=None,
            min_prefix_prob=0.0,
            uniform_adaptive_lengths=True,
        ),
        drafter=drafter,
        input_batch=SimpleNamespace(req_ids=["r0", "r1", "r2"]),
        dcut_logged_first_plan=True,
        dcut_next_draft_lens={},
    )

    monkeypatch_module._update_dcut_next_draft_lens(runner, draft_token_ids)

    assert len(set(runner.dcut_next_draft_lens.values())) == 1
    assert runner.dcut_next_draft_lens == {"r0": 3, "r1": 3, "r2": 3}


def test_apply_dcut_draft_lens_does_not_mutate_scheduler_without_opt_in(monkeypatch):
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
        dcut_logged_safe_apply_bypass=False,
        dcut_config=SimpleNamespace(apply_adaptive_lengths=True, mutate_scheduler_output=False),
    )

    updated = monkeypatch_module._apply_dcut_draft_lens(runner, scheduler_output)

    assert updated is scheduler_output
    assert scheduler_output.scheduled_spec_decode_tokens == {"r0": [10, 11, 12]}
    assert scheduler_output.num_scheduled_tokens == {"r0": 4}
    assert runner.dcut_next_draft_lens == {}
    assert runner.dcut_logged_safe_apply_bypass


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


def test_selected_token_probs_from_logits_matches_softmax(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch_module = import_monkeypatch_with_fake_vllm(monkeypatch)
    logits = torch.tensor([[0.1, 0.9, -0.2], [1.0, 0.5, -0.5]])
    draft_token_ids = torch.tensor([1, 0])

    selected_probs = monkeypatch_module._selected_token_probs_from_logits(logits, draft_token_ids)
    expected_probs = torch.softmax(logits.float(), dim=-1).gather(dim=-1, index=draft_token_ids.view(-1, 1)).view(-1)

    assert torch.allclose(selected_probs, expected_probs)


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

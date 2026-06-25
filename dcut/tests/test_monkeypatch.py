# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from dataclasses import dataclass

from dcut.verify_adaptive_config import VerifyAdaptiveConfig


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _install_fake_vllm_logger():
    vllm_module = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: _Logger()
    sys.modules.setdefault("vllm", vllm_module)
    sys.modules["vllm.logger"] = logger_module


def test_proposer_patch_does_not_replace_logits_processor_child_module(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    class FakeDraftTokenIds:
        def reshape(self, *args):
            return self

    class FakeLogitsProcessor:
        def __init__(self):
            self.called = False

        def forward(self, *args, **kwargs):
            self.called = True
            return "logits"

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class FakeModel:
        def __init__(self):
            self.logits_processor = FakeLogitsProcessor()
            self.lm_head = object()

        def __setattr__(self, name, value):
            if name == "logits_processor" and hasattr(self, "logits_processor"):
                msg = "torch.nn.Module child replacement should not happen"
                raise TypeError(msg)
            super().__setattr__(name, value)

    class FakeProposer:
        def __init__(self):
            self.runner = object()
            self.model = FakeModel()

        def _run_merged_draft(self):
            self.model.logits_processor(self.model.lm_head, "hidden_states")
            return FakeDraftTokenIds()

    module = types.SimpleNamespace(AscendSpecDecodeBaseProposer=FakeProposer)
    assert dcut_monkeypatch._patch_proposer_module(module)

    recorded = {}
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)
    monkeypatch.setattr(
        dcut_monkeypatch,
        "_record_selected_token_probs",
        lambda proposer, logits, draft_token_ids: recorded.setdefault("logits", logits),
    )

    proposer = FakeProposer()
    proposer._run_merged_draft()

    assert proposer.model.logits_processor.called
    assert recorded["logits"] == "logits"


def test_record_probs_skips_acl_graph_capture(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    class FakeForwardContext:
        capturing = True

    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module._forward_context = FakeForwardContext()
    forward_context_module.get_forward_context = lambda: forward_context_module._forward_context
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_module)

    class FakeRunner:
        dcut_logged_skip_capture = False

    class FakeProposer:
        runner = FakeRunner()
        method = "dflash"
        parallel_drafting = False

    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)

    dcut_monkeypatch._record_selected_token_probs(FakeProposer(), object(), object())

    assert not hasattr(FakeProposer, "latest_draft_token_probs")
    assert FakeProposer.runner.dcut_logged_skip_capture


def test_acl_graph_capture_check_treats_missing_forward_context_as_not_capturing(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    def raise_if_called():
        msg = "Forward context is not set"
        raise AssertionError(msg)

    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module._forward_context = None
    forward_context_module.get_forward_context = raise_if_called
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_module)

    assert not dcut_monkeypatch._in_acl_graph_capture()


def test_apply_dcut_draft_lens_observe_only_leaves_scheduler_output(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    scheduler_output = types.SimpleNamespace(
        scheduled_spec_decode_tokens={"req-0": [1, 2, 3, 4]},
    )
    runner = types.SimpleNamespace(
        dcut_next_draft_lens={"req-0": 2},
        dcut_config=types.SimpleNamespace(apply_truncation=False),
        dcut_logged_observe_only=False,
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)

    result = dcut_monkeypatch._apply_dcut_draft_lens(runner, scheduler_output)

    assert result is scheduler_output
    assert scheduler_output.scheduled_spec_decode_tokens["req-0"] == [1, 2, 3, 4]
    assert runner.dcut_next_draft_lens == {}
    assert runner.dcut_logged_observe_only


def test_apply_dcut_draft_lens_updates_scheduler_token_counts(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    @dataclass(frozen=True)
    class SchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = SchedulerOutput(
        scheduled_spec_decode_tokens={"req-0": [11, 12, 13, 14], "req-1": [21, 22]},
        num_scheduled_tokens={"req-0": 5, "req-1": 3},
        total_num_scheduled_tokens=8,
    )
    runner = types.SimpleNamespace(
        dcut_next_draft_lens={"req-0": 2},
        dcut_config=types.SimpleNamespace(apply_truncation=True),
        dcut_logged_first_truncation=False,
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)

    result = dcut_monkeypatch._apply_dcut_draft_lens(runner, scheduler_output)

    assert result.scheduled_spec_decode_tokens == {
        "req-0": [11, 12],
        "req-1": [21, 22],
    }
    assert result.num_scheduled_tokens == {"req-0": 3, "req-1": 3}
    assert result.total_num_scheduled_tokens == 6
    assert scheduler_output.num_scheduled_tokens == {"req-0": 5, "req-1": 3}
    assert runner.dcut_next_draft_lens == {}
    assert runner.dcut_logged_first_truncation



def test_apply_dcut_draft_lens_rolls_back_scheduler_request_counters(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    @dataclass(frozen=True)
    class SchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    request = types.SimpleNamespace(
        num_computed_tokens=10,
        num_output_placeholders=4,
    )
    scheduler_output = SchedulerOutput(
        scheduled_spec_decode_tokens={"req-0": [11, 12, 13, 14]},
        num_scheduled_tokens={"req-0": 5},
        total_num_scheduled_tokens=5,
    )
    runner = types.SimpleNamespace(
        requests={"req-0": request},
        dcut_next_draft_lens={"req-0": 2},
        dcut_config=types.SimpleNamespace(apply_truncation=True),
        dcut_logged_first_truncation=False,
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)

    result = dcut_monkeypatch._apply_dcut_draft_lens(runner, scheduler_output)

    assert result.scheduled_spec_decode_tokens == {"req-0": [11, 12]}
    assert result.num_scheduled_tokens == {"req-0": 3}
    assert result.total_num_scheduled_tokens == 3
    assert request.num_computed_tokens == 8
    assert request.num_output_placeholders == 2


def test_apply_dcut_draft_lens_removes_zero_length_spec_entries(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    @dataclass(frozen=True)
    class SchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = SchedulerOutput(
        scheduled_spec_decode_tokens={"req-0": [11, 12, 13]},
        num_scheduled_tokens={"req-0": 4},
        total_num_scheduled_tokens=4,
    )
    runner = types.SimpleNamespace(
        dcut_next_draft_lens={"req-0": 0},
        dcut_config=types.SimpleNamespace(apply_truncation=True),
        dcut_logged_first_truncation=False,
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)

    result = dcut_monkeypatch._apply_dcut_draft_lens(runner, scheduler_output)

    assert result.scheduled_spec_decode_tokens == {}
    assert result.num_scheduled_tokens == {"req-0": 1}
    assert result.total_num_scheduled_tokens == 1


def test_apply_dcut_draft_lens_preserves_accepted_token_segment(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    @dataclass(frozen=True)
    class SchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = SchedulerOutput(
        scheduled_spec_decode_tokens={"req-0": [11, 12, 13, 14]},
        num_scheduled_tokens={"req-0": 5},
        total_num_scheduled_tokens=5,
    )
    runner = types.SimpleNamespace(
        dcut_next_draft_lens={"req-0": 1},
        dcut_config=types.SimpleNamespace(apply_truncation=True),
        dcut_logged_first_truncation=False,
        dcut_logged_acceptance_floor=False,
        input_batch=types.SimpleNamespace(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            num_accepted_tokens_cpu=[3],
        ),
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)

    result = dcut_monkeypatch._apply_dcut_draft_lens(runner, scheduler_output)

    assert result.scheduled_spec_decode_tokens == {"req-0": [11, 12]}
    assert result.num_scheduled_tokens == {"req-0": 3}
    assert result.total_num_scheduled_tokens == 3
    assert runner.dcut_logged_acceptance_floor


def test_update_dcut_next_draft_lens_logs_every_configured_plan(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    class FakeProbs:
        def detach(self):
            return self

        def to(self, device):
            return self

        def tolist(self):
            return [[0.9, 0.8, 0.7, 0.6]]

    class FakeDraftTokenIds:
        shape = (1, 4)

    logs = []
    runner = types.SimpleNamespace(
        drafter=types.SimpleNamespace(latest_draft_token_probs=FakeProbs()),
        input_batch=types.SimpleNamespace(req_ids=["req-0"]),
        dcut_config=VerifyAdaptiveConfig(apply_truncation=True, log_every_n_plans=1),
        dcut_next_draft_lens={},
        dcut_logged_first_plan=True,
        dcut_plan_count=0,
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)
    monkeypatch.setattr(dcut_monkeypatch, "_in_acl_graph_capture", lambda: False)
    monkeypatch.setattr(
        dcut_monkeypatch,
        "_log_info",
        lambda message, *args, **kwargs: logs.append(message % args),
    )

    dcut_monkeypatch._update_dcut_next_draft_lens(runner, FakeDraftTokenIds())

    assert runner.dcut_plan_count == 1
    assert any("verifier_tokens=" in log and "draft_lens=" in log for log in logs)


def test_proposer_patch_observe_only_does_not_wrap_logits(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    class FakeDraftTokenIds:
        def reshape(self, *args):
            return self

    class FakeModel:
        def __init__(self):
            self.compute_logits = self._compute_logits
            self.compute_logits_calls = 0

        def _compute_logits(self, *args, **kwargs):
            self.compute_logits_calls += 1
            return "logits"

    class FakeRunner:
        dcut_config = types.SimpleNamespace(apply_truncation=False)

    class FakeProposer:
        def __init__(self):
            self.runner = FakeRunner()
            self.model = FakeModel()
            self.compute_logits_seen_by_draft = None

        def _run_merged_draft(self):
            self.compute_logits_seen_by_draft = self.model.compute_logits
            self.model.compute_logits("hidden_states")
            return FakeDraftTokenIds()

    module = types.SimpleNamespace(AscendSpecDecodeBaseProposer=FakeProposer)
    assert dcut_monkeypatch._patch_proposer_module(module)

    recorded = {}
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)
    monkeypatch.setattr(
        dcut_monkeypatch,
        "_record_selected_token_probs",
        lambda proposer, logits, draft_token_ids: recorded.setdefault("logits", logits),
    )

    proposer = FakeProposer()
    original_compute_logits = proposer.model.compute_logits
    proposer._run_merged_draft()

    assert proposer.compute_logits_seen_by_draft is original_compute_logits
    assert proposer.model.compute_logits is original_compute_logits
    assert proposer.model.compute_logits_calls == 1
    assert recorded == {}


def test_update_dcut_next_draft_lens_observe_only_does_not_plan(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    class FakeDraftTokenIds:
        shape = (1, 4)

    runner = types.SimpleNamespace(
        drafter=types.SimpleNamespace(latest_draft_token_probs=object()),
        input_batch=types.SimpleNamespace(req_ids=["req-0"]),
        dcut_config=VerifyAdaptiveConfig(apply_truncation=False),
        dcut_next_draft_lens={"stale": 1},
        dcut_plan_count=0,
    )
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)
    monkeypatch.setattr(dcut_monkeypatch, "_in_acl_graph_capture", lambda: False)

    dcut_monkeypatch._update_dcut_next_draft_lens(runner, FakeDraftTokenIds())

    assert runner.dcut_next_draft_lens == {}
    assert runner.dcut_plan_count == 0

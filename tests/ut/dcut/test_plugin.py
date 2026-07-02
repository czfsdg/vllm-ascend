# SPDX-License-Identifier: Apache-2.0

import dcut.plugin as plugin


class FakeSpeculativeConfig:
    method = "dflash"
    num_speculative_tokens = 4


class FakeInputBatch:
    num_reqs = 32


class FakeRunner:
    speculative_config = FakeSpeculativeConfig()
    input_batch = FakeInputBatch()

    def __init__(self):
        self.initialized = True

    def propose_draft_token_ids(self):
        return [[1, 2]]


def test_patch_runner_class_is_lazy_and_adds_plan_only_state(monkeypatch):
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "0")
    monkeypatch.setattr(plugin, "_PATCHED", False)

    plugin._patch_runner_class(FakeRunner)
    runner = FakeRunner()
    result = runner.propose_draft_token_ids()

    assert runner.initialized
    assert result == [[1, 2]]
    assert runner.dcut_cost_table.max_verify_len == 4
    assert runner.dcut_last_decision.requested_len == 4
    assert runner.dcut_last_decision.batch_size == 32


def test_accuracy_safe_mode_returns_no_draft_tokens(monkeypatch):
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "1")
    monkeypatch.setattr(plugin, "_PATCHED", False)

    plugin._patch_runner_class(FakeRunner)
    runner = FakeRunner()

    assert runner.propose_draft_token_ids() is None
    assert runner.dcut_accuracy_safe_mode is True
    assert "dflash" in runner.dcut_target_only_methods


def test_repeated_identical_decisions_are_throttled(monkeypatch):
    class FreshSpeculativeConfig:
        method = "dflash"
        num_speculative_tokens = 4

    class FreshInputBatch:
        num_reqs = 1

    class FreshRunner:
        speculative_config = FreshSpeculativeConfig()
        input_batch = FreshInputBatch()

        def __init__(self):
            self.initialized = True

        def propose_draft_token_ids(self):
            return [[1, 2]]

    logs = []
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "0")
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.setattr(plugin, "_visible_log", lambda level, message, *args: logs.append(message % args))

    plugin._patch_runner_class(FreshRunner)
    runner = FreshRunner()

    for _ in range(3):
        assert runner.propose_draft_token_ids() == [[1, 2]]

    decision_logs = [log for log in logs if "cut-policy decision" in log]
    assert len(decision_logs) == 1
    assert "repeat_count=1" in decision_logs[0]
    assert runner.dcut_decision_count == 3


def test_acceptance_rate_uses_runtime_counters(monkeypatch):
    class FreshSpeculativeConfig:
        method = "dflash"
        num_speculative_tokens = 4

    class FreshInputBatch:
        num_reqs = 1

    class FreshRunner:
        speculative_config = FreshSpeculativeConfig()
        input_batch = FreshInputBatch()

        def __init__(self):
            self.spec_decode_metrics = {
                "num_accepted_tokens": 2,
                "num_draft_tokens": 4,
            }

        def propose_draft_token_ids(self):
            return [[1, 2]]

    logs = []
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "0")
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.setattr(plugin, "_visible_log", lambda level, message, *args: logs.append(message % args))

    plugin._patch_runner_class(FreshRunner)
    runner = FreshRunner()

    assert runner.propose_draft_token_ids() == [[1, 2]]
    assert runner.dcut_last_decision.acceptance_rate == 0.5
    decision_logs = [log for log in logs if "cut-policy decision" in log]
    assert "acceptance_source=runtime_counters" in decision_logs[0]

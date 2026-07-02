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


def test_repeated_identical_decisions_are_logged(monkeypatch):
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
    monkeypatch.setattr(
        plugin,
        "_visible_log",
        lambda level, message, *args: logs.append(message % args),
    )

    plugin._patch_runner_class(FreshRunner)
    runner = FreshRunner()

    for _ in range(3):
        assert runner.propose_draft_token_ids() == [[1, 2]]

    decision_logs = [log for log in logs if "cut-policy decision" in log]
    assert len(decision_logs) == 3
    assert "repeat_count=1" in decision_logs[0]
    assert "repeat_count=3" in decision_logs[-1]
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
    monkeypatch.setattr(
        plugin,
        "_visible_log",
        lambda level, message, *args: logs.append(message % args),
    )

    plugin._patch_runner_class(FreshRunner)
    runner = FreshRunner()

    assert runner.propose_draft_token_ids() == [[1, 2]]
    assert runner.dcut_last_decision.acceptance_rate == 0.5
    decision_logs = [log for log in logs if "cut-policy decision" in log]
    assert "acceptance_source=runtime_counters" in decision_logs[0]


def test_acceptance_rate_uses_metrics_log_bridge(monkeypatch, tmp_path):
    metrics_path = tmp_path / "dcut_metrics.json"
    monkeypatch.setattr(plugin, "METRICS_FILE_PATH", str(metrics_path))

    record = type(
        "Record",
        (),
        {
            "getMessage": lambda self: (
                "SpecDecoding metrics: Mean acceptance length: 2.99, "
                "Accepted throughput: 4.48 tokens/s, Drafted throughput: "
                "15.78 tokens/s, Accepted: 147 tokens, Drafted: 518 "
                "tokens, Avg Draft acceptance rate: 28.4%"
            ),
        },
    )()
    plugin._DcutMetricsLogHandler().emit(record)

    class Runner:
        pass

    runner = Runner()
    rate, source = plugin._acceptance_rate_from_runner(runner)

    assert round(rate, 3) == round(147 / 518, 3)
    assert source == "spec_decoding_metrics_log"


def test_acceptance_rate_uses_per_step_sampled_token_lengths(monkeypatch):
    class FreshSpeculativeConfig:
        method = "dflash"
        num_speculative_tokens = 4

    class FreshInputBatch:
        num_reqs = 2

    class FreshRunner:
        speculative_config = FreshSpeculativeConfig()
        input_batch = FreshInputBatch()

        def __init__(self):
            pass

        def propose_draft_token_ids(self, sampled_token_ids):
            return [[1, 2]]

    logs = []
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "0")
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.setattr(
        plugin,
        "_visible_log",
        lambda level, message, *args: logs.append(message % args),
    )

    plugin._patch_runner_class(FreshRunner)
    runner = FreshRunner()

    sampled_token_ids = [[1, 2, 3], [4]]
    assert runner.propose_draft_token_ids(sampled_token_ids) == [[1, 2]]
    assert runner.dcut_last_decision.acceptance_rate == 0.25
    assert runner.dcut_last_accepted_tokens == 2
    assert runner.dcut_last_drafted_tokens == 8
    decision_logs = [log for log in logs if "cut-policy decision" in log]
    assert "acceptance_source=sampled_token_lengths" in decision_logs[0]
    assert "accepted_tokens=2" in decision_logs[0]
    assert "drafted_tokens=8" in decision_logs[0]
    assert "batch_dcut_plan=[#0:accept=0.500,cut=4,#1:accept=0.000,cut=4]" in decision_logs[0]


def test_acceptance_rate_prefers_bookkeeping_valid_sampled_tokens(monkeypatch):
    class FreshSpeculativeConfig:
        method = "dflash"
        num_speculative_tokens = 4

    class FreshInputBatch:
        num_reqs = 2

    class FreshRunner:
        speculative_config = FreshSpeculativeConfig()
        input_batch = FreshInputBatch()

        def __init__(self):
            pass

        def _bookkeeping_sync(self):
            return (None, [[101, 102, 103], [201, 202]], None)

        def propose_draft_token_ids(self, sampled_token_ids):
            return sampled_token_ids

    logs = []
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "0")
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.setattr(
        plugin,
        "_visible_log",
        lambda level, message, *args: logs.append(message % args),
    )

    plugin._patch_runner_class(FreshRunner)
    runner = FreshRunner()

    runner._bookkeeping_sync()
    assert runner.propose_draft_token_ids(object()) is not None

    assert runner.dcut_last_decision.acceptance_rate == 0.375
    assert runner.dcut_last_accepted_tokens == 3
    assert runner.dcut_last_drafted_tokens == 8
    assert runner.dcut_last_per_request_acceptance == [0.5, 0.25]
    decision_logs = [log for log in logs if "cut-policy decision" in log]
    assert "acceptance_source=bookkeeping_valid_sampled_tokens" in decision_logs[0]
    assert "accepted_tokens=3" in decision_logs[0]
    assert "drafted_tokens=8" in decision_logs[0]
    assert "per_request_acceptance=[0.5, 0.25]" in decision_logs[0]
    assert "batch_dcut_plan=[#0:accept=0.500,cut=4,#1:accept=0.250,cut=4]" in decision_logs[0]

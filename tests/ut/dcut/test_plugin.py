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

    def _adaptive_profile_run(self, scheduled_tokens, *args):
        return "fake_profile", float(sum(scheduled_tokens)), sum(scheduled_tokens)


def test_patch_runner_class_is_lazy_and_adds_plan_only_state(monkeypatch):
    logs = []
    monkeypatch.setenv("DCUT_ACCURACY_SAFE_MODE", "0")
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.setattr(
        plugin,
        "_visible_log",
        lambda level, message, *args: logs.append(message % args),
    )

    plugin._patch_runner_class(FakeRunner)
    runner = FakeRunner()
    result = runner.propose_draft_token_ids()

    assert runner.initialized
    assert result == [[1, 2]]
    assert runner.dcut_cost_table.max_verify_len == 4
    assert runner.dcut_last_decision.requested_len == 4
    assert runner.dcut_last_decision.batch_size == 32
    prediction_logs = [log for log in logs if "[dcut][cost-table][prediction]" in log]
    assert "source=analytic_estimate" in prediction_logs[0]
    assert "predicted_table=[batch_size_buckets=" in prediction_logs[0]
    warmup_logs = [log for log in logs if "[dcut][cost-table][warmup]" in log]
    assert "entries=" in warmup_logs[0]
    assert "source=analytic_estimate" in warmup_logs[0]
    assert "warmed_table=[batch_size_buckets=" in warmup_logs[0]
    startup_logs = [log for log in logs if "[dcut][cost-table][startup] source=npu_profile" in log]
    assert startup_logs


def test_default_adaptive_profile_run_uses_dummy_run():
    class Runner:
        def __init__(self):
            self.dummy_calls = []

        def _dummy_run(self, **kwargs):
            self.dummy_calls.append(kwargs)

    runner = Runner()

    source, avg_ms, num_tokens = plugin._default_adaptive_profile_run(
        runner,
        scheduled_tokens=[3, 3],
        warmup_seq_lens=[],
        n_warmup_iters=1,
        n_measure_iters=2,
    )

    assert source == "dummy_run"
    assert avg_ms >= 0.0
    assert num_tokens == 6
    assert len(runner.dummy_calls) == 3
    assert all(call["num_tokens"] == 6 for call in runner.dummy_calls)
    assert all(call["is_profile"] is True for call in runner.dummy_calls)
    assert all(call["force_attention"] is True for call in runner.dummy_calls)
    assert all(call["profile_seq_lens"] == 3 for call in runner.dummy_calls)


def test_patch_runner_installs_default_dummy_profile_hook(monkeypatch):
    class FreshSpeculativeConfig:
        method = "dflash"
        num_speculative_tokens = 2

    class FreshInputBatch:
        num_reqs = 2

    class Runner:
        speculative_config = FreshSpeculativeConfig()
        input_batch = FreshInputBatch()

        def __init__(self):
            self.dummy_calls = []

        def _dummy_run(self, **kwargs):
            self.dummy_calls.append(kwargs)

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

    plugin._patch_runner_class(Runner)
    runner = Runner()

    assert callable(runner._adaptive_profile_run)
    assert runner.dummy_calls
    assert any("[dcut][cost-table][startup] source=npu_profile unit=ms" in log for log in logs)


def test_startup_cost_table_profiles_with_runner_hook():
    class RunnerWithProfile:
        def __init__(self):
            self.dcut_cost_table = plugin.DcutCostTable(
                max_verify_len=2,
                max_q_tokens=4,
                max_batch_size=2,
                q_bucket_size=1,
            )
            self.profile_calls = []

        def _adaptive_profile_run(self, scheduled_tokens, *args):
            self.profile_calls.append(tuple(scheduled_tokens))
            return "eager", float(sum(scheduled_tokens) * 10), sum(scheduled_tokens)

    runner = RunnerWithProfile()

    source = plugin._profile_startup_cost_table(runner)

    assert source == "npu_profile"
    assert runner.profile_calls
    assert runner.dcut_cost_table.profile_counts[(2, 4)] == 5
    assert runner.dcut_cost_table.get(4, 2).target_cost == 40.0
    assert runner.dcut_cost_table.get(4, 2).draft_cost == 0.0


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

    decision_logs = [log for log in logs if "[dcut][plan]" in log]
    assert len(decision_logs) == 3
    assert "plan_count=1" in decision_logs[0]
    assert "plan_count=3" in decision_logs[-1]
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
    decision_logs = [log for log in logs if "[dcut][plan]" in log]
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
    assert runner.dcut_last_decision.acceptance_rate == 0.5
    assert runner.dcut_last_accepted_tokens == 4
    assert runner.dcut_last_drafted_tokens == 8
    assert runner.dcut_last_accepted_draft_tokens == 2
    decision_logs = [log for log in logs if "[dcut][plan]" in log]
    assert "acceptance_source=sampled_token_lengths" in decision_logs[0]
    assert "batch_dcut_plan=[#0:accept=0.750,cut=4,#1:accept=0.250,cut=4]" in decision_logs[0]
    assert "candidate_scores=[" in decision_logs[0]
    result_logs = [log for log in logs if "[dcut][result]" in log]
    assert "effective_tokens=4" in result_logs[0]
    assert "accepted_draft_tokens=2" in result_logs[0]


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
    q_tokens = plugin._q_tokens_for_decision(plugin._batch_size_from_runner(runner), 4)
    q_bucket = runner.dcut_cost_table.q_bucket_for(q_tokens)
    batch_bucket = runner.dcut_cost_table.batch_bucket_for(plugin._batch_size_from_runner(runner))
    profile_count_before = runner.dcut_cost_table.profile_counts[(batch_bucket, q_bucket)]

    runner.dcut_verify_start_time = plugin.time.perf_counter()
    runner.dcut_last_planned_cut = 4
    runner._bookkeeping_sync()
    assert runner.dcut_cost_table.profile_counts[(batch_bucket, q_bucket)] == profile_count_before
    assert runner.propose_draft_token_ids(object()) is not None

    assert runner.dcut_last_decision.acceptance_rate == 0.625
    assert runner.dcut_last_accepted_tokens == 5
    assert runner.dcut_last_drafted_tokens == 8
    assert runner.dcut_last_accepted_draft_tokens == 3
    assert runner.dcut_last_per_request_acceptance == [0.75, 0.5]
    decision_logs = [log for log in logs if "[dcut][plan]" in log]
    assert "acceptance_source=bookkeeping_valid_sampled_tokens" in decision_logs[0]
    assert "batch_dcut_plan=[#0:accept=0.750,cut=4,#1:accept=0.500,cut=4]" in decision_logs[0]
    assert "candidate_scores=[" in decision_logs[0]
    result_logs = [log for log in logs if "[dcut][result]" in log]
    assert "effective_tokens=5" in result_logs[0]
    assert "accepted_draft_tokens=3" in result_logs[0]
    assert "per_request_acceptance=[0.75, 0.5]" in result_logs[0]
    assert "verify_elapsed_ms=" in result_logs[0]
    assert "draft_elapsed_ms=" in result_logs[0]
    assert "target_elapsed_ms=" in result_logs[0]
    assert "planned_cut=4" in result_logs[0]
    profile_logs = [log for log in logs if "[dcut][cost-table][profile]" in log]
    assert profile_logs == []

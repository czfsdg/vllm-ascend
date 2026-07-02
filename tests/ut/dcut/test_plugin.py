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

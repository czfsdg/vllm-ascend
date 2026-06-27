# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dcut"))

import dcut.controller as controller_module
import numpy as np
import pytest
from dcut.config import VerifyAdaptiveConfig
from dcut.controller import VerifyAdaptiveController, choose_query_lens_discrete


def test_choose_query_lens_discrete_global_topk():
    result = choose_query_lens_discrete(
        probs=np.array([[0.9, 0.8, 0.1], [0.6, 0.5, 0.4]]),
        base_batch_size=2,
        q_levels=[2, 4, 6, 8],
        cost_lookup={2: 1.0, 4: 1.1, 6: 2.0, 8: 4.0}.__getitem__,
        max_draft_len=3,
        collect_records=True,
    )

    assert result["best_Q"] == 4
    assert result["best_S"] == 2
    assert result["draft_lens"] == [2, 0]
    assert result["records"] is not None


def test_verify_adaptive_config_ignores_unknown_keys_and_validates():
    cfg = VerifyAdaptiveConfig.from_dict({
        "min_warmup_batch_size": 1,
        "query_len_step_per_req": 1,
        "batch_size_step": 1,
        "budget_ratios": [0.25, 0.5, 1.0],
        "log_decision_details": True,
        "log_decision_interval": 2,
        "unknown": "ignored",
    })

    cfg.validate(num_speculative_tokens=4)
    assert cfg.min_warmup_batch_size == 1
    assert cfg.batch_size_step == 1
    assert cfg.budget_ratios == [0.25, 0.5, 1.0]
    assert cfg.log_decision_details is True
    assert cfg.log_decision_interval == 2


def test_choose_query_lens_discrete_requires_meaningful_score_gain():
    probs = np.array([[0.9, 0.01, 0.01], [0.8, 0.01, 0.01]])

    without_threshold = choose_query_lens_discrete(
        probs=probs,
        base_batch_size=2,
        q_levels=[4, 6],
        cost_lookup={4: 1.0, 6: 1.0}.__getitem__,
        max_draft_len=3,
    )
    with_threshold = choose_query_lens_discrete(
        probs=probs,
        base_batch_size=2,
        q_levels=[4, 6],
        cost_lookup={4: 1.0, 6: 1.0}.__getitem__,
        max_draft_len=3,
        min_score_improvement_ratio=0.01,
    )

    assert without_threshold["best_Q"] == 6
    assert with_threshold["best_Q"] == 4
    assert with_threshold["draft_lens"] == [1, 1]


def test_verify_adaptive_config_rejects_invalid_query_range():
    cfg = VerifyAdaptiveConfig(min_query_len_per_req=8, max_query_len_per_req=4)

    with pytest.raises(ValueError, match="min_query_len_per_req"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_decision_log_interval():
    cfg = VerifyAdaptiveConfig(log_decision_details=True, log_decision_interval=0)

    with pytest.raises(ValueError, match="log_decision_interval"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_min_score_improvement_ratio():
    cfg = VerifyAdaptiveConfig(min_score_improvement_ratio=-0.1)

    with pytest.raises(ValueError, match="min_score_improvement_ratio"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_budget_ratio():
    cfg = VerifyAdaptiveConfig(budget_ratios=[0.25, 1.5])

    with pytest.raises(ValueError, match="budget_ratios"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_batch_size_step():
    cfg = VerifyAdaptiveConfig(batch_size_step=0)

    with pytest.raises(ValueError, match="batch_size_step"):
        cfg.validate(num_speculative_tokens=4)


def test_measure_runner_profiles_target_only(monkeypatch):
    calls = []

    class FakeRunner:

        def _dummy_run(self, *args, **kwargs):
            calls.append((args, kwargs))

    controller = SimpleNamespace(
        config=SimpleNamespace(
            n_warmup_iters=2,
            n_measure_iters=3,
            warmup_seq_lens=4096,
        )
    )
    monkeypatch.setattr(
        controller_module.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )

    VerifyAdaptiveController._measure_runner(controller, FakeRunner(), 17)

    assert len(calls) == 5
    assert all(args == (17,) for args, _ in calls)
    assert all(kwargs["skip_drafter"] is True for _, kwargs in calls)
    assert all(kwargs["is_profile"] is True for _, kwargs in calls)
    assert all(kwargs["uniform_decode"] is True for _, kwargs in calls)
    assert all(kwargs["profile_seq_lens"] == 4096 for _, kwargs in calls)

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dcut"))

import dcut.controller as controller_module
import dcut.monkeypatch as dcut_monkeypatch
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
        "log_verifier_timing": True,
        "log_verifier_timing_interval": 3,
        "log_attention_query_shape": True,
        "log_attention_query_shape_interval": 4,
        "log_attention_timing": True,
        "log_attention_timing_interval": 5,
        "log_verifier_breakdown": True,
        "log_verifier_breakdown_interval": 6,
        "log_model_forward_module_breakdown": True,
        "log_model_forward_module_top_k": 7,
        "log_function_input_shapes": True,
        "log_function_input_shapes_max_items": 5,
        "profile_in_profile_run": True,
        "min_cost_reduction_ratio": 0.07,
        "unknown": "ignored",
    })

    cfg.validate(num_speculative_tokens=4)
    assert cfg.min_warmup_batch_size == 1
    assert cfg.batch_size_step == 1
    assert cfg.budget_ratios == [0.25, 0.5, 1.0]
    assert cfg.log_decision_details is True
    assert cfg.log_decision_interval == 2
    assert cfg.log_verifier_timing is True
    assert cfg.log_verifier_timing_interval == 3
    assert cfg.log_attention_query_shape is True
    assert cfg.log_attention_query_shape_interval == 4
    assert cfg.log_attention_timing is True
    assert cfg.log_attention_timing_interval == 5
    assert cfg.log_verifier_breakdown is True
    assert cfg.log_verifier_breakdown_interval == 6
    assert cfg.log_model_forward_module_breakdown is True
    assert cfg.log_model_forward_module_top_k == 7
    assert cfg.log_function_input_shapes is True
    assert cfg.log_function_input_shapes_max_items == 5
    assert cfg.profile_in_profile_run is True
    assert cfg.min_cost_reduction_ratio == 0.07


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


def test_verify_adaptive_config_rejects_invalid_verifier_timing_interval():
    cfg = VerifyAdaptiveConfig(log_verifier_timing=True, log_verifier_timing_interval=0)

    with pytest.raises(ValueError, match="log_verifier_timing_interval"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_attention_query_shape_interval():
    cfg = VerifyAdaptiveConfig(log_attention_query_shape=True, log_attention_query_shape_interval=0)

    with pytest.raises(ValueError, match="log_attention_query_shape_interval"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_attention_timing_interval():
    cfg = VerifyAdaptiveConfig(log_attention_timing=True, log_attention_timing_interval=0)

    with pytest.raises(ValueError, match="log_attention_timing_interval"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_verifier_breakdown_interval():
    cfg = VerifyAdaptiveConfig(log_verifier_breakdown=True, log_verifier_breakdown_interval=0)

    with pytest.raises(ValueError, match="log_verifier_breakdown_interval"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_model_forward_module_top_k():
    cfg = VerifyAdaptiveConfig(log_model_forward_module_breakdown=True, log_model_forward_module_top_k=0)

    with pytest.raises(ValueError, match="log_model_forward_module_top_k"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_function_input_shapes_max_items():
    cfg = VerifyAdaptiveConfig(log_function_input_shapes=True, log_function_input_shapes_max_items=0)

    with pytest.raises(ValueError, match="log_function_input_shapes_max_items"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_min_score_improvement_ratio():
    cfg = VerifyAdaptiveConfig(min_score_improvement_ratio=-0.1)

    with pytest.raises(ValueError, match="min_score_improvement_ratio"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_min_cost_reduction_ratio():
    cfg = VerifyAdaptiveConfig(min_cost_reduction_ratio=-0.1)

    with pytest.raises(ValueError, match="min_cost_reduction_ratio"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_budget_ratio():
    cfg = VerifyAdaptiveConfig(budget_ratios=[0.25, 1.5])

    with pytest.raises(ValueError, match="budget_ratios"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_batch_size_step():
    cfg = VerifyAdaptiveConfig(batch_size_step=0)

    with pytest.raises(ValueError, match="batch_size_step"):
        cfg.validate(num_speculative_tokens=4)



def test_should_log_attention_query_shape_respects_interval():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_attention_query_shape=True,
            log_attention_query_shape_interval=2,
        ),
        _attention_query_shape_count=0,
    )

    assert VerifyAdaptiveController.should_log_attention_query_shape(controller) is True
    assert VerifyAdaptiveController.should_log_attention_query_shape(controller) is False
    assert VerifyAdaptiveController.should_log_attention_query_shape(controller) is True


def test_should_log_attention_query_shape_disabled():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_attention_query_shape=False,
            log_attention_query_shape_interval=1,
        ),
        _attention_query_shape_count=0,
    )

    assert VerifyAdaptiveController.should_log_attention_query_shape(controller) is False
    assert controller._attention_query_shape_count == 0


def test_should_log_attention_timing_respects_interval():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_attention_timing=True,
            log_attention_timing_interval=2,
        ),
        _attention_timing_count=0,
    )

    assert VerifyAdaptiveController.should_log_attention_timing(controller) is True
    assert VerifyAdaptiveController.should_log_attention_timing(controller) is False
    assert VerifyAdaptiveController.should_log_attention_timing(controller) is True


def test_should_log_attention_timing_disabled():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_attention_timing=False,
            log_attention_timing_interval=1,
        ),
        _attention_timing_count=0,
    )

    assert VerifyAdaptiveController.should_log_attention_timing(controller) is False
    assert controller._attention_timing_count == 0


def test_should_log_verifier_breakdown_respects_interval():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_verifier_breakdown=True,
            log_verifier_breakdown_interval=2,
        ),
        _verifier_breakdown_count=0,
    )

    assert VerifyAdaptiveController.should_log_verifier_breakdown(controller) is True
    assert VerifyAdaptiveController.should_log_verifier_breakdown(controller) is False
    assert VerifyAdaptiveController.should_log_verifier_breakdown(controller) is True


def test_should_log_verifier_breakdown_disabled():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_verifier_breakdown=False,
            log_verifier_breakdown_interval=1,
        ),
        _verifier_breakdown_count=0,
    )

    assert VerifyAdaptiveController.should_log_verifier_breakdown(controller) is False
    assert controller._verifier_breakdown_count == 0


def test_should_log_verifier_timing_respects_interval():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_verifier_timing=True,
            log_verifier_timing_interval=2,
        ),
        _verifier_timing_count=0,
    )

    assert VerifyAdaptiveController.should_log_verifier_timing(controller) is True
    assert VerifyAdaptiveController.should_log_verifier_timing(controller) is False
    assert VerifyAdaptiveController.should_log_verifier_timing(controller) is True


def test_should_log_verifier_timing_disabled():
    controller = SimpleNamespace(
        config=SimpleNamespace(
            log_verifier_timing=False,
            log_verifier_timing_interval=1,
        ),
        _verifier_timing_count=0,
    )

    assert VerifyAdaptiveController.should_log_verifier_timing(controller) is False
    assert controller._verifier_timing_count == 0

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
            profile_in_profile_run=False,
        )
    )
    monkeypatch.setattr(
        controller_module.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )

    controller.max_query_len_per_req = 8

    VerifyAdaptiveController._measure_runner(controller, FakeRunner(), 3, 17)

    assert len(calls) == 5
    assert all(args == (17,) for args, _ in calls)
    assert all(kwargs["skip_drafter"] is True for _, kwargs in calls)
    assert all(kwargs["is_profile"] is False for _, kwargs in calls)
    assert all(kwargs["uniform_decode"] is True for _, kwargs in calls)
    assert all(kwargs["profile_seq_lens"] == 4096 for _, kwargs in calls)
    assert all(kwargs["profile_num_scheduled_tokens"] == [6, 6, 5] for _, kwargs in calls)


def test_build_profile_query_lens_uses_requested_batch_size():
    controller = SimpleNamespace(max_query_len_per_req=8)

    assert VerifyAdaptiveController._build_profile_query_lens(controller, 9, 57) == [7, 7, 7, 6, 6, 6, 6, 6, 6]
    assert VerifyAdaptiveController._build_profile_query_lens(controller, 12, 96) == [8] * 12


def test_build_profile_query_lens_rejects_invalid_token_count():
    controller = SimpleNamespace(max_query_len_per_req=8)

    with pytest.raises(ValueError, match=">= batch_size"):
        VerifyAdaptiveController._build_profile_query_lens(controller, 4, 3)
    with pytest.raises(ValueError, match="exceeds batch capacity"):
        VerifyAdaptiveController._build_profile_query_lens(controller, 4, 33)


def test_filter_query_levels_without_cost_gain_keeps_full_budget_when_flat():
    controller = SimpleNamespace(
        config=SimpleNamespace(min_cost_reduction_ratio=0.05),
        _sorted_bs=[4],
        _sorted_sql_per_bs={4: [11, 18, 25, 32]},
        _cost_table={
            (4, 11): 0.028,
            (4, 18): 0.0275,
            (4, 25): 0.0278,
            (4, 32): 0.0282,
        },
    )

    VerifyAdaptiveController._filter_query_levels_without_cost_gain(controller)

    assert controller._sorted_sql_per_bs[4] == [32]


def test_filter_query_levels_without_cost_gain_keeps_lower_budgets_when_useful():
    controller = SimpleNamespace(
        config=SimpleNamespace(min_cost_reduction_ratio=0.05),
        _sorted_bs=[4],
        _sorted_sql_per_bs={4: [11, 18, 25, 32]},
        _cost_table={
            (4, 11): 0.020,
            (4, 18): 0.024,
            (4, 25): 0.026,
            (4, 32): 0.028,
        },
    )

    VerifyAdaptiveController._filter_query_levels_without_cost_gain(controller)

    assert controller._sorted_sql_per_bs[4] == [11, 18, 25, 32]


def test_log_verifier_timing_groups_stats_by_shape(monkeypatch):
    records = []
    runner = SimpleNamespace(
        _dcut_controller=SimpleNamespace(config=SimpleNamespace(log_decision_max_records=2)),
        _dcut_verifier_timing_stats={},
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "req0": [1, 2, 3],
            "req1": [4, 5],
        },
        num_scheduled_tokens={
            "req0": 4,
            "req1": 3,
        },
        total_num_scheduled_tokens=7,
    )

    monkeypatch.setattr(
        dcut_monkeypatch.logger,
        "info",
        lambda message, *args: records.append(message % args),
    )

    dcut_monkeypatch._dcut_log_verifier_timing(runner, scheduler_output, 14.0)
    dcut_monkeypatch._dcut_log_verifier_timing(runner, scheduler_output, 21.0)

    assert runner._dcut_verifier_timing_stats[(2, 7, 5, 3)] == (2, 35.0, 14.0, 21.0)
    assert "query_lens=[4, 3]" in records[-1]
    assert "shape_count=2" in records[-1]
    assert "shape_avg_ms=17.500" in records[-1]

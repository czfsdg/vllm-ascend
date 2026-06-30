# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
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
    cfg = VerifyAdaptiveConfig.from_dict(
        {
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
            "n_profile_presweep_iters": 2,
            "fixed_cut_ratio": 0.25,
            "apply_runtime_cuts": True,
            "min_cost_reduction_ratio": 0.07,
            "unknown": "ignored",
        }
    )

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
    assert cfg.n_profile_presweep_iters == 2
    assert cfg.fixed_cut_ratio == 0.25
    assert cfg.apply_runtime_cuts is True
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


def test_verify_adaptive_config_rejects_invalid_fixed_cut_ratio():
    cfg = VerifyAdaptiveConfig(fixed_cut_ratio=1.0)

    with pytest.raises(ValueError, match="fixed_cut_ratio"):
        cfg.validate(num_speculative_tokens=4)


def test_process_draft_output_fixed_cut_ratio_uses_batch_budget():
    class FakeSelectedProbs:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        @property
        def shape(self):
            return self.values.shape

        def __getitem__(self, key):
            return FakeSelectedProbs(self.values[key])

        def numpy(self):
            return self.values

    controller = VerifyAdaptiveController.__new__(VerifyAdaptiveController)
    controller.config = VerifyAdaptiveConfig(fixed_cut_ratio=0.25)
    controller.max_query_len_per_req = 8
    controller._adaptive_draft_lens = {}
    controller._sorted_bs = []

    controller.process_draft_output(
        FakeSelectedProbs(
            np.array(
                [
                    [0.99] * 7,
                    [0.8] * 7,
                    [0.1] * 7,
                ],
                dtype=np.float32,
            )
        ),
        ["req0", "req1", "req2"],
        {"req0", "req1", "req2"},
        batch_size=3,
    )

    assert controller._adaptive_draft_lens == {
        "req0": 7,
        "req1": 7,
        "req2": 1,
    }


def test_truncate_scheduler_output_consumes_adaptive_plan():
    @dataclass
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    class FakeController:
        config = SimpleNamespace(log_decision_details=False, apply_runtime_cuts=True)
        max_query_len_per_req = 8

        def __init__(self):
            self.plans = {
                "req0": 1,
                "req1": 2,
            }
            self.invalidated = []

        def get_adaptive_draft_len(self, req_id):
            return self.plans.get(req_id)

        def invalidate(self, req_id):
            self.invalidated.append(req_id)
            self.plans.pop(req_id, None)

    controller = FakeController()
    runner = SimpleNamespace(_dcut_controller=controller)
    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={
            "req0": [10, 11, 12],
            "req1": [20, 21],
        },
        num_scheduled_tokens={
            "req0": 4,
            "req1": 3,
        },
        total_num_scheduled_tokens=7,
    )

    truncated = dcut_monkeypatch._dcut_truncate_scheduler_output(runner, scheduler_output)

    assert truncated.scheduled_spec_decode_tokens == {
        "req0": [10],
        "req1": [20, 21],
    }
    assert truncated.num_scheduled_tokens == {
        "req0": 2,
        "req1": 3,
    }
    assert truncated.total_num_scheduled_tokens == 5
    assert controller.invalidated == ["req0", "req1"]
    assert controller.plans == {}


def test_truncate_scheduler_output_skips_cut_when_runtime_cuts_disabled():
    @dataclass
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    class FakeController:
        config = SimpleNamespace(log_decision_details=False, apply_runtime_cuts=False)
        max_query_len_per_req = 8

        def __init__(self):
            self.plans = {"req0": 1}

        def get_adaptive_draft_len(self, req_id):
            return self.plans.get(req_id)

        def invalidate(self, req_id):
            self.plans.pop(req_id, None)

    controller = FakeController()
    runner = SimpleNamespace(_dcut_controller=controller)
    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"req0": [10, 11, 12]},
        num_scheduled_tokens={"req0": 4},
        total_num_scheduled_tokens=4,
    )

    assert dcut_monkeypatch._dcut_truncate_scheduler_output(runner, scheduler_output) is scheduler_output
    assert controller.plans == {}


def test_truncate_scheduler_output_skips_mixed_prefill_batch():
    @dataclass
    class FakeSchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    class FakeController:
        config = SimpleNamespace(log_decision_details=False, apply_runtime_cuts=True)
        max_query_len_per_req = 8

        def __init__(self):
            self.plans = {
                "decode_req": 1,
            }

        def get_adaptive_draft_len(self, req_id):
            return self.plans.get(req_id)

        def invalidate(self, req_id):
            self.plans.pop(req_id, None)

    controller = FakeController()
    runner = SimpleNamespace(_dcut_controller=controller, _dcut_mixed_prefill_skip_warnings=0)
    scheduler_output = FakeSchedulerOutput(
        scheduled_spec_decode_tokens={"decode_req": [10, 11, 12]},
        num_scheduled_tokens={
            "decode_req": 4,
            "prefill_req": 976,
        },
        total_num_scheduled_tokens=980,
    )

    assert dcut_monkeypatch._dcut_truncate_scheduler_output(runner, scheduler_output) is scheduler_output
    assert controller.plans == {}
    assert runner._dcut_mixed_prefill_skip_warnings == 1


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


def test_verify_adaptive_config_rejects_invalid_profile_presweep_iters():
    cfg = VerifyAdaptiveConfig(n_profile_presweep_iters=-1)

    with pytest.raises(ValueError, match="n_profile_presweep_iters"):
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


def test_presweep_profile_shapes_runs_before_measurement(monkeypatch):
    calls = []
    controller = SimpleNamespace(
        config=SimpleNamespace(n_profile_presweep_iters=2),
        max_query_len_per_req=8,
    )
    monkeypatch.setattr(
        controller_module.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: calls.append(("sync",))),
        raising=False,
    )

    def fake_run_profile_iteration(_runner, num_tokens, profile_query_lens):
        calls.append((num_tokens, profile_query_lens))

    controller._run_profile_iteration = fake_run_profile_iteration

    VerifyAdaptiveController._presweep_profile_shapes(controller, object(), [(2, 9), (3, 17)])

    assert calls == [
        (9, [5, 4]),
        (17, [6, 6, 5]),
        (9, [5, 4]),
        (17, [6, 6, 5]),
        ("sync",),
    ]


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


def test_layer_block_display_name_only_matches_layer_blocks():
    assert dcut_monkeypatch._dcut_layer_block_display_name("language_model.model.layers.0") == "layers.0"
    assert dcut_monkeypatch._dcut_layer_block_display_name("model.layers.12") == "layers.12"
    assert dcut_monkeypatch._dcut_layer_block_display_name("language_model.model.layers.0.mlp") is None
    assert dcut_monkeypatch._dcut_layer_block_display_name("language_model.embed_tokens") is None


def test_top_level_phase_sum_excludes_nested_model_forward_details():
    assert (
        dcut_monkeypatch._dcut_top_level_phase_sum(
            {
                "model_forward": 10.0,
                "model_forward.model_call": 8.0,
                "model_forward.update_full_graph_params": 0.5,
                "compute_logits": 2.0,
            }
        )
        == 12.0
    )


def test_model_forward_call_patch_records_nested_phase(monkeypatch):
    class FakeModel:
        def forward(self, value):
            return value + 1

    model = FakeModel()
    runner = SimpleNamespace(model=model)
    context = {
        "phases": {},
        "log_input_shapes": False,
    }

    monkeypatch.setattr(
        dcut_monkeypatch.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )

    dcut_monkeypatch._dcut_patch_model_forward_call(runner)
    token = dcut_monkeypatch._VERIFIER_BREAKDOWN_CONTEXT.set(context)
    try:
        assert model.forward(1) == 2
    finally:
        dcut_monkeypatch._VERIFIER_BREAKDOWN_CONTEXT.reset(token)

    assert "model_forward.model_call" in context["phases"]
    assert context["phases"]["model_forward.model_call"] >= 0.0


def test_verifier_breakdown_logs_model_forward_shape_stats(monkeypatch):
    records = []
    runner = SimpleNamespace(_dcut_model_forward_timing_stats={})
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "req0": [1, 2, 3],
            "req1": [4, 5, 6],
        },
        num_scheduled_tokens={
            "req0": 4,
            "req1": 4,
        },
        total_num_scheduled_tokens=8,
    )
    context = {
        "module_classes": {"FakeLayer": 3.0},
        "module_names": {"layers.0:FakeLayer": 3.0},
        "module_stack": {},
        "layer_blocks": {"layers.0": 5.0},
        "layer_block_stack": {},
        "phases": {"model_forward": 20.0, "compute_logits": 2.0},
        "module_top_k": 2,
        "spec_tokens": 6,
        "spec_reqs": 2,
        "runner": runner,
    }
    token = dcut_monkeypatch._VERIFIER_BREAKDOWN_CONTEXT.set(context)

    monkeypatch.setattr(
        dcut_monkeypatch.logger,
        "info",
        lambda message, *args: records.append(message % args),
    )

    dcut_monkeypatch._dcut_finish_verifier_breakdown(scheduler_output, token, 25.0)

    assert runner._dcut_model_forward_timing_stats[(2, 8, 6, 4)] == (1, 20.0, 20.0, 20.0)
    assert "model_forward_shape_stats={'elapsed_ms': 20.0" in records[-1]
    assert "'ms_per_token': 2.5" in records[-1]
    assert "'shape_count': 1" in records[-1]
    assert "layer_blocks=[('layers.0', 5.0)]" in records[-1]
    assert "module_classes=[('FakeLayer', 3.0)]" in records[-1]


def test_patch_runner_rejects_profile_shape_for_older_dummy_run(monkeypatch):

    class FakeRunner:
        def __init__(self):
            self.drafter = "drafter"

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return None

        def sample_tokens(self, grammar_output):
            return None

        def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
            return None

        def _dummy_run(self, *args, uniform_decode=False):
            return None

    monkeypatch.setattr(dcut_monkeypatch, "_dcut_init_controller", lambda runner: None)
    monkeypatch.setattr(dcut_monkeypatch, "_dcut_patch_model_compute_logits", lambda runner: None)
    monkeypatch.setattr(dcut_monkeypatch, "_dcut_patch_model_forward_modules", lambda runner: None)

    dcut_monkeypatch._patch_runner(FakeRunner)
    runner = FakeRunner()

    assert runner._dcut_profile_num_scheduled_tokens_supported is False
    with pytest.raises(RuntimeError, match="misleading flat cost table"):
        runner._dummy_run(8, uniform_decode=True, skip_drafter=True, profile_num_scheduled_tokens=[4, 4])


def test_profile_cost_skips_older_dummy_run_signature(monkeypatch):
    errors = []
    controller = SimpleNamespace(profile_cost_table=lambda runner: pytest.fail("profile_cost_table should be skipped"))
    runner = SimpleNamespace(
        _dcut_controller=controller,
        _dcut_profile_num_scheduled_tokens_supported=False,
    )

    monkeypatch.setattr(dcut_monkeypatch.logger, "warning", lambda message, *args: errors.append(message % args))

    dcut_monkeypatch._dcut_profile_cost(runner)

    assert "skip cost profiling" in errors[0]


def test_gdn_module_detection_targets_linear_attention_blocks():
    class GatedDeltaNetAttention:
        pass

    class NestedNorm:
        pass

    assert dcut_monkeypatch._dcut_should_trace_gdn_module(
        "language_model.model.layers.0.linear_attn", GatedDeltaNetAttention()
    )
    assert dcut_monkeypatch._dcut_should_trace_gdn_module("any.path", GatedDeltaNetAttention())
    assert not dcut_monkeypatch._dcut_should_trace_gdn_module(
        "language_model.model.layers.0.linear_attn.norm", NestedNorm()
    )


def test_top_gdn_records_rounds_and_limits_metadata():
    records = [
        {
            "name": "layers.0.linear_attn",
            "class": "GatedDeltaNetAttention",
            "elapsed_ms": 1.23456,
            "inputs": {"hidden_states": {"shape": (8, 4096)}},
            "output": {"shape": (8, 4096)},
            "metadata": {"spec_state_indices_tensor": {"shape": (1, 8)}},
        },
        {
            "name": "layers.1.linear_attn",
            "class": "GatedDeltaNetAttention",
            "elapsed_ms": 3.98765,
            "inputs": {"hidden_states": {"shape": (16, 4096)}},
            "output": {"shape": (16, 4096)},
            "metadata": {"spec_state_indices_tensor": {"shape": (2, 8)}},
        },
    ]

    top = dcut_monkeypatch._dcut_top_gdn_records(records, top_k=1)

    assert top == [
        {
            "name": "layers.1.linear_attn",
            "class": "GatedDeltaNetAttention",
            "elapsed_ms": 3.988,
            "inputs": {"hidden_states": {"shape": (16, 4096)}},
            "output": {"shape": (16, 4096)},
            "metadata": {"spec_state_indices_tensor": {"shape": (2, 8)}},
        }
    ]


def test_gdn_input_summary_reads_keyword_arguments():
    hidden = SimpleNamespace(shape=(8, 4096))
    output = SimpleNamespace(shape=(8, 4096))

    summary = dcut_monkeypatch._dcut_summarize_gdn_inputs((), {"hidden_states": hidden, "output": output})

    assert summary == {
        "hidden_states": {"type": "SimpleNamespace", "shape": (8, 4096)},
        "output": {"type": "SimpleNamespace", "shape": (8, 4096)},
    }


def test_gdn_metadata_summary_selects_module_prefix(monkeypatch):
    class Module:
        prefix = "language_model.model.layers.0.linear_attn"

    metadata = SimpleNamespace(
        num_actual_tokens=8,
        num_spec_decodes=1,
        max_query_len=8,
        actual_seq_lengths_q=[8],
        spec_state_indices_tensor=SimpleNamespace(shape=(1, 8)),
    )
    context = SimpleNamespace(attn_metadata={Module.prefix: metadata})
    monkeypatch.setattr(dcut_monkeypatch, "get_forward_context", lambda: context)

    summary = dcut_monkeypatch._dcut_summarize_gdn_metadata(Module())

    assert summary["num_actual_tokens"] == 8
    assert summary["num_spec_decodes"] == 1
    assert summary["max_query_len"] == 8
    assert summary["actual_seq_lengths_q"] == {"type": "list", "len": 1, "first": 8, "last": 8}
    assert summary["spec_state_indices_tensor"] == {"type": "SimpleNamespace", "shape": (1, 8)}

# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "runtime_cost.py"
CONFIG_PATH = MODULE_PATH.with_name("verify_adaptive_config.py")
RUNNER_PATH = MODULE_PATH.with_name("patch_runner.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("dcut_runtime_cost", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_cost_learns_full_cost_per_query_bucket() -> None:
    module = _load_module()
    calibrator = module.RuntimeCostCalibrator(
        alpha=0.25,
        samples_per_bucket=2,
    )

    residual, estimate = calibrator.observe(
        batch_size=32,
        query_len=64,
        mode="pure_spec_graph",
        full_step_s=0.080,
        profiled_step_s=0.040,
    )
    assert math.isclose(residual, 0.040)
    assert math.isclose(estimate.overhead_s, 0.040)
    assert math.isclose(estimate.full_step_s, 0.080)
    assert calibrator.needs_sample(32, 64, "pure_spec_graph")

    residual, estimate = calibrator.observe(
        batch_size=32,
        query_len=64,
        mode="pure_spec_graph",
        full_step_s=0.060,
        profiled_step_s=0.040,
    )
    assert math.isclose(residual, 0.020)
    assert math.isclose(estimate.overhead_s, 0.035)
    assert math.isclose(estimate.full_step_s, 0.075)
    assert not calibrator.needs_sample(32, 64, "pure_spec_graph")
    assert calibrator.needs_sample(32, 512, "pure_spec_graph")

    residual, estimate = calibrator.observe(
        batch_size=16,
        query_len=32,
        mode="mixed",
        full_step_s=0.010,
        profiled_step_s=0.020,
    )
    assert residual == 0.0
    assert estimate.overhead_s == 0.0
    assert math.isclose(estimate.full_step_s, 0.010)


def test_runtime_cost_uses_same_query_aggregate_until_mode_is_observed() -> None:
    module = _load_module()
    calibrator = module.RuntimeCostCalibrator(
        alpha=0.5,
        samples_per_bucket=4,
    )
    calibrator.observe(
        batch_size=32,
        query_len=256,
        mode="pure_spec_graph",
        full_step_s=0.075,
        profiled_step_s=0.050,
    )

    fallback = calibrator.get(32, 256, "mixed")
    assert fallback.samples == 1
    assert math.isclose(fallback.overhead_s, 0.025)
    assert math.isclose(fallback.full_step_s, 0.075)
    assert calibrator.get(32, 64, "mixed").samples == 0

    calibrator.observe(
        batch_size=32,
        query_len=256,
        mode="mixed",
        full_step_s=0.100,
        profiled_step_s=0.050,
    )
    exact = calibrator.get(32, 256, "mixed")
    assert exact.samples == 1
    assert math.isclose(exact.overhead_s, 0.050)
    assert math.isclose(exact.full_step_s, 0.100)


def _load_config_module():
    spec = importlib.util.spec_from_file_location(
        "dcut_verify_adaptive_config",
        CONFIG_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_cost_config_validates_calibration_settings() -> None:
    module = _load_config_module()
    config = module.VerifyAdaptiveConfig.from_dict(
        {
            "runtime_cost_calibration": True,
            "runtime_cost_samples_per_bucket": 4,
            "runtime_cost_ewma_alpha": 0.5,
            "runtime_cost_dump_path": "/tmp/dcut-runtime.jsonl",
        }
    )
    config.validate(num_speculative_tokens=15)
    assert config.runtime_cost_samples_per_bucket == 4
    assert math.isclose(config.runtime_cost_ewma_alpha, 0.5)

    config.runtime_cost_samples_per_bucket = 0
    try:
        config.validate(num_speculative_tokens=15)
    except ValueError as exc:
        assert "samples_per_bucket" in str(exc)
    else:
        raise AssertionError("zero calibration samples must be rejected")


def test_runtime_breakdown_covers_the_full_iteration() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for component in (
        "scheduler_and_ipc_gap",
        "target_execute_total",
        "gdn_replay_prepare",
        "sampling",
        "bookkeeping",
        "draft_model",
        "draft_id_copy",
        "selected_probs_queue",
        "adaptive_decision",
        "device_drain",
    ):
        assert f'"{component}"' in source
    assert "runner_step_s + scheduler_gap_s + dcut_prepare_s" in source
    assert '(_prepare_start_s - _previous_end_s) * 1e3' in source
    assert "full_step_s=full_iteration_s" in source
    assert "sum_query_len=_sum_query_len" in source
    assert "_ctrl.should_measure_runtime_step" in source

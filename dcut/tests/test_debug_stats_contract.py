# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


PATCH_RUNNER = Path(__file__).resolve().parents[1] / "patch_runner.py"


def _execute_source() -> str:
    source = PATCH_RUNNER.read_text(encoding="utf-8")
    start = source.index("    def execute_model(")
    end = source.index("    _orig_sample_tokens", start)
    return source[start:end]


def test_fast_path_execution_order_is_unchanged() -> None:
    source = _execute_source()
    fast_start = source.index("        if not debug_stats:")
    fast_end = source.index(
        "        # Optional slow-path debug timing.",
        fast_start,
    )
    fast_path = source[fast_start:fast_end]

    first_sync = fast_path.index("_torch_fast.npu.synchronize()")
    execute = fast_path.index("_dcut_execute_with_gdn_prefill_route(")
    second_sync = fast_path.index(
        "_torch_fast.npu.synchronize()",
        first_sync + 1,
    )
    assert first_sync < execute < second_sync


def test_debug_stats_name_gap_as_inter_call_time() -> None:
    source = _execute_source()

    assert '"prev_step": _prev_step' in source
    assert '"gap_sample_valid": _has_gap_sample' in source
    assert '"inter_call_gap_after_prev_step_ms"' in source
    assert '"stats_io_after_prev_step_ms"' in source
    assert '"gap_ms"' not in source
    assert '"scheduler_ms"' not in source


def test_debug_stats_partition_runner_time() -> None:
    source = _execute_source()
    expected_fields = {
        "classify_ms",
        "adaptive_probs_process_ms",
        "drafter_enable_ms",
        "truncate_ms",
        "prob_capture_reset_ms",
        "pre_cpu_other_ms",
        "pre_cpu_total_ms",
        "pre_sync_ms",
        "pre_total_ms",
        "execute_call_ms",
        "post_sync_ms",
        "fwd_ms",
        "post_cpu_ms",
    }

    for field in expected_fields:
        assert f'"{field}"' in source


def test_debug_file_has_one_distributed_writer_and_logs_all_batches() -> None:
    source = _execute_source()

    assert 'if _fwd_stats_out and _rank_info["is_writer"]:' in source
    assert 'if _fwd_stats_out and _has_spec:' not in source
    assert '"world_rank": _rank_info["world_rank"]' in source
    assert '"has_prefill": _has_prefill' in source
    assert '"mixed_batch": _has_prefill and _has_spec' in source
    assert '"runtime_mode": _runtime_mode' in source
    assert '"is_eager": _is_eager' in source
    assert '"gdn_graph_safe": _graph_safe' in source

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dcut"))

import numpy as np
import pytest
from dcut.config import VerifyAdaptiveConfig
from dcut.controller import choose_query_lens_discrete


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
            "unknown": "ignored",
        }
    )

    cfg.validate(num_speculative_tokens=4)
    assert cfg.min_warmup_batch_size == 1


def test_verify_adaptive_config_rejects_invalid_query_range():
    cfg = VerifyAdaptiveConfig(min_query_len_per_req=8, max_query_len_per_req=4)

    with pytest.raises(ValueError, match="min_query_len_per_req"):
        cfg.validate(num_speculative_tokens=4)


def test_dcut_align_selected_probs_skips_ambiguous_partial_rows():
    import torch
    from dcut.monkeypatch import _dcut_align_selected_probs

    class Runner:
        num_spec_tokens = 3

        class InputBatch:
            req_ids = ["r0", "r1", "r2", "r3"]

        input_batch = InputBatch()

    probs = torch.ones((2, 3), dtype=torch.float32)

    assert _dcut_align_selected_probs(Runner(), probs, 4, {"r1"}) is None


def test_dcut_align_selected_probs_maps_active_only_rows():
    import torch
    from dcut.monkeypatch import _dcut_align_selected_probs

    class Runner:
        num_spec_tokens = 2

        class InputBatch:
            req_ids = ["prefill", "decode0", "decode1"]

        input_batch = InputBatch()

    probs = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)

    aligned = _dcut_align_selected_probs(Runner(), probs, 3, {"decode0", "decode1"})

    assert aligned is not None
    assert aligned.tolist() == [[0.0, 0.0], [0.9, 0.8], [0.7, 0.6]]


def test_dcut_truncates_scheduler_output_before_execute_model():
    from dataclasses import dataclass
    from types import SimpleNamespace

    from dcut.monkeypatch import _dcut_truncate_scheduler_output

    @dataclass
    class SchedulerOutput:
        scheduled_spec_decode_tokens: dict[str, list[int]]
        num_scheduled_tokens: dict[str, int]
        total_num_scheduled_tokens: int

    scheduler_output = SchedulerOutput(
        scheduled_spec_decode_tokens={"r0": [10, 11, 12], "r1": [20, 21, 22]},
        num_scheduled_tokens={"r0": 4, "r1": 4},
        total_num_scheduled_tokens=8,
    )
    controller = SimpleNamespace(
        get_adaptive_draft_len=lambda req_id: {"r0": 1, "r1": 0}[req_id],
    )
    runner = SimpleNamespace(
        _dcut_controller=controller,
        input_batch=SimpleNamespace(req_id_to_index={}, num_accepted_tokens_cpu=None),
    )

    truncated = _dcut_truncate_scheduler_output(runner, scheduler_output)

    assert truncated is not scheduler_output
    assert truncated.scheduled_spec_decode_tokens == {"r0": [10]}
    assert truncated.num_scheduled_tokens == {"r0": 2, "r1": 1}
    assert truncated.total_num_scheduled_tokens == 3

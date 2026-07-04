# SPDX-License-Identifier: Apache-2.0
"""Unit tests for standalone D-Cut adaptive verifier helpers."""

import sys
import types

import numpy as np

# verify_adaptive_controller imports vLLM distributed/logger modules at import
# time.  Provide minimal stubs so the pure selection function can be unit-tested
# without a full vLLM install.
vllm = types.ModuleType("vllm")
vllm_distributed = types.ModuleType("vllm.distributed")
vllm_logger = types.ModuleType("vllm.logger")


def _fake_group():
    return types.SimpleNamespace(rank_in_group=0, world_size=1, is_first_rank=True)


vllm_distributed.get_pp_group = _fake_group
vllm_distributed.get_tp_group = _fake_group
vllm_logger.init_logger = lambda name: types.SimpleNamespace(info=lambda *a, **k: None)
sys.modules.setdefault("vllm", vllm)
sys.modules.setdefault("vllm.distributed", vllm_distributed)
sys.modules.setdefault("vllm.logger", vllm_logger)

from dcut.verify_adaptive_config import VerifyAdaptiveConfig  # noqa: E402
from dcut.verify_adaptive_controller import choose_query_lens_discrete  # noqa: E402


def test_verify_adaptive_config_ignores_unknown_keys():
    cfg = VerifyAdaptiveConfig.from_dict({"enabled": True, "unknown": 1, "n_measure_iters": 2})
    assert cfg.enabled is True
    assert cfg.n_measure_iters == 2


def test_verify_adaptive_config_validation_rejects_bad_values():
    cfg = VerifyAdaptiveConfig(min_query_len_per_req=1)
    try:
        cfg.validate(num_speculative_tokens=3)
    except ValueError as exc:
        assert "min_query_len_per_req" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_choose_query_lens_discrete_uses_profiled_cost():
    probs = np.array([[0.9, 0.5, 0.1], [0.8, 0.7, 0.1]], dtype=np.float64)
    result = choose_query_lens_discrete(
        probs=probs,
        base_batch_size=2,
        q_levels=[2, 4, 6],
        cost_lookup={2: 1.0, 4: 1.1, 6: 10.0}.__getitem__,
        max_draft_len=3,
    )
    assert result["best_Q"] == 4
    assert result["draft_lens"] == [1, 1]

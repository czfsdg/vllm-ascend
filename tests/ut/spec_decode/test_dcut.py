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
    cfg = VerifyAdaptiveConfig.from_dict({
        "min_warmup_batch_size": 1,
        "query_len_step_per_req": 1,
        "log_decision_details": True,
        "log_decision_interval": 2,
        "unknown": "ignored",
    })

    cfg.validate(num_speculative_tokens=4)
    assert cfg.min_warmup_batch_size == 1
    assert cfg.log_decision_details is True
    assert cfg.log_decision_interval == 2


def test_verify_adaptive_config_rejects_invalid_query_range():
    cfg = VerifyAdaptiveConfig(min_query_len_per_req=8, max_query_len_per_req=4)

    with pytest.raises(ValueError, match="min_query_len_per_req"):
        cfg.validate(num_speculative_tokens=4)


def test_verify_adaptive_config_rejects_invalid_decision_log_interval():
    cfg = VerifyAdaptiveConfig(log_decision_details=True, log_decision_interval=0)

    with pytest.raises(ValueError, match="log_decision_interval"):
        cfg.validate(num_speculative_tokens=4)

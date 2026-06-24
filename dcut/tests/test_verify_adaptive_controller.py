# SPDX-License-Identifier: Apache-2.0

from dcut.verify_adaptive_config import VerifyAdaptiveConfig
from dcut.verify_adaptive_controller import choose_query_lens_discrete


def test_choose_query_lens_discrete_prefers_global_high_confidence_prefixes():
    result = choose_query_lens_discrete(
        probs=[[0.9, 0.8], [0.5, 0.5]],
        base_batch_size=2,
        q_levels=[2, 4, 6],
        cost_lookup=lambda q: {2: 1.0, 4: 1.2, 6: 2.0}[q],
        max_draft_len=2,
    )

    assert result["draft_lens"] == [2, 0]
    assert result["best_Q"] == 4


def test_config_query_len_levels_include_baseline_and_cap():
    config = VerifyAdaptiveConfig(
        min_query_len_per_req=2,
        max_query_len_per_req=None,
        query_len_step_per_req=2,
    )

    assert config.query_len_levels(max_draft_len=5) == [1, 2, 4, 6]

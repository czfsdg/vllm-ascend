# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

VALIDATION_SCRIPT = Path(__file__).with_name("validate_gdn_ops_npu.py")


def test_npu_validation_covers_parity_and_graph_replay() -> None:
    validation = VALIDATION_SCRIPT.read_text(encoding="utf-8")

    for op_name in (
        "npu_causal_conv1d_custom",
        "npu_recurrent_gated_delta_rule",
        "npu_dcut_causal_conv1d",
        "npu_dcut_recurrent_gated_delta_rule",
    ):
        assert op_name in validation
    assert "num_accepted_tokens_opt=case.state_offsets + 1" in validation
    assert "ssm_state_indices=case.original_state_indices" in validation
    assert "ssm_state_indices=case.dcut_state_indices" in validation
    assert validation.count("graph.replay()") == 2
    assert "previous_round_longer=True" in validation
    assert "DCUT_NPU_VALIDATION_PASS" in validation

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DCUT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = DCUT_ROOT / "kernel"


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def test_piecewise_gdn_core_is_moved_inside_graph() -> None:
    patch = _read("patch_gdn_v023.py")
    core = _read("gdn_forward_v023.py")
    install = _read("install.py")
    runner = _read("patch_runner.py")

    assert "target_class._forward_core = dcut_forward_core" in patch
    assert "target_class.forward =" not in patch
    assert "torch.ops.vllm.qwen_gdn_attention_core" in core
    assert 'GDN_PIECEWISE_SPLITTING_OP = "vllm::qwen_gdn_attention_core"' in patch
    assert "CompilationConfig._attention_ops = _without_gdn_piecewise_split" in patch
    assert "compilation_config.splitting_ops = _without_gdn_piecewise_split" in patch
    assert "_enable_gdn_piecewise_graph()" in install
    assert runner.index("_enable_gdn_piecewise_graph(vllm_config)") < runner.index(
        "_orig_init(self, *a, **k)"
    )
    assert "npu_dcut_causal_conv1d" in core
    assert "npu_dcut_recurrent_gated_delta_rule" in core
    assert "ssm_state_indices=spec_state_indices_tensor.flatten()" not in core


def test_recurrent_kernel_uses_fixed_request_rows() -> None:
    for relative_path in (
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_kernel/recurrent_gated_delta_rule.h",
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_kernel/arch35/recurrent_gated_delta_rule.h",
    ):
        kernel = _read(relative_path)
        assert "batch_i * stateIndexStride_ + (seq_i - seq0)" in kernel
        assert "stateTokenIdx += acceptedTokenNum - 1" in kernel
        assert "acceptedTokenNum > static_cast<int32_t>(stateIndexStride_)" in kernel


def test_conv_kernel_accepts_zero_based_state_offsets() -> None:
    wrapper = _read("kernel/dcut_causal_conv1d/op_kernel/dcut_causal_conv1d.cpp")
    kernel = _read("kernel/dcut_causal_conv1d/vendor/op_kernel/causal_conv1d.h")

    assert "DCUT_CAUSAL_CONV_DIRECT_STATE_OFFSETS" in wrapper
    assert "stateTokenOffset = ReadNumAcceptedTokensValue(seq);" in kernel


def test_torch_registration_has_graph_metadata() -> None:
    binding = _read("kernel/torch_extension/dcut_torch_binding.cpp")

    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert "Tensor(a!) state" in binding
    assert "Tensor(b!) conv_state" in binding


def test_truncation_has_no_previous_acceptance_floor() -> None:
    truncate = _read("truncate.py")

    assert "_get_gdn_min_draft_lens" not in truncate
    assert "gdn_min_draft_lens" not in truncate

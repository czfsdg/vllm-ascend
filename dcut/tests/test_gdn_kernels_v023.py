# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DCUT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = DCUT_ROOT / "kernel"
REPO_ROOT = DCUT_ROOT.parent


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def test_piecewise_gdn_core_is_switch_gated() -> None:
    patch = _read("patch_gdn_v023.py")
    core = _read("gdn_forward_v023.py")
    install = _read("install.py")
    runner = _read("patch_runner.py")
    envs = (REPO_ROOT / "vllm_ascend" / "envs.py").read_text(
        encoding="utf-8"
    )

    assert "target_class._forward_core = dcut_forward_core" in patch
    assert "target_class.forward =" not in patch
    assert "torch.ops.vllm.qwen_gdn_attention_core" in core
    assert 'GDN_PIECEWISE_SPLITTING_OP = "vllm::qwen_gdn_attention_core"' in patch
    assert '"VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE": lambda: bool(' in envs
    assert 'os.getenv("VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE", "0")' in envs
    assert patch.index(
        "if not _gdn_piecewise_graph_enabled():"
    ) < patch.index(
        "from vllm.config.compilation import CompilationConfig"
    )
    assert "CompilationConfig._attention_ops = _without_gdn_piecewise_split" in patch
    assert "compilation_config.splitting_ops = _without_gdn_piecewise_split" in patch
    assert "_enable_gdn_piecewise_graph()" in install
    assert runner.index("_enable_gdn_piecewise_graph(vllm_config)") < runner.index(
        "_orig_init(self, *a, **k)"
    )
    assert "npu_dcut_causal_conv1d" in core
    assert "npu_dcut_recurrent_gated_delta_rule" in core
    assert "ssm_state_indices=spec_state_indices_tensor.flatten()" not in core
    assert "_dcut_get_gdn_piecewise_spec_bufs" in core
    assert "_dcut_prepare_gdn_piecewise_replay" in runner
    assert "forward_context.cudagraph_runtime_mode" in runner
    assert "CUDAGraphMode.NONE" in runner
    assert '"_dcut_piecewise_capture_dummy"' in runner
    assert "_should_build_dummy_attn_metadata" in runner
    assert "is_graph_capturing" in runner
    assert "_dcut_gdn_piecewise_capture_sizes" in runner
    assert "captured during startup; using eager" in runner
    assert "captured PIECEWISE GDN token bucket" in runner


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


def test_conv_host_registers_explicit_dcut_name() -> None:
    tiling = _read(
        "kernel/dcut_causal_conv1d/op_host/dcut_causal_conv1d_tiling.cpp"
    )
    infershape = _read(
        "kernel/dcut_causal_conv1d/op_host/dcut_causal_conv1d_infershape.cpp"
    )

    assert "IMPL_OP_OPTILING(DcutCausalConv1d)" in tiling
    assert "IMPL_OP_INFERSHAPE(DcutCausalConv1d)" in infershape
    for source in (tiling, infershape):
        assert "#define CausalConv1d" not in source
        assert "causal_conv1d_tiling.cpp\"" not in source
        assert "causal_conv1d_infershape.cpp\"" not in source


def test_recurrent_host_registers_explicit_dcut_name() -> None:
    tiling = _read(
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_host/"
        "recurrent_gated_delta_rule_tiling.cpp"
    )
    arch35 = _read(
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_host/arch35/"
        "recurrent_gated_delta_rule_tiling_arch35.cpp"
    )
    infershape = _read(
        "kernel/dcut_recurrent_gated_delta_rule/op_host/"
        "dcut_recurrent_gated_delta_rule_infershape.cpp"
    )

    assert "IMPL_OP_OPTILING(DcutRecurrentGatedDeltaRule)" in tiling
    assert (
        "REGISTER_OPS_TILING_TEMPLATE(DcutRecurrentGatedDeltaRule, "
        "DcutRecurrentGatedDeltaRuleTiling, 0)"
    ) in tiling
    assert (
        "REGISTER_OPS_TILING_TEMPLATE(DcutRecurrentGatedDeltaRule,"
    ) in arch35
    assert "DcutRecurrentGatedDeltaRuleTilingArch35" in arch35
    assert "IMPL_OP_INFERSHAPE(DcutRecurrentGatedDeltaRule)" in infershape

    assert "IMPL_OP_OPTILING(RecurrentGatedDeltaRule)" not in tiling
    assert (
        "REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule"
    ) not in tiling
    assert (
        "REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule"
    ) not in arch35
    assert "#define RecurrentGatedDeltaRule" not in infershape
    assert "recurrent_gated_delta_rule_infershape.cpp\"" not in infershape


def test_torch_registration_has_graph_metadata() -> None:
    binding = _read("kernel/torch_extension/dcut_torch_binding.cpp")
    conv_wrapper = _read(
        "kernel/dcut_causal_conv1d/dcut_causal_conv1d_torch_adpt.h"
    )

    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert "Tensor(a!) state" in binding
    assert "Tensor(b!) conv_state" in binding
    assert "DCUT_CAUSAL_CONV_RUN_MODE = 1" in conv_wrapper
    assert "pad_slot_id, 1, output" not in conv_wrapper


def test_acl_workspace_size_is_valid_tensor_shape() -> None:
    adapter = (
        REPO_ROOT / "csrc" / "aclnn_torch_adapter" / "op_api_common.h"
    ).read_text(encoding="utf-8")

    assert "at::empty({static_cast<int64_t>(workspace_size)}" in adapter


def test_torch_registration_links_npu_bridge() -> None:
    cmake = _read("kernel/torch_extension/CMakeLists.txt")

    assert '"${REPO_ROOT}/csrc/aclnn_torch_adapter/NPUBridge.cpp"' in cmake


def test_recurrent_opapi_uses_explicit_dcut_dfx_names() -> None:
    l0_source = _read(
        "kernel/dcut_recurrent_gated_delta_rule/op_host/op_api/"
        "dcut_recurrent_gated_delta_rule.cpp"
    )
    aclnn_source = _read(
        "kernel/dcut_recurrent_gated_delta_rule/op_host/op_api/"
        "aclnn_dcut_recurrent_gated_delta_rule.cpp"
    )
    aclnn_header = _read(
        "kernel/dcut_recurrent_gated_delta_rule/op_host/op_api/"
        "aclnn_dcut_recurrent_gated_delta_rule.h"
    )

    assert "L0_DFX(DcutRecurrentGatedDeltaRule" in l0_source
    assert "L2_DFX_PHASE_1(" in aclnn_source
    assert "aclnnDcutRecurrentGatedDeltaRule," in aclnn_source
    assert '#include "opdev/make_op_executor.h"' in aclnn_source
    assert "uint64_t* workspaceSize" in aclnn_source
    assert "*workspaceSize = unique_executor->GetWorkspaceSize()" in aclnn_source
    assert (
        "L2_DFX_PHASE_2(aclnnDcutRecurrentGatedDeltaRule)" in aclnn_source
    )
    assert "uint64_t workspaceSize" in aclnn_source
    assert (
        "CommonOpExecutorRun(workspace, workspaceSize, executor, stream)"
        in aclnn_source
    )
    assert "l0op::DcutRecurrentGatedDeltaRule(" in aclnn_source
    assert "aclnnDcutRecurrentGatedDeltaRuleGetWorkspaceSize(" in aclnn_header
    for source in (l0_source, aclnn_source):
        assert "#define L0_DFX" not in source
        assert "#define RecurrentGatedDeltaRule" not in source
        assert "../../../../../csrc/attention/" not in source


def test_truncation_has_no_previous_acceptance_floor() -> None:
    truncate = _read("truncate.py")

    assert "_get_gdn_min_draft_lens" not in truncate
    assert "gdn_min_draft_lens" not in truncate

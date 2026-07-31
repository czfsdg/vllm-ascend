# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DCUT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DCUT_ROOT.parent


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def test_piecewise_switch_is_registered_and_default_off() -> None:
    envs = (REPO_ROOT / "vllm_ascend" / "envs.py").read_text(
        encoding="utf-8"
    )
    piecewise = _read("patch_piecewise.py")

    assert '"VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE": lambda: bool(' in envs
    assert 'os.getenv("VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE", "0")' in envs
    assert 'ENV_DCUT_CONFIG = "VLLM_DCUT_CONFIG"' in piecewise
    assert 'ENV_GDN_PIECEWISE = "VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE"' in piecewise
    assert "if not os.environ.get(ENV_DCUT_CONFIG):" in piecewise
    assert "return [op for op in ops if op != _TARGET_OP]" in piecewise


def test_runner_uses_fail_closed_phase_and_parallel_routing() -> None:
    runner = _read("patch_runner.py")

    assert "_dcut_prepare_gdn_piecewise_replay" in runner
    assert "_dcut_gdn_piecewise_enabled" in runner
    assert "meta.spec_sequence_masks is None" not in runner
    assert 'getattr(self, "pcp_size", 1) == 1' in runner
    assert 'getattr(self, "dcp_size", 1) == 1' in runner
    assert "CUDAGraphMode.NONE" in runner
    assert "runtime_mode_overridden" in runner
    assert "forward_context.cudagraph_runtime_mode = (" in runner
    assert "original_runtime_mode" in runner
    assert "captured during startup; using eager" in runner
    assert "_dcut_piecewise_capture_dummy" in runner
    assert "_should_build_dummy_attn_metadata" in runner


def test_forward_requires_real_capture_and_masks_padding() -> None:
    core = _read("gdn_forward_v023.py")
    buffers = _read("gdn_buffers.py")

    assert "torch.npu.is_current_stream_capturing()" in core
    assert "GDN core entered an active PIECEWISE ACLGraph" in core
    assert 'piecewise_spec_bufs["qsl"]' in core
    assert 'piecewise_spec_bufs["ssi"]' in core
    assert 'piecewise_spec_bufs["nat"]' in core
    assert 'piecewise_spec_bufs["asl"]' in core
    assert 'piecewise_spec_bufs["token_mask"]' in core
    assert "_dcut_gdn_piecewise_spec_key" in buffers
    assert "id(model_instance)" in buffers
    assert "meta.num_prefills) != 0" in buffers
    assert "meta.num_decodes) != 0" in buffers


def test_full_attention_remains_an_eager_piecewise_boundary() -> None:
    install = _read("install.py")
    attention = _read("patch_attention.py")

    assert "and not _patch_attention()" in install
    assert 'patch_marker = "_dcut_piecewise_fia_patched"' in attention
    assert "mode == CUDAGraphMode.PIECEWISE" in attention
    assert "ctx.capturing = False" in attention
    assert "ctx.capturing = orig_capturing" in attention

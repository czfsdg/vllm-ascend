# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from dcut import patch_gdn_v023 as gdn_patch
from dcut.patch_runner import (
    _dcut_piecewise_capture_dummy_enabled,
)

ENV_NAME = gdn_patch.ENV_GDN_PIECEWISE


def test_gdn_piecewise_graph_switch_defaults_to_disabled(monkeypatch) -> None:
    from vllm_ascend import envs

    monkeypatch.delenv(ENV_NAME, raising=False)
    assert envs.VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE is False

    monkeypatch.setenv(ENV_NAME, "1")
    assert envs.VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE is True

    monkeypatch.setenv(ENV_NAME, "0")
    assert envs.VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE is False


def test_switch_falls_back_when_installed_envs_lacks_registration(
    monkeypatch,
) -> None:
    from vllm_ascend import envs

    monkeypatch.delitem(envs.env_variables, ENV_NAME)
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert gdn_patch._gdn_piecewise_graph_enabled() is False

    monkeypatch.setenv(ENV_NAME, "1")
    assert gdn_patch._gdn_piecewise_graph_enabled() is True

    monkeypatch.setenv(ENV_NAME, "0")
    assert gdn_patch._gdn_piecewise_graph_enabled() is False


def test_disabled_switch_preserves_gdn_splitting_boundary(
    monkeypatch,
) -> None:
    from vllm.config.compilation import CompilationConfig

    splitting_ops = [
        "vllm::unrelated_attention",
        gdn_patch.GDN_PIECEWISE_SPLITTING_OP,
    ]
    monkeypatch.setattr(
        gdn_patch,
        "_gdn_piecewise_graph_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        CompilationConfig,
        "_attention_ops",
        list(splitting_ops),
    )
    live_compilation_config = SimpleNamespace(
        splitting_ops=list(splitting_ops)
    )
    vllm_config = SimpleNamespace(
        compilation_config=live_compilation_config
    )

    assert gdn_patch._enable_gdn_piecewise_graph(vllm_config)
    assert CompilationConfig._attention_ops == splitting_ops
    assert live_compilation_config.splitting_ops == splitting_ops


def test_enabled_switch_removes_only_gdn_splitting_boundary(
    monkeypatch,
) -> None:
    from vllm.config.compilation import CompilationConfig

    unrelated_op = "vllm::unrelated_attention"
    splitting_ops = [
        unrelated_op,
        gdn_patch.GDN_PIECEWISE_SPLITTING_OP,
    ]
    monkeypatch.setattr(
        gdn_patch,
        "_gdn_piecewise_graph_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        CompilationConfig,
        "_attention_ops",
        list(splitting_ops),
    )
    live_compilation_config = SimpleNamespace(
        splitting_ops=list(splitting_ops)
    )
    vllm_config = SimpleNamespace(
        compilation_config=live_compilation_config
    )

    assert gdn_patch._enable_gdn_piecewise_graph(vllm_config)
    assert CompilationConfig._attention_ops == [unrelated_op]
    assert live_compilation_config.splitting_ops == [unrelated_op]


def test_piecewise_dummy_metadata_is_only_for_startup_capture(
    monkeypatch,
) -> None:
    from vllm.config import CUDAGraphMode

    monkeypatch.setattr(
        "dcut.patch_runner._gdn_piecewise_graph_enabled",
        lambda: True,
    )
    runner = SimpleNamespace(
        _dcut_in_real_warmup=True,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE
        ),
    )

    assert _dcut_piecewise_capture_dummy_enabled(
        runner,
        CUDAGraphMode.PIECEWISE,
        is_graph_capturing=True,
    )
    assert not _dcut_piecewise_capture_dummy_enabled(
        runner,
        CUDAGraphMode.PIECEWISE,
        is_graph_capturing=False,
    )
    assert not _dcut_piecewise_capture_dummy_enabled(
        runner,
        CUDAGraphMode.PIECEWISE,
        is_profile=True,
        is_graph_capturing=True,
    )
    assert not _dcut_piecewise_capture_dummy_enabled(
        runner,
        CUDAGraphMode.NONE,
        is_graph_capturing=True,
    )

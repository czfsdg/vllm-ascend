# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

DCUT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DCUT_ROOT.parent


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def _load_local_graph_functions(namespace: dict):
    path = DCUT_ROOT / "gdn_forward_v023.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_dcut_gdn_local_graph_key",
        "_dcut_mark_local_graph_captured",
        "_dcut_run_gdn_local_graph",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *functions,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


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
    assert "_ensure_gdn_splitting_op" in piecewise
    assert "result.append(_TARGET_OP)" in piecewise


def test_runner_keeps_outer_piecewise_and_routes_gdn_locally() -> None:
    runner = _read("patch_runner.py")

    assert "_dcut_prepare_gdn_piecewise_replay" in runner
    assert "_dcut_gdn_piecewise_enabled" in runner
    assert "meta.spec_sequence_masks is None" not in runner
    assert 'getattr(self, "pcp_size", 1) == 1' in runner
    assert 'getattr(self, "dcp_size", 1) == 1' in runner
    assert "CUDAGraphMode.NONE" not in runner
    assert "runtime_mode_overridden" not in runner
    assert "forward_context.cudagraph_runtime_mode = (" not in runner
    assert "_dcut_gdn_local_graph_safe" in runner
    assert "_dcut_gdn_local_graph_capture_requested" in runner
    assert "only GDN boundaries use eager" in runner
    model_forward = runner[
        runner.index("    def _model_forward("):
        runner.index("    def execute_model(")
    ]
    reset = "forward_context._dcut_gdn_local_graph_safe = False"
    mode_check = "== CUDAGraphMode.PIECEWISE"
    assert reset in model_forward
    assert model_forward.index(reset) < model_forward.index(mode_check)
    assert "_dcut_piecewise_capture_dummy" in runner
    assert "_should_build_dummy_attn_metadata" in runner


def test_forward_captures_locally_and_zeroes_padding_in_kernel() -> None:
    core = _read("gdn_forward_v023.py")
    buffers = _read("gdn_buffers.py")

    assert "torch.npu.is_current_stream_capturing()" in core
    assert "torch.npu.NPUGraph()" in core
    assert "graph.replay()" in core
    assert "current_platform.get_global_graph_pool()" in core
    assert "_dcut_gdn_local_graph_key" in core
    assert "qwen_gdn_attention_core must" in core
    assert 'piecewise_spec_bufs["qsl"]' in core
    assert 'piecewise_spec_bufs["ssi"]' in core
    assert 'piecewise_spec_bufs["nat"]' in core
    assert 'piecewise_spec_bufs["asl"]' not in core
    assert 'piecewise_spec_bufs["token_mask"]' not in core
    assert "zero_padded_output=piecewise_spec_bufs is not None" in core
    assert "_dcut_gdn_piecewise_spec_key" in buffers
    assert "_dcut_gdn_local_graph_expected_prefixes" in buffers
    assert "id(model_instance)" in buffers
    assert "meta.num_prefills) != 0" in buffers
    assert "meta.num_decodes) != 0" in buffers


def test_unmatched_runtime_addresses_are_captured_then_replayed() -> None:
    class FakeTensor:
        def __init__(self, address: int):
            self.shape = (128,)
            self._address = address

        def data_ptr(self):
            return self._address

    class FakeGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    class FakeNpu:
        capturing = False

        @classmethod
        def is_current_stream_capturing(cls):
            return cls.capturing

        @staticmethod
        def NPUGraph():
            return FakeGraph()

        @staticmethod
        def graph(graph, pool):
            return nullcontext()

    class FakeCore:
        calls = 0

        @classmethod
        def _forward_core_impl(cls, *args):
            cls.calls += 1

    functions = _load_local_graph_functions(
        {
            "torch": SimpleNamespace(npu=FakeNpu),
            "current_platform": SimpleNamespace(
                get_global_graph_pool=lambda: object()
            ),
            "AscendGatedDeltaNetAttention": FakeCore,
            "logger": SimpleNamespace(warning=lambda *args: None),
        }
    )
    run_graph = functions["_dcut_run_gdn_local_graph"]
    attention = SimpleNamespace(prefix="layers.0.mixer")
    context = SimpleNamespace(
        _dcut_gdn_local_graph_capture_requested=False,
        _dcut_gdn_local_graph_captured_prefixes=set(),
    )
    tensors = tuple(FakeTensor(address) for address in range(1, 5))

    assert run_graph(attention, *tensors, context, {})
    graph = next(iter(attention._dcut_gdn_local_graph_entries.values()))
    assert FakeCore.calls == 1
    assert graph.replays == 0

    assert run_graph(attention, *tensors, context, {})
    assert FakeCore.calls == 1
    assert graph.replays == 1


def test_runtime_local_graph_never_nests_inside_outer_capture() -> None:
    source = _read("gdn_forward_v023.py")

    assert "graph is None and torch.npu.is_current_stream_capturing()" in source
    assert "if graph is None and not capture_requested" not in source


def test_full_attention_remains_an_eager_piecewise_boundary() -> None:
    install = _read("install.py")
    attention = _read("patch_attention.py")

    assert "and not _patch_attention()" in install
    assert 'patch_marker = "_dcut_piecewise_fia_patched"' in attention
    assert "mode == CUDAGraphMode.PIECEWISE" in attention
    assert "ctx.capturing = False" in attention
    assert "ctx.capturing = orig_capturing" in attention

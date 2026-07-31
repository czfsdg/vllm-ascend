# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

CORE_PATH = Path(__file__).resolve().parents[1] / "gdn_forward_v023.py"
_ROUTER_FUNCTIONS = {
    "_dcut_gdn_local_graph_key",
    "_dcut_mark_local_graph_captured",
    "_dcut_run_gdn_local_graph",
}


class _FakeTensor:
    def __init__(self, num_tokens: int, address: int):
        self.shape = (num_tokens,)
        self._address = address

    def data_ptr(self) -> int:
        return self._address


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self) -> None:
        self.replay_count += 1


class _FakeNPU:
    def __init__(self):
        self.graphs = []
        self.stream_capturing = False

    def NPUGraph(self):
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph

    def is_current_stream_capturing(self) -> bool:
        return self.stream_capturing

    @contextmanager
    def graph(self, graph, pool):
        assert graph in self.graphs
        assert pool == "global-pool"
        yield


class _FakeAttentionImpl:
    @staticmethod
    def _forward_core_impl(attention, *args) -> None:
        attention.impl_calls += 1


def _load_router():
    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in _ROUTER_FUNCTIONS
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    fake_npu = _FakeNPU()
    warnings = []
    namespace = {
        "torch": SimpleNamespace(npu=fake_npu, Tensor=_FakeTensor),
        "current_platform": SimpleNamespace(
            get_global_graph_pool=lambda: "global-pool"
        ),
        "AscendGatedDeltaNetAttention": _FakeAttentionImpl,
        "logger": SimpleNamespace(
            warning=lambda *args, **kwargs: warnings.append(args)
        ),
    }
    exec(compile(module, str(CORE_PATH), "exec"), namespace)
    return namespace, fake_npu, warnings


def test_local_graph_router_captures_replays_and_falls_back() -> None:
    namespace, fake_npu, warnings = _load_router()
    run_local_graph = namespace["_dcut_run_gdn_local_graph"]

    attention = SimpleNamespace(prefix="layers.0.mixer", impl_calls=0)
    context = SimpleNamespace(
        _dcut_gdn_local_graph_capture_requested=True,
        _dcut_gdn_local_graph_captured_prefixes=set(),
    )
    tensors = (
        _FakeTensor(128, 100),
        _FakeTensor(128, 200),
        _FakeTensor(128, 300),
        _FakeTensor(128, 400),
    )

    assert run_local_graph(
        attention, *tensors, context, {"qsl": object()}
    )
    assert attention.impl_calls == 1
    assert context._dcut_gdn_local_graph_captured_prefixes == {
        attention.prefix
    }
    assert len(fake_npu.graphs) == 1

    context._dcut_gdn_local_graph_capture_requested = False
    assert run_local_graph(
        attention, *tensors, context, {"qsl": object()}
    )
    assert fake_npu.graphs[0].replay_count == 1
    assert attention.impl_calls == 1

    different_output = _FakeTensor(128, 401)
    assert not run_local_graph(
        attention,
        *tensors[:3],
        different_output,
        context,
        {"qsl": object()},
    )
    assert len(fake_npu.graphs) == 1
    assert warnings


def test_local_graph_router_rejects_nested_capture() -> None:
    namespace, fake_npu, _ = _load_router()
    run_local_graph = namespace["_dcut_run_gdn_local_graph"]
    fake_npu.stream_capturing = True

    attention = SimpleNamespace(prefix="layers.0.mixer", impl_calls=0)
    context = SimpleNamespace(
        _dcut_gdn_local_graph_capture_requested=True,
        _dcut_gdn_local_graph_captured_prefixes=set(),
    )
    tensors = tuple(_FakeTensor(64, address) for address in range(4))

    try:
        run_local_graph(
            attention, *tensors, context, {"qsl": object()}
        )
    except RuntimeError as exc:
        assert "cannot nest the local GDN graph" in str(exc)
    else:
        raise AssertionError("nested local graph capture was not rejected")

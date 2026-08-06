# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

PATCH_PATH = Path(__file__).resolve().parents[1] / "patch_gdn_v023.py"


def _load_prefill_routers():
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_dcut_gdn_has_prefill", "_dcut_gdn_use_native_core"}
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return (
        namespace["_dcut_gdn_has_prefill"],
        namespace["_dcut_gdn_use_native_core"],
    )


def test_prefill_batch_routes_to_native_gdn_core() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=1),
            "layers.1.mixer": SimpleNamespace(num_prefills=1),
        }
    )

    assert has_prefill(context)
    assert has_prefill(context, "layers.0.mixer")
    assert use_native(context, "layers.0.mixer")


def test_pure_spec_batch_keeps_dcut_gdn_core() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=0),
        }
    )

    assert not has_prefill(context)
    assert not has_prefill(context, "layers.0.mixer")
    assert not use_native(context, "layers.0.mixer")


def test_scheduler_non_prefill_overrides_synthetic_gdn_prefill() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        _dcut_gdn_native_batch=False,
        attn_metadata={
            # The native GDN builder reports ordinary decode rows here when
            # speculative and non-speculative decode coexist.
            "layers.0.mixer": SimpleNamespace(num_prefills=3),
        },
    )

    assert has_prefill(context, "layers.0.mixer")
    assert not use_native(context, "layers.0.mixer")


def test_reused_context_can_return_to_pure_spec_route() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        _dcut_gdn_native_batch=True,
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=0),
        },
    )

    assert not has_prefill(context)
    assert use_native(context, "layers.0.mixer")

    context._dcut_gdn_native_batch = False
    assert not use_native(context, "layers.0.mixer")


def test_prefill_metadata_uses_native_builder() -> None:
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_gdn_spec_metadata_builder"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)

    calls = []

    def native_builder(self, attn_metadata):
        calls.append(attn_metadata)
        return "native"

    class MetadataBuilder:
        _attach_spec_decode_metadata = native_builder

    fake_module = SimpleNamespace(
        AscendGDNAttentionMetadataBuilder=MetadataBuilder
    )
    namespace["_patch_gdn_spec_metadata_builder"](fake_module)
    metadata = SimpleNamespace(num_prefills=1)

    assert MetadataBuilder()._attach_spec_decode_metadata(metadata) == "native"
    assert calls == [metadata]

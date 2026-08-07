# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

PATCH_PATH = Path(__file__).resolve().parents[1] / "patch_runner.py"
FORCE_EAGER_ARG_POSITION = 6


def _load_force_prefill_eager():
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dcut_force_prefill_eager"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_FORCE_EAGER_ARG_POSITION": FORCE_EAGER_ARG_POSITION,
    }
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return namespace["_dcut_force_prefill_eager"]


def _record_force_eager(calls):
    def determine(runner, *args, **kwargs):
        if len(args) > FORCE_EAGER_ARG_POSITION:
            force_eager = args[FORCE_EAGER_ARG_POSITION]
        else:
            force_eager = kwargs.get("force_eager", False)
        calls.append(force_eager)
        return force_eager

    return determine


def test_pure_prefill_forces_eager_with_keyword_argument() -> None:
    force_prefill_eager = _load_force_prefill_eager()
    calls = []
    runner = SimpleNamespace(_dcut_gdn_scheduler_has_prefill=True)

    result = force_prefill_eager(
        runner,
        _record_force_eager(calls),
        8,
        1,
        object(),
        8,
        False,
        force_eager=False,
    )

    assert result is True
    assert calls == [True]


def test_mixed_batch_forces_eager_with_positional_argument() -> None:
    force_prefill_eager = _load_force_prefill_eager()
    calls = []
    runner = SimpleNamespace(_dcut_gdn_scheduler_has_prefill=True)

    result = force_prefill_eager(
        runner,
        _record_force_eager(calls),
        9,
        2,
        object(),
        8,
        False,
        False,
        False,
    )

    assert result is True
    assert calls == [True]


def test_pure_decode_preserves_piecewise_eligibility() -> None:
    force_prefill_eager = _load_force_prefill_eager()
    calls = []
    runner = SimpleNamespace(_dcut_gdn_scheduler_has_prefill=False)

    result = force_prefill_eager(
        runner,
        _record_force_eager(calls),
        8,
        1,
        object(),
        8,
        False,
        force_eager=False,
    )

    assert result is False
    assert calls == [False]


def test_runner_installs_prefill_eager_dispatch_wrapper() -> None:
    source = PATCH_PATH.read_text(encoding="utf-8")

    assert "_dcut_gdn_scheduler_has_prefill" in source
    assert "_orig_determine_batch_execution_and_padding" in source
    assert (
        "R._determine_batch_execution_and_padding = ("
        in source
    )

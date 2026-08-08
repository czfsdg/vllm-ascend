# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch

DCUT_DIR = Path(__file__).resolve().parents[1]


def _load_functions(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
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


def _load_utils():
    names = {
        "_env_flag",
        "_dcut_process_probs_stage",
        "_dcut_reuse_argmax_enabled",
        "_dcut_selected_token_probs",
        "_dcut_can_reuse_argmax_for_probs",
        "_dcut_selected_probs_from_graph",
        "_dcut_selected_probs_from_reused_logits",
    }
    return _load_functions(
        DCUT_DIR / "utils.py",
        names,
        {
            "os": os,
            "torch": torch,
            "ENV_PROCESS_PROBS_STAGE": "VLLM_DCUT_PROCESS_PROBS_STAGE",
            "ENV_REUSE_ARGMAX": "VLLM_DCUT_REUSE_ARGMAX",
        },
    )


def test_deferred_processing_and_argmax_reuse_defaults(monkeypatch) -> None:
    utils = _load_utils()
    monkeypatch.delenv("VLLM_DCUT_PROCESS_PROBS_STAGE", raising=False)
    monkeypatch.delenv("VLLM_DCUT_REUSE_ARGMAX", raising=False)

    assert utils["_dcut_process_probs_stage"]() == "pre_truncate"
    assert utils["_dcut_reuse_argmax_enabled"]() is True

    monkeypatch.setenv("VLLM_DCUT_PROCESS_PROBS_STAGE", "post-sample")
    monkeypatch.setenv("VLLM_DCUT_REUSE_ARGMAX", "0")
    assert utils["_dcut_process_probs_stage"]() == "post_sample"
    assert utils["_dcut_reuse_argmax_enabled"]() is False


class _Drafter:
    method = "dflash"
    _dcut_run_merged_patched = True


def test_reused_logits_match_selected_softmax_probability(monkeypatch) -> None:
    utils = _load_utils()
    monkeypatch.delenv("VLLM_DCUT_REUSE_ARGMAX", raising=False)
    logits = torch.tensor(
        [[1.0, 3.0, -2.0], [2.0, 0.5, 1.0]],
        dtype=torch.float32,
    )
    token_ids = torch.tensor([[1, 0]])
    drafter = _Drafter()
    drafter._dcut_last_logits_for_probs = logits
    drafter._dcut_last_draft_ran_python = True

    actual = utils["_dcut_selected_probs_from_reused_logits"](
        drafter,
        token_ids,
    )
    expected = logits.softmax(dim=-1).gather(
        -1,
        token_ids.reshape(-1, 1),
    ).reshape_as(token_ids)

    assert actual is not None
    torch.testing.assert_close(actual, expected)


def test_graph_selected_probs_are_selected_by_output_bucket() -> None:
    utils = _load_utils()
    small = torch.tensor([[0.8, 0.6]], dtype=torch.float32)
    large = torch.tensor(
        [[0.9, 0.7], [0.5, 0.3]],
        dtype=torch.float32,
    )
    draft_token_ids = torch.zeros((1, 2), dtype=torch.int64)
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=False,
        _dcut_graph_selected_probs_ready=True,
        _dcut_graph_selected_probs_by_output_ptr={
            int(draft_token_ids.data_ptr()): small,
        },
        # Deliberately point the same-shape fallback at another bucket. The
        # fixed output address must win when multiple graphs share a shape.
        _dcut_graph_selected_probs_by_shape={
            (1, 2): large,
        },
        _dcut_graph_selected_probs_by_numel={
            2: large,
        },
    )

    actual = utils["_dcut_selected_probs_from_graph"](
        drafter,
        draft_token_ids,
    )

    assert actual is not None
    torch.testing.assert_close(actual, small)


def test_eager_draft_does_not_reuse_graph_selected_probs() -> None:
    utils = _load_utils()
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=True,
        _dcut_graph_selected_probs_ready=True,
        _dcut_graph_selected_probs_by_output_ptr={},
        _dcut_graph_selected_probs_by_shape={
            (1, 2): torch.ones((1, 2)),
        },
        _dcut_graph_selected_probs_by_numel={},
    )

    actual = utils["_dcut_selected_probs_from_graph"](
        drafter,
        torch.zeros((1, 2), dtype=torch.int64),
    )

    assert actual is None


def test_pre_truncate_waits_once_and_filters_finished_requests() -> None:
    process = _load_functions(
        DCUT_DIR / "probs.py",
        {"_maybe_process_adaptive_probs"},
        {},
    )["_maybe_process_adaptive_probs"]

    class Event:
        synchronized = 0

        def query(self):
            return False

        def synchronize(self):
            self.synchronized += 1

    class Controller:
        call = None

        def process_draft_output(self, **kwargs):
            self.call = kwargs

    event = Event()
    controller = Controller()
    runner = SimpleNamespace(
        _adaptive_probs_pending=True,
        _adaptive_probs_event=event,
        _dcut_skip_unready_probs=False,
        _adaptive_num_reqs=2,
        _adaptive_active={"keep", "finished"},
        _adaptive_req_ids=["keep", "finished"],
        _adaptive_probs_pinned=torch.ones((2, 3)),
        _verify_adaptive_controller=controller,
        input_batch=SimpleNamespace(
            req_ids=["keep", "new"],
            num_reqs=2,
        ),
    )

    process(runner, stage="pre_truncate")

    assert event.synchronized == 1
    assert runner._adaptive_probs_pending is False
    assert controller.call is not None
    assert controller.call["active_draft_req_ids"] == {"keep"}


def test_prepare_clears_transient_graph_logits_pointer() -> None:
    prepare = _load_functions(
        DCUT_DIR / "probs.py",
        {"_dcut_prepare_prob_capture"},
        {},
    )["_dcut_prepare_prob_capture"]
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=True,
        _dcut_last_logits_for_probs=object(),
        _last_selected_probs=object(),
    )
    runner = SimpleNamespace(drafter=drafter)

    prepare(runner, SimpleNamespace())

    assert drafter._dcut_last_draft_ran_python is False
    assert drafter._dcut_last_logits_for_probs is None
    assert drafter._last_selected_probs is None

def test_no_low_batch_or_eager_fallback_was_migrated() -> None:
    sources = "\n".join(
        (DCUT_DIR / name).read_text(encoding="utf-8")
        for name in (
            "controller.py",
            "drafter.py",
            "globals.py",
            "patch_proposer.py",
            "patch_runner.py",
            "probs.py",
            "truncate.py",
            "utils.py",
        )
    )
    assert "MIN_SPEC_BATCH" not in sources
    assert "min_spec_batch" not in sources
    assert "use_cuda_graph = False" not in sources


def test_pending_probs_are_processed_before_truncation() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    execute_start = source.index("    def execute_model(")
    execute_end = source.index("    _orig_sample_tokens", execute_start)
    execute_source = source[execute_start:execute_end]

    process_at = execute_source.index("_maybe_process_adaptive_probs")
    truncate_at = execute_source.index("scheduler_output = _dcut_truncate")
    assert process_at < truncate_at


def test_scheduler_prefill_route_is_scoped_to_execute_model() -> None:
    route = _load_functions(
        DCUT_DIR / "patch_runner.py",
        {"_dcut_execute_with_gdn_prefill_route"},
        {},
    )["_dcut_execute_with_gdn_prefill_route"]
    runner = SimpleNamespace()
    observed = []

    def execute(active_runner, scheduler_output, intermediate_tensors):
        observed.append(active_runner._dcut_gdn_scheduler_has_prefill)
        return scheduler_output, intermediate_tensors

    result = route(runner, execute, "schedule", "intermediate", False)

    assert result == ("schedule", "intermediate")
    assert observed == [False]
    assert not hasattr(runner, "_dcut_gdn_scheduler_has_prefill")

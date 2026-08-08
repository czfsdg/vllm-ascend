# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


DCUT_DIR = Path(__file__).resolve().parents[1]


def _load_bypass():
    path = DCUT_DIR / "probs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dcut_bypass_prob_capture_for_prefill"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_dcut_bypass_prob_capture_for_prefill"]


def _load_drafter_enable():
    path = DCUT_DIR / "controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dcut_enable_drafter_probs"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_dcut_patch_drafter_instance": lambda drafter: None,
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_dcut_enable_drafter_probs"]


def test_prefill_bypass_disables_drafter_and_drops_stale_decision() -> None:
    class Controller:
        cleared = 0

        def clear_adaptive_decision(self):
            self.cleared += 1

    controller = Controller()
    drafter = SimpleNamespace(
        needs_draft_probs=True,
        _dcut_last_draft_ran_python=True,
        _dcut_last_logits_for_probs=object(),
        _last_selected_probs=object(),
    )
    runner = SimpleNamespace(
        drafter=drafter,
        _adaptive_probs_pending=True,
        _adaptive_num_reqs=2,
        _adaptive_req_ids=["old-0", "old-1"],
        _adaptive_active={"old-0", "old-1"},
        _verify_adaptive_controller=controller,
    )

    _load_bypass()(runner)

    assert drafter.needs_draft_probs is False
    assert drafter._dcut_last_draft_ran_python is False
    assert drafter._dcut_last_logits_for_probs is None
    assert drafter._last_selected_probs is None
    assert runner._adaptive_probs_pending is False
    assert runner._adaptive_num_reqs == 0
    assert runner._adaptive_req_ids == []
    assert runner._adaptive_active == set()
    assert controller.cleared == 1


def test_decode_after_prefill_reenables_drafter_probabilities() -> None:
    controller = SimpleNamespace(clear_adaptive_decision=lambda: None)
    drafter = SimpleNamespace(
        needs_draft_probs=True,
        _dcut_last_draft_ran_python=True,
        _dcut_last_logits_for_probs=object(),
        _last_selected_probs=object(),
        method="dflash",
        parallel_drafting=False,
    )
    runner = SimpleNamespace(
        drafter=drafter,
        _adaptive_probs_pending=False,
        _adaptive_num_reqs=0,
        _adaptive_req_ids=[],
        _adaptive_active=set(),
        _verify_adaptive_controller=controller,
        _dcut_logged_drafter_probs=True,
    )

    _load_bypass()(runner)
    _load_drafter_enable()(runner)

    assert drafter.needs_draft_probs is True


def test_prefill_execute_branch_skips_all_probability_work() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    execute_start = source.index("    def execute_model(")
    execute_end = source.index("    _orig_sample_tokens", execute_start)
    execute_source = source[execute_start:execute_end]

    prefill_start = execute_source.index(
        "        if _ctrl is not None and _has_prefill:"
    )
    decode_start = execute_source.index(
        "        if _ctrl is not None and not _has_prefill:",
        prefill_start,
    )
    prefill_branch = execute_source[prefill_start:decode_start]

    assert "_dcut_bypass_prob_capture_for_prefill(self)" in prefill_branch
    assert "_maybe_process_adaptive_probs" not in prefill_branch
    assert "_dcut_enable_drafter_probs" not in prefill_branch
    assert "_dcut_truncate" not in prefill_branch
    assert "_dcut_prepare_prob_capture" not in prefill_branch

    decode_end = execute_source.index(
        "        if not debug_stats:", decode_start
    )
    decode_branch = execute_source[decode_start:decode_end]
    assert "_dcut_enable_drafter_probs(self)" in decode_branch
    assert "_dcut_prepare_prob_capture(self, scheduler_output)" in decode_branch

    assert '"drafter_needs_draft_probs"' in execute_source
    assert '"adaptive_probs_pending_after_step"' in execute_source
    assert '"prob_capture_skipped_for_prefill"' in execute_source
    assert '"draft_ran_python"' in execute_source


def test_runner_callbacks_do_not_queue_probs_for_prefill() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    sample_start = source.index("    def sample_tokens(")
    copy_start = source.index(
        "    def _copy_draft_token_ids_to_cpu(",
        sample_start,
    )
    update_start = source.index("    def _update_states(", copy_start)

    sample_source = source[sample_start:copy_start]
    copy_source = source[copy_start:update_start]
    guard = 'getattr(self, "_dcut_skip_current_prob_capture", False)'

    assert guard in sample_source
    assert sample_source.index(guard) < sample_source.index(
        "_maybe_process_adaptive_probs"
    )
    assert guard in copy_source
    assert copy_source.index(guard) < copy_source.index("_dcut_queue_probs")

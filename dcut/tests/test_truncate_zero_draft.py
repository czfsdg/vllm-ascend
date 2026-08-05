# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

TRUNCATE_PATH = Path(__file__).resolve().parents[1] / "truncate.py"
REPO_ROOT = TRUNCATE_PATH.parents[1]


@dataclass
class _SchedulerOutput:
    scheduled_spec_decode_tokens: dict[str, list[int]]
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_new_reqs: list[object] = field(default_factory=list)


def _load_truncate(target_draft_lens: list[int]):
    tree = ast.parse(TRUNCATE_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_dcut_has_prefill", "_dcut_truncate"}
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)

    trim_records = []
    namespace = {
        "replace": replace,
        "_dcut_get_target_draft_lens": (
            lambda controller, original: target_draft_lens
        ),
        "_dcut_record_trim": lambda *args: trim_records.append(args[1:]),
    }
    exec(compile(module, str(TRUNCATE_PATH), "exec"), namespace)
    return namespace["_dcut_truncate"], trim_records


def test_mixed_prefill_decode_batch_is_not_truncated() -> None:
    truncate, trim_records = _load_truncate([1])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"decode": [10, 11, 12]},
        num_scheduled_tokens={"decode": 4, "prefill": 8},
        total_num_scheduled_tokens=12,
        scheduled_new_reqs=[SimpleNamespace(req_id="prefill")],
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
    )

    assert result is original
    assert result.scheduled_spec_decode_tokens["decode"] == [10, 11, 12]
    assert trim_records == []


def test_single_token_prefill_tail_is_not_truncated() -> None:
    truncate, trim_records = _load_truncate([1])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"decode": [10, 11, 12]},
        num_scheduled_tokens={"decode": 4, "prefill": 1},
        total_num_scheduled_tokens=5,
    )
    input_batch = SimpleNamespace(
        req_id_to_index={"decode": 0, "prefill": 1},
        num_computed_tokens_cpu=[32, 7],
        num_prompt_tokens=[32, 8],
    )

    result = truncate(
        SimpleNamespace(
            _verify_adaptive_controller=object(),
            input_batch=input_batch,
        ),
        original,
    )

    assert result is original
    assert result.scheduled_spec_decode_tokens["decode"] == [10, 11, 12]
    assert trim_records == []


def test_zero_draft_decision_stays_on_spec_path_without_adding_tokens() -> None:
    truncate, trim_records = _load_truncate([0, 2])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "zero": [10, 11, 12],
            "kept": [20, 21, 22],
        },
        num_scheduled_tokens={"zero": 4, "kept": 4, "ordinary": 1},
        total_num_scheduled_tokens=9,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
    )

    assert result is not original
    assert result.scheduled_spec_decode_tokens == {
        "zero": [],
        "kept": [20, 21],
    }
    assert result.num_scheduled_tokens == {
        "zero": 1,
        "kept": 3,
        "ordinary": 1,
    }
    assert result.total_num_scheduled_tokens == 5
    assert "ordinary" not in result.scheduled_spec_decode_tokens
    assert original.scheduled_spec_decode_tokens["zero"] == [10, 11, 12]
    assert trim_records == [(6, 4, 2)]


def test_all_zero_draft_decisions_keep_a_truthy_spec_mapping() -> None:
    truncate, _ = _load_truncate([0, 0])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"first": [1, 2], "second": [3]},
        num_scheduled_tokens={"first": 3, "second": 2},
        total_num_scheduled_tokens=5,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
    )

    assert result.scheduled_spec_decode_tokens == {
        "first": [],
        "second": [],
    }
    assert result.scheduled_spec_decode_tokens
    assert result.num_scheduled_tokens == {"first": 1, "second": 1}
    assert result.total_num_scheduled_tokens == 2


def test_gdn_uses_zero_as_spec_and_minus_one_as_non_spec() -> None:
    runner = (
        REPO_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
    ).read_text(encoding="utf-8")
    builder = (
        REPO_ROOT / "vllm_ascend" / "ops" / "gdn_attn_builder.py"
    ).read_text(encoding="utf-8")

    assert "num_decode_draft_tokens = np.full(num_reqs, -1" in runner
    assert "num_decode_draft_tokens[req_idx] = draft_len" in runner
    assert (
        "torch.ge(\n"
        "                num_decode_draft_tokens_cpu,\n"
        "                0,"
    ) in builder

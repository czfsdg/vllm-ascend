# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from dcut.truncate import _dcut_truncate


class _Controller:
    def __init__(self, draft_lens=None):
        self.draft_lens = draft_lens or {}

    def get_adaptive_draft_len(self, req_id):
        return self.draft_lens.get(req_id, 1)


class _Event:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


@dataclass
class _SchedulerOutput:
    scheduled_spec_decode_tokens: dict[str, list[int]]
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int


def _runner(
    *,
    has_gdn: bool,
    accepted_tokens: list[int],
    draft_lens=None,
    method="eagle",
    req_id_to_index=None,
):
    event = _Event()
    if req_id_to_index is None:
        req_id_to_index = {"request-0": 0}
    runner = SimpleNamespace(
        _has_gdn=has_gdn,
        num_accepted_tokens_event=event,
        input_batch=SimpleNamespace(
            req_id_to_index=req_id_to_index,
            num_accepted_tokens_cpu=accepted_tokens,
        ),
        speculative_config=SimpleNamespace(method=method),
        _verify_adaptive_controller=_Controller(draft_lens),
        _dcut_stat_full=0,
        _dcut_stat_trimmed=0,
        _dcut_stat_reqs=0,
        _dcut_stat_steps=0,
        _dcut_stat_log_every=0,
    )
    return runner, event


def test_gdn_truncation_does_not_floor_cut_to_previous_acceptance():
    runner, event = _runner(has_gdn=True, accepted_tokens=[4])
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"request-0": [11, 12, 13, 14]},
        num_scheduled_tokens={"request-0": 5},
        total_num_scheduled_tokens=5,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens["request-0"] == [11]
    assert truncated.num_scheduled_tokens["request-0"] == 2
    assert truncated.total_num_scheduled_tokens == 2
    assert event.synchronize_calls == 0


def test_non_gdn_truncation_does_not_apply_state_floor():
    runner, event = _runner(has_gdn=False, accepted_tokens=[4])
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"request-0": [11, 12, 13, 14]},
        num_scheduled_tokens={"request-0": 5},
        total_num_scheduled_tokens=5,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens["request-0"] == [11]
    assert truncated.num_scheduled_tokens["request-0"] == 2
    assert truncated.total_num_scheduled_tokens == 2
    assert event.synchronize_calls == 0


def test_dflash_truncation_keeps_per_request_lengths():
    runner, _ = _runner(
        has_gdn=False,
        accepted_tokens=[0, 0],
        draft_lens={"request-0": 1, "request-1": 3},
        method="dflash",
        req_id_to_index={"request-0": 0, "request-1": 1},
    )
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "request-0": [11, 12, 13, 14],
            "request-1": [21, 22, 23, 24],
        },
        num_scheduled_tokens={"request-0": 5, "request-1": 5},
        total_num_scheduled_tokens=10,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens == {
        "request-0": [11],
        "request-1": [21, 22, 23],
    }
    assert truncated.num_scheduled_tokens == {
        "request-0": 2,
        "request-1": 4,
    }
    assert truncated.total_num_scheduled_tokens == 6


def test_dflash_truncates_requests_with_different_available_lengths():
    runner, _ = _runner(
        has_gdn=False,
        accepted_tokens=[0, 0],
        draft_lens={"request-0": 1, "request-1": 1},
        method="dflash",
        req_id_to_index={"request-0": 0, "request-1": 1},
    )
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "request-0": [11, 12, 13, 14],
            "request-1": [21, 22],
        },
        num_scheduled_tokens={"request-0": 5, "request-1": 3},
        total_num_scheduled_tokens=8,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens == {
        "request-0": [11],
        "request-1": [21],
    }
    assert truncated.num_scheduled_tokens == {
        "request-0": 2,
        "request-1": 2,
    }
    assert truncated.total_num_scheduled_tokens == 4


def test_zero_length_cut_removes_spec_decode_entry():
    runner, _ = _runner(
        has_gdn=False,
        accepted_tokens=[0],
        draft_lens={"request-0": 0},
    )
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"request-0": [11, 12, 13]},
        num_scheduled_tokens={"request-0": 4},
        total_num_scheduled_tokens=4,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens == {}
    assert truncated.num_scheduled_tokens == {"request-0": 1}
    assert truncated.total_num_scheduled_tokens == 1


def test_dflash_gdn_cuts_remain_per_request_without_state_floors():
    runner, event = _runner(
        has_gdn=True,
        accepted_tokens=[4, 2],
        draft_lens={"request-0": 1, "request-1": 1},
        method="dflash",
        req_id_to_index={"request-0": 0, "request-1": 1},
    )
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "request-0": [11, 12, 13, 14],
            "request-1": [21, 22, 23, 24],
        },
        num_scheduled_tokens={"request-0": 5, "request-1": 5},
        total_num_scheduled_tokens=10,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens == {
        "request-0": [11],
        "request-1": [21],
    }
    assert truncated.num_scheduled_tokens == {
        "request-0": 2,
        "request-1": 2,
    }
    assert truncated.total_num_scheduled_tokens == 4
    assert event.synchronize_calls == 0

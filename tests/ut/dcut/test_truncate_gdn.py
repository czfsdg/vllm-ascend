# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from dcut.truncate import _dcut_truncate


class _Controller:
    def get_adaptive_draft_len(self, req_id):
        del req_id
        return 1


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


def _runner(*, has_gdn: bool, accepted_tokens: list[int]):
    event = _Event()
    runner = SimpleNamespace(
        _has_gdn=has_gdn,
        num_accepted_tokens_event=event,
        input_batch=SimpleNamespace(
            req_id_to_index={"request-0": 0},
            num_accepted_tokens_cpu=accepted_tokens,
        ),
        _verify_adaptive_controller=_Controller(),
        _dcut_stat_full=0,
        _dcut_stat_trimmed=0,
        _dcut_stat_reqs=0,
        _dcut_stat_steps=0,
        _dcut_stat_log_every=0,
    )
    return runner, event


def test_gdn_truncation_keeps_previous_accepted_state_slot():
    runner, event = _runner(has_gdn=True, accepted_tokens=[4])
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"request-0": [11, 12, 13, 14]},
        num_scheduled_tokens={"request-0": 5},
        total_num_scheduled_tokens=5,
    )

    truncated = _dcut_truncate(runner, output)

    assert truncated.scheduled_spec_decode_tokens["request-0"] == [11, 12, 13]
    assert truncated.num_scheduled_tokens["request-0"] == 4
    assert truncated.total_num_scheduled_tokens == 4
    assert event.synchronize_calls == 1


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

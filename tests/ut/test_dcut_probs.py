# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from dcut.probs import _maybe_process_adaptive_probs


class _PendingEvent:
    def __init__(self):
        self.synchronized = False

    def query(self):
        return False

    def synchronize(self):
        self.synchronized = True


def test_maybe_process_adaptive_probs_does_not_block_on_pending_event():
    event = _PendingEvent()
    runner = SimpleNamespace(
        _adaptive_probs_pending=True,
        _adaptive_probs_event=event,
        _adaptive_num_reqs=1,
        _adaptive_active={"req-1"},
        _verify_adaptive_controller=MagicMock(),
        _adaptive_probs_pinned=torch.ones(1, 1),
        _adaptive_req_ids=["req-1"],
    )

    _maybe_process_adaptive_probs(runner)

    assert runner._adaptive_probs_pending is True
    assert event.synchronized is False
    runner._verify_adaptive_controller.process_draft_output.assert_not_called()


def test_maybe_process_adaptive_probs_processes_ready_event():
    event = MagicMock()
    event.query.return_value = True
    controller = MagicMock()
    probs = torch.ones(1, 1)
    runner = SimpleNamespace(
        _adaptive_probs_pending=True,
        _adaptive_probs_event=event,
        _adaptive_num_reqs=1,
        _adaptive_active={"req-1"},
        _verify_adaptive_controller=controller,
        _adaptive_probs_pinned=probs,
        _adaptive_req_ids=["req-1"],
    )

    _maybe_process_adaptive_probs(runner)

    assert runner._adaptive_probs_pending is False
    event.synchronize.assert_not_called()
    controller.process_draft_output.assert_called_once_with(
        selected_probs=probs[:1],
        req_ids=["req-1"],
        active_draft_req_ids={"req-1"},
        batch_size=1,
    )

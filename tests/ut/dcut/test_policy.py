# SPDX-License-Identifier: Apache-2.0

from dcut.cost_table import DcutCostTable
from dcut.policy import DcutPolicy


def test_cost_table_builds_one_entry_per_verify_len():
    table = DcutCostTable(max_verify_len=4)

    assert [entry.verify_len for entry in table.entries] == [1, 2, 3, 4]
    assert "k=4" in table.summary()


def test_policy_never_exceeds_requested_len():
    policy = DcutPolicy(DcutCostTable(max_verify_len=8))

    decision = policy.decide(requested_len=4, batch_size=32, acceptance_rate=0.5)

    assert 1 <= decision.selected_len <= 4
    assert decision.requested_len == 4
    assert decision.batch_size == 32



def test_policy_cuts_under_high_concurrency():
    policy = DcutPolicy(DcutCostTable(max_verify_len=4))

    decision = policy.decide(requested_len=4, batch_size=32, acceptance_rate=0.75)

    assert decision.selected_len < decision.requested_len
    assert decision.reason == "high_concurrency_cost_cut"

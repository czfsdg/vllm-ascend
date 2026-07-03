# SPDX-License-Identifier: Apache-2.0

from dcut.config import _parse_config
from dcut.cost_table import DcutCostTable
from dcut.policy import DcutPolicy


def _profile_all(table: DcutCostTable) -> DcutCostTable:
    for verify_len in range(1, table.max_verify_len + 1):
        entry = table.get(verify_len)
        for _ in range(table.min_profile_samples):
            table.update_profile(
                verify_len,
                target_cost=entry.target_cost,
                draft_cost=entry.draft_cost,
            )
    return table


def test_cost_table_builds_one_entry_per_verify_len():
    table = DcutCostTable(max_verify_len=4)

    assert [entry.verify_len for entry in table.entries] == [1, 2, 3, 4]
    assert [entry.verify_len for entry in table.warmup()] == [1, 2, 3, 4]
    profiled = table.update_profile(verify_len=3, target_cost=10.0, draft_cost=2.0)
    assert profiled.total_cost == 12.0
    assert table.get(3).target_cost == 10.0
    assert table.profile_counts == {3: 1}
    assert table.profiled_lens == set()
    table.update_profile(verify_len=3, target_cost=8.0, draft_cost=1.0)
    assert table.get(3).total_cost == 9.0
    assert table.profiled_lens == {3}
    assert "k=4" in table.summary()


def test_policy_never_exceeds_requested_len():
    policy = DcutPolicy(DcutCostTable(max_verify_len=8))

    decision = policy.decide(requested_len=4, batch_size=32, acceptance_rate=0.5)

    assert 1 <= decision.selected_len <= 4
    assert decision.requested_len == 4
    assert decision.batch_size == 32


def test_policy_cuts_under_high_concurrency():
    policy = DcutPolicy(_profile_all(DcutCostTable(max_verify_len=4)))

    decision = policy.decide(requested_len=4, batch_size=32, acceptance_rate=0.75)

    assert decision.selected_len < decision.requested_len
    assert decision.reason == "high_concurrency_cost_cut"


def test_config_overrides_policy_and_cost_table_values():
    config = _parse_config(
        {
            "cost_table": {
                "target_base_cost": 2.0,
                "target_token_cost": 0.5,
                "draft_token_cost": 0.25,
            },
            "policy": {
                "default_acceptance_rate": 0.6,
                "high_concurrency_batch": 8,
            },
            "safety": {
                "accuracy_safe_mode": False,
                "target_only_methods": ["dflash", "draft_model"],
            },
        }
    )

    table = DcutCostTable(
        max_verify_len=2,
        target_base_cost=config.target_base_cost,
        target_token_cost=config.target_token_cost,
        draft_token_cost=config.draft_token_cost,
    )
    policy = DcutPolicy(
        table,
        default_acceptance_rate=config.default_acceptance_rate,
        high_concurrency_batch=config.high_concurrency_batch,
    )

    assert table.get(1).target_cost == 2.5
    assert policy.default_acceptance_rate == 0.6
    assert policy.high_concurrency_batch == 8
    assert config.accuracy_safe_mode is False
    assert config.target_only_methods == ("dflash", "draft_model")


def test_policy_uses_acceptance_to_change_selected_len():
    policy = DcutPolicy(_profile_all(DcutCostTable(max_verify_len=7)))

    low_acceptance = policy.decide(requested_len=7, batch_size=16, acceptance_rate=0.3)
    high_acceptance = policy.decide(requested_len=7, batch_size=16, acceptance_rate=0.6)

    assert low_acceptance.selected_len < high_acceptance.selected_len


def test_policy_profiles_each_len_before_scoring():
    table = DcutCostTable(max_verify_len=3)
    policy = DcutPolicy(table)

    first = policy.decide(requested_len=3, batch_size=1, acceptance_rate=0.5)
    assert first.selected_len == 1
    assert first.reason == "npu_profile_warmup"

    table.update_profile(1, target_cost=100.0, draft_cost=10.0)
    second = policy.decide(requested_len=3, batch_size=1, acceptance_rate=0.5)
    assert second.selected_len == 1
    assert second.reason == "npu_profile_warmup"

    table.update_profile(1, target_cost=10.0, draft_cost=1.0)
    third = policy.decide(requested_len=3, batch_size=1, acceptance_rate=0.5)
    assert third.selected_len == 2
    assert table.get(1).total_cost == 11.0

# SPDX-License-Identifier: Apache-2.0
"""D-Cut policy for selecting a verifier cut length."""

from __future__ import annotations

from dataclasses import dataclass

from dcut.cost_table import DcutCostTable

MIN_HIGH_CONCURRENCY_BATCH = 16
DEFAULT_ACCEPTANCE_RATE = 0.75


@dataclass(frozen=True)
class CutDecision:
    requested_len: int
    selected_len: int
    batch_size: int
    acceptance_rate: float
    score: float
    reason: str


class DcutPolicy:
    """Selects verifier step length from cost and coarse runtime signals."""

    def __init__(self, cost_table: DcutCostTable) -> None:
        self.cost_table = cost_table

    def decide(
        self,
        requested_len: int,
        batch_size: int,
        acceptance_rate: float | None = None,
    ) -> CutDecision:
        bounded_requested_len = min(max(requested_len, 1), self.cost_table.max_verify_len)
        bounded_batch_size = max(batch_size, 1)
        observed_acceptance = DEFAULT_ACCEPTANCE_RATE if acceptance_rate is None else acceptance_rate
        observed_acceptance = min(max(observed_acceptance, 0.0), 1.0)

        best_len = 1
        best_score = float("inf")
        for verify_len in range(1, bounded_requested_len + 1):
            cost = self.cost_table.get(verify_len)
            expected_accepts = max(verify_len * observed_acceptance, 1e-6)
            concurrency = max(bounded_batch_size - 1, 0) / MIN_HIGH_CONCURRENCY_BATCH
            length_penalty = 1.0 + concurrency * max(verify_len - 1, 0) / bounded_requested_len
            score = cost.total_cost * length_penalty / expected_accepts
            if score < best_score:
                best_score = score
                best_len = verify_len

        if best_len == bounded_requested_len:
            reason = "keep_full_spec_len"
        elif bounded_batch_size >= MIN_HIGH_CONCURRENCY_BATCH:
            reason = "high_concurrency_cost_cut"
        else:
            reason = "low_acceptance_cost_cut"

        return CutDecision(
            requested_len=bounded_requested_len,
            selected_len=best_len,
            batch_size=bounded_batch_size,
            acceptance_rate=observed_acceptance,
            score=best_score,
            reason=reason,
        )

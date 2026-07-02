# SPDX-License-Identifier: Apache-2.0
"""Cost table utilities for D-Cut adaptive verification planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostEntry:
    """Estimated per-step cost for one verifier length."""

    verify_len: int
    target_cost: float
    draft_cost: float
    total_cost: float


class DcutCostTable:
    """Small deterministic cost table used by the D-Cut policy.

    The first implementation intentionally uses a conservative analytic model
    instead of device profiling so that it can be enabled from an editable
    plugin without mutating the packaged vLLM Ascend image. Later work can
    replace ``target_cost`` and ``draft_cost`` with calibrated NPU timings while
    keeping this API stable.
    """

    def __init__(
        self,
        max_verify_len: int,
        target_base_cost: float = 1.0,
        target_token_cost: float = 0.18,
        draft_token_cost: float = 0.08,
    ) -> None:
        if max_verify_len < 1:
            raise ValueError("max_verify_len must be >= 1")
        self.max_verify_len = max_verify_len
        self.target_base_cost = target_base_cost
        self.target_token_cost = target_token_cost
        self.draft_token_cost = draft_token_cost
        self._entries = tuple(self._build_entry(i) for i in range(1, max_verify_len + 1))

    @property
    def entries(self) -> tuple[CostEntry, ...]:
        return self._entries

    def get(self, verify_len: int) -> CostEntry:
        bounded_len = min(max(verify_len, 1), self.max_verify_len)
        return self._entries[bounded_len - 1]

    def summary(self) -> str:
        return ", ".join(
            f"k={entry.verify_len}:target={entry.target_cost:.3f},"
            f"draft={entry.draft_cost:.3f},total={entry.total_cost:.3f}"
            for entry in self._entries
        )

    def _build_entry(self, verify_len: int) -> CostEntry:
        target_cost = self.target_base_cost + self.target_token_cost * verify_len
        draft_cost = self.draft_token_cost * verify_len
        return CostEntry(
            verify_len=verify_len,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )

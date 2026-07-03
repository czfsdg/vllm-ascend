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
    """Cost table used by the D-Cut policy.

    Analytic entries are only bootstrap values. Runtime NPU profiling updates
    entries with measured timings before the policy treats a verify length as
    calibrated.
    """

    def __init__(
        self,
        max_verify_len: int,
        target_base_cost: float = 1.0,
        target_token_cost: float = 0.18,
        draft_token_cost: float = 0.08,
        min_profile_samples: int = 2,
    ) -> None:
        if max_verify_len < 1:
            raise ValueError("max_verify_len must be >= 1")
        self.max_verify_len = max_verify_len
        self.target_base_cost = target_base_cost
        self.target_token_cost = target_token_cost
        self.draft_token_cost = draft_token_cost
        self.min_profile_samples = max(min_profile_samples, 1)
        self.profile_counts: dict[int, int] = {}
        self.profiled_lens: set[int] = set()
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

    def warmup(self) -> tuple[CostEntry, ...]:
        """Touch all bootstrap entries before the first D-Cut decision."""
        return tuple(self.get(entry.verify_len) for entry in self._entries)

    def needs_profile(self, verify_len: int) -> bool:
        bounded_len = min(max(verify_len, 1), self.max_verify_len)
        return self.profile_counts.get(bounded_len, 0) < self.min_profile_samples

    def update_profile(
        self,
        verify_len: int,
        target_cost: float,
        draft_cost: float,
    ) -> CostEntry:
        """Replace one entry with measured runtime profiling costs."""
        bounded_len = min(max(verify_len, 1), self.max_verify_len)
        target_cost = max(target_cost, 0.0)
        draft_cost = max(draft_cost, 0.0)
        measured_entry = CostEntry(
            verify_len=bounded_len,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )
        current_entry = self.get(bounded_len)
        profile_count = self.profile_counts.get(bounded_len, 0) + 1
        self.profile_counts[bounded_len] = profile_count

        if profile_count == 1 or measured_entry.total_cost <= current_entry.total_cost:
            entry = measured_entry
        else:
            entry = current_entry

        entries = list(self._entries)
        entries[bounded_len - 1] = entry
        self._entries = tuple(entries)
        if profile_count >= self.min_profile_samples:
            self.profiled_lens.add(bounded_len)
        return entry

    def _build_entry(self, verify_len: int) -> CostEntry:
        target_cost = self.target_base_cost + self.target_token_cost * verify_len
        draft_cost = self.draft_token_cost * verify_len
        return CostEntry(
            verify_len=verify_len,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )

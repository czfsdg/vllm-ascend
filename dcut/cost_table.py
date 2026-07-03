# SPDX-License-Identifier: Apache-2.0
"""Cost table utilities for D-Cut adaptive verification planning."""

from __future__ import annotations

from dataclasses import dataclass

PROFILE_STATE_BOOTSTRAP = "bootstrap"
PROFILE_STATE_PROFILING = "profiling"
PROFILE_STATE_READY = "ready"
DEFAULT_PROFILE_SAMPLES = 5
DEFAULT_TRIM_LARGEST_SAMPLES = 2


@dataclass(frozen=True)
class CostEntry:
    """Estimated per-step cost for one verifier length."""

    verify_len: int
    target_cost: float
    draft_cost: float
    total_cost: float


class DcutCostTable:
    """Cost table used by the D-Cut policy.

    Analytic entries are only bootstrap values. Runtime NPU profiling collects
    multiple samples per verifier length, stores a robust trimmed mean, and only
    then marks that length ready for normal D-Cut planning.
    """

    def __init__(
        self,
        max_verify_len: int,
        target_base_cost: float = 1.0,
        target_token_cost: float = 0.18,
        draft_token_cost: float = 0.08,
        min_profile_samples: int = DEFAULT_PROFILE_SAMPLES,
        trim_largest_samples: int = DEFAULT_TRIM_LARGEST_SAMPLES,
    ) -> None:
        if max_verify_len < 1:
            raise ValueError("max_verify_len must be >= 1")
        self.max_verify_len = max_verify_len
        self.target_base_cost = target_base_cost
        self.target_token_cost = target_token_cost
        self.draft_token_cost = draft_token_cost
        self.min_profile_samples = max(min_profile_samples, 1)
        self.trim_largest_samples = max(trim_largest_samples, 0)
        self.profile_samples: dict[int, list[CostEntry]] = {}
        self.profile_counts: dict[int, int] = {}
        self.profile_states: dict[int, str] = {
            verify_len: PROFILE_STATE_BOOTSTRAP for verify_len in range(1, max_verify_len + 1)
        }
        self.profiled_lens: set[int] = set()
        self.last_measured_entry: CostEntry | None = None
        self.last_profile_updated = False
        self._entries = tuple(self._build_entry(i) for i in range(1, max_verify_len + 1))

    @property
    def entries(self) -> tuple[CostEntry, ...]:
        return self._entries

    def get(self, verify_len: int) -> CostEntry:
        bounded_len = min(max(verify_len, 1), self.max_verify_len)
        return self._entries[bounded_len - 1]

    def profile_state(self, verify_len: int) -> str:
        bounded_len = min(max(verify_len, 1), self.max_verify_len)
        return self.profile_states[bounded_len]

    def summary(self) -> str:
        return ", ".join(
            f"k={entry.verify_len}:state={self.profile_state(entry.verify_len)},"
            f"samples={self.profile_counts.get(entry.verify_len, 0)},"
            f"target={entry.target_cost:.3f},draft={entry.draft_cost:.3f},"
            f"total={entry.total_cost:.3f}"
            for entry in self._entries
        )

    def warmup(self) -> tuple[CostEntry, ...]:
        """Touch all bootstrap entries before the first D-Cut decision."""
        return tuple(self.get(entry.verify_len) for entry in self._entries)

    def needs_profile(self, verify_len: int) -> bool:
        return self.profile_state(verify_len) != PROFILE_STATE_READY

    def update_profile(
        self,
        verify_len: int,
        target_cost: float,
        draft_cost: float,
    ) -> CostEntry:
        """Collect one measured sample and publish a trimmed mean when ready."""
        bounded_len = min(max(verify_len, 1), self.max_verify_len)
        target_cost = max(target_cost, 0.0)
        draft_cost = max(draft_cost, 0.0)
        measured_entry = CostEntry(
            verify_len=bounded_len,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )
        samples = self.profile_samples.setdefault(bounded_len, [])
        samples.append(measured_entry)
        profile_count = len(samples)
        self.profile_counts[bounded_len] = profile_count
        self.last_measured_entry = measured_entry

        if profile_count < self.min_profile_samples:
            self.profile_states[bounded_len] = PROFILE_STATE_PROFILING
            self.last_profile_updated = False
            return self.get(bounded_len)

        entry = self._trimmed_mean_entry(bounded_len, samples)
        entries = list(self._entries)
        entries[bounded_len - 1] = entry
        self._entries = tuple(entries)
        self.profile_states[bounded_len] = PROFILE_STATE_READY
        self.profiled_lens.add(bounded_len)
        self.last_profile_updated = True
        return entry

    def _trimmed_mean_entry(self, verify_len: int, samples: list[CostEntry]) -> CostEntry:
        sorted_samples = sorted(samples, key=lambda entry: entry.total_cost)
        retained_count = max(len(sorted_samples) - self.trim_largest_samples, 1)
        retained_samples = sorted_samples[:retained_count]
        target_cost = sum(entry.target_cost for entry in retained_samples) / retained_count
        draft_cost = sum(entry.draft_cost for entry in retained_samples) / retained_count
        return CostEntry(
            verify_len=verify_len,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
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

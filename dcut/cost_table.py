# SPDX-License-Identifier: Apache-2.0
"""Cost table utilities for D-Cut adaptive verification planning."""

from __future__ import annotations

import random
from dataclasses import dataclass

PROFILE_STATE_BOOTSTRAP = "bootstrap"
PROFILE_STATE_PROFILING = "profiling"
PROFILE_STATE_READY = "ready"
DEFAULT_PROFILE_SAMPLES = 5
DEFAULT_TRIM_LARGEST_SAMPLES = 2
DEFAULT_Q_BUCKET_SIZE = 8
BOOTSTRAP_SIMULATION_SEED = 0
BOOTSTRAP_SIMULATION_JITTER = 0.08


@dataclass(frozen=True)
class CostEntry:
    """Estimated cost for one total speculative-token Q bucket."""

    q_tokens: int
    target_cost: float
    draft_cost: float
    total_cost: float


class DcutCostTable:
    """Cost table used by the D-Cut policy.

    The table is keyed only by Q bucket, where Q is total speculative-token work
    for a candidate cut: ``batch_size * candidate_verify_len`` rounded up to the
    configured bucket size. Runtime NPU profiling collects multiple samples per
    Q bucket, stores a robust trimmed mean, and only then marks that bucket ready
    for normal D-Cut planning.
    """

    def __init__(
        self,
        max_verify_len: int,
        target_base_cost: float = 1.0,
        target_token_cost: float = 0.18,
        draft_token_cost: float = 0.08,
        min_profile_samples: int = DEFAULT_PROFILE_SAMPLES,
        trim_largest_samples: int = DEFAULT_TRIM_LARGEST_SAMPLES,
        max_q_tokens: int | None = None,
        q_bucket_size: int = DEFAULT_Q_BUCKET_SIZE,
    ) -> None:
        if max_verify_len < 1:
            raise ValueError("max_verify_len must be >= 1")
        self.max_verify_len = max_verify_len
        self.q_bucket_size = max(q_bucket_size, 1)
        raw_max_q_tokens = max(int(max_q_tokens or max_verify_len), 1)
        self.max_q_tokens = ((raw_max_q_tokens + self.q_bucket_size - 1) // self.q_bucket_size) * self.q_bucket_size
        self.target_base_cost = target_base_cost
        self.target_token_cost = target_token_cost
        self.draft_token_cost = draft_token_cost
        self.min_profile_samples = max(min_profile_samples, 1)
        self.trim_largest_samples = max(trim_largest_samples, 0)
        self.q_buckets = tuple(range(self.q_bucket_size, self.max_q_tokens + 1, self.q_bucket_size))
        self.profile_samples: dict[int, list[CostEntry]] = {}
        self.profile_counts: dict[int, int] = {}
        self.profile_states: dict[int, str] = {q_bucket: PROFILE_STATE_BOOTSTRAP for q_bucket in self.q_buckets}
        self.profiled_lens: set[int] = set()
        self.last_measured_entry: CostEntry | None = None
        self.last_profile_updated = False
        self._entries = {q_bucket: self._build_entry(q_bucket) for q_bucket in self.q_buckets}

    @property
    def entries(self) -> tuple[CostEntry, ...]:
        return tuple(self._entries[q_bucket] for q_bucket in self.q_buckets)

    def get(self, q_tokens: int | None = None) -> CostEntry:
        return self._entries[self._bucket_q(q_tokens)]

    def profile_state(self, q_tokens: int | None = None) -> str:
        return self.profile_states[self._bucket_q(q_tokens)]

    def summary(self, q_tokens: int | None = None) -> str:
        if q_tokens is None:
            return (
                f"q_bucket_size={self.q_bucket_size},max_q_tokens={self.max_q_tokens},q_buckets={len(self.q_buckets)}"
            )
        q_bucket = self._bucket_q(q_tokens)
        entry = self.get(q_bucket)
        return (
            f"q={q_bucket}:state={self.profile_state(q_bucket)},"
            f"samples={self.profile_counts.get(q_bucket, 0)},"
            f"target={entry.target_cost:.3f},draft={entry.draft_cost:.3f},"
            f"total={entry.total_cost:.3f}"
        )

    def warmup(self) -> tuple[CostEntry, ...]:
        """Touch all bootstrap entries before the first D-Cut decision."""
        return self.entries

    def simulate_bootstrap_profiles(self) -> None:
        """Seed every Q bucket with deterministic synthetic draft/target samples."""
        rng = random.Random(BOOTSTRAP_SIMULATION_SEED)
        for q_bucket in self.q_buckets:
            base_entry = self._build_entry(q_bucket)
            for _ in range(self.min_profile_samples):
                jitter = 1.0 + rng.uniform(-BOOTSTRAP_SIMULATION_JITTER, BOOTSTRAP_SIMULATION_JITTER)
                self.update_profile(
                    q_tokens=q_bucket,
                    target_cost=base_entry.target_cost * jitter,
                    draft_cost=base_entry.draft_cost * jitter,
                )

    def needs_profile(self, q_tokens: int | None = None) -> bool:
        return self.profile_state(q_tokens) != PROFILE_STATE_READY

    def update_profile(
        self,
        target_cost: float,
        draft_cost: float,
        q_tokens: int | None = None,
    ) -> CostEntry:
        """Collect one measured sample and publish a trimmed mean when ready."""
        q_bucket = self._bucket_q(q_tokens)
        target_cost = max(target_cost, 0.0)
        draft_cost = max(draft_cost, 0.0)
        measured_entry = CostEntry(
            q_tokens=q_bucket,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )
        samples = self.profile_samples.setdefault(q_bucket, [])
        samples.append(measured_entry)
        profile_count = len(samples)
        self.profile_counts[q_bucket] = profile_count
        self.last_measured_entry = measured_entry

        if profile_count < self.min_profile_samples:
            self.profile_states[q_bucket] = PROFILE_STATE_PROFILING
            self.last_profile_updated = False
            return self.get(q_bucket)

        entry = self._trimmed_mean_entry(q_bucket, samples)
        self._entries[q_bucket] = entry
        self.profile_states[q_bucket] = PROFILE_STATE_READY
        self.profiled_lens.add(q_bucket)
        self.last_profile_updated = True
        return entry

    def q_bucket_for(self, q_tokens: int | None) -> int:
        return self._bucket_q(q_tokens)

    def _trimmed_mean_entry(self, q_bucket: int, samples: list[CostEntry]) -> CostEntry:
        sorted_samples = sorted(samples, key=lambda entry: entry.total_cost)
        retained_count = max(len(sorted_samples) - self.trim_largest_samples, 1)
        retained_samples = sorted_samples[:retained_count]
        target_cost = sum(entry.target_cost for entry in retained_samples) / retained_count
        draft_cost = sum(entry.draft_cost for entry in retained_samples) / retained_count
        return CostEntry(
            q_tokens=q_bucket,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )

    def _build_entry(self, q_bucket: int) -> CostEntry:
        target_cost = self.target_base_cost + self.target_token_cost * q_bucket
        draft_cost = self.draft_token_cost * q_bucket
        return CostEntry(
            q_tokens=q_bucket,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )

    def _bucket_q(self, q_tokens: int | None) -> int:
        q_value = max(int(q_tokens or self.q_bucket_size), 1)
        q_bucket = ((q_value + self.q_bucket_size - 1) // self.q_bucket_size) * self.q_bucket_size
        return min(max(q_bucket, self.q_bucket_size), self.max_q_tokens)

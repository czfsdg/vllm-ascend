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
    """Estimated or measured cost for one ``(batch_size, Q)`` bucket."""

    q_tokens: int
    target_cost: float
    draft_cost: float
    total_cost: float
    batch_size: int = 0


class DcutCostTable:
    """Cost table used by the D-Cut policy.

    The table is keyed by ``(batch_size_bucket, q_bucket)``. ``q_bucket`` is
    total speculative-token work for a candidate cut:
    ``batch_size * candidate_verify_len`` rounded up to the configured bucket
    size.  Each bucket stores multiple startup profiling samples and publishes a
    trimmed mean after enough samples are collected.  During serving, the table
    is read-only.
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
        max_batch_size: int | None = None,
    ) -> None:
        if max_verify_len < 1:
            raise ValueError("max_verify_len must be >= 1")
        self.max_verify_len = max_verify_len
        self.q_bucket_size = max(q_bucket_size, 1)
        raw_max_q_tokens = max(int(max_q_tokens or max_verify_len), 1)
        self.max_q_tokens = ((raw_max_q_tokens + self.q_bucket_size - 1) // self.q_bucket_size) * self.q_bucket_size
        self.max_batch_size = max(int(max_batch_size or 0), 0)
        self.batch_size_buckets = self._build_batch_size_buckets(self.max_batch_size)
        self.target_base_cost = target_base_cost
        self.target_token_cost = target_token_cost
        self.draft_token_cost = draft_token_cost
        self.min_profile_samples = max(min_profile_samples, 1)
        self.trim_largest_samples = max(trim_largest_samples, 0)
        self.q_buckets = tuple(range(self.q_bucket_size, self.max_q_tokens + 1, self.q_bucket_size))
        self.profile_samples: dict[tuple[int, int], list[CostEntry]] = {}
        self.profile_counts: dict[tuple[int, int], int] = {}
        self.profile_states: dict[tuple[int, int], str] = {
            (batch_size, q_bucket): PROFILE_STATE_BOOTSTRAP
            for batch_size in self.batch_size_buckets
            for q_bucket in self.q_buckets
        }
        self.profiled_lens: set[tuple[int, int]] = set()
        self.last_measured_entry: CostEntry | None = None
        self.last_profile_updated = False
        self._entries = {
            (batch_size, q_bucket): self._build_entry(q_bucket, batch_size)
            for batch_size in self.batch_size_buckets
            for q_bucket in self.q_buckets
        }

    @property
    def entries(self) -> tuple[CostEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def get(self, q_tokens: int | None = None, batch_size: int | None = None) -> CostEntry:
        return self._entries[self._bucket_key(q_tokens, batch_size)]

    def profile_state(self, q_tokens: int | None = None, batch_size: int | None = None) -> str:
        return self.profile_states[self._bucket_key(q_tokens, batch_size)]

    def summary(self, q_tokens: int | None = None, batch_size: int | None = None) -> str:
        if q_tokens is None:
            return (
                f"batch_size_buckets={len(self.batch_size_buckets)},"
                f"q_bucket_size={self.q_bucket_size},max_q_tokens={self.max_q_tokens},"
                f"q_buckets={len(self.q_buckets)}"
            )
        batch_bucket, q_bucket = self._bucket_key(q_tokens, batch_size)
        entry = self.get(q_bucket, batch_bucket)
        return (
            f"bs={batch_bucket},q={q_bucket}:state={self.profile_state(q_bucket, batch_bucket)},"
            f"samples={self.profile_counts.get((batch_bucket, q_bucket), 0)},"
            f"target={entry.target_cost:.3f},draft={entry.draft_cost:.3f},"
            f"total={entry.total_cost:.3f}"
        )

    def warmup(self) -> tuple[CostEntry, ...]:
        """Touch all bootstrap entries before the first D-Cut decision."""
        return self.entries

    def simulate_bootstrap_profiles(self) -> None:
        """Seed every ``(batch_size, Q)`` bucket with deterministic estimates."""
        rng = random.Random(BOOTSTRAP_SIMULATION_SEED)
        for batch_size, q_bucket in self._entries:
            base_entry = self._build_entry(q_bucket, batch_size)
            for _ in range(self.min_profile_samples):
                jitter = 1.0 + rng.uniform(-BOOTSTRAP_SIMULATION_JITTER, BOOTSTRAP_SIMULATION_JITTER)
                self.update_profile(
                    q_tokens=q_bucket,
                    batch_size=batch_size,
                    target_cost=base_entry.target_cost * jitter,
                    draft_cost=base_entry.draft_cost * jitter,
                )

    def needs_profile(self, q_tokens: int | None = None, batch_size: int | None = None) -> bool:
        return self.profile_state(q_tokens, batch_size) != PROFILE_STATE_READY

    def update_profile(
        self,
        target_cost: float,
        draft_cost: float,
        q_tokens: int | None = None,
        batch_size: int | None = None,
    ) -> CostEntry:
        """Collect one startup sample and publish a trimmed mean when ready."""
        batch_bucket, q_bucket = self._bucket_key(q_tokens, batch_size)
        target_cost = max(target_cost, 0.0)
        draft_cost = max(draft_cost, 0.0)
        measured_entry = CostEntry(
            batch_size=batch_bucket,
            q_tokens=q_bucket,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )
        key = (batch_bucket, q_bucket)
        samples = self.profile_samples.setdefault(key, [])
        samples.append(measured_entry)
        profile_count = len(samples)
        self.profile_counts[key] = profile_count
        self.last_measured_entry = measured_entry

        if profile_count < self.min_profile_samples:
            self.profile_states[key] = PROFILE_STATE_PROFILING
            self.last_profile_updated = False
            return self.get(q_bucket, batch_bucket)

        entry = self._trimmed_mean_entry(batch_bucket, q_bucket, samples)
        self._entries[key] = entry
        self.profile_states[key] = PROFILE_STATE_READY
        self.profiled_lens.add(key)
        self.last_profile_updated = True
        return entry

    def q_bucket_for(self, q_tokens: int | None) -> int:
        return self._bucket_q(q_tokens)

    def batch_bucket_for(self, batch_size: int | None) -> int:
        return self._bucket_batch(batch_size)

    def _trimmed_mean_entry(self, batch_bucket: int, q_bucket: int, samples: list[CostEntry]) -> CostEntry:
        sorted_samples = sorted(samples, key=lambda entry: entry.total_cost)
        retained_count = max(len(sorted_samples) - self.trim_largest_samples, 1)
        retained_samples = sorted_samples[:retained_count]
        target_cost = sum(entry.target_cost for entry in retained_samples) / retained_count
        draft_cost = sum(entry.draft_cost for entry in retained_samples) / retained_count
        return CostEntry(
            batch_size=batch_bucket,
            q_tokens=q_bucket,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )

    def _build_entry(self, q_bucket: int, batch_size: int = 0) -> CostEntry:
        batch_factor = max(batch_size, 1)
        target_cost = self.target_base_cost + self.target_token_cost * q_bucket
        draft_cost = self.draft_token_cost * q_bucket / batch_factor
        return CostEntry(
            batch_size=batch_size,
            q_tokens=q_bucket,
            target_cost=target_cost,
            draft_cost=draft_cost,
            total_cost=target_cost + draft_cost,
        )

    def _bucket_key(self, q_tokens: int | None, batch_size: int | None) -> tuple[int, int]:
        return self._bucket_batch(batch_size), self._bucket_q(q_tokens)

    def _bucket_batch(self, batch_size: int | None) -> int:
        if not self.batch_size_buckets:
            return 0
        if batch_size is None:
            return self.batch_size_buckets[-1]
        batch_value = max(int(batch_size), 1)
        for batch_bucket in self.batch_size_buckets:
            if batch_bucket >= batch_value:
                return batch_bucket
        return self.batch_size_buckets[-1]

    def _bucket_q(self, q_tokens: int | None) -> int:
        q_value = max(int(q_tokens or self.q_bucket_size), 1)
        q_bucket = ((q_value + self.q_bucket_size - 1) // self.q_bucket_size) * self.q_bucket_size
        return min(max(q_bucket, self.q_bucket_size), self.max_q_tokens)

    @staticmethod
    def _build_batch_size_buckets(max_batch_size: int) -> tuple[int, ...]:
        if max_batch_size <= 0:
            return (0,)
        buckets = [1]
        candidate = 2
        while candidate < max_batch_size:
            buckets.append(candidate)
            candidate *= 2
        if buckets[-1] != max_batch_size:
            buckets.append(max_batch_size)
        return tuple(dict.fromkeys(buckets))

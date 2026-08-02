# SPDX-License-Identifier: Apache-2.0
"""Runtime full-step cost calibration for adaptive D-Cut decisions."""

from __future__ import annotations

from dataclasses import dataclass

ALL_RUNTIME_MODES = "all"


@dataclass
class RuntimeCostEstimate:
    """EWMA estimate measured from complete real serving iterations."""

    samples: int = 0
    full_step_s: float = 0.0
    profiled_step_s: float = 0.0
    overhead_s: float = 0.0


class RuntimeCostCalibrator:
    """Learn real cost per ``(batch bucket, query bucket, runtime mode)``.

    Startup profiling remains a fallback for query buckets not observed yet.
    Once a real bucket is sampled, decisions use its complete synchronized
    iteration cost directly instead of adding a batch-wide residual to every
    query length. This keeps short-query profile errors from contaminating
    longer query buckets.
    """

    def __init__(self, *, alpha: float, samples_per_bucket: int) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if samples_per_bucket < 1:
            raise ValueError("samples_per_bucket must be >= 1.")
        self.alpha = alpha
        self.samples_per_bucket = samples_per_bucket
        self._estimates: dict[
            tuple[int, int, str], RuntimeCostEstimate
        ] = {}

    def needs_sample(
        self,
        batch_size: int,
        query_len: int,
        mode: str,
    ) -> bool:
        estimate = self._estimates.get((batch_size, query_len, mode))
        return estimate is None or estimate.samples < self.samples_per_bucket

    def observe(
        self,
        *,
        batch_size: int,
        query_len: int,
        mode: str,
        full_step_s: float,
        profiled_step_s: float,
    ) -> tuple[float, RuntimeCostEstimate]:
        residual_s = max(float(full_step_s) - float(profiled_step_s), 0.0)
        exact = self._update(
            (batch_size, query_len, mode),
            full_step_s=float(full_step_s),
            profiled_step_s=float(profiled_step_s),
            residual_s=residual_s,
        )
        if mode != ALL_RUNTIME_MODES:
            self._update(
                (batch_size, query_len, ALL_RUNTIME_MODES),
                full_step_s=float(full_step_s),
                profiled_step_s=float(profiled_step_s),
                residual_s=residual_s,
            )
        return residual_s, exact

    def get(
        self,
        batch_size: int,
        query_len: int,
        mode: str,
    ) -> RuntimeCostEstimate:
        exact = self._estimates.get((batch_size, query_len, mode))
        if exact is not None and exact.samples:
            return exact
        aggregate = self._estimates.get(
            (batch_size, query_len, ALL_RUNTIME_MODES)
        )
        if aggregate is not None:
            return aggregate
        return RuntimeCostEstimate()

    def snapshot(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "batch_size": batch_size,
                "query_len": query_len,
                "mode": mode,
                "samples": estimate.samples,
                "full_step_s": estimate.full_step_s,
                "full_step_ms": estimate.full_step_s * 1e3,
                "profiled_step_s": estimate.profiled_step_s,
                "profiled_step_ms": estimate.profiled_step_s * 1e3,
                "overhead_s": estimate.overhead_s,
                "overhead_ms": estimate.overhead_s * 1e3,
            }
            for (
                batch_size,
                query_len,
                mode,
            ), estimate in sorted(self._estimates.items())
        ]

    def _update(
        self,
        key: tuple[int, int, str],
        *,
        full_step_s: float,
        profiled_step_s: float,
        residual_s: float,
    ) -> RuntimeCostEstimate:
        estimate = self._estimates.setdefault(key, RuntimeCostEstimate())
        if estimate.samples == 0:
            estimate.full_step_s = full_step_s
            estimate.profiled_step_s = profiled_step_s
            estimate.overhead_s = residual_s
        else:
            estimate.full_step_s = (
                self.alpha * full_step_s
                + (1.0 - self.alpha) * estimate.full_step_s
            )
            estimate.profiled_step_s = (
                self.alpha * profiled_step_s
                + (1.0 - self.alpha) * estimate.profiled_step_s
            )
            estimate.overhead_s = (
                self.alpha * residual_s
                + (1.0 - self.alpha) * estimate.overhead_s
            )
        estimate.samples += 1
        return estimate

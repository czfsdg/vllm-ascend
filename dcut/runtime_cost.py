# SPDX-License-Identifier: Apache-2.0
"""Runtime fixed-cost calibration for adaptive D-Cut decisions."""

from __future__ import annotations

from dataclasses import dataclass

ALL_RUNTIME_MODES = "all"


@dataclass
class RuntimeCostEstimate:
    """EWMA estimate of cost missing from the startup NPU profiles."""

    samples: int = 0
    overhead_s: float = 0.0


class RuntimeCostCalibrator:
    """Learn full-step residual cost per ``(batch bucket, runtime mode)``.

    Startup profiling measures target and full-drafter device work for every
    candidate Q. A short synchronized calibration on real serving steps then
    measures the complete runner pipeline. Their non-negative difference is
    the fixed cost paid once per speculative iteration (sampling, bookkeeping,
    metadata/copies, D-Cut and graph dispatch/host work).
    """

    def __init__(self, *, alpha: float, samples_per_bucket: int) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if samples_per_bucket < 1:
            raise ValueError("samples_per_bucket must be >= 1.")
        self.alpha = alpha
        self.samples_per_bucket = samples_per_bucket
        self._estimates: dict[tuple[int, str], RuntimeCostEstimate] = {}

    def needs_sample(self, batch_size: int, mode: str) -> bool:
        estimate = self._estimates.get((batch_size, mode))
        return estimate is None or estimate.samples < self.samples_per_bucket

    def observe(
        self,
        *,
        batch_size: int,
        mode: str,
        full_step_s: float,
        profiled_step_s: float,
    ) -> tuple[float, RuntimeCostEstimate]:
        residual_s = max(float(full_step_s) - float(profiled_step_s), 0.0)
        exact = self._update((batch_size, mode), residual_s)
        if mode != ALL_RUNTIME_MODES:
            self._update((batch_size, ALL_RUNTIME_MODES), residual_s)
        return residual_s, exact

    def get(self, batch_size: int, mode: str) -> RuntimeCostEstimate:
        exact = self._estimates.get((batch_size, mode))
        if exact is not None and exact.samples:
            return exact
        aggregate = self._estimates.get((batch_size, ALL_RUNTIME_MODES))
        if aggregate is not None:
            return aggregate
        return RuntimeCostEstimate()

    def snapshot(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "batch_size": batch_size,
                "mode": mode,
                "samples": estimate.samples,
                "overhead_s": estimate.overhead_s,
                "overhead_ms": estimate.overhead_s * 1e3,
            }
            for (batch_size, mode), estimate in sorted(self._estimates.items())
        ]

    def _update(
        self,
        key: tuple[int, str],
        sample_s: float,
    ) -> RuntimeCostEstimate:
        estimate = self._estimates.setdefault(key, RuntimeCostEstimate())
        if estimate.samples == 0:
            estimate.overhead_s = sample_s
        else:
            estimate.overhead_s = (
                self.alpha * sample_s
                + (1.0 - self.alpha) * estimate.overhead_s
            )
        estimate.samples += 1
        return estimate

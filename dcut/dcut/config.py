# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class VerifyAdaptiveConfig:
    """D-Cut config.

    ``query_len = 1(anchor token) + draft_len``. Unknown JSON keys are ignored
    so config files can be shared across plugin versions.
    """

    warmup_batch_sizes: list[int] = field(default_factory=list)
    budget_ratios: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    min_warmup_batch_size: int = 2
    max_warmup_batch_size: int | None = None
    query_len_step_per_req: int = 2
    max_query_len_per_req: int | None = None
    min_query_len_per_req: int = 2
    warmup_seq_lens: int = 4096
    n_warmup_iters: int = 3
    n_measure_iters: int = 5
    cost_table_dump_path: str | None = None
    log_decision_details: bool = False
    log_decision_interval: int = 1
    log_decision_max_records: int = 8
    min_score_improvement_ratio: float = 0.01
    enabled: bool = True

    @classmethod
    def from_json(cls, path: str) -> VerifyAdaptiveConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifyAdaptiveConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def validate(self, num_speculative_tokens: int) -> None:
        max_query_len = self.max_query_len_per_req or num_speculative_tokens + 1
        if self.min_query_len_per_req < 2:
            raise ValueError("min_query_len_per_req must be >= 2.")
        if self.query_len_step_per_req < 1:
            raise ValueError("query_len_step_per_req must be >= 1.")
        if self.min_query_len_per_req > max_query_len:
            raise ValueError(
                f"min_query_len_per_req ({self.min_query_len_per_req}) > "
                f"effective max_query_len_per_req ({max_query_len}).")
        if self.warmup_seq_lens < 1:
            raise ValueError("warmup_seq_lens must be >= 1.")
        if self.n_warmup_iters < 0:
            raise ValueError("n_warmup_iters must be >= 0.")
        if self.n_measure_iters < 1:
            raise ValueError("n_measure_iters must be >= 1.")
        if self.log_decision_interval < 1:
            raise ValueError("log_decision_interval must be >= 1.")
        if self.log_decision_max_records < 1:
            raise ValueError("log_decision_max_records must be >= 1.")
        if self.min_score_improvement_ratio < 0.0:
            raise ValueError("min_score_improvement_ratio must be >= 0.0.")
        if any(ratio <= 0.0 or ratio > 1.0 for ratio in self.budget_ratios):
            raise ValueError("budget_ratios entries must be in (0.0, 1.0].")
        if self.min_warmup_batch_size < 1:
            raise ValueError("min_warmup_batch_size must be >= 1.")
        if self.max_warmup_batch_size is not None and self.max_warmup_batch_size < 1:
            raise ValueError("max_warmup_batch_size must be >= 1.")
        if any(bs < 1 for bs in self.warmup_batch_sizes):
            raise ValueError("warmup_batch_sizes entries must be >= 1.")

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VerifyAdaptiveConfig:
    enabled: bool = True
    min_query_len_per_req: int = 2
    max_query_len_per_req: int | None = None
    query_len_step_per_req: int = 2
    min_warmup_batch_size: int = 2
    max_warmup_batch_size: int | None = None
    warmup_batch_sizes: list[int] | None = None
    warmup_seq_lens: int = 4096
    n_warmup_iters: int = 3
    n_measure_iters: int = 5
    cost_table: dict[str, float] | None = None
    apply_adaptive_lengths: bool = True
    min_prefix_prob: float = 0.05
    min_adaptive_draft_len: int = 2
    uniform_adaptive_lengths: bool = True
    mutate_scheduler_output: bool = True
    log_concurrency_interval_s: float = 5.0
    log_runtime_events: bool = False
    debug_scheduler_state: bool = False

    @classmethod
    def from_file(cls, path: str) -> VerifyAdaptiveConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("D-Cut config must be a JSON object")
        return cls(**data)

    def query_len_levels(self, max_draft_len: int) -> list[int]:
        max_query_len = self.max_query_len_per_req or (max_draft_len + 1)
        max_query_len = max(1, min(max_query_len, max_draft_len + 1))
        levels = [1]
        start = max(2, self.min_query_len_per_req)
        step = max(1, self.query_len_step_per_req)
        levels.extend(range(start, max_query_len + 1, step))
        if levels[-1] != max_query_len:
            levels.append(max_query_len)
        return sorted(set(levels))

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_query_len_per_req": self.min_query_len_per_req,
            "max_query_len_per_req": self.max_query_len_per_req,
            "query_len_step_per_req": self.query_len_step_per_req,
            "warmup_seq_lens": self.warmup_seq_lens,
            "n_warmup_iters": self.n_warmup_iters,
            "n_measure_iters": self.n_measure_iters,
            "cost_table_entries": 0 if self.cost_table is None else len(self.cost_table),
            "apply_adaptive_lengths": self.apply_adaptive_lengths,
            "min_prefix_prob": self.min_prefix_prob,
            "min_adaptive_draft_len": self.min_adaptive_draft_len,
            "uniform_adaptive_lengths": self.uniform_adaptive_lengths,
            "mutate_scheduler_output": self.mutate_scheduler_output,
            "log_concurrency_interval_s": self.log_concurrency_interval_s,
            "log_runtime_events": self.log_runtime_events,
            "debug_scheduler_state": self.debug_scheduler_state,
        }

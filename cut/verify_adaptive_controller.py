# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import bisect
import json
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .verify_adaptive_config import VerifyAdaptiveConfig


def choose_query_lens_discrete(
    probs: "list[list[float]] | np.ndarray",
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
) -> dict[str, Any]:
    """Choose adaptive draft lengths using profiled verifier cost levels."""
    active_count = len(probs)
    mat = np.asarray(probs, dtype=np.float64).reshape(active_count,
                                                       -1)[:, :max_draft_len]
    gains = np.cumprod(mat, axis=1)
    seq_ids = np.repeat(np.arange(active_count), gains.shape[1])
    flat_gains = gains.ravel()
    order = np.argsort(-flat_gains, kind="stable")
    sorted_seq = seq_ids[order]
    prefix_gain = np.concatenate(([0.0], np.cumsum(flat_gains[order])))
    total_available = flat_gains.shape[0]
    best_score = -math.inf
    best_q, best_s = base_batch_size, 0
    records: list[dict[str, Any]] | None = [] if collect_records else None
    for query_len in q_levels:
        spec_slots = query_len - base_batch_size
        if spec_slots < 0:
            continue
        spec_slots = min(spec_slots, total_available)
        cost = cost_lookup(query_len)
        if cost <= 0.0:
            continue
        score = (base_batch_size + prefix_gain[spec_slots]) / cost
        if records is not None:
            records.append({
                "Q": query_len,
                "S": int(spec_slots),
                "score": score,
                "cost": cost,
            })
        if score > best_score:
            best_score, best_q, best_s = score, query_len, spec_slots
    draft_lens = np.bincount(sorted_seq[:best_s],
                             minlength=active_count).tolist()
    return {
        "draft_lens": draft_lens,
        "best_Q": best_q,
        "best_S": int(best_s),
        "best_score": best_score,
        "records": records,
    }


class VerifyAdaptiveController:
    """Per-request draft-length selector for verifier steps."""

    def __init__(self, config: VerifyAdaptiveConfig, num_spec_tokens: int,
                 max_batch_size: int, device: torch.device) -> None:
        config.validate(num_spec_tokens)
        self.config = config
        self.num_spec_tokens = num_spec_tokens
        self.max_batch_size = max_batch_size
        self.device = device
        self.max_query_len_per_req = (config.max_query_len_per_req
                                      if config.max_query_len_per_req
                                      is not None else num_spec_tokens + 1)
        self._batch_size_levels = self._build_batch_size_levels()
        self._query_len_levels = self._build_query_len_levels()
        self._cost_table: dict[tuple[int, int], float] = {}
        self._cost_records: list[dict[str, Any]] = []
        self._sorted_bs: list[int] = []
        self._sorted_sql_per_bs: dict[int, list[int]] = {}
        self._adaptive_draft_lens: dict[str, int] = {}

    def _build_batch_size_levels(self) -> list[int]:
        if self.config.warmup_batch_sizes:
            return sorted(set(self.config.warmup_batch_sizes))
        cap = (self.config.max_warmup_batch_size
               if self.config.max_warmup_batch_size is not None
               else self.max_batch_size)
        levels = list(range(self.config.min_warmup_batch_size, cap + 1, 2))
        if not levels or levels[-1] < cap:
            levels.append(cap)
        return levels

    def _build_query_len_levels(self) -> list[int]:
        levels = list(
            range(self.config.min_query_len_per_req,
                  self.max_query_len_per_req + 1,
                  self.config.query_len_step_per_req))
        if not levels or levels[-1] < self.max_query_len_per_req:
            levels.append(self.max_query_len_per_req)
        return sorted(set(levels))

    def profile_cost_table(self, runner: Any) -> None:
        if not self.config.enabled:
            return
        max_tokens = getattr(runner, "max_num_tokens", None)
        for bs in self._batch_size_levels:
            self._sorted_sql_per_bs[bs] = []
            for ql in self._query_len_levels:
                num_tokens = bs * ql
                if max_tokens is not None and num_tokens > max_tokens:
                    continue
                runtime_mode, avg_ms, padded_tokens = runner._adaptive_profile_run(
                    [ql] * bs, self.config.warmup_seq_lens,
                    self.config.n_warmup_iters, self.config.n_measure_iters)
                elapsed_s = avg_ms / 1e3
                self._cost_table[(bs, num_tokens)] = elapsed_s
                self._cost_records.append({
                    "batch_size": bs,
                    "query_len_per_req": ql,
                    "sum_query_len": num_tokens,
                    "padded_tokens": padded_tokens,
                    "seq_lens": self.config.warmup_seq_lens,
                    "runtime_mode": runtime_mode,
                    "avg_ms": avg_ms,
                    "cost_s": elapsed_s,
                })
                self._sorted_sql_per_bs[bs].append(num_tokens)
        self._sorted_bs = [
            bs for bs in sorted(self._sorted_sql_per_bs)
            if self._sorted_sql_per_bs[bs]
        ]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()
        self._dump_cost_table_if_requested()

    def process_draft_output(self, selected_probs: torch.Tensor,
                             req_ids: list[str], active_draft_req_ids: set[str],
                             batch_size: int) -> None:
        if not self.config.enabled or not active_draft_req_ids or not self._sorted_bs:
            return
        n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
        all_probs_np = selected_probs[:n_rows].numpy()
        active_indices = [
            i for i in range(n_rows) if req_ids[i] in active_draft_req_ids
        ]
        if not active_indices:
            return
        bs_key = _ceil_lookup(batch_size, self._sorted_bs)
        q_levels = self._sorted_sql_per_bs.get(bs_key) or []
        if not q_levels:
            return
        result = choose_query_lens_discrete(
            probs=all_probs_np[active_indices],
            base_batch_size=batch_size,
            q_levels=q_levels,
            cost_lookup=lambda q: self._cost_table[(bs_key, q)],
            max_draft_len=self.max_query_len_per_req - 1,
        )
        for req_id, draft_len in zip([req_ids[i] for i in active_indices],
                                     result["draft_lens"]):
            self._adaptive_draft_lens[req_id] = draft_len

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        return self._adaptive_draft_lens.get(req_id)

    def invalidate(self, req_id: str) -> None:
        self._adaptive_draft_lens.pop(req_id, None)

    def _dump_cost_table_if_requested(self) -> None:
        dump_path = (os.getenv("VLLM_DCUT_COST_TABLE_OUT")
                     or self.config.cost_table_dump_path)
        if not dump_path:
            return
        rows = []
        for (bs, sum_query_len), cost_s in sorted(self._cost_table.items()):
            rows.append({
                "batch_size": bs,
                "sum_query_len": sum_query_len,
                "query_len_per_req": sum_query_len // bs
                if bs > 0 and sum_query_len % bs == 0 else None,
                "cost_s": cost_s,
                "cost_ms": cost_s * 1e3,
            })
        payload = {
            "schema_version": 1,
            "num_spec_tokens": self.num_spec_tokens,
            "max_batch_size": self.max_batch_size,
            "warmup_seq_lens": self.config.warmup_seq_lens,
            "n_warmup_iters": self.config.n_warmup_iters,
            "n_measure_iters": self.config.n_measure_iters,
            "batch_size_levels": self._batch_size_levels,
            "query_len_levels": self._query_len_levels,
            "cost_table": rows,
            "profile_records": self._cost_records,
        }
        dirname = os.path.dirname(dump_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = f"{dump_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, dump_path)


def _ceil_lookup(val: int, sorted_keys: list[int]) -> int:
    idx = bisect.bisect_left(sorted_keys, val)
    return sorted_keys[min(idx, len(sorted_keys) - 1)]

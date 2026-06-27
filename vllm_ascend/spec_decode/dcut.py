# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import bisect
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import logger

from vllm_ascend import envs


@dataclass
class VerifyAdaptiveConfig:
    """Config for D-Cut adaptive verifier step-length.

    ``query_len = 1 (anchor) + draft_len``. Unknown JSON keys are ignored.
    """

    warmup_batch_sizes: list[int] = field(default_factory=list)
    min_warmup_batch_size: int = 2
    max_warmup_batch_size: int | None = None
    query_len_step_per_req: int = 2
    max_query_len_per_req: int | None = None
    min_query_len_per_req: int = 2
    warmup_seq_lens: int = 4096
    n_warmup_iters: int = 3
    n_measure_iters: int = 5
    cost_table_dump_path: str | None = None
    enabled: bool = True

    @classmethod
    def from_json(cls, path: str) -> VerifyAdaptiveConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifyAdaptiveConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self, num_speculative_tokens: int) -> None:
        eff_max_q = self.max_query_len_per_req or num_speculative_tokens + 1
        if self.min_query_len_per_req < 2:
            raise ValueError("min_query_len_per_req must be >= 2.")
        if self.query_len_step_per_req < 1:
            raise ValueError("query_len_step_per_req must be >= 1.")
        if self.min_query_len_per_req > eff_max_q:
            raise ValueError(
                f"min_query_len_per_req ({self.min_query_len_per_req}) > "
                f"effective max_query_len_per_req ({eff_max_q}).")
        if self.warmup_seq_lens < 1:
            raise ValueError("warmup_seq_lens must be >= 1.")
        if self.n_warmup_iters < 0:
            raise ValueError("n_warmup_iters must be >= 0.")
        if self.n_measure_iters < 1:
            raise ValueError("n_measure_iters must be >= 1.")
        if self.min_warmup_batch_size < 1:
            raise ValueError("min_warmup_batch_size must be >= 1.")
        if self.max_warmup_batch_size is not None and self.max_warmup_batch_size < 1:
            raise ValueError("max_warmup_batch_size must be >= 1.")
        if any(bs < 1 for bs in self.warmup_batch_sizes):
            raise ValueError("warmup_batch_sizes entries must be >= 1.")


def choose_query_lens_discrete(
    probs: list[list[float]] | np.ndarray,
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
) -> dict[str, Any]:
    """D-Cut discrete marginal-gain scan over measured sum-query levels."""
    active = len(probs)
    mat = np.asarray(probs, dtype=np.float64).reshape(active, -1)[:, :max_draft_len]
    gains = np.cumprod(mat, axis=1)
    seq_ids = np.repeat(np.arange(active), gains.shape[1])
    flat_gains = gains.ravel()
    order = np.argsort(-flat_gains, kind="stable")
    sorted_seq = seq_ids[order]
    prefix_gain = np.concatenate(([0.0], np.cumsum(flat_gains[order])))
    total_available = flat_gains.shape[0]
    best_score = -math.inf
    best_q = base_batch_size
    best_s = 0
    records: list[dict[str, Any]] | None = [] if collect_records else None
    for q_level in q_levels:
        speculative_slots = q_level - base_batch_size
        if speculative_slots < 0:
            continue
        speculative_slots = min(speculative_slots, total_available)
        cost = cost_lookup(q_level)
        if cost <= 0.0:
            continue
        score = (base_batch_size + prefix_gain[speculative_slots]) / cost
        if records is not None:
            records.append({"Q": q_level, "S": int(speculative_slots), "score": score, "cost": cost})
        if score > best_score:
            best_score = score
            best_q = q_level
            best_s = speculative_slots
    return {
        "draft_lens": np.bincount(sorted_seq[:best_s], minlength=active).tolist(),
        "best_Q": best_q,
        "best_S": int(best_s),
        "best_score": best_score,
        "records": records,
    }


class VerifyAdaptiveController:
    """Per-request D-Cut draft-length selector for verifier steps."""

    def __init__(self, config: VerifyAdaptiveConfig, num_spec_tokens: int, max_batch_size: int) -> None:
        config.validate(num_spec_tokens)
        self.config = config
        self.num_spec_tokens = num_spec_tokens
        self.max_batch_size = max_batch_size
        self.max_query_len_per_req = config.max_query_len_per_req or num_spec_tokens + 1
        self._batch_size_levels = self._build_batch_size_levels()
        self._query_len_levels = self._build_query_len_levels()
        self._cost_table: dict[tuple[int, int], float] = {}
        self._cost_records: list[dict[str, Any]] = []
        self._sorted_bs: list[int] = []
        self._sorted_sql_per_bs: dict[int, list[int]] = {}
        self._adaptive_draft_lens: dict[str, int] = {}
        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info("D-Cut: bs_levels=%s ql_levels=%s", self._batch_size_levels, self._query_len_levels)

    @classmethod
    def from_env(cls, num_spec_tokens: int, max_batch_size: int) -> VerifyAdaptiveController | None:
        cfg_path = envs.VLLM_ASCEND_DCUT_CONFIG
        if not cfg_path:
            return None
        return cls(VerifyAdaptiveConfig.from_json(cfg_path), num_spec_tokens, max_batch_size)

    def _build_batch_size_levels(self) -> list[int]:
        if self.config.warmup_batch_sizes:
            return sorted(set(self.config.warmup_batch_sizes))
        cap = self.config.max_warmup_batch_size or self.max_batch_size
        levels = list(range(self.config.min_warmup_batch_size, cap + 1, 2))
        if not levels or levels[-1] < cap:
            levels.append(cap)
        return levels

    def _build_query_len_levels(self) -> list[int]:
        levels = list(range(self.config.min_query_len_per_req, self.max_query_len_per_req + 1,
                            self.config.query_len_step_per_req))
        if not levels or levels[-1] < self.max_query_len_per_req:
            levels.append(self.max_query_len_per_req)
        return sorted(set(levels))

    def profile_cost_table(self, runner: Any) -> None:
        if not self.config.enabled:
            return
        max_tokens = getattr(runner, "max_num_tokens", None)
        for batch_size in self._batch_size_levels:
            self._sorted_sql_per_bs[batch_size] = []
            for query_len in self._query_len_levels:
                num_tokens = batch_size * query_len
                if max_tokens is not None and num_tokens > max_tokens:
                    continue
                avg_ms = self._measure_runner(runner, num_tokens)
                self._cost_table[(batch_size, num_tokens)] = avg_ms / 1e3
                self._sorted_sql_per_bs[batch_size].append(num_tokens)
                self._cost_records.append({
                    "batch_size": batch_size,
                    "query_len_per_req": query_len,
                    "sum_query_len": num_tokens,
                    "avg_ms": avg_ms,
                    "cost_s": avg_ms / 1e3,
                })
        self._sorted_bs = [bs for bs in sorted(self._sorted_sql_per_bs) if self._sorted_sql_per_bs[bs]]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            self._cost_table = tp_group.broadcast_object(self._cost_table, src=0)
        logger.info("D-Cut: cost table ready (%d entries).", len(self._cost_table))
        self._dump_cost_table_if_requested()

    def _measure_runner(self, runner: Any, num_tokens: int) -> float:
        for _ in range(self.config.n_warmup_iters):
            runner._dummy_run(num_tokens, uniform_decode=True, is_profile=True,
                              profile_seq_lens=self.config.warmup_seq_lens)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(self.config.n_measure_iters):
            runner._dummy_run(num_tokens, uniform_decode=True, is_profile=True,
                              profile_seq_lens=self.config.warmup_seq_lens)
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / self.config.n_measure_iters

    def process_draft_output(self, selected_probs: torch.Tensor, req_ids: list[str], active_draft_req_ids: set[str],
                             batch_size: int) -> None:
        if not self.config.enabled or not active_draft_req_ids or not self._sorted_bs:
            return
        n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
        active_indices = [i for i in range(n_rows) if req_ids[i] in active_draft_req_ids]
        if not active_indices:
            return
        active_probs = selected_probs[:n_rows].numpy()[active_indices]
        active_req_ids = [req_ids[i] for i in active_indices]
        bs_key = _ceil_lookup(batch_size, self._sorted_bs)
        q_levels = self._sorted_sql_per_bs.get(bs_key) or []
        if not q_levels:
            return
        result = choose_query_lens_discrete(active_probs, batch_size, q_levels,
                                            lambda q: self._cost_table[(bs_key, q)],
                                            self.max_query_len_per_req - 1)
        for req_id, draft_len in zip(active_req_ids, result["draft_lens"]):
            self._adaptive_draft_lens[req_id] = draft_len
        logger.debug("D-Cut: bs_key=%d best_Q=%d best_S=%d score=%.4f draft_lens=%s", bs_key,
                     result["best_Q"], result["best_S"], result["best_score"], result["draft_lens"])

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        return self._adaptive_draft_lens.get(req_id)

    def invalidate(self, req_id: str) -> None:
        self._adaptive_draft_lens.pop(req_id, None)

    def _dump_cost_table_if_requested(self) -> None:
        dump_path = envs.VLLM_ASCEND_DCUT_COST_TABLE_OUT or self.config.cost_table_dump_path
        if not dump_path or get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return
        rows = [{"batch_size": bs, "sum_query_len": q, "cost_s": cost_s, "cost_ms": cost_s * 1e3}
                for (bs, q), cost_s in sorted(self._cost_table.items())]
        payload = {"schema_version": 1, "num_spec_tokens": self.num_spec_tokens,
                   "max_batch_size": self.max_batch_size, "cost_table": rows,
                   "profile_records": self._cost_records}
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

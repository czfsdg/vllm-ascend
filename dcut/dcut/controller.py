# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import bisect
import json
import math
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import logger

from dcut.config import VerifyAdaptiveConfig


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def dcut_enabled() -> bool:
    return _env_enabled("VLLM_ASCEND_ENABLE_DCUT") or _env_enabled("DCUT_ENABLE")


def choose_query_lens_discrete(
    probs: list[list[float]] | np.ndarray,
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
    min_score_improvement_ratio: float = 0.0,
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
    best_score = -math.inf
    best_q = base_batch_size
    best_s = 0
    records: list[dict[str, Any]] | None = [] if collect_records else None
    for q_level in q_levels:
        speculative_slots = min(max(q_level - base_batch_size, 0), flat_gains.shape[0])
        cost = cost_lookup(q_level)
        if cost <= 0.0:
            continue
        score = (base_batch_size + prefix_gain[speculative_slots]) / cost
        if records is not None:
            records.append({"Q": q_level, "S": int(speculative_slots), "score": score, "cost": cost})
        if _is_meaningful_score_improvement(score, best_score, min_score_improvement_ratio):
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
    """Per-request D-Cut draft-length selector."""

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
        self._decision_count = 0
        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info("D-Cut: bs_levels=%s ql_levels=%s budget_ratios=%s",
                        self._batch_size_levels, self._query_len_levels, self.config.budget_ratios)

    @classmethod
    def from_env(cls, num_spec_tokens: int, max_batch_size: int) -> VerifyAdaptiveController | None:
        cfg_path = os.getenv("VLLM_ASCEND_DCUT_CONFIG") or os.getenv("VLLM_DCUT_CONFIG") or os.getenv("DCUT_CONFIG")
        if cfg_path:
            config = VerifyAdaptiveConfig.from_json(cfg_path)
            logger.info("D-Cut: loaded config from %s", cfg_path)
        elif dcut_enabled():
            config = VerifyAdaptiveConfig()
            logger.info("D-Cut: using built-in default config")
        else:
            return None
        return cls(config, num_spec_tokens, max_batch_size)

    def _build_batch_size_levels(self) -> list[int]:
        if self.config.warmup_batch_sizes:
            return sorted(set(self.config.warmup_batch_sizes))
        cap = self.config.max_warmup_batch_size or self.max_batch_size
        levels = list(range(self.config.min_warmup_batch_size, cap + 1, self.config.batch_size_step))
        if not levels or levels[-1] < cap:
            levels.append(cap)
        return levels

    def _build_query_len_levels(self) -> list[int]:
        levels = list(range(self.config.min_query_len_per_req, self.max_query_len_per_req + 1,
                            self.config.query_len_step_per_req))
        if not levels or levels[-1] < self.max_query_len_per_req:
            levels.append(self.max_query_len_per_req)
        return sorted(set(levels))

    def _build_sum_query_len_levels(self, batch_size: int) -> list[int]:
        max_draft_tokens = batch_size * (self.max_query_len_per_req - 1)
        if self.config.budget_ratios:
            levels = [batch_size + math.ceil(ratio * max_draft_tokens)
                      for ratio in self.config.budget_ratios]
            full_budget = batch_size + max_draft_tokens
            if full_budget not in levels:
                levels.append(full_budget)
            return sorted(set(levels))
        return sorted(set(batch_size * query_len for query_len in self._query_len_levels))

    def profile_cost_table(self, runner: Any) -> None:
        if not self.config.enabled:
            return
        max_tokens = getattr(runner, "max_num_tokens", None)
        for batch_size in self._batch_size_levels:
            self._sorted_sql_per_bs[batch_size] = []
            for num_tokens in self._build_sum_query_len_levels(batch_size):
                if max_tokens is not None and num_tokens > max_tokens:
                    continue
                avg_ms = self._measure_runner(runner, num_tokens)
                self._cost_table[(batch_size, num_tokens)] = avg_ms / 1000.0
                self._sorted_sql_per_bs[batch_size].append(num_tokens)
                self._cost_records.append({
                    "batch_size": batch_size,
                    "query_len_per_req": num_tokens / batch_size,
                    "sum_query_len": num_tokens,
                    "draft_budget_tokens": max(num_tokens - batch_size, 0),
                    "draft_budget_ratio": max(num_tokens - batch_size, 0) /
                    max(batch_size * (self.max_query_len_per_req - 1), 1),
                    "avg_ms": avg_ms,
                    "cost_s": avg_ms / 1000.0,
                })
        self._sorted_bs = [bs for bs in sorted(self._sorted_sql_per_bs) if self._sorted_sql_per_bs[bs]]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()
        self._dump_cost_table_if_requested()
        logger.info("D-Cut: cost table ready (%d entries).", len(self._cost_table))

    def _measure_runner(self, runner: Any, num_tokens: int) -> float:
        for _ in range(self.config.n_warmup_iters):
            runner._dummy_run(num_tokens, uniform_decode=True, is_profile=True,
                              skip_drafter=True, profile_seq_lens=self.config.warmup_seq_lens)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(self.config.n_measure_iters):
            runner._dummy_run(num_tokens, uniform_decode=True, is_profile=True,
                              skip_drafter=True, profile_seq_lens=self.config.warmup_seq_lens)
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000.0 / self.config.n_measure_iters

    def process_draft_output(self, selected_probs: torch.Tensor, req_ids: list[str], active_draft_req_ids: set[str],
                             batch_size: int) -> None:
        if not self.config.enabled:
            logger.info_once("D-Cut: skip adaptive planning because the controller is disabled.")
            return
        if not active_draft_req_ids:
            logger.debug("D-Cut: skip adaptive planning because there are no active decode requests.")
            return
        if not self._sorted_bs:
            logger.warning_once(
                "D-Cut: skip adaptive planning because the verifier cost table is empty. "
                "Check D-Cut cost profiling during engine initialization."
            )
            return
        n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
        active_indices = [i for i in range(n_rows) if req_ids[i] in active_draft_req_ids]
        if not active_indices:
            logger.info_once(
                "D-Cut: skip adaptive planning because captured probability rows do not match active request ids."
            )
            return
        active_probs = selected_probs[:n_rows].numpy()[active_indices]
        active_req_ids = [req_ids[i] for i in active_indices]
        bs_key = _ceil_lookup(batch_size, self._sorted_bs)
        q_levels = self._sorted_sql_per_bs.get(bs_key) or []
        if not q_levels:
            logger.warning_once(
                "D-Cut: skip adaptive planning because no query levels are available for bs_key=%d. "
                "Known batch-size levels are %s.",
                bs_key,
                self._sorted_bs,
            )
            return
        log_details = self._should_log_decision_details()
        result = choose_query_lens_discrete(active_probs, batch_size, q_levels,
                                            lambda q: self._cost_table[(bs_key, q)],
                                            self.max_query_len_per_req - 1,
                                            collect_records=log_details,
                                            min_score_improvement_ratio=self.config.min_score_improvement_ratio)
        for req_id, draft_len in zip(active_req_ids, result["draft_lens"]):
            self._adaptive_draft_lens[req_id] = draft_len
        logger.info(
            "D-Cut: planned adaptive draft lengths batch_size=%d active_reqs=%d best_Q=%d best_S=%d draft_lens=%s",
            batch_size,
            len(active_req_ids),
            result["best_Q"],
            result["best_S"],
            result["draft_lens"],
        )
        if log_details:
            logger.info(
                "D-Cut decision details: bs_key=%d max_draft_len=%d min_score_improvement_ratio=%.4f "
                "prob_mean_by_pos=%s candidates=%s",
                bs_key,
                self.max_query_len_per_req - 1,
                self.config.min_score_improvement_ratio,
                _rounded_list(active_probs.mean(axis=0)),
                _format_decision_records(result["records"], self.config.log_decision_max_records),
            )

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        return self._adaptive_draft_lens.get(req_id)

    def invalidate(self, req_id: str) -> None:
        self._adaptive_draft_lens.pop(req_id, None)

    def _dump_cost_table_if_requested(self) -> None:
        dump_path = (
            os.getenv("VLLM_ASCEND_DCUT_COST_TABLE_OUT")
            or os.getenv("VLLM_DCUT_COST_TABLE_OUT")
            or os.getenv("DCUT_COST_TABLE_OUT")
            or self.config.cost_table_dump_path
        )
        if not dump_path or get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return
        rows = [{"batch_size": bs, "sum_query_len": q, "cost_s": cost_s, "cost_ms": cost_s * 1000.0}
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

    def _should_log_decision_details(self) -> bool:
        if not self.config.log_decision_details:
            return False
        self._decision_count += 1
        return (self._decision_count - 1) % self.config.log_decision_interval == 0


def _is_meaningful_score_improvement(score: float, best_score: float, min_improvement_ratio: float) -> bool:
    if not math.isfinite(best_score):
        return True
    return score > best_score * (1.0 + min_improvement_ratio)


def _ceil_lookup(val: int, sorted_keys: list[int]) -> int:
    idx = bisect.bisect_left(sorted_keys, val)
    return sorted_keys[min(idx, len(sorted_keys) - 1)]


def _rounded_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]


def _format_decision_records(records: list[dict[str, Any]] | None, limit: int) -> list[dict[str, float | int]]:
    if not records:
        return []
    return [{
        "Q": int(record["Q"]),
        "S": int(record["S"]),
        "cost_ms": round(float(record["cost"]) * 1000.0, 4),
        "score": round(float(record["score"]), 6),
    } for record in records[:limit]]

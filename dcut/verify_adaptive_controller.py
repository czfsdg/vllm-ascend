# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import bisect
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger

from .verify_adaptive_config import VerifyAdaptiveConfig

logger = init_logger(f"vllm.{__name__}")


def choose_query_lens_discrete(
    probs: list[list[float]] | np.ndarray,
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
    min_draft_len: int = 0,
) -> dict[str, Any]:
    """Discrete marginal-gain scan over measured sum-query-len levels."""
    active_count = len(probs)
    mat = np.asarray(probs, dtype=np.float64).reshape(active_count, -1)[:, :max_draft_len]
    gains = np.cumprod(mat, axis=1)
    min_draft_len = min(max(int(min_draft_len), 0), max_draft_len)
    initial_slots = active_count * min_draft_len
    initial_gain = float(gains[:, :min_draft_len].sum()) if min_draft_len > 0 else 0.0

    remaining_gains = gains[:, min_draft_len:]
    if remaining_gains.shape[1] > 0:
        seq_ids = np.repeat(np.arange(active_count), remaining_gains.shape[1])
        flat_gains = remaining_gains.ravel()
        order = np.argsort(-flat_gains, kind="stable")
        sorted_seq = seq_ids[order]
        prefix_gain = np.concatenate(([0.0], np.cumsum(flat_gains[order])))
        total_extra_available = flat_gains.shape[0]
    else:
        sorted_seq = np.array([], dtype=np.int64)
        prefix_gain = np.array([0.0], dtype=np.float64)
        total_extra_available = 0

    best_score = -math.inf
    best_q, best_s, best_extra_s = base_batch_size, initial_slots, 0
    records: list[dict[str, Any]] | None = [] if collect_records else None

    for query_len_sum in q_levels:
        draft_slots = query_len_sum - base_batch_size
        if draft_slots < initial_slots:
            continue
        extra_slots = min(draft_slots - initial_slots, total_extra_available)
        effective_slots = initial_slots + extra_slots
        cost = cost_lookup(query_len_sum)
        if cost <= 0.0:
            continue
        score = (base_batch_size + initial_gain + prefix_gain[extra_slots]) / cost
        if records is not None:
            records.append({"Q": query_len_sum, "S": int(effective_slots), "score": score, "cost": cost})
        if score > best_score:
            best_score = score
            best_q, best_s, best_extra_s = query_len_sum, effective_slots, extra_slots

    draft_lens = np.full(active_count, min_draft_len, dtype=np.int64)
    if best_extra_s > 0:
        draft_lens += np.bincount(sorted_seq[:best_extra_s], minlength=active_count)
    draft_lens = draft_lens.tolist()
    return {
        "draft_lens": draft_lens,
        "best_Q": best_q,
        "best_S": int(best_s),
        "best_score": best_score,
        "records": records,
    }


class VerifyAdaptiveController:
    """Per-request draft-length selector for the verifier step."""

    def __init__(
        self,
        config: VerifyAdaptiveConfig,
        num_spec_tokens: int,
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        config.validate(num_spec_tokens)
        self.config = config
        self.num_spec_tokens = num_spec_tokens
        self.max_batch_size = max_batch_size
        self.device = device
        self.max_query_len_per_req = (
            config.max_query_len_per_req if config.max_query_len_per_req is not None else num_spec_tokens + 1
        )
        self._batch_size_levels = self._build_batch_size_levels()
        self._query_len_levels = self._build_query_len_levels()
        self._cost_table: dict[tuple[int, int], float] = {}
        self._cost_records: list[dict[str, Any]] = []
        self._sorted_bs: list[int] = []
        self._sorted_sql_per_bs: dict[int, list[int]] = {}
        self._adaptive_draft_lens: dict[str, int] = {}
        self._last_decision: dict[str, Any] | None = None
        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: bs_levels=%s ql_levels=%s",
                self._batch_size_levels,
                self._query_len_levels,
            )

    def _build_batch_size_levels(self) -> list[int]:
        if self.config.warmup_batch_sizes:
            return sorted(set(self.config.warmup_batch_sizes))
        cap = (
            self.config.max_warmup_batch_size if self.config.max_warmup_batch_size is not None else self.max_batch_size
        )
        start = self.config.min_warmup_batch_size
        levels = list(range(start, cap + 1, 2))
        if not levels or levels[-1] < cap:
            levels.append(cap)
        return levels

    def _build_query_len_levels(self) -> list[int]:
        min_q = self.config.min_query_len_per_req
        max_q = self.max_query_len_per_req
        step = self.config.query_len_step_per_req
        levels = list(range(min_q, max_q + 1, step))
        if not levels or levels[-1] < max_q:
            levels.append(max_q)
        return sorted(set(levels))

    def _fill_fallback_cost_table(self, reason: str, warn: bool = True) -> None:
        self._cost_table.clear()
        self._cost_records.clear()
        self._sorted_sql_per_bs.clear()
        for bs in self._batch_size_levels:
            self._sorted_sql_per_bs[bs] = []
            for ql in self._query_len_levels:
                sum_query_len = bs * ql
                cost_s = float(sum_query_len) / 1e6
                self._cost_table[(bs, sum_query_len)] = cost_s
                self._sorted_sql_per_bs[bs].append(sum_query_len)
                self._cost_records.append(
                    {
                        "batch_size": bs,
                        "query_len_per_req": ql,
                        "sum_query_len": sum_query_len,
                        "padded_tokens": sum_query_len,
                        "runtime_mode": "fallback",
                        "avg_ms": cost_s * 1e3,
                        "cost_s": cost_s,
                        "fallback_reason": reason,
                    }
                )
        self._sorted_bs = [bs for bs in sorted(self._sorted_sql_per_bs.keys()) if self._sorted_sql_per_bs[bs]]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()
        log_fn = logger.warning if warn else logger.info
        log_fn("VerifyAdaptiveController: using fallback cost table: %s", reason)

    def profile_cost_table(self, runner: Any) -> None:
        if not self.config.enabled:
            return
        if self.config.skip_runtime_profiling:
            self._fill_fallback_cost_table("skip_runtime_profiling=true", warn=False)
            self._dump_cost_table_if_requested()
            return
        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: profiling %d ITL cost points (%d bs × %d ql).",
                len(self._batch_size_levels) * len(self._query_len_levels),
                len(self._batch_size_levels),
                len(self._query_len_levels),
            )
        max_tokens = getattr(runner, "max_num_tokens", None)
        for bs in self._batch_size_levels:
            self._sorted_sql_per_bs[bs] = []
            for ql in self._query_len_levels:
                num_tokens = bs * ql
                if max_tokens is not None and num_tokens > max_tokens:
                    logger.info("profile skip: bs=%d ql=%d num_tokens=%d > %d", bs, ql, num_tokens, max_tokens)
                    continue
                try:
                    runtime_mode, avg_ms, padded_tokens = runner._adaptive_profile_run(
                        [ql] * bs,
                        self.config.warmup_seq_lens,
                        self.config.n_warmup_iters,
                        self.config.n_measure_iters,
                    )
                except Exception as exc:
                    if not self.config.fallback_on_profile_error:
                        raise
                    self._fill_fallback_cost_table(str(exc))
                    self._dump_cost_table_if_requested()
                    return
                elapsed_s = avg_ms / 1e3
                self._cost_table[(bs, num_tokens)] = elapsed_s
                self._cost_records.append(
                    {
                        "batch_size": bs,
                        "query_len_per_req": ql,
                        "sum_query_len": num_tokens,
                        "padded_tokens": padded_tokens,
                        "seq_lens": self.config.warmup_seq_lens,
                        "runtime_mode": runtime_mode,
                        "avg_ms": avg_ms,
                        "cost_s": elapsed_s,
                    }
                )
                self._sorted_sql_per_bs[bs].append(num_tokens)
                if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
                    logger.info(
                        "profile bs=%-4d ql=%-4d sql=%-6d padded=%-6d seq_lens=%-6d mode=%-6s avg=%.3f ms",
                        bs,
                        ql,
                        num_tokens,
                        padded_tokens,
                        self.config.warmup_seq_lens,
                        runtime_mode,
                        avg_ms,
                    )
        self._sorted_bs = [bs for bs in sorted(self._sorted_sql_per_bs.keys()) if self._sorted_sql_per_bs[bs]]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            self._cost_table = tp_group.broadcast_object(self._cost_table, src=0)
        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info("VerifyAdaptiveController: cost table ready (%d entries).", len(self._cost_table))
        self._dump_cost_table_if_requested()

    def process_draft_output(
        self,
        selected_probs: torch.Tensor,
        req_ids: list[str],
        active_draft_req_ids: set[str],
        batch_size: int,
    ) -> None:
        if not self.config.enabled or not active_draft_req_ids or not self._sorted_bs:
            return
        n_rows = min(selected_probs.shape[0], len(req_ids), batch_size)
        all_probs_np = selected_probs[:n_rows].numpy()
        active_indices = [i for i in range(n_rows) if req_ids[i] in active_draft_req_ids]
        if not active_indices:
            return
        active_probs = all_probs_np[active_indices]
        active_req_ids = [req_ids[i] for i in active_indices]
        bs_key = _ceil_lookup(batch_size, self._sorted_bs)
        q_levels = self._sorted_sql_per_bs.get(bs_key) or []
        if not q_levels:
            return
        result = choose_query_lens_discrete(
            probs=active_probs,
            base_batch_size=batch_size,
            q_levels=q_levels,
            cost_lookup=lambda q: self._cost_table[(bs_key, q)],
            max_draft_len=self.max_query_len_per_req - 1,
            min_draft_len=self.config.min_draft_len_per_req,
        )
        draft_lens = [min(self.num_spec_tokens, int(draft_len)) for draft_len in result["draft_lens"]]
        effective_s = sum(draft_lens)
        for req_id, draft_len in zip(active_req_ids, draft_lens):
            self._adaptive_draft_lens[req_id] = draft_len
        draft_len_hist = dict(sorted(Counter(draft_lens).items()))
        self._last_decision = {
            "batch_size": batch_size,
            "active_count": len(active_req_ids),
            "bs_key": bs_key,
            "best_Q": result["best_Q"],
            "best_S": result["best_S"],
            "effective_S": effective_s,
            "best_score": result["best_score"],
            "draft_len_hist": draft_len_hist,
            "draft_lens_by_req": dict(zip(active_req_ids, draft_lens)),
        }
        logger.info(
            "D-Cut decision: bs=%d active=%d bs_key=%d best_Q=%d best_S=%d effective_S=%d score=%.4f draft_len_hist=%s",
            batch_size,
            len(active_req_ids),
            bs_key,
            result["best_Q"],
            result["best_S"],
            effective_s,
            result["best_score"],
            draft_len_hist,
        )

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        return self._adaptive_draft_lens.get(req_id)

    def invalidate(self, req_id: str) -> None:
        self._adaptive_draft_lens.pop(req_id, None)

    def _dump_cost_table_if_requested(self) -> None:
        dump_path = os.getenv("VLLM_DCUT_COST_TABLE_OUT") or self.config.cost_table_dump_path
        if not dump_path:
            return
        if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return
        rows = []
        for (bs, sum_query_len), cost_s in sorted(self._cost_table.items()):
            rows.append(
                {
                    "batch_size": bs,
                    "sum_query_len": sum_query_len,
                    "query_len_per_req": sum_query_len // bs if bs > 0 and sum_query_len % bs == 0 else None,
                    "cost_s": cost_s,
                    "cost_ms": cost_s * 1e3,
                }
            )
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
        logger.info("VerifyAdaptiveController: dumped cost table to %s", dump_path)


def _ceil_lookup(val: int, sorted_keys: list[int]) -> int:
    idx = bisect.bisect_left(sorted_keys, val)
    return sorted_keys[min(idx, len(sorted_keys) - 1)]

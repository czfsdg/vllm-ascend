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
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger

from .verify_adaptive_config import VerifyAdaptiveConfig

logger = init_logger(__name__)


def choose_query_lens_discrete(
    probs: list[list[float]] | np.ndarray,
    base_batch_size: int,
    q_levels: list[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
    collect_records: bool = False,
) -> dict[str, Any]:
    """Discrete marginal-gain scan over the measured sum_query_len levels."""
    A = len(probs)
    mat = np.asarray(probs, dtype=np.float64).reshape(A, -1)[:, :max_draft_len]
    gains = np.cumprod(mat, axis=1)
    seq_ids = np.repeat(np.arange(A), gains.shape[1])
    flat_gains = gains.ravel()
    order = np.argsort(-flat_gains, kind="stable")
    sorted_seq = seq_ids[order]
    prefix_gain = np.concatenate(([0.0], np.cumsum(flat_gains[order])))
    total_available = flat_gains.shape[0]
    best_score = -math.inf
    best_Q, best_S = base_batch_size, 0
    records: list[dict[str, Any]] | None = [] if collect_records else None

    for Q in q_levels:
        S = Q - base_batch_size
        if S < 0:
            continue
        S = min(S, total_available)
        cost = cost_lookup(Q)
        if cost <= 0.0:
            continue
        score = (base_batch_size + prefix_gain[S]) / cost
        if records is not None:
            records.append({"Q": Q, "S": int(S), "score": score, "cost": cost})
        if score > best_score:
            best_score, best_Q, best_S = score, Q, S

    draft_lens = np.bincount(sorted_seq[:best_S], minlength=A).tolist()
    return {
        "draft_lens": draft_lens,
        "best_Q": best_Q,
        "best_S": int(best_S),
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
        if get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank:
            logger.info(
                "VerifyAdaptiveController: bs_levels=%s ql_levels=%s", self._batch_size_levels, self._query_len_levels
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

    def profile_cost_table(self, runner: Any) -> None:
        if not self.config.enabled:
            logger.info("VerifyAdaptiveController: disabled; skip cost profiling.")
            return
        max_tokens = getattr(runner, "max_num_tokens", None)
        logger.info(
            "VerifyAdaptiveController: begin cost profiling bs_levels=%s ql_levels=%s "
            "warmup_seq_lens=%s n_warmup_iters=%s n_measure_iters=%s "
            "max_tokens=%s json_out=%s markdown_out=%s",
            self._batch_size_levels,
            self._query_len_levels,
            self.config.warmup_seq_lens,
            self.config.n_warmup_iters,
            self.config.n_measure_iters,
            max_tokens,
            os.getenv("VLLM_DCUT_COST_TABLE_OUT") or self.config.cost_table_dump_path,
            os.getenv("VLLM_DCUT_COST_TABLE_MD_OUT") or self.config.cost_table_markdown_path,
        )
        candidates: list[tuple[int, int, int]] = []
        for bs in self._batch_size_levels:
            self._sorted_sql_per_bs[bs] = []
            for ql in self._query_len_levels:
                num_tokens = bs * ql
                if max_tokens is not None and num_tokens > max_tokens:
                    logger.info("profile skip: bs=%d ql=%d num_tokens=%d > %d", bs, ql, num_tokens, max_tokens)
                    continue
                candidates.append((bs, ql, num_tokens))

        if not candidates:
            logger.warning("VerifyAdaptiveController: no valid cost-profile candidates.")
            return

        logger.info(
            "VerifyAdaptiveController: graph prewarm pass START candidates=%d",
            len(candidates),
        )
        for bs, ql, _ in candidates:
            runner._adaptive_profile_run(
                [ql] * bs,
                self.config.warmup_seq_lens,
                max(1, self.config.n_warmup_iters),
                0,
            )
        logger.info("VerifyAdaptiveController: graph prewarm pass END")

        raw_records: list[dict[str, Any]] = []
        for bs, ql, num_tokens in candidates:
            runtime_mode, avg_ms, padded_tokens, timing_stats = runner._adaptive_profile_run(
                [ql] * bs,
                self.config.warmup_seq_lens,
                self.config.n_warmup_iters,
                self.config.n_measure_iters,
            )
            raw_cost_ms = float(timing_stats.get("median_ms", avg_ms))
            record = {
                "batch_size": bs,
                "query_len_per_req": ql,
                "sum_query_len": num_tokens,
                "padded_tokens": padded_tokens,
                "seq_lens": self.config.warmup_seq_lens,
                "runtime_mode": runtime_mode,
                "raw_avg_ms": float(timing_stats.get("avg_ms", avg_ms)),
                "raw_median_ms": raw_cost_ms,
                "raw_min_ms": float(timing_stats.get("min_ms", avg_ms)),
                "raw_max_ms": float(timing_stats.get("max_ms", avg_ms)),
                "raw_std_ms": float(timing_stats.get("std_ms", 0.0)),
                "samples_ms": timing_stats.get("samples_ms", []),
            }
            raw_records.append(record)

        bucket_costs: dict[tuple[int, int], list[float]] = {}
        bucket_modes: dict[tuple[int, int], str] = {}
        for record in raw_records:
            key = (int(record["batch_size"]), int(record["padded_tokens"]))
            bucket_costs.setdefault(key, []).append(float(record["raw_median_ms"]))
            bucket_modes[key] = str(record["runtime_mode"])
        bucket_cost_ms = {
            key: float(np.median(np.asarray(values, dtype=np.float64))) for key, values in bucket_costs.items()
        }

        bucket_representative: dict[tuple[int, int], int] = {}
        for record in raw_records:
            bs = int(record["batch_size"])
            bucket_key = (bs, int(record["padded_tokens"]))
            num_tokens = int(record["sum_query_len"])
            bucket_representative[bucket_key] = max(bucket_representative.get(bucket_key, 0), num_tokens)

        for record in raw_records:
            bs = int(record["batch_size"])
            num_tokens = int(record["sum_query_len"])
            bucket_key = (bs, int(record["padded_tokens"]))
            cost_ms = bucket_cost_ms[bucket_key]
            elapsed_s = cost_ms / 1e3
            representative_num_tokens = bucket_representative[bucket_key]
            is_representative = num_tokens == representative_num_tokens
            logger.info(
                "VerifyAdaptiveController: profile row bs=%d query_len=%d "
                "sum_query_len=%d runtime_mode=%s padded_tokens=%d "
                "cost_ms=%.6f raw_median_ms=%.6f raw_avg_ms=%.6f raw_std_ms=%.6f "
                "cost_table_representative=%s representative_sum_query_len=%d",
                bs,
                int(record["query_len_per_req"]),
                num_tokens,
                bucket_modes[bucket_key],
                int(record["padded_tokens"]),
                cost_ms,
                float(record["raw_median_ms"]),
                float(record["raw_avg_ms"]),
                float(record["raw_std_ms"]),
                is_representative,
                representative_num_tokens,
            )
            record["avg_ms"] = cost_ms
            record["cost_ms"] = cost_ms
            record["cost_s"] = elapsed_s
            record["bucket_key"] = {
                "batch_size": bs,
                "padded_tokens": int(record["padded_tokens"]),
            }
            record["cost_table_representative"] = is_representative
            record["representative_sum_query_len"] = representative_num_tokens
            self._cost_records.append(record)
            if is_representative:
                self._cost_table[(bs, num_tokens)] = elapsed_s
                self._sorted_sql_per_bs[bs].append(num_tokens)

        self._sorted_bs = [bs for bs in sorted(self._sorted_sql_per_bs) if self._sorted_sql_per_bs[bs]]
        for bs in self._sorted_bs:
            self._sorted_sql_per_bs[bs].sort()
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            self._cost_table = tp_group.broadcast_object(self._cost_table, src=0)
        logger.info("VerifyAdaptiveController: cost table ready (%d entries).", len(self._cost_table))
        self._log_cost_table()
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
        )
        for req_id, draft_len in zip(active_req_ids, result["draft_lens"]):
            self._adaptive_draft_lens[req_id] = draft_len

    def get_adaptive_draft_len(self, req_id: str) -> int | None:
        return self._adaptive_draft_lens.get(req_id)

    def invalidate(self, req_id: str) -> None:
        self._adaptive_draft_lens.pop(req_id, None)

    def _cost_table_rows(self) -> list[dict[str, Any]]:
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
        return rows

    def _format_cost_table_markdown(self, rows: list[dict[str, Any]]) -> str:
        lines = [
            "# D-Cut verifier cost table",
            "",
            "- runtime target: Qwen3.5 GDN graph capture",
            f"- num_spec_tokens: {self.num_spec_tokens}",
            f"- max_batch_size: {self.max_batch_size}",
            f"- warmup_seq_lens: {self.config.warmup_seq_lens}",
            f"- n_warmup_iters: {self.config.n_warmup_iters}",
            f"- n_measure_iters: {self.config.n_measure_iters}",
            "",
            "| batch_size | query_len_per_req | sum_query_len | cost_ms | cost_s |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['batch_size']} | {row['query_len_per_req']} | "
                f"{row['sum_query_len']} | {row['cost_ms']:.6f} | "
                f"{row['cost_s']:.9f} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _log_cost_table(self) -> None:
        if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return
        rows = self._cost_table_rows()
        if not rows:
            logger.warning("VerifyAdaptiveController: empty cost table.")
            return
        logger.info("D-Cut verifier cost table (Qwen3.5 GDN graph):\n%s", self._format_cost_table_markdown(rows))

    def _dump_cost_table_if_requested(self) -> None:
        dump_path = os.getenv("VLLM_DCUT_COST_TABLE_OUT") or self.config.cost_table_dump_path
        markdown_path = os.getenv("VLLM_DCUT_COST_TABLE_MD_OUT") or self.config.cost_table_markdown_path
        if not dump_path and not markdown_path:
            return
        if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
            return
        rows = self._cost_table_rows()
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
        if dump_path:
            dirname = os.path.dirname(dump_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            tmp_path = f"{dump_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, dump_path)
            logger.info("VerifyAdaptiveController: dumped JSON cost table to %s", dump_path)
        if markdown_path:
            dirname = os.path.dirname(markdown_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            tmp_path = f"{markdown_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(self._format_cost_table_markdown(rows))
            os.replace(tmp_path, markdown_path)
            logger.info("VerifyAdaptiveController: dumped Markdown cost table to %s", markdown_path)


def _ceil_lookup(val: int, sorted_keys: list[int]) -> int:
    idx = bisect.bisect_left(sorted_keys, val)
    return sorted_keys[min(idx, len(sorted_keys) - 1)]

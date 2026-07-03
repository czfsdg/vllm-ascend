# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin entry point for D-Cut planning."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import time
from functools import wraps
from numbers import Real
from types import ModuleType

from dcut.config import load_dcut_config
from dcut.cost_table import DcutCostTable
from dcut.policy import DcutPolicy

logger = logging.getLogger("dcut")


def _synchronize_npu_for_profile() -> None:
    """Synchronize async NPU work before reading wall-clock profile timers."""
    if importlib.util.find_spec("torch") is None:
        return
    torch = importlib.import_module("torch")
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "synchronize"):
        npu.synchronize()


def _visible_log(level: str, message: str, *args) -> None:
    formatted = message % args if args else message
    print(formatted, flush=True)
    log_method = getattr(logger, level)
    log_method(message, *args)


_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
ACCEPTANCE_EMA_ALPHA = 0.2
UNKNOWN_ACCEPTANCE_RATE = 0.0
METRICS_FILE_NAME = f"dcut_acceptance_metrics_{os.getuid()}.json"
METRICS_FILE_PATH = os.path.join(tempfile.gettempdir(), METRICS_FILE_NAME)
SPEC_DECODING_METRICS_PATTERN = re.compile(
    r"SpecDecoding metrics:.*Accepted:\s*([0-9,]+)\s*tokens,"
    r"\s*Drafted:\s*([0-9,]+)\s*tokens"
)
ACCEPTED_COUNTER_NAMES = (
    "num_accepted_tokens",
    "accepted_tokens",
    "accepted",
    "total_accepted_tokens",
)
DRAFTED_COUNTER_NAMES = (
    "num_draft_tokens",
    "num_drafted_tokens",
    "drafted_tokens",
    "drafted",
    "total_draft_tokens",
)
ACCEPTANCE_COUNTER_ROOTS = (
    "spec_decode_metrics",
    "spec_decode_stats",
    "speculative_decode_metrics",
    "metrics",
)
_PATCHED = False
_IMPORT_HOOK_INSTALLED = False
_METRICS_LOG_HANDLER_INSTALLED = False
_ORIGINAL_IMPORT = builtins.__import__


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _batch_size_from_runner(runner) -> int:
    input_batch = getattr(runner, "input_batch", None)
    return int(getattr(input_batch, "num_reqs", 1) or 1)


def _num_speculative_tokens(runner) -> int:
    speculative_config = getattr(runner, "speculative_config", None)
    return int(getattr(speculative_config, "num_speculative_tokens", 1) or 1)


def _spec_method(runner) -> str:
    speculative_config = getattr(runner, "speculative_config", None)
    return str(getattr(speculative_config, "method", "") or "")


def _should_use_target_only(runner) -> bool:
    if not getattr(runner, "dcut_accuracy_safe_mode", True):
        return False
    target_only_methods = getattr(runner, "dcut_target_only_methods", ("dflash",))
    return _spec_method(runner) in target_only_methods


def _read_field(container, name: str):
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    # Avoid tensor.item() here because this runs in the decode path and a device
    # tensor conversion would force a CPU/NPU synchronization.
    return None


def _find_first_counter(container, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _to_float(_read_field(container, name))
        if value is not None:
            return value
    return None


def _write_acceptance_metrics(accepted: float, drafted: float, source: str) -> None:
    if drafted <= 0:
        return
    payload = {
        "accepted": accepted,
        "drafted": drafted,
        "acceptance_rate": accepted / drafted,
        "source": source,
        "timestamp": time.time(),
    }
    tmp_path = f"{METRICS_FILE_PATH}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file)
    os.replace(tmp_path, METRICS_FILE_PATH)


def _read_acceptance_metrics_file() -> tuple[float, str, float, float] | None:
    try:
        with open(METRICS_FILE_PATH, encoding="utf-8") as metrics_file:
            payload = json.load(metrics_file)
    except (OSError, json.JSONDecodeError):
        return None

    acceptance_rate = _to_float(payload.get("acceptance_rate"))
    if acceptance_rate is None:
        accepted = _to_float(payload.get("accepted"))
        drafted = _to_float(payload.get("drafted"))
        if accepted is None or drafted is None or drafted <= 0:
            return None
        acceptance_rate = accepted / drafted
    accepted = _to_float(payload.get("accepted"))
    drafted = _to_float(payload.get("drafted"))
    return (
        min(max(acceptance_rate, 0.0), 1.0),
        str(payload.get("source", "metrics_file")),
        accepted if accepted is not None else -1.0,
        drafted if drafted is not None else -1.0,
    )


class _DcutMetricsLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = SPEC_DECODING_METRICS_PATTERN.search(message)
        if match is None:
            return
        accepted = float(match.group(1).replace(",", ""))
        drafted = float(match.group(2).replace(",", ""))
        try:
            _write_acceptance_metrics(
                accepted,
                drafted,
                "spec_decoding_metrics_log",
            )
        except OSError:
            self.handleError(record)


def _install_metrics_log_bridge() -> None:
    global _METRICS_LOG_HANDLER_INSTALLED
    if _METRICS_LOG_HANDLER_INSTALLED:
        return
    handler = _DcutMetricsLogHandler()
    handler.setLevel(logging.INFO)
    for logger_name in ("", "vllm", "vllm.v1", "vllm.v1.metrics"):
        logging.getLogger(logger_name).addHandler(handler)
    _METRICS_LOG_HANDLER_INSTALLED = True


def _read_acceptance_counters(runner) -> tuple[float, float] | None:
    for root_name in ACCEPTANCE_COUNTER_ROOTS:
        root = _read_field(runner, root_name)
        if root is None:
            continue
        accepted = _find_first_counter(root, ACCEPTED_COUNTER_NAMES)
        drafted = _find_first_counter(root, DRAFTED_COUNTER_NAMES)
        if accepted is not None and drafted is not None and drafted > 0:
            return accepted, drafted

    accepted = _find_first_counter(runner, ACCEPTED_COUNTER_NAMES)
    drafted = _find_first_counter(runner, DRAFTED_COUNTER_NAMES)
    if accepted is not None and drafted is not None and drafted > 0:
        return accepted, drafted
    return None


def _counts_from_sampled_token_lists(sampled_token_ids) -> list[int] | None:
    if not isinstance(sampled_token_ids, list):
        return None
    if not sampled_token_ids:
        return None
    counts: list[int] = []
    for sampled_tokens in sampled_token_ids:
        if isinstance(sampled_tokens, list):
            counts.append(len(sampled_tokens))
    return counts or None


def _counts_from_runner_cpu_buffer(runner) -> list[int] | None:
    counts_cpu = getattr(runner, "valid_sampled_token_count_cpu", None)
    if counts_cpu is None:
        return None
    event = getattr(runner, "valid_sampled_token_count_event", None)
    if event is not None:
        event.synchronize()
    if str(getattr(counts_cpu, "device", "")) != "cpu":
        return None
    num_reqs = _batch_size_from_runner(runner)
    counts = counts_cpu[:num_reqs].tolist()
    return [int(count) for count in counts]


def _acceptance_stats_from_sampled_counts(
    counts: list[int] | None,
    requested_len: int,
) -> tuple[float, int, int, list[float], int] | None:
    if not counts or requested_len <= 0:
        return None

    # valid_sampled_token_ids includes accepted draft tokens plus the
    # target-model bonus token. D-Cut should plan with effective output tokens
    # per verification step, so include the bonus token in this acceptance rate
    # and keep the draft-only count only for debugging against vLLM metrics.
    accepted_draft_by_request = [max(count - 1, 0) for count in counts]
    effective_by_request = [min(max(count, 0), requested_len) for count in counts]
    effective_tokens = sum(effective_by_request)
    accepted_draft_tokens = sum(accepted_draft_by_request)
    drafted = requested_len * len(counts)
    if drafted <= 0:
        return None
    per_request_acceptance = [
        min(max(effective_count / requested_len, 0.0), 1.0) for effective_count in effective_by_request
    ]
    return (
        min(max(effective_tokens / drafted, 0.0), 1.0),
        effective_tokens,
        drafted,
        per_request_acceptance,
        accepted_draft_tokens,
    )


def _set_acceptance_stats(
    runner,
    accepted: float | int,
    drafted: float | int,
    per_request_acceptance: list[float] | None = None,
    accepted_draft_tokens: float | int | None = None,
) -> None:
    runner.dcut_last_accepted_tokens = accepted
    runner.dcut_last_drafted_tokens = drafted
    runner.dcut_last_per_request_acceptance = per_request_acceptance
    runner.dcut_last_accepted_draft_tokens = accepted_draft_tokens


def _acceptance_rate_from_runner(
    runner,
    sampled_token_ids=None,
    requested_len: int = 1,
) -> tuple[float | None, str]:
    pending_rate = getattr(runner, "dcut_pending_acceptance_rate", None)
    if pending_rate is not None:
        runner.dcut_pending_acceptance_rate = None
        runner.dcut_observed_acceptance_rate = pending_rate
        return pending_rate, getattr(
            runner,
            "dcut_pending_acceptance_source",
            "bookkeeping_valid_sampled_tokens",
        )

    sampled_counts = _counts_from_sampled_token_lists(sampled_token_ids)
    sampled_stats = _acceptance_stats_from_sampled_counts(sampled_counts, requested_len)
    if sampled_stats is not None:
        sampled_rate, accepted, drafted, per_request_acceptance, accepted_draft_tokens = sampled_stats
        runner.dcut_observed_acceptance_rate = sampled_rate
        _set_acceptance_stats(
            runner,
            accepted,
            drafted,
            per_request_acceptance,
            accepted_draft_tokens,
        )
        return sampled_rate, "sampled_token_lengths"

    cpu_counts = _counts_from_runner_cpu_buffer(runner)
    cpu_stats = _acceptance_stats_from_sampled_counts(cpu_counts, requested_len)
    if cpu_stats is not None:
        cpu_rate, accepted, drafted, per_request_acceptance, accepted_draft_tokens = cpu_stats
        runner.dcut_observed_acceptance_rate = cpu_rate
        _set_acceptance_stats(
            runner,
            accepted,
            drafted,
            per_request_acceptance,
            accepted_draft_tokens,
        )
        return cpu_rate, "valid_sampled_token_count_cpu"

    counters = _read_acceptance_counters(runner)
    if counters is not None:
        accepted, drafted = counters
        previous_counters = getattr(runner, "dcut_acceptance_counter_snapshot", None)
        runner.dcut_acceptance_counter_snapshot = counters
        if previous_counters is not None:
            accepted_delta = accepted - previous_counters[0]
            drafted_delta = drafted - previous_counters[1]
            if accepted_delta >= 0 and drafted_delta > 0:
                raw_rate = accepted_delta / drafted_delta
                previous_rate = getattr(runner, "dcut_observed_acceptance_rate", raw_rate)
                smoothed_rate = ACCEPTANCE_EMA_ALPHA * raw_rate + (1.0 - ACCEPTANCE_EMA_ALPHA) * previous_rate
                runner.dcut_observed_acceptance_rate = smoothed_rate
                _set_acceptance_stats(runner, accepted_delta, drafted_delta)
                return smoothed_rate, "runtime_counter_delta"

        raw_rate = accepted / drafted
        runner.dcut_observed_acceptance_rate = raw_rate
        _set_acceptance_stats(runner, accepted, drafted)
        return raw_rate, "runtime_counters"

    metrics_file_rate = _read_acceptance_metrics_file()
    if metrics_file_rate is not None:
        rate, source, accepted, drafted = metrics_file_rate
        runner.dcut_observed_acceptance_rate = rate
        _set_acceptance_stats(runner, accepted, drafted)
        return rate, source

    _set_acceptance_stats(runner, -1, -1)
    return UNKNOWN_ACCEPTANCE_RATE, "unavailable"


def _log_cost_table_prediction(cost_table: DcutCostTable, config_source: str) -> None:
    _visible_log(
        "info",
        "[dcut][cost-table][prediction] source=analytic config=%s "
        "max_verify_len=%d target_base_cost=%.3f target_token_cost=%.3f "
        "draft_token_cost=%.3f predicted_table=[%s]",
        config_source or "<defaults>",
        cost_table.max_verify_len,
        cost_table.target_base_cost,
        cost_table.target_token_cost,
        cost_table.draft_token_cost,
        cost_table.summary(),
    )


def _log_cost_table_warmup(cost_table: DcutCostTable) -> None:
    warmed_entries = cost_table.warmup()
    _visible_log(
        "info",
        "[dcut][cost-table][warmup] entries=%d warmed_table=[%s]",
        len(warmed_entries),
        cost_table.summary(),
    )


def _log_cost_table_profile(runner, verify_len: int) -> None:
    cost_table = getattr(runner, "dcut_cost_table", None)
    if cost_table is None:
        return
    entry = cost_table.get(verify_len)
    _visible_log(
        "info",
        "[dcut][cost-table][profile] source=npu_runtime verify_len=%d "
        "target_ms=%.3f draft_ms=%.3f total_ms=%.3f profiled_lens=%s "
        "profiled_table=[%s]",
        entry.verify_len,
        entry.target_cost,
        entry.draft_cost,
        entry.total_cost,
        sorted(cost_table.profiled_lens),
        cost_table.summary(),
    )


def _format_batch_cut_plan(runner, decision) -> str:
    per_request_acceptance = getattr(runner, "dcut_last_per_request_acceptance", None)
    if per_request_acceptance:
        plan_items = [
            f"#{index}:accept={acceptance:.3f},cut={decision.selected_len}"
            for index, acceptance in enumerate(per_request_acceptance)
        ]
        return "[" + ",".join(plan_items) + "]"
    return f"[batch_size={decision.batch_size},cut={decision.selected_len}]"


def _format_candidate_scores(runner, decision) -> str:
    cost_table = getattr(runner, "dcut_cost_table", None)
    if cost_table is None:
        return "[]"
    batch_size = max(decision.batch_size, 1)
    high_concurrency_batch = max(
        getattr(getattr(runner, "dcut_policy", None), "high_concurrency_batch", 1),
        1,
    )
    concurrency = max(batch_size - 1, 0) / high_concurrency_batch
    items = []
    for verify_len in range(1, decision.requested_len + 1):
        cost = cost_table.get(verify_len)
        expected_accepts = max(1.0 + max(verify_len - 1, 0) * decision.acceptance_rate, 1e-6)
        length_penalty = 1.0 + concurrency * max(verify_len - 1, 0) / decision.requested_len
        score = cost.total_cost * length_penalty / expected_accepts
        marker = "*" if verify_len == decision.selected_len else ""
        items.append(f"k={verify_len}:score={score:.3f}{marker}")
    return "[" + ",".join(items) + "]"


def _decision_signature(decision) -> tuple[int, int, int, float, str]:
    return (
        decision.requested_len,
        decision.selected_len,
        decision.batch_size,
        round(decision.acceptance_rate, 6),
        decision.reason,
    )


def _log_dcut_plan(runner, decision, acceptance_source: str) -> None:
    plan_count = int(getattr(runner, "dcut_plan_count", 0)) + 1
    runner.dcut_plan_count = plan_count
    runner.dcut_decision_count = plan_count
    runner.dcut_last_logged_decision_signature = _decision_signature(decision)
    _visible_log(
        "info",
        "[dcut][plan] requested_len=%d selected_len=%d batch_size=%d "
        "acceptance_basis=%.3f acceptance_source=%s score=%.6f reason=%s "
        "batch_dcut_plan=%s candidate_scores=%s mode=plan_only plan_count=%d",
        decision.requested_len,
        decision.selected_len,
        decision.batch_size,
        decision.acceptance_rate,
        acceptance_source,
        decision.score,
        decision.reason,
        _format_batch_cut_plan(runner, decision),
        _format_candidate_scores(runner, decision),
        plan_count,
    )


def _log_acceptance_result(runner, acceptance_source: str) -> None:
    result_count = int(getattr(runner, "dcut_result_count", 0)) + 1
    runner.dcut_result_count = result_count
    _visible_log(
        "info",
        "[dcut][result] acceptance_source=%s acceptance_rate=%.3f "
        "effective_tokens=%s drafted_tokens=%s accepted_draft_tokens=%s "
        "per_request_acceptance=%s verify_elapsed_ms=%s draft_elapsed_ms=%s "
        "target_elapsed_ms=%s planned_cut=%s result_count=%d",
        acceptance_source,
        getattr(runner, "dcut_observed_acceptance_rate", UNKNOWN_ACCEPTANCE_RATE),
        getattr(runner, "dcut_last_accepted_tokens", "unknown"),
        getattr(runner, "dcut_last_drafted_tokens", "unknown"),
        getattr(runner, "dcut_last_accepted_draft_tokens", None),
        getattr(runner, "dcut_last_per_request_acceptance", None),
        getattr(runner, "dcut_last_verify_elapsed_ms", None),
        getattr(runner, "dcut_last_draft_elapsed_ms", None),
        getattr(runner, "dcut_last_target_elapsed_ms", None),
        getattr(runner, "dcut_last_planned_cut", None),
        result_count,
    )


def _patch_runner_class(npu_model_runner: type) -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_init = npu_model_runner.__init__
    original_propose = npu_model_runner.propose_draft_token_ids
    original_bookkeeping = getattr(npu_model_runner, "_bookkeeping_sync", None)

    @wraps(original_init)
    def init_with_dcut(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not _env_flag("DCUT_ENABLE", "1"):
            return
        config, config_source = load_dcut_config()
        max_verify_len = _num_speculative_tokens(self)
        self.dcut_cost_table = DcutCostTable(
            max_verify_len=max_verify_len,
            target_base_cost=config.target_base_cost,
            target_token_cost=config.target_token_cost,
            draft_token_cost=config.draft_token_cost,
        )
        _log_cost_table_prediction(self.dcut_cost_table, config_source)
        _log_cost_table_warmup(self.dcut_cost_table)
        self.dcut_policy = DcutPolicy(
            self.dcut_cost_table,
            default_acceptance_rate=UNKNOWN_ACCEPTANCE_RATE,
            high_concurrency_batch=config.high_concurrency_batch,
        )
        self.dcut_accuracy_safe_mode = config.accuracy_safe_mode
        self.dcut_target_only_methods = config.target_only_methods
        self.dcut_observed_acceptance_rate = None
        self.dcut_acceptance_counter_snapshot = None
        self.dcut_decision_count = 0
        self.dcut_plan_count = 0
        self.dcut_result_count = 0
        self.dcut_last_logged_decision_signature = None
        self.dcut_pending_acceptance_rate = None
        self.dcut_pending_acceptance_source = None
        self.dcut_last_per_request_acceptance = None
        self.dcut_last_accepted_draft_tokens = None
        self.dcut_verify_start_time = None
        self.dcut_last_verify_elapsed_ms = None
        self.dcut_last_draft_elapsed_ms = None
        self.dcut_last_target_elapsed_ms = None
        self.dcut_last_planned_cut = None
        _visible_log(
            "info",
            "[dcut][cost-table] initialized: enabled=%s config=%s max_verify_len=%d "
            "accuracy_safe_mode=%s target_only_methods=%s table=[%s]",
            True,
            config_source or "<defaults>",
            max_verify_len,
            config.accuracy_safe_mode,
            config.target_only_methods,
            self.dcut_cost_table.summary(),
        )

    @wraps(original_propose)
    def propose_with_dcut(self, *args, **kwargs):
        policy = getattr(self, "dcut_policy", None)
        if policy is not None and _env_flag("DCUT_ENABLE", "1"):
            requested_len = _num_speculative_tokens(self)
            sampled_token_ids = args[0] if args else kwargs.get("sampled_token_ids")
            acceptance_rate, acceptance_source = _acceptance_rate_from_runner(
                self,
                sampled_token_ids=sampled_token_ids,
                requested_len=requested_len,
            )
            if acceptance_source in {
                "sampled_token_lengths",
                "valid_sampled_token_count_cpu",
            }:
                _log_acceptance_result(self, acceptance_source)
            decision = policy.decide(
                requested_len=requested_len,
                batch_size=_batch_size_from_runner(self),
                acceptance_rate=acceptance_rate,
            )
            self.dcut_last_decision = decision
            _log_dcut_plan(self, decision, acceptance_source)
            _synchronize_npu_for_profile()
            self.dcut_verify_start_time = time.perf_counter()
            self.dcut_last_planned_cut = decision.selected_len
            if _should_use_target_only(self):
                if not getattr(self, "dcut_target_only_warning_emitted", False):
                    _visible_log(
                        "warning",
                        "[dcut] accuracy-safe target-only fallback is active for "
                        "speculative method=%s. Returning no draft tokens to preserve "
                        "target-model output quality. Set DCUT_ACCURACY_SAFE_MODE=0 "
                        "after DFlash acceptance/accuracy is verified.",
                        _spec_method(self),
                    )
                    self.dcut_target_only_warning_emitted = True
                self.dcut_last_draft_elapsed_ms = 0.0
                return None
            _synchronize_npu_for_profile()
            draft_start_time = time.perf_counter()
            proposed = original_propose(self, *args, **kwargs)
            _synchronize_npu_for_profile()
            self.dcut_last_draft_elapsed_ms = round(
                (time.perf_counter() - draft_start_time) * 1000.0,
                3,
            )
            return proposed
        return original_propose(self, *args, **kwargs)

    if callable(original_bookkeeping):

        @wraps(original_bookkeeping)
        def bookkeeping_with_dcut(self, *args, **kwargs):
            result = original_bookkeeping(self, *args, **kwargs)
            policy = getattr(self, "dcut_policy", None)
            if policy is None or not _env_flag("DCUT_ENABLE", "1"):
                return result
            if not isinstance(result, tuple) or len(result) < 2:
                return result

            valid_sampled_token_ids = result[1]
            requested_len = _num_speculative_tokens(self)
            counts = _counts_from_sampled_token_lists(valid_sampled_token_ids)
            stats = _acceptance_stats_from_sampled_counts(counts, requested_len)
            if stats is None:
                return result

            rate, accepted, drafted, per_request_acceptance, accepted_draft_tokens = stats
            self.dcut_pending_acceptance_rate = rate
            self.dcut_pending_acceptance_source = "bookkeeping_valid_sampled_tokens"
            self.dcut_observed_acceptance_rate = rate
            _set_acceptance_stats(
                self,
                accepted,
                drafted,
                per_request_acceptance,
                accepted_draft_tokens,
            )
            verify_start_time = getattr(self, "dcut_verify_start_time", None)
            if verify_start_time is not None:
                _synchronize_npu_for_profile()
                self.dcut_last_verify_elapsed_ms = round(
                    (time.perf_counter() - verify_start_time) * 1000.0,
                    3,
                )
                self.dcut_verify_start_time = None
                draft_elapsed_ms = getattr(self, "dcut_last_draft_elapsed_ms", 0.0) or 0.0
                self.dcut_last_target_elapsed_ms = round(
                    max(self.dcut_last_verify_elapsed_ms - draft_elapsed_ms, 0.0),
                    3,
                )
                planned_cut = getattr(self, "dcut_last_planned_cut", None)
                if planned_cut is not None:
                    self.dcut_cost_table.update_profile(
                        planned_cut,
                        self.dcut_last_target_elapsed_ms,
                        draft_elapsed_ms,
                    )
                    _log_cost_table_profile(self, planned_cut)
            _log_acceptance_result(self, "bookkeeping_valid_sampled_tokens")
            return result

        npu_model_runner._bookkeeping_sync = bookkeeping_with_dcut

    npu_model_runner.__init__ = init_with_dcut
    npu_model_runner.propose_draft_token_ids = propose_with_dcut
    _PATCHED = True
    _visible_log("info", "[dcut] NPUModelRunner patch installed")


def _maybe_patch_loaded_runner(module: ModuleType | None = None) -> bool:
    runner_module = module or sys.modules.get(_RUNNER_MODULE)
    npu_model_runner = getattr(runner_module, "NPUModelRunner", None)
    if npu_model_runner is None:
        return False
    _patch_runner_class(npu_model_runner)
    return True


def _install_import_hook() -> None:
    global _IMPORT_HOOK_INSTALLED
    if _IMPORT_HOOK_INSTALLED:
        return

    def dcut_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        if name == _RUNNER_MODULE or name.startswith(f"{_RUNNER_MODULE}."):
            _maybe_patch_loaded_runner(sys.modules.get(_RUNNER_MODULE))
        return module

    builtins.__import__ = dcut_import
    _IMPORT_HOOK_INSTALLED = True


def register() -> None:
    """Register D-Cut as a vLLM general plugin.

    vLLM loads general plugins while building CLI arguments, before importing
    the Ascend model runner is safe. Therefore registration only installs an
    import hook and patches ``NPUModelRunner`` after vLLM imports it normally.
    """

    if not _env_flag("DCUT_ENABLE", "1"):
        _visible_log("info", "[dcut] plugin disabled by DCUT_ENABLE=0")
        return
    _install_metrics_log_bridge()
    if not _maybe_patch_loaded_runner():
        if _IMPORT_HOOK_INSTALLED:
            return
        _install_import_hook()
        _visible_log("info", "[dcut] plugin registered: waiting_for=%s", _RUNNER_MODULE)

# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin entry point for D-Cut planning."""

from __future__ import annotations

import builtins
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


def _read_acceptance_metrics_file() -> tuple[float, str] | None:
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
    return min(max(acceptance_rate, 0.0), 1.0), str(payload.get("source", "metrics_file"))


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


def _acceptance_rate_from_runner(runner) -> tuple[float | None, str]:
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
                return smoothed_rate, "runtime_counter_delta"

        raw_rate = accepted / drafted
        runner.dcut_observed_acceptance_rate = raw_rate
        return raw_rate, "runtime_counters"

    metrics_file_rate = _read_acceptance_metrics_file()
    if metrics_file_rate is not None:
        runner.dcut_observed_acceptance_rate = metrics_file_rate[0]
        return metrics_file_rate

    return UNKNOWN_ACCEPTANCE_RATE, "unavailable"


def _decision_signature(decision) -> tuple[int, int, int, float, str]:
    return (
        decision.requested_len,
        decision.selected_len,
        decision.batch_size,
        round(decision.acceptance_rate, 6),
        decision.reason,
    )


def _log_cut_decision(runner, decision, acceptance_source: str) -> None:
    decision_count = int(getattr(runner, "dcut_decision_count", 0)) + 1
    runner.dcut_decision_count = decision_count
    runner.dcut_last_logged_decision_signature = _decision_signature(decision)
    _visible_log(
        "info",
        "[dcut] cut-policy decision: requested_len=%d selected_len=%d "
        "batch_size=%d acceptance_rate=%.3f score=%.6f reason=%s "
        "acceptance_source=%s mode=plan_only repeat_count=%d",
        decision.requested_len,
        decision.selected_len,
        decision.batch_size,
        decision.acceptance_rate,
        decision.score,
        decision.reason,
        acceptance_source,
        decision_count,
    )


def _patch_runner_class(npu_model_runner: type) -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_init = npu_model_runner.__init__
    original_propose = npu_model_runner.propose_draft_token_ids

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
        self.dcut_last_logged_decision_signature = None
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
            acceptance_rate, acceptance_source = _acceptance_rate_from_runner(self)
            decision = policy.decide(
                requested_len=requested_len,
                batch_size=_batch_size_from_runner(self),
                acceptance_rate=acceptance_rate,
            )
            self.dcut_last_decision = decision
            _log_cut_decision(self, decision, acceptance_source)
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
                return None
        return original_propose(self, *args, **kwargs)

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

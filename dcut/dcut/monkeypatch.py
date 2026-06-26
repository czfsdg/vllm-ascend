# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import os
import sys
import time
from dataclasses import replace
from functools import wraps
from typing import Any

from vllm.logger import init_logger

from dcut.verify_adaptive_config import VerifyAdaptiveConfig
from dcut.verify_adaptive_controller import choose_query_lens_discrete, make_cost_lookup

logger = init_logger(__name__)


def _emit_dcut_log(message: str, *args: Any) -> None:
    rendered = message % args if args else message
    logger.info(rendered)
    print(f"[DCUT] {rendered}", flush=True)


_INSTALLED = False
_IMPORT_HOOK_INSTALLED = False
_ORIGINAL_IMPORT = builtins.__import__
_MODEL_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_PROPOSER_MODULE = "vllm_ascend.spec_decode.llm_base_proposer"
_CONFIG_ENV_NAMES = ("VLLM_DCUT_CONFIG", "VLLM_ASCEND_DCUT_CONFIG")


def _get_config_path() -> str | None:
    for env_name in _CONFIG_ENV_NAMES:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def _load_config() -> VerifyAdaptiveConfig | None:
    config_path = _get_config_path()
    if not config_path:
        _emit_dcut_log(
            "D-Cut adaptive verify dormant: set one of %s to a config JSON to enable.",
            ", ".join(_CONFIG_ENV_NAMES),
        )
        return None
    try:
        config = VerifyAdaptiveConfig.from_file(config_path)
    except Exception:
        logger.exception("Failed to load D-Cut config from %s; adaptive verify is disabled.", config_path)
        return None
    if not config.enabled:
        _emit_dcut_log("D-Cut adaptive verify dormant: config disables it.")
        return None
    return config


def _is_supported_runner(runner: Any) -> bool:
    speculative_config = getattr(runner, "speculative_config", None)
    if speculative_config is None:
        return False
    method = getattr(speculative_config, "method", None)
    parallel_drafting = bool(getattr(speculative_config, "parallel_drafting", False))
    return method == "dflash" or (method == "draft_model" and parallel_drafting)


def _ensure_runner_state(runner: Any) -> bool:
    if getattr(runner, "_dcut_state_initialized", False):
        return bool(getattr(runner, "dcut_adaptive_enabled", False))

    runner._dcut_state_initialized = True
    runner.dcut_adaptive_enabled = False
    runner.dcut_config = None
    runner.dcut_next_draft_lens = {}
    runner.dcut_logged_first_plan = False
    runner.dcut_logged_first_truncation = False
    runner.dcut_logged_safe_mode = False
    runner.dcut_last_concurrency_log_ts = 0.0

    config = _load_config()
    if config is None:
        return False
    if not _is_supported_runner(runner):
        speculative_config = getattr(runner, "speculative_config", None)
        _emit_dcut_log(
            "D-Cut adaptive verify dormant: method=%s parallel_drafting=%s model_type=%s is unsupported.",
            getattr(speculative_config, "method", None),
            bool(getattr(speculative_config, "parallel_drafting", False)),
            getattr(getattr(getattr(runner, "model_config", None), "hf_config", None), "model_type", None),
        )
        return False
    if getattr(runner, "use_async_scheduling", False):
        logger.warning("D-Cut adaptive verify is enabled with async scheduling; probs bookkeeping may be skipped.")

    runner.dcut_config = config
    runner.dcut_adaptive_enabled = True
    _emit_dcut_log("D-Cut adaptive verify ENABLED (config=%s)", config.to_log_dict())
    return True


def _should_log_runtime_events(runner: Any) -> bool:
    config = getattr(runner, "dcut_config", None)
    return bool(getattr(config, "log_runtime_events", True))


def _get_concurrency_log_interval(runner: Any) -> float:
    config = getattr(runner, "dcut_config", None)
    if config is not None:
        return float(getattr(config, "log_concurrency_interval_s", 5.0))
    return float(os.getenv("VLLM_DCUT_LOG_CONCURRENCY_INTERVAL_S", "5.0"))


def _log_concurrency(runner: Any, scheduler_output: Any) -> None:
    if not _should_log_runtime_events(runner):
        return
    interval_s = _get_concurrency_log_interval(runner)
    if interval_s <= 0:
        return
    now = time.monotonic()
    last_log_ts = float(getattr(runner, "dcut_last_concurrency_log_ts", 0.0))
    if now - last_log_ts < interval_s:
        return
    runner.dcut_last_concurrency_log_ts = now

    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
    scheduled_spec_decode_tokens = getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {}
    active_reqs = int(getattr(getattr(runner, "input_batch", None), "num_reqs", 0) or 0)
    scheduled_reqs = len(num_scheduled_tokens)
    spec_reqs = len(scheduled_spec_decode_tokens)
    total_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    max_scheduled_tokens = max(num_scheduled_tokens.values(), default=0)
    _emit_dcut_log(
        "D-Cut concurrency: active_reqs=%d scheduled_reqs=%d spec_reqs=%d "
        "total_scheduled_tokens=%d max_scheduled_tokens_per_req=%d",
        active_reqs,
        scheduled_reqs,
        spec_reqs,
        total_scheduled_tokens,
        max_scheduled_tokens,
    )


def _min_configured_draft_len(runner: Any) -> int:
    config = getattr(runner, "dcut_config", None)
    return max(0, int(getattr(config, "min_adaptive_draft_len", 2)))


def _min_safe_draft_len(runner: Any, req_id: Any) -> int:
    input_batch = getattr(runner, "input_batch", None)
    req_id_to_index = getattr(input_batch, "req_id_to_index", {}) or {}
    req_idx = req_id_to_index.get(req_id)
    if req_idx is None:
        return 0
    accepted_tokens = getattr(input_batch, "num_accepted_tokens_cpu", None)
    if accepted_tokens is None:
        return 0
    try:
        return max(0, int(accepted_tokens[req_idx]) - 1)
    except Exception:
        return 0


def _scheduled_cached_req_ids(scheduler_output: Any) -> list[Any]:
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    return list(getattr(cached, "req_ids", []) or [])


def _normalize_scheduled_token_counts(
    scheduler_output: Any,
    scheduled_spec_decode_tokens: dict[Any, list[int]],
    num_scheduled_tokens: dict[Any, int],
) -> tuple[dict[Any, int], bool]:
    normalized = dict(num_scheduled_tokens)
    changed = False
    for req_id, draft_token_ids in scheduled_spec_decode_tokens.items():
        expected = len(draft_token_ids) + 1
        if normalized.get(req_id) != expected:
            normalized[req_id] = expected
            changed = True
    for req_id in _scheduled_cached_req_ids(scheduler_output):
        current = int(normalized.get(req_id, 0))
        if current <= 0:
            normalized[req_id] = 1
            changed = True
    return normalized, changed


def _update_scheduler_output(scheduler_output: Any, **updates: Any) -> Any:
    try:
        for name, value in updates.items():
            setattr(scheduler_output, name, value)
        return scheduler_output
    except Exception:
        return replace(scheduler_output, **updates)


def _apply_dcut_draft_lens(runner: Any, scheduler_output: Any) -> Any:
    if not _ensure_runner_state(runner) or not runner.dcut_next_draft_lens:
        return scheduler_output
    if not getattr(getattr(runner, "dcut_config", None), "apply_adaptive_lengths", False):
        return scheduler_output
    scheduled = scheduler_output.scheduled_spec_decode_tokens
    if not scheduled:
        return scheduler_output

    updated = {}
    updated_num_scheduled_tokens = scheduler_output.num_scheduled_tokens.copy()
    changed = False
    total_removed = 0
    for req_id, draft_token_ids in scheduled.items():
        target_len = runner.dcut_next_draft_lens.get(req_id)
        if target_len is None:
            updated[req_id] = draft_token_ids
            new_num_scheduled_tokens = len(draft_token_ids) + 1
            changed = changed or updated_num_scheduled_tokens.get(req_id) != new_num_scheduled_tokens
            updated_num_scheduled_tokens[req_id] = new_num_scheduled_tokens
            continue
        original_len = len(draft_token_ids)
        min_safe_len = max(_min_safe_draft_len(runner, req_id), _min_configured_draft_len(runner))
        target_len = max(min_safe_len, min(int(target_len), original_len))
        removed = original_len - target_len
        if target_len > 0:
            updated[req_id] = draft_token_ids[:target_len]
            new_num_scheduled_tokens = target_len + 1
        else:
            new_num_scheduled_tokens = 1
        changed = changed or removed > 0 or updated_num_scheduled_tokens.get(req_id) != new_num_scheduled_tokens
        updated_num_scheduled_tokens[req_id] = new_num_scheduled_tokens
        total_removed += removed
    runner.dcut_next_draft_lens = {}
    if not changed:
        return scheduler_output
    applied_draft_lens = [len(draft_token_ids) for draft_token_ids in updated.values()]
    if _should_log_runtime_events(runner):
        _emit_dcut_log(
            "D-Cut apply: requests=%d removed_tokens=%d applied_draft_lens_hist=%s",
            len(updated),
            total_removed,
            _format_draft_lens_hist(applied_draft_lens),
        )
        if not getattr(runner, "dcut_logged_first_truncation", False):
            _emit_dcut_log(
                "D-Cut adaptive verify ACTIVE: truncated scheduled draft tokens for the first time (requests=%d).",
                len(updated),
            )
            runner.dcut_logged_first_truncation = True
    logger.debug("D-Cut: truncated scheduled spec-decode tokens for %d requests.", len(updated))
    updated_num_scheduled_tokens, normalized_changed = _normalize_scheduled_token_counts(
        scheduler_output, updated, updated_num_scheduled_tokens
    )
    changed = changed or normalized_changed
    updated_total_num_scheduled_tokens = sum(updated_num_scheduled_tokens.values())
    if updated_total_num_scheduled_tokens <= 0:
        logger.warning("D-Cut: skip truncation because updated scheduled-token total is non-positive.")
        return scheduler_output
    return _update_scheduler_output(
        scheduler_output,
        scheduled_spec_decode_tokens=updated,
        num_scheduled_tokens=updated_num_scheduled_tokens,
        total_num_scheduled_tokens=updated_total_num_scheduled_tokens,
    )


def _record_selected_token_probs(proposer: Any, logits: Any, draft_token_ids: Any) -> None:
    runner = getattr(proposer, "runner", None)
    if runner is None or not _ensure_runner_state(runner):
        return
    if getattr(proposer, "method", None) != "dflash" and not getattr(proposer, "parallel_drafting", False):
        return
    try:
        import torch

        num_indices = min(logits.shape[0], draft_token_ids.numel())
        logits = logits[:num_indices]
        draft_token_ids = draft_token_ids[:num_indices].to(torch.long)
        probs = torch.softmax(logits.float(), dim=-1)
        selected_probs = probs.gather(dim=-1, index=draft_token_ids.view(-1, 1)).view(-1)
        proposer.latest_draft_token_probs = selected_probs.view(-1, proposer.num_speculative_tokens)
    except Exception:
        logger.exception("D-Cut: failed to record selected draft token probabilities.")


def _format_draft_lens_hist(draft_lens: list[int]) -> str:
    hist: dict[int, int] = {}
    for draft_len in draft_lens:
        hist[int(draft_len)] = hist.get(int(draft_len), 0) + 1
    return ",".join(f"{draft_len}:{count}" for draft_len, count in sorted(hist.items()))


def _update_dcut_next_draft_lens(runner: Any, draft_token_ids: Any) -> None:
    if not _ensure_runner_state(runner) or draft_token_ids is None:
        return
    if not getattr(getattr(runner, "dcut_config", None), "apply_adaptive_lengths", False):
        return
    drafter = getattr(runner, "drafter", None)
    drafter_probs = getattr(drafter, "latest_draft_token_probs", None)
    if drafter_probs is None:
        return
    try:
        probs_cpu = drafter_probs.detach().to("cpu").tolist()
    except Exception:
        logger.exception("D-Cut: failed to copy draft probabilities; falling back to vanilla DFlash.")
        runner.dcut_next_draft_lens = {}
        return

    req_ids = list(runner.input_batch.req_ids)
    probs_cpu = probs_cpu[: len(req_ids)]
    if not probs_cpu:
        runner.dcut_next_draft_lens = {}
        return
    max_draft_len = int(draft_token_ids.shape[1])
    base_batch_size = max(1, len(probs_cpu))
    q_levels = [
        base_batch_size + (query_len_per_req - 1) * base_batch_size
        for query_len_per_req in runner.dcut_config.query_len_levels(max_draft_len)
    ]
    result = choose_query_lens_discrete(
        probs=probs_cpu,
        base_batch_size=base_batch_size,
        q_levels=q_levels,
        cost_lookup=make_cost_lookup(runner.dcut_config.cost_table, base_batch_size),
        max_draft_len=max_draft_len,
        min_prefix_prob=runner.dcut_config.min_prefix_prob,
    )
    draft_lens = [int(draft_len) for draft_len in result["draft_lens"]]
    runner.dcut_next_draft_lens = {req_id: draft_len for req_id, draft_len in zip(req_ids, draft_lens, strict=False)}
    total_draft_lens = sum(draft_lens)
    max_draft_lens = max(draft_lens, default=0)
    min_draft_lens = min(draft_lens, default=0)
    avg_draft_lens = total_draft_lens / max(1, len(draft_lens))
    best_q = int(result["best_Q"])
    query_len_per_req = best_q / max(1, base_batch_size)
    if _should_log_runtime_events(runner):
        _emit_dcut_log(
            "D-Cut plan: batch=%d verify_query_tokens=%d query_len_per_req=%.2f "
            "draft_lens_total=%d draft_lens_avg=%.2f draft_lens_min=%d "
            "draft_lens_max=%d draft_lens_hist=%s",
            len(req_ids),
            best_q,
            query_len_per_req,
            total_draft_lens,
            avg_draft_lens,
            min_draft_lens,
            max_draft_lens,
            _format_draft_lens_hist(draft_lens),
        )
        if not getattr(runner, "dcut_logged_first_plan", False):
            _emit_dcut_log(
                "D-Cut adaptive verify ACTIVE: computed first adaptive draft-length "
                "plan (batch=%d, best_Q=%s, max_draft_len=%d).",
                len(req_ids),
                result["best_Q"],
                max_draft_len,
            )
            runner.dcut_logged_first_plan = True
    logger.debug("D-Cut: selected best_Q=%s draft_lens=%s", result["best_Q"], runner.dcut_next_draft_lens)


def _patch_runner_module(module: Any) -> bool:
    NPUModelRunner = getattr(module, "NPUModelRunner", None)
    if NPUModelRunner is None or getattr(NPUModelRunner, "_dcut_patched", False):
        return False

    original_execute_model = NPUModelRunner.execute_model
    original_propose_draft_token_ids = NPUModelRunner.propose_draft_token_ids

    @wraps(original_execute_model)
    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        _log_concurrency(self, scheduler_output)
        scheduler_output = _apply_dcut_draft_lens(self, scheduler_output)
        return original_execute_model(self, scheduler_output, *args, **kwargs)

    @wraps(original_propose_draft_token_ids)
    def propose_draft_token_ids(self: Any, *args: Any, **kwargs: Any) -> Any:
        draft_token_ids = original_propose_draft_token_ids(self, *args, **kwargs)
        _update_dcut_next_draft_lens(self, draft_token_ids)
        return draft_token_ids

    NPUModelRunner.execute_model = execute_model
    NPUModelRunner.propose_draft_token_ids = propose_draft_token_ids
    NPUModelRunner._dcut_patched = True
    _emit_dcut_log("D-Cut adaptive-verify patched NPUModelRunner.")
    return True


def _patch_proposer_module(module: Any) -> bool:
    AscendSpecDecodeBaseProposer = getattr(module, "AscendSpecDecodeBaseProposer", None)
    if AscendSpecDecodeBaseProposer is None or getattr(AscendSpecDecodeBaseProposer, "_dcut_patched", False):
        return False

    original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft

    @wraps(original_run_merged_draft)
    def _run_merged_draft(self: Any, *args: Any, **kwargs: Any) -> Any:
        runner = getattr(self, "runner", None)
        if (
            runner is None
            or not _ensure_runner_state(runner)
            or not getattr(getattr(runner, "dcut_config", None), "apply_adaptive_lengths", False)
        ):
            return original_run_merged_draft(self, *args, **kwargs)

        captured_logits = None
        logits_processor_hook = None
        original_compute_logits = getattr(self.model, "compute_logits", None)
        original_logits_processor = getattr(self.model, "logits_processor", None)

        def capture_logits(logits: Any) -> Any:
            nonlocal captured_logits
            captured_logits = logits
            return logits

        def compute_logits_wrapper(*inner_args: Any, **inner_kwargs: Any) -> Any:
            return capture_logits(original_compute_logits(*inner_args, **inner_kwargs))

        def logits_processor_wrapper(*inner_args: Any, **inner_kwargs: Any) -> Any:
            return capture_logits(original_logits_processor(*inner_args, **inner_kwargs))

        def logits_processor_hook_fn(module: Any, inputs: Any, output: Any) -> None:
            capture_logits(output)

        if original_compute_logits is not None:
            self.model.compute_logits = compute_logits_wrapper
        if original_logits_processor is not None:
            if hasattr(original_logits_processor, "register_forward_hook"):
                logits_processor_hook = original_logits_processor.register_forward_hook(logits_processor_hook_fn)
            else:
                self.model.logits_processor = logits_processor_wrapper
        try:
            draft_token_ids = original_run_merged_draft(self, *args, **kwargs)
        finally:
            if original_compute_logits is not None:
                self.model.compute_logits = original_compute_logits
            if logits_processor_hook is not None:
                logits_processor_hook.remove()
            elif original_logits_processor is not None:
                self.model.logits_processor = original_logits_processor

        if captured_logits is not None and draft_token_ids is not None:
            _record_selected_token_probs(self, captured_logits, draft_token_ids.reshape(-1))
        return draft_token_ids

    AscendSpecDecodeBaseProposer._run_merged_draft = _run_merged_draft
    AscendSpecDecodeBaseProposer._dcut_patched = True
    _emit_dcut_log("D-Cut adaptive-verify patched AscendSpecDecodeBaseProposer.")
    return True


def _try_patch_loaded_modules() -> None:
    runner_module = sys.modules.get(_MODEL_RUNNER_MODULE)
    if runner_module is not None:
        _patch_runner_module(runner_module)

    proposer_module = sys.modules.get(_PROPOSER_MODULE)
    if proposer_module is not None:
        _patch_proposer_module(proposer_module)


def _install_import_hook() -> None:
    global _IMPORT_HOOK_INSTALLED
    if _IMPORT_HOOK_INSTALLED:
        return

    def dcut_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        if name in (_MODEL_RUNNER_MODULE, _PROPOSER_MODULE) or name.startswith("vllm_ascend."):
            try:
                _try_patch_loaded_modules()
            except Exception:
                logger.exception("D-Cut adaptive-verify delayed patch failed; plugin remains dormant.")
        return module

    builtins.__import__ = dcut_import
    _IMPORT_HOOK_INSTALLED = True
    _emit_dcut_log("D-Cut adaptive-verify delayed import hook installed.")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        _emit_dcut_log("D-Cut adaptive-verify plugin already installed; skipping duplicate install.")
        return
    _emit_dcut_log(
        "D-Cut adaptive-verify plugin install requested (VLLM_PLUGINS=%s, config_env=%s).",
        os.getenv("VLLM_PLUGINS", "<unset>"),
        _get_config_path() or "<unset>",
    )
    _install_import_hook()
    _try_patch_loaded_modules()
    _INSTALLED = True
    _emit_dcut_log(
        "D-Cut adaptive-verify plugin installed for vLLM Ascend "
        "(patches are applied lazily after Ascend runner modules load)."
    )

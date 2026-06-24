# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import replace
from functools import wraps
from typing import Any

from vllm.logger import init_logger

from dcut.verify_adaptive_config import VerifyAdaptiveConfig
from dcut.verify_adaptive_controller import choose_query_lens_discrete

logger = init_logger(__name__)
_INSTALLED = False
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
        return None
    try:
        config = VerifyAdaptiveConfig.from_file(config_path)
    except Exception:
        logger.exception("Failed to load D-Cut config from %s; adaptive verify is disabled.", config_path)
        return None
    if not config.enabled:
        logger.info("D-Cut adaptive verify dormant: config disables it.")
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

    config = _load_config()
    if config is None:
        return False
    if not _is_supported_runner(runner):
        speculative_config = getattr(runner, "speculative_config", None)
        logger.info(
            "D-Cut adaptive verify dormant: method=%s parallel_drafting=%s is unsupported.",
            getattr(speculative_config, "method", None),
            bool(getattr(speculative_config, "parallel_drafting", False)),
        )
        return False
    if getattr(runner, "use_async_scheduling", False):
        logger.warning("D-Cut adaptive verify is enabled with async scheduling; probs bookkeeping may be skipped.")

    runner.dcut_config = config
    runner.dcut_adaptive_enabled = True
    logger.info("D-Cut adaptive verify ENABLED (config=%s)", config.to_log_dict())
    return True


def _apply_dcut_draft_lens(runner: Any, scheduler_output: Any) -> Any:
    if not _ensure_runner_state(runner) or not runner.dcut_next_draft_lens:
        return scheduler_output
    scheduled = scheduler_output.scheduled_spec_decode_tokens
    if not scheduled:
        return scheduler_output

    updated = {}
    changed = False
    for req_id, draft_token_ids in scheduled.items():
        target_len = runner.dcut_next_draft_lens.get(req_id)
        if target_len is None:
            updated[req_id] = draft_token_ids
            continue
        target_len = max(0, min(int(target_len), len(draft_token_ids)))
        updated[req_id] = draft_token_ids[:target_len]
        changed = changed or target_len != len(draft_token_ids)
    runner.dcut_next_draft_lens = {}
    if not changed:
        return scheduler_output
    logger.debug("D-Cut: truncated scheduled spec-decode tokens for %d requests.", len(updated))
    return replace(scheduler_output, scheduled_spec_decode_tokens=updated)


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
        vocab_size = logits.shape[-1]
        if bool(torch.any((draft_token_ids < 0) | (draft_token_ids >= vocab_size)).item()):
            logger.debug("D-Cut: selected draft ids are outside logits vocab; skip probs for this step.")
            return
        probs = torch.softmax(logits.float(), dim=-1)
        selected_probs = probs.gather(dim=-1, index=draft_token_ids.view(-1, 1)).view(-1)
        proposer.latest_draft_token_probs = selected_probs.view(-1, proposer.num_speculative_tokens)
    except Exception:
        logger.exception("D-Cut: failed to record selected draft token probabilities.")


def _update_dcut_next_draft_lens(runner: Any, draft_token_ids: Any) -> None:
    if not _ensure_runner_state(runner) or draft_token_ids is None:
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
        cost_lookup=lambda q: float(q),
        max_draft_len=max_draft_len,
    )
    runner.dcut_next_draft_lens = {
        req_id: int(draft_len)
        for req_id, draft_len in zip(req_ids, result["draft_lens"], strict=False)
    }
    logger.debug("D-Cut: selected best_Q=%s draft_lens=%s", result["best_Q"], runner.dcut_next_draft_lens)


def _patch_runner() -> None:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_dcut_patched", False):
        return

    original_execute_model = NPUModelRunner.execute_model
    original_propose_draft_token_ids = NPUModelRunner.propose_draft_token_ids

    @wraps(original_execute_model)
    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
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


def _patch_proposer() -> None:
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    if getattr(AscendSpecDecodeBaseProposer, "_dcut_patched", False):
        return

    original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft

    @wraps(original_run_merged_draft)
    def _run_merged_draft(self: Any, *args: Any, **kwargs: Any) -> Any:
        runner = getattr(self, "runner", None)
        if runner is None or not _ensure_runner_state(runner):
            return original_run_merged_draft(self, *args, **kwargs)

        captured_logits = None
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

        if original_compute_logits is not None:
            self.model.compute_logits = compute_logits_wrapper
        if original_logits_processor is not None:
            self.model.logits_processor = logits_processor_wrapper
        try:
            draft_token_ids = original_run_merged_draft(self, *args, **kwargs)
        finally:
            if original_compute_logits is not None:
                self.model.compute_logits = original_compute_logits
            if original_logits_processor is not None:
                self.model.logits_processor = original_logits_processor

        if captured_logits is not None and draft_token_ids is not None:
            _record_selected_token_probs(self, captured_logits, draft_token_ids.reshape(-1))
        return draft_token_ids

    AscendSpecDecodeBaseProposer._run_merged_draft = _run_merged_draft
    AscendSpecDecodeBaseProposer._dcut_patched = True


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_runner()
    _patch_proposer()
    _INSTALLED = True
    logger.info("D-Cut adaptive-verify monkey patch installed for vLLM Ascend.")

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from dataclasses import replace
from functools import wraps
from types import ModuleType

import torch
from vllm.logger import logger

from dcut.controller import VerifyAdaptiveController, dcut_enabled

_PATCHED_MODULES: set[str] = set()
_HOOK_INSTALLED = False
_TARGET_MODULES = {
    "vllm_ascend.spec_decode.llm_base_proposer",
    "vllm_ascend.worker.model_runner_v1",
    "vllm_ascend.worker.worker",
}


class _DcutPatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped_loader: importlib.abc.Loader) -> None:
        self.fullname = fullname
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_module(module.__name__, module)


class _DcutPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if fullname not in _TARGET_MODULES or fullname in _PATCHED_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _DcutPatchLoader(fullname, spec.loader)
        return spec


def apply_patch() -> None:
    """Install lazy D-Cut monkeypatches into the already-installed vllm-ascend."""
    global _HOOK_INSTALLED
    if not dcut_enabled():
        logger.info("D-Cut plugin loaded but disabled. Set DCUT_ENABLE=1 or VLLM_ASCEND_ENABLE_DCUT=1 to enable it.")
        return
    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _DcutPatchFinder())
        _HOOK_INSTALLED = True
        logger.info("D-Cut lazy monkeypatch hook installed. Enable flag detected.")
    for module_name in _TARGET_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            _patch_module(module_name, module)


def _patch_module(module_name: str, module: ModuleType) -> None:
    if module_name in _PATCHED_MODULES:
        return
    if module_name == "vllm_ascend.spec_decode.llm_base_proposer":
        _patch_proposer(module.AscendSpecDecodeBaseProposer)
    elif module_name == "vllm_ascend.worker.model_runner_v1":
        _patch_runner(module.NPUModelRunner)
    elif module_name == "vllm_ascend.worker.worker":
        _patch_worker(module.NPUWorker)
    _PATCHED_MODULES.add(module_name)
    logger.info("D-Cut patched module: %s", module_name)


def _patch_proposer(cls):
    if getattr(cls, "_dcut_patched", False):
        return
    original_init = cls.__init__
    original_compute = cls.compute_draft_token_ids

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.needs_draft_probs = False
        self._last_selected_probs = None

    def take_last_selected_probs(self):
        probs = getattr(self, "_last_selected_probs", None)
        self._last_selected_probs = None
        return probs

    @wraps(original_compute)
    def compute_draft_token_ids(self, hidden_states):
        logits = self.model.logits_processor(self.model.lm_head, hidden_states)
        next_token = _greedy_sample_from_tp_logits(logits)
        if getattr(self, "needs_draft_probs", False) and getattr(self, "parallel_drafting", False):
            chosen = logits.gather(-1, next_token.long().unsqueeze(-1)).squeeze(-1)
            self._last_selected_probs = (chosen - logits.logsumexp(dim=-1)).exp().view(
                -1, self.num_speculative_tokens
            ).contiguous()
        if not hasattr(self.model, "draft_id_to_target_id") or self.model.draft_id_to_target_id is None:
            return next_token
        bias = torch.index_select(self.model.draft_id_to_target_id, dim=0, index=next_token.view(-1)).view(
            next_token.shape
        )
        return next_token + bias

    cls.__init__ = __init__
    cls.take_last_selected_probs = take_last_selected_probs
    cls.compute_draft_token_ids = compute_draft_token_ids
    cls._dcut_patched = True


def _greedy_sample_from_tp_logits(logits: torch.Tensor) -> torch.Tensor:
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    _batch, vocab_local = logits.shape
    local_max_logits, local_max_indices = logits.max(dim=-1)
    local_global_idx = local_max_indices + tp_group.rank_in_group * vocab_local
    gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
    gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)
    global_max_rank = gathered_logits.argmax(dim=-1)
    return gathered_global_idx.gather(dim=-1, index=global_max_rank.unsqueeze(-1)).squeeze(-1)


def _patch_runner(cls):
    if getattr(cls, "_dcut_patched", False):
        return
    original_init = cls.__init__
    original_execute_model = cls.execute_model
    original_sample_tokens = cls.sample_tokens
    original_copy_draft = cls._copy_draft_token_ids_to_cpu

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._dcut_controller = None
        self._dcut_probs_event = None
        self._dcut_probs_pinned = None
        self._dcut_probs_pending = False
        self._dcut_num_reqs = 0
        self._dcut_req_ids = []
        self._dcut_active = set()
        self._dcut_missing_probs_warnings = 0
        self._dcut_fallback_probs_warnings = 0
        _dcut_init_controller(self)

    @wraps(original_execute_model)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        scheduler_output = _dcut_truncate_scheduler_output(self, scheduler_output)
        return original_execute_model(self, scheduler_output, intermediate_tensors)

    @wraps(original_sample_tokens)
    def sample_tokens(self, grammar_output):
        _dcut_maybe_process_probs(self)
        return original_sample_tokens(self, grammar_output)

    @wraps(original_copy_draft)
    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        original_copy_draft(self, scheduler_output, zeros_only=zeros_only)
        _dcut_queue_probs(self, zeros_only)

    cls.__init__ = __init__
    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    cls._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    cls.profile_dcut_cost = _dcut_profile_cost
    cls._dcut_patched = True


def _supports_dcut(runner) -> bool:
    spec_cfg = getattr(runner, "speculative_config", None)
    if spec_cfg is None:
        return False
    method = getattr(spec_cfg, "method", None)
    return method == "dflash" or (method == "draft_model" and getattr(spec_cfg, "parallel_drafting", False))


def _dcut_init_controller(runner) -> None:
    if not _supports_dcut(runner):
        return
    num_spec = getattr(runner, "num_spec_tokens", 0) or 0
    if num_spec <= 0:
        return
    try:
        controller = VerifyAdaptiveController.from_env(num_spec, runner.scheduler_config.max_num_seqs)
    except Exception as exc:
        logger.error("D-Cut: failed to initialize adaptive verifier; disabled: %s", exc)
        return
    if controller is None:
        return
    runner._dcut_controller = controller
    drafter = getattr(runner, "drafter", None)
    if drafter is not None and hasattr(drafter, "needs_draft_probs"):
        drafter.needs_draft_probs = True
    runner._dcut_probs_event = torch.npu.Event()
    runner._dcut_probs_pinned = torch.empty(
        (runner.max_num_reqs, num_spec), dtype=torch.float32, device="cpu", pin_memory=runner.pin_memory)
    if (
        getattr(runner.speculative_config, "method", None) == "dflash"
        and not getattr(getattr(runner, "ascend_config", None), "enable_reduce_sample", False)
    ):
        logger.warning(
            "D-Cut: dflash is running with enable_reduce_sample=False. "
            "Selected draft probabilities may be unavailable, so D-Cut cannot plan/cut. "
            "Enable it with --additional-config '{\"enable_reduce_sample\": true}' "
            "or set DCUT_FALLBACK_PROB for cost-only fallback diagnostics."
        )
    logger.info(
        "D-Cut adaptive verifier enabled: method=%s num_spec_tokens=%d max_num_seqs=%d",
        getattr(runner.speculative_config, "method", None),
        num_spec,
        runner.scheduler_config.max_num_seqs,
    )


def _dcut_profile_cost(runner) -> None:
    controller = getattr(runner, "_dcut_controller", None)
    if controller is not None:
        controller.profile_cost_table(runner)


def _dcut_truncate_scheduler_output(runner, scheduler_output):
    controller = getattr(runner, "_dcut_controller", None)
    if controller is None or not scheduler_output.scheduled_spec_decode_tokens:
        return scheduler_output
    new_spec = scheduler_output.scheduled_spec_decode_tokens.copy()
    new_num_sched = scheduler_output.num_scheduled_tokens.copy()
    tokens_delta = 0
    for req_id, draft_toks in list(new_spec.items()):
        adaptive_len = controller.get_adaptive_draft_len(req_id)
        if adaptive_len is not None and adaptive_len < len(draft_toks):
            min_draft_len = _dcut_min_safe_draft_len(runner, req_id)
            if adaptive_len < min_draft_len:
                logger.debug(
                    "D-Cut: clamping draft cut for req_id=%s from %d to %d "
                    "to keep the verifier segment compatible with already-accepted tokens.",
                    req_id,
                    adaptive_len,
                    min_draft_len,
                )
                adaptive_len = min(min_draft_len, len(draft_toks))
            if adaptive_len >= len(draft_toks):
                continue
            diff = len(draft_toks) - adaptive_len
            tokens_delta += diff
            new_num_sched[req_id] -= diff
            if adaptive_len == 0:
                del new_spec[req_id]
            else:
                new_spec[req_id] = draft_toks[:adaptive_len]
    if tokens_delta <= 0:
        return scheduler_output
    logger.debug(
        "D-Cut: cut scheduled speculative tokens reqs=%d tokens_before=%d tokens_after=%d delta=%d",
        len(scheduler_output.scheduled_spec_decode_tokens),
        scheduler_output.total_num_scheduled_tokens,
        scheduler_output.total_num_scheduled_tokens - tokens_delta,
        tokens_delta,
    )
    return replace(
        scheduler_output,
        scheduled_spec_decode_tokens=new_spec,
        num_scheduled_tokens=new_num_sched,
        total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens - tokens_delta,
    )


def _dcut_min_safe_draft_len(runner, req_id: str) -> int:
    """Return the minimum draft length that keeps varlen verifier inputs safe.

    Ascend hybrid/Mamba paths pass ``num_accepted_tokens`` to custom kernels.
    A scheduled speculative segment contains one target token plus the draft
    tokens. If D-Cut shrinks a segment below ``num_accepted_tokens``, kernels
    such as GDN causal conv can fail during tiling. Therefore the adaptive
    draft length must stay at least ``num_accepted_tokens - 1``.
    """
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return 0
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    accepted_tokens_cpu = getattr(input_batch, "num_accepted_tokens_cpu", None)
    if not req_id_to_index or accepted_tokens_cpu is None:
        return 0
    req_index = req_id_to_index.get(req_id)
    if req_index is None or req_index < 0 or req_index >= len(accepted_tokens_cpu):
        return 0
    return max(int(accepted_tokens_cpu[req_index]) - 1, 0)


def _dcut_queue_probs(runner, zeros_only: bool) -> None:
    if zeros_only or getattr(runner, "_dcut_probs_pending", False):
        return
    if getattr(runner, "_dcut_controller", None) is None or runner._dcut_probs_pinned is None:
        return
    drafter = getattr(runner, "drafter", None)
    if drafter is None or not hasattr(drafter, "take_last_selected_probs"):
        return
    probs = drafter.take_last_selected_probs()
    if probs is None:
        fallback_prob = _get_fallback_prob()
        if fallback_prob is None:
            _dcut_clear_active_plans(runner)
            warnings = getattr(runner, "_dcut_missing_probs_warnings", 0)
            if warnings < 5:
                logger.warning(
                    "D-Cut: no selected draft probabilities captured for this step; skip adaptive cut. "
                    "For dflash, enable --additional-config '{\"enable_reduce_sample\": true}' "
                    "to use real probabilities, or set DCUT_FALLBACK_PROB to test cost-only cutting."
                )
                runner._dcut_missing_probs_warnings = warnings + 1
            return
        probs = torch.full(
            (runner.input_batch.num_reqs, runner.num_spec_tokens),
            fallback_prob,
            dtype=torch.float32,
            device=runner.device,
        )
        warnings = getattr(runner, "_dcut_fallback_probs_warnings", 0)
        if warnings < 5:
            logger.warning(
                "D-Cut: using fallback draft probability %.4f because real selected probabilities were unavailable.",
                fallback_prob,
            )
            runner._dcut_fallback_probs_warnings = warnings + 1
    num_reqs = runner.input_batch.num_reqs
    runner._dcut_probs_pending = True
    runner._dcut_num_reqs = num_reqs
    runner._dcut_req_ids = runner.input_batch.req_ids.copy()
    runner._dcut_active = _dcut_active_decode_req_ids(runner)
    runner._dcut_probs_pinned[:num_reqs].copy_(probs, non_blocking=True)
    runner._dcut_probs_event.record()


def _dcut_active_decode_req_ids(runner) -> set[str]:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return set()
    return {
        input_batch.req_ids[i]
        for i in range(input_batch.num_reqs)
        if input_batch.num_computed_tokens_cpu[i] >= input_batch.num_prompt_tokens[i]
    }


def _dcut_clear_active_plans(runner) -> None:
    controller = getattr(runner, "_dcut_controller", None)
    if controller is None:
        return
    for req_id in _dcut_active_decode_req_ids(runner):
        controller.invalidate(req_id)


def _get_fallback_prob() -> float | None:
    import os

    raw_value = os.getenv("DCUT_FALLBACK_PROB")
    if raw_value is None:
        return None
    value = float(raw_value)
    if value <= 0.0 or value > 1.0:
        raise ValueError("DCUT_FALLBACK_PROB must be in the range (0, 1].")
    return value


def _dcut_maybe_process_probs(runner) -> None:
    if not getattr(runner, "_dcut_probs_pending", False):
        return
    if not runner._dcut_probs_event.query():
        runner._dcut_probs_event.synchronize()
    runner._dcut_probs_pending = False
    if runner._dcut_active and runner._dcut_controller is not None:
        logger.debug(
            "D-Cut: processing draft probabilities batch_size=%d active_decode_reqs=%d",
            runner._dcut_num_reqs,
            len(runner._dcut_active),
        )
        runner._dcut_controller.process_draft_output(
            selected_probs=runner._dcut_probs_pinned[: runner._dcut_num_reqs],
            req_ids=runner._dcut_req_ids,
            active_draft_req_ids=runner._dcut_active,
            batch_size=runner._dcut_num_reqs,
        )


def _patch_worker(cls):
    if getattr(cls, "_dcut_patched", False):
        return
    original_compile = cls.compile_or_warm_up_model

    @wraps(original_compile)
    def compile_or_warm_up_model(self):
        result = original_compile(self)
        if hasattr(self.model_runner, "profile_dcut_cost"):
            try:
                self.model_runner.profile_dcut_cost()
            except Exception as exc:
                logger.error("D-Cut: cost profiling failed; falling back to full verification: %s", exc)
        return result

    cls.compile_or_warm_up_model = compile_or_warm_up_model
    cls._dcut_patched = True

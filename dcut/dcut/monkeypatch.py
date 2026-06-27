# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from functools import wraps

import torch
from vllm.logger import logger

from dcut.controller import VerifyAdaptiveController, dcut_enabled

_PATCHED = False


def apply_patch() -> None:
    """Install D-Cut monkeypatches into the already-installed vllm-ascend."""
    global _PATCHED
    if _PATCHED:
        return
    if not dcut_enabled():
        logger.info("D-Cut plugin loaded but disabled. Set DCUT_ENABLE=1 or VLLM_ASCEND_ENABLE_DCUT=1 to enable it.")
        _PATCHED = True
        return

    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker

    _patch_proposer(AscendSpecDecodeBaseProposer)
    _patch_runner(NPUModelRunner)
    _patch_worker(NPUWorker)
    _PATCHED = True
    logger.info("D-Cut plugin monkeypatches installed. Enable flag detected.")


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
        _dcut_init_controller(self)

    @wraps(original_execute_model)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        scheduler_output = _dcut_truncate_scheduler_output(self, scheduler_output)
        return original_execute_model(self, scheduler_output, intermediate_tensors)

    @wraps(original_sample_tokens)
    def sample_tokens(self, grammar_output):
        _dcut_maybe_process_probs(self)
        output = original_sample_tokens(self, grammar_output)
        controller = getattr(self, "_dcut_controller", None)
        if controller is not None and hasattr(self, "execute_model_state"):
            # Best-effort cleanup happens in execute_model wrapper for scheduled outputs;
            # finished request ids are not available after original_sample_tokens returns
            # on all vLLM versions, so stale entries are also harmlessly overwritten.
            pass
        return output

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
            diff = len(draft_toks) - adaptive_len
            tokens_delta += diff
            new_num_sched[req_id] -= diff
            if adaptive_len == 0:
                del new_spec[req_id]
            else:
                new_spec[req_id] = draft_toks[:adaptive_len]
    if tokens_delta <= 0:
        return scheduler_output
    logger.info(
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
        return
    num_reqs = runner.input_batch.num_reqs
    runner._dcut_probs_pending = True
    runner._dcut_num_reqs = num_reqs
    runner._dcut_req_ids = runner.input_batch.req_ids.copy()
    runner._dcut_active = {
        runner.input_batch.req_ids[i]
        for i in range(num_reqs)
        if runner.input_batch.num_computed_tokens_cpu[i] >= runner.input_batch.num_prompt_tokens[i]
    }
    runner._dcut_probs_pinned[:num_reqs].copy_(probs, non_blocking=True)
    runner._dcut_probs_event.record()


def _dcut_maybe_process_probs(runner) -> None:
    if not getattr(runner, "_dcut_probs_pending", False):
        return
    if not runner._dcut_probs_event.query():
        runner._dcut_probs_event.synchronize()
    runner._dcut_probs_pending = False
    if runner._dcut_active and runner._dcut_controller is not None:
        logger.info(
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

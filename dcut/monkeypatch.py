# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch installer for D-Cut adaptive verifier step-length on vLLM 0.22.x.

Ported as a self-contained vLLM general plugin. Enable with:

    cd dcut && pip install -e .
    export DCUT_ENABLE=1
    export DCUT_CONFIG=/path/to/verify_adaptive_config.json

The drafter still proposes ``num_speculative_tokens`` every step, while the
verifier checks a batch-adaptive subset selected from draft probabilities and a
profiled verifier ITL cost table.
"""

from __future__ import annotations

import builtins
import inspect
import os
import sys
import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices

from .verify_adaptive_config import VerifyAdaptiveConfig
from .verify_adaptive_controller import VerifyAdaptiveController

logger = init_logger(f"vllm.{__name__}")
_INSTALLED = False
ENV_CONFIG = "DCUT_CONFIG"
ENV_ENABLE = "DCUT_ENABLE"


def _supports_adaptive_verify(spec_cfg: Any) -> bool:
    if spec_cfg is None:
        return False
    method = getattr(spec_cfg, "method", None)
    parallel = getattr(spec_cfg, "parallel_drafting", False)
    return method == "dflash" or (method == "draft_model" and parallel)


def _dcut_init_controller(self) -> None:
    self._verify_adaptive_controller = None
    self._adaptive_probs_event = None
    self._adaptive_probs_pinned = None
    self._adaptive_probs_pending = False
    self._adaptive_num_reqs = 0
    self._adaptive_req_ids = []
    self._adaptive_active = set()
    self._dcut_last_cut_records = []

    dcut_enabled = os.environ.get(ENV_ENABLE, "0").lower() in {"1", "true", "yes", "on"}
    cfg_path = os.environ.get(ENV_CONFIG) or None
    if not dcut_enabled and not cfg_path:
        return
    if dcut_enabled and not cfg_path:
        logger.warning("DCUT_ENABLE is set but DCUT_CONFIG is empty; D-Cut disabled.")
        return
    spec_cfg = getattr(self, "speculative_config", None)
    if not _supports_adaptive_verify(spec_cfg):
        logger.warning(
            "DCUT_CONFIG is set but the speculative method does not support adaptive verifier step-length "
            "(requires dflash or draft_model+parallel_drafting); D-Cut disabled."
        )
        return
    num_spec = getattr(self, "num_spec_tokens", 0) or 0
    if num_spec <= 0:
        logger.warning("D-Cut: num_spec_tokens <= 0; disabled.")
        return

    acfg = VerifyAdaptiveConfig.from_json(cfg_path)
    self._verify_adaptive_controller = VerifyAdaptiveController(
        config=acfg,
        num_spec_tokens=num_spec,
        max_batch_size=self.scheduler_config.max_num_seqs,
        device=self.device,
    )
    drafter = getattr(self, "drafter", None)
    if drafter is not None and hasattr(drafter, "needs_draft_probs"):
        drafter.needs_draft_probs = True
    self._adaptive_probs_event = torch.cuda.Event()
    self._adaptive_probs_pinned = torch.empty(
        (self.max_num_reqs, num_spec),
        dtype=torch.float32,
        device="cpu",
        pin_memory=self.pin_memory,
    )
    logger.info("D-Cut adaptive verify ENABLED (config=%s).", cfg_path)


def _dcut_truncate(self, scheduler_output):
    ctrl = self._verify_adaptive_controller
    self._dcut_last_cut_records = []
    if ctrl is None or not scheduler_output.scheduled_spec_decode_tokens:
        return scheduler_output

    new_spec = scheduler_output.scheduled_spec_decode_tokens.copy()
    new_num_sched = scheduler_output.num_scheduled_tokens.copy()
    tokens_delta = 0

    for req_id, draft_toks in list(new_spec.items()):
        original_len = len(draft_toks)
        adaptive_len = ctrl.get_adaptive_draft_len(req_id)
        if adaptive_len is None:
            adaptive_len = original_len
        cut_len = min(adaptive_len, original_len)
        self._dcut_last_cut_records.append(
            {
                "req_id": req_id,
                "original_len": original_len,
                "verify_len": cut_len,
                "cut_tokens": original_len - cut_len,
            }
        )
        if cut_len < original_len:
            diff = original_len - cut_len
            tokens_delta += diff
            new_num_sched[req_id] -= diff
            if cut_len == 0:
                del new_spec[req_id]
            else:
                new_spec[req_id] = draft_toks[:cut_len]

    if tokens_delta > 0:
        scheduler_output = replace(
            scheduler_output,
            scheduled_spec_decode_tokens=new_spec,
            num_scheduled_tokens=new_num_sched,
            total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens - tokens_delta,
        )
    return scheduler_output


def _dcut_queue_probs(self, zeros_only: bool) -> None:
    if (
        zeros_only
        or self._adaptive_probs_pending
        or self._adaptive_probs_pinned is None
        or self._adaptive_probs_event is None
    ):
        return
    drafter = getattr(self, "drafter", None)
    if drafter is None or not hasattr(drafter, "take_last_selected_probs"):
        return
    probs = drafter.take_last_selected_probs()
    if probs is None:
        return
    num_reqs = self.input_batch.num_reqs
    self._adaptive_probs_pending = True
    self._adaptive_num_reqs = num_reqs
    self._adaptive_req_ids = self.input_batch.req_ids.copy()
    self._adaptive_active = {
        self.input_batch.req_ids[i]
        for i in range(num_reqs)
        if self.input_batch.num_computed_tokens_cpu[i] >= self.input_batch.num_prompt_tokens[i]
    }
    self._adaptive_probs_pinned[:num_reqs].copy_(probs, non_blocking=True)
    self._adaptive_probs_event.record()


def _maybe_process_adaptive_probs(self) -> None:
    if not self._adaptive_probs_pending:
        return
    assert self._adaptive_probs_event is not None
    if not self._adaptive_probs_event.query():
        self._adaptive_probs_event.synchronize()
    self._adaptive_probs_pending = False
    num_reqs = self._adaptive_num_reqs
    active = self._adaptive_active
    if active and self._verify_adaptive_controller is not None:
        assert self._adaptive_probs_pinned is not None
        self._verify_adaptive_controller.process_draft_output(
            selected_probs=self._adaptive_probs_pinned[:num_reqs],
            req_ids=self._adaptive_req_ids,
            active_draft_req_ids=active,
            batch_size=num_reqs,
        )


def profile_adaptive_cost(self) -> None:
    if getattr(self, "_verify_adaptive_controller", None) is not None:
        self._verify_adaptive_controller.profile_cost_table(self)


@torch.inference_mode()
def _adaptive_profile_run(
    self,
    scheduled_tokens: list[int],
    seq_lens: int = 1024,
    n_warmup: int = 3,
    n_measure: int = 5,
):
    num_scheduled_tokens = np.array(scheduled_tokens, dtype=np.int32)
    num_reqs = len(scheduled_tokens)
    num_tokens_unpadded = int(num_scheduled_tokens.sum())
    max_query_len = int(num_scheduled_tokens.max())
    assert num_tokens_unpadded <= self.max_num_tokens, (
        f"adaptive profile: num_tokens={num_tokens_unpadded} > max_num_tokens={self.max_num_tokens}"
    )

    _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=False,
            force_eager=False,
        )
    )
    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        self.vllm_config.parallel_config.num_ubatches,
    )
    slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_tokens_unpadded=num_tokens_unpadded,
        ubatch_slices=ubatch_slices_padded,
    )
    if slot_mappings_by_group is not None:
        for slot_mapping in slot_mappings_by_group.values():
            slot_mapping.fill_(-1)

    with self.synchronize_input_prep():
        self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)
        cum_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
        self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
        self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(cum_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        self.input_batch.block_table.commit_block_table(num_reqs_padded)
        if self.speculative_config is not None:
            draft_len = max_query_len - 1
            self.num_decode_draft_tokens.np[:num_reqs] = draft_len
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
            self.num_accepted_tokens.gpu[:num_reqs] = max_query_len
            self.num_accepted_tokens.gpu[num_reqs:].fill_(1)

    pad_attn = _cudagraph_mode == CUDAGraphMode.FULL
    build_attn_kwargs = {
        "num_tokens": num_tokens_unpadded,
        "num_tokens_padded": num_tokens_padded if pad_attn else None,
        "num_reqs": num_reqs,
        "num_reqs_padded": num_reqs_padded,
        "max_query_len": max_query_len,
        "ubatch_slices": ubatch_slices_padded if pad_attn else ubatch_slices,
        "for_cudagraph_capture": False,
        "slot_mappings": slot_mappings_by_group,
        "use_spec_decode": self.speculative_config is not None,
        "num_scheduled_tokens_np": num_scheduled_tokens,
    }
    build_attn_signature = inspect.signature(self._build_attention_metadata)
    build_attn_kwargs = {k: v for k, v in build_attn_kwargs.items() if k in build_attn_signature.parameters}
    attn_metadata, _ = self._build_attention_metadata(**build_attn_kwargs)
    if ubatch_slices_padded is not None:
        num_tokens_padded = ubatch_slices_padded[0].num_tokens
    if num_tokens_across_dp is not None:
        num_tokens_across_dp[:] = num_tokens_padded

    model_kwargs = self._init_model_kwargs()
    use_embeds = self.enable_prompt_embeds or (self.supports_mm_inputs and not self.model_config.is_encoder_decoder)
    if use_embeds:
        input_ids = None
        inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
    else:
        input_ids = self.input_ids.gpu[:num_tokens_padded]
        inputs_embeds = None
    if self.uses_mrope:
        positions = self.mrope_positions.gpu[:, :num_tokens_padded]
    elif self.uses_xdrope_dim > 0:
        positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
    else:
        positions = self.positions[:num_tokens_padded]
    intermediate_tensors = None
    if not get_pp_group().is_first_rank:
        if self.intermediate_tensors is None:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )
        intermediate_tensors = self.sync_and_gather_intermediate_tensors(num_tokens_padded, None, False)

    mode_names = {CUDAGraphMode.FULL: "FCG", CUDAGraphMode.PIECEWISE: "PCG", CUDAGraphMode.NONE: "eager"}
    avg_ms = 0.0
    with set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=_cudagraph_mode,
        batch_descriptor=batch_desc,
        ubatch_slices=ubatch_slices_padded,
        slot_mapping=slot_mappings,
    ):

        def _forward() -> None:
            self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        for _ in range(max(n_warmup, 0)):
            _forward()
        torch.cuda.synchronize()
        if n_measure > 0:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(n_measure):
                _forward()
            end_ev.record()
            torch.cuda.synchronize()
            avg_ms = start_ev.elapsed_time(end_ev) / n_measure
    return mode_names.get(_cudagraph_mode, str(_cudagraph_mode)), avg_ms, int(num_tokens_padded)


def _patch_proposer() -> None:
    import vllm.v1.spec_decode.llm_base_proposer as m

    proposer_cls = m.SpecDecodeBaseProposer
    if getattr(proposer_cls, "_dcut_patched", False):
        return
    proposer_cls.needs_draft_probs = False
    proposer_cls._last_selected_probs = None

    @staticmethod
    def _gather_selected_probs(logits, token_ids, full_probs):
        idx = token_ids.long().unsqueeze(-1)
        if full_probs is not None:
            return full_probs.gather(-1, idx).squeeze(-1)
        chosen = logits.gather(-1, idx).squeeze(-1)
        return (chosen - logits.logsumexp(dim=-1)).exp()

    def take_last_selected_probs(self):
        return getattr(self, "_last_selected_probs", None)

    orig_sample = proposer_cls._sample_draft_tokens

    def _sample_draft_tokens(self, hidden_states, sampling_metadata):
        self._last_selected_probs = None
        out = orig_sample(self, hidden_states, sampling_metadata)
        if getattr(self, "needs_draft_probs", False):
            token_ids = out[0]
            full_probs = out[1] if len(out) > 1 else None
            logits = None if full_probs is not None else self.model.compute_logits(hidden_states)
            selected = proposer_cls._gather_selected_probs(logits, token_ids, full_probs)
            self._last_selected_probs = selected.view(-1, self.num_speculative_tokens).contiguous()
        return out

    proposer_cls._gather_selected_probs = _gather_selected_probs
    proposer_cls.take_last_selected_probs = take_last_selected_probs
    proposer_cls._sample_draft_tokens = _sample_draft_tokens
    proposer_cls._dcut_patched = True


def _log_dcut_verify_result(runner, batch_size: int, elapsed_ms: float) -> None:
    records = getattr(runner, "_dcut_last_cut_records", []) or []
    total_before = sum(record["original_len"] for record in records)
    total_after = sum(record["verify_len"] for record in records)
    controller = getattr(runner, "_verify_adaptive_controller", None)
    decision = getattr(controller, "_last_decision", None) if controller is not None else None
    message = (
        "D-Cut target verify finished: "
        f"batch_size={batch_size}, "
        f"cut draft tokens {total_before} -> {total_after}, "
        f"elapsed={elapsed_ms:.3f} ms, decision={decision}, details={records}"
    )
    print(message, flush=True)
    logger.info(message)


def _patch_runner_class(runner_cls) -> None:
    if runner_cls.__dict__.get("_dcut_patched", False):
        return

    orig_init = runner_cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _dcut_init_controller(self)

    orig_exec = runner_cls.execute_model

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        if getattr(self, "_verify_adaptive_controller", None) is not None:
            scheduler_output = _dcut_truncate(self, scheduler_output)
            batch_size = len(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})
            start = time.perf_counter()
            out = orig_exec(self, scheduler_output, intermediate_tensors)
            elapsed_ms = (time.perf_counter() - start) * 1e3
            _log_dcut_verify_result(self, batch_size, elapsed_ms)
            return out
        return orig_exec(self, scheduler_output, intermediate_tensors)

    orig_sample_tokens = runner_cls.sample_tokens

    def sample_tokens(self, *args, **kwargs):
        out = orig_sample_tokens(self, *args, **kwargs)
        if getattr(self, "_adaptive_probs_pending", False):
            _maybe_process_adaptive_probs(self)
        return out

    orig_copy = runner_cls._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        orig_copy(self, scheduler_output, zeros_only)
        if getattr(self, "_verify_adaptive_controller", None) is not None:
            _dcut_queue_probs(self, zeros_only)

    orig_update = runner_cls._update_states

    def _update_states(self, scheduler_output):
        ret = orig_update(self, scheduler_output)
        ctrl = getattr(self, "_verify_adaptive_controller", None)
        if ctrl is not None:
            for req_id in scheduler_output.finished_req_ids:
                ctrl.invalidate(req_id)
        return ret

    runner_cls.__init__ = __init__
    runner_cls.execute_model = execute_model
    runner_cls.sample_tokens = sample_tokens
    runner_cls._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    runner_cls._update_states = _update_states
    runner_cls._adaptive_profile_run = _adaptive_profile_run
    runner_cls.profile_adaptive_cost = profile_adaptive_cost
    runner_cls._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    runner_cls._dcut_patched = True
    logger.info("D-Cut patched runner class: %s.%s", runner_cls.__module__, runner_cls.__name__)


def _patch_runner() -> None:
    import vllm.v1.worker.gpu_model_runner as m

    _patch_runner_class(m.GPUModelRunner)
    _patch_loaded_ascend_classes()


def _patch_worker_class(worker_cls) -> None:
    if worker_cls.__dict__.get("_dcut_patched", False):
        return
    orig_warmup = worker_cls.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *args, **kwargs):
        ret = orig_warmup(self, *args, **kwargs)
        runner = getattr(self, "model_runner", None)
        if runner is not None and hasattr(runner, "profile_adaptive_cost"):
            runner.profile_adaptive_cost()
        return ret

    worker_cls.compile_or_warm_up_model = compile_or_warm_up_model
    worker_cls._dcut_patched = True
    logger.info("D-Cut patched worker class: %s.%s", worker_cls.__module__, worker_cls.__name__)


def _patch_worker() -> None:
    import vllm.v1.worker.gpu_worker as m

    _patch_worker_class(m.Worker)
    _patch_loaded_ascend_classes()


def _patch_loaded_ascend_classes() -> None:
    ascend_runner_m = sys.modules.get("vllm_ascend.worker.model_runner_v1")
    if ascend_runner_m is not None and hasattr(ascend_runner_m, "NPUModelRunner"):
        _patch_runner_class(ascend_runner_m.NPUModelRunner)

    ascend_worker_m = sys.modules.get("vllm_ascend.worker.worker")
    if ascend_worker_m is not None and hasattr(ascend_worker_m, "NPUWorker"):
        _patch_worker_class(ascend_worker_m.NPUWorker)


def _install_ascend_import_hook() -> None:
    if getattr(builtins, "_dcut_import_hook_installed", False):
        return

    original_import = builtins.__import__

    def dcut_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name in {"vllm_ascend.worker.model_runner_v1", "vllm_ascend.worker.worker"} or name.startswith(
            "vllm_ascend.worker"
        ):
            _patch_loaded_ascend_classes()
        return module

    builtins.__import__ = dcut_import
    builtins._dcut_import_hook_installed = True
    builtins._dcut_original_import = original_import
    logger.info("D-Cut Ascend import hook installed; NPU runner/worker will be patched after normal import.")


def install(*args, **kwargs) -> None:
    """vLLM general-plugin entrypoint."""
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_proposer()
    _patch_runner()
    _patch_worker()
    _install_ascend_import_hook()
    _INSTALLED = True
    logger.info(
        "D-Cut adaptive-verify monkey patch installed "
        "(active only if DCUT_ENABLE=1 plus DCUT_CONFIG is set + method is dflash/PARD)."
    )

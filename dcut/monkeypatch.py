# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch installer for D-Cut adaptive verifier step-length on vLLM 0.22.x.

Ported from vllm-project/vllm PR #44885 as a self-contained vLLM general
plugin. The drafter still proposes ``num_speculative_tokens`` each step, while
the verifier checks an adaptive subset chosen by a profiled cost table and draft
confidence scores.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import types
from contextlib import suppress
from dataclasses import replace

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
ENV_CONFIG = "VLLM_DCUT_CONFIG"
ENV_TRIM_STATS_OUT = "VLLM_DCUT_TRIM_STATS_OUT"
ENV_STAT_EVERY = "VLLM_DCUT_STAT_EVERY"
ENV_PROFILE_FORCE_EAGER = "VLLM_DCUT_PROFILE_FORCE_EAGER"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("D-Cut: invalid %s=%r; using %d", name, value, default)
        return default


def _supports_adaptive_verify(spec_cfg) -> bool:
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
    self._dcut_trim_stats_out = os.environ.get(ENV_TRIM_STATS_OUT) or None
    self._dcut_stat_every = max(1, _env_int(ENV_STAT_EVERY, 1))
    self._dcut_step_idx = 0
    self._dcut_total_trimmed_tokens = 0
    self._dcut_total_trimmed_reqs = 0
    self._dcut_profile_force_eager = _env_flag(ENV_PROFILE_FORCE_EAGER, False)
    self._dcut_cost_profile_done = False
    self._dcut_cost_profile_failed = False

    cfg_path = os.environ.get(ENV_CONFIG) or None
    if not cfg_path:
        return
    spec_cfg = getattr(self, "speculative_config", None)
    if not _supports_adaptive_verify(spec_cfg):
        logger.warning(
            "VLLM_DCUT_CONFIG is set but the speculative method does not "
            "support adaptive verifier step-length; D-Cut disabled."
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
    logger.info(
        "D-Cut adaptive verify ENABLED (config=%s trim_stats=%s stat_every=%d profile_force_eager=%s).",
        cfg_path,
        self._dcut_trim_stats_out,
        self._dcut_stat_every,
        self._dcut_profile_force_eager,
    )


def _dcut_write_trim_stats(
    self,
    *,
    batch_size: int,
    trimmed_tokens: int,
    trimmed_reqs: int,
    scheduled_reqs: int,
    total_scheduled_tokens: int,
) -> None:
    self._dcut_step_idx += 1
    self._dcut_total_trimmed_tokens += trimmed_tokens
    self._dcut_total_trimmed_reqs += trimmed_reqs
    if not self._dcut_trim_stats_out:
        return
    if trimmed_tokens == 0 and self._dcut_step_idx % self._dcut_stat_every != 0:
        return
    dirname = os.path.dirname(self._dcut_trim_stats_out)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    line = (
        f"step={self._dcut_step_idx} batch_size={batch_size} "
        f"scheduled_reqs={scheduled_reqs} "
        f"total_scheduled_tokens={total_scheduled_tokens} "
        f"trimmed_reqs={trimmed_reqs} trimmed_tokens={trimmed_tokens} "
        f"total_trimmed_reqs={self._dcut_total_trimmed_reqs} "
        f"total_trimmed_tokens={self._dcut_total_trimmed_tokens}\n"
    )
    try:
        with open(self._dcut_trim_stats_out, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        logger.warning("D-Cut: failed to write trim stats to %s: %s", self._dcut_trim_stats_out, exc)


def _dcut_truncate(self, scheduler_output):
    ctrl = self._verify_adaptive_controller
    if ctrl is None or not scheduler_output.scheduled_spec_decode_tokens:
        return scheduler_output
    new_spec = scheduler_output.scheduled_spec_decode_tokens.copy()
    new_num_sched = scheduler_output.num_scheduled_tokens.copy()
    tokens_delta = 0
    trimmed_reqs = 0
    scheduled_reqs = len(new_spec)
    original_total_tokens = scheduler_output.total_num_scheduled_tokens
    for req_id, draft_toks in list(new_spec.items()):
        adaptive_len = ctrl.get_adaptive_draft_len(req_id)
        if adaptive_len is not None and adaptive_len < len(draft_toks):
            diff = len(draft_toks) - adaptive_len
            tokens_delta += diff
            trimmed_reqs += 1
            new_num_sched[req_id] -= diff
            if adaptive_len == 0:
                del new_spec[req_id]
            else:
                new_spec[req_id] = draft_toks[:adaptive_len]
    if tokens_delta > 0:
        scheduler_output = replace(
            scheduler_output,
            scheduled_spec_decode_tokens=new_spec,
            num_scheduled_tokens=new_num_sched,
            total_num_scheduled_tokens=(scheduler_output.total_num_scheduled_tokens - tokens_delta),
        )
    _dcut_write_trim_stats(
        self,
        batch_size=getattr(self.input_batch, "num_reqs", scheduled_reqs),
        trimmed_tokens=tokens_delta,
        trimmed_reqs=trimmed_reqs,
        scheduled_reqs=scheduled_reqs,
        total_scheduled_tokens=original_total_tokens,
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
    if self._adaptive_active and self._verify_adaptive_controller is not None:
        assert self._adaptive_probs_pinned is not None
        self._verify_adaptive_controller.process_draft_output(
            selected_probs=self._adaptive_probs_pinned[: self._adaptive_num_reqs],
            req_ids=self._adaptive_req_ids,
            active_draft_req_ids=self._adaptive_active,
            batch_size=self._adaptive_num_reqs,
        )


def profile_adaptive_cost(self) -> None:
    ctrl = getattr(self, "_verify_adaptive_controller", None)
    if ctrl is not None:
        if getattr(self, "_dcut_cost_profile_done", False):
            logger.info("D-Cut cost profiling SKIP: already done.")
            return
        if getattr(self, "_dcut_cost_profile_failed", False):
            logger.info("D-Cut cost profiling SKIP: previous attempt failed.")
            return
        logger.info(
            "D-Cut cost profiling START: config=%s cost_table_out=%s "
            "trim_stats_out=%s stat_every=%s profile_force_eager=%s cudagraph_mode=%s",
            os.getenv(ENV_CONFIG),
            os.getenv("VLLM_DCUT_COST_TABLE_OUT"),
            os.getenv(ENV_TRIM_STATS_OUT),
            os.getenv(ENV_STAT_EVERY),
            getattr(self, "_dcut_profile_force_eager", None),
            getattr(getattr(self, "compilation_config", None), "cudagraph_mode", None),
        )
        try:
            ctrl.profile_cost_table(self)
            self._dcut_cost_profile_done = True
            logger.info(
                "D-Cut cost profiling END: entries=%d json_out=%s markdown_out=%s",
                len(getattr(ctrl, "_cost_table", {})),
                os.getenv("VLLM_DCUT_COST_TABLE_OUT"),
                os.getenv("VLLM_DCUT_COST_TABLE_MD_OUT"),
            )
        except Exception:
            self._dcut_cost_profile_failed = True
            logger.exception("D-Cut cost profiling FAILED; adaptive trimming will stay disabled until restart.")
            raise


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
    assert num_tokens_unpadded <= self.max_num_tokens
    logger.info(
        "D-Cut adaptive profile dispatch: scheduled_tokens=%s sum_query_len=%d "
        "force_eager=%s configured_cudagraph_mode=%s",
        scheduled_tokens,
        num_tokens_unpadded,
        getattr(self, "_dcut_profile_force_eager", False),
        getattr(getattr(self, "compilation_config", None), "cudagraph_mode", None),
    )
    _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = self._determine_batch_execution_and_padding(
        num_tokens=num_tokens_unpadded,
        num_reqs=num_reqs,
        num_scheduled_tokens_np=num_scheduled_tokens,
        max_num_scheduled_tokens=max_query_len,
        use_cascade_attn=False,
        allow_microbatching=False,
        force_eager=getattr(self, "_dcut_profile_force_eager", False),
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
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)
    with self.synchronize_input_prep():
        self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)
        cum_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
        self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
        self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(cum_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        if getattr(self, "_has_gdn", False):
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(cum_num_tokens[-1])
            self.gdn_query_start_loc.copy_to_gpu()
        self.input_batch.block_table.commit_block_table(num_reqs_padded)
        if self.speculative_config is not None:
            draft_len = max_query_len - 1
            self.num_decode_draft_tokens.np[:num_reqs] = draft_len
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
            self.num_accepted_tokens.gpu[:num_reqs] = max_query_len
            self.num_accepted_tokens.gpu[num_reqs:].fill_(1)
    ascend_attention_mod = sys.modules.get("vllm_ascend.attention.attention_v1")
    if ascend_attention_mod is not None and hasattr(ascend_attention_mod, "AscendAttentionState"):
        self.attn_state = ascend_attention_mod.AscendAttentionState.DecodeOnly
    pad_attn = _cudagraph_mode == CUDAGraphMode.FULL
    attn_kwargs = {
        "num_tokens": num_tokens_unpadded,
        "num_tokens_padded": num_tokens_padded if pad_attn else None,
        "num_reqs": num_reqs_padded,
        "num_reqs_padded": num_reqs_padded,
        "max_query_len": max_query_len,
        "ubatch_slices": ubatch_slices_padded if pad_attn else ubatch_slices,
        "for_cudagraph_capture": False,
        "slot_mappings": slot_mappings_by_group,
        "use_spec_decode": self.speculative_config is not None,
        "num_scheduled_tokens_np": num_scheduled_tokens,
    }
    build_attn_params = inspect.signature(self._build_attention_metadata).parameters
    attn_kwargs = {name: value for name, value in attn_kwargs.items() if name in build_attn_params}
    attn_metadata, _ = self._build_attention_metadata(**attn_kwargs)
    if ubatch_slices_padded is not None:
        num_tokens_padded = ubatch_slices_padded[0].num_tokens
    if num_tokens_across_dp is not None:
        num_tokens_across_dp[:] = num_tokens_padded
    model_kwargs = self._init_model_kwargs()
    use_embeds = self.enable_prompt_embeds or (self.supports_mm_inputs and not self.model_config.is_encoder_decoder)
    input_ids = None if use_embeds else self.input_ids.gpu[:num_tokens_padded]
    inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded] if use_embeds else None
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
    _mode_names = {
        CUDAGraphMode.FULL: "FCG",
        CUDAGraphMode.PIECEWISE: "PCG",
        CUDAGraphMode.NONE: "eager",
    }
    avg_ms = 0.0
    ascend_ctx_mod = sys.modules.get("vllm_ascend.ascend_forward_context")
    if ascend_ctx_mod is not None and hasattr(ascend_ctx_mod, "set_ascend_forward_context"):
        forward_context_manager = ascend_ctx_mod.set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            in_profile_run=False,
            num_actual_tokens=num_tokens_padded,
            aclgraph_runtime_mode=_cudagraph_mode,
            batch_descriptor=batch_desc,
            model_instance=self.model,
            input_ids=input_ids,
        )
    else:
        forward_context_manager = set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=_cudagraph_mode,
            batch_descriptor=batch_desc,
            ubatch_slices=ubatch_slices_padded,
            slot_mapping=slot_mappings,
        )

    logger.info(
        "D-Cut adaptive profile run: runtime_mode=%s batch_descriptor=%s padded_tokens=%d padded_reqs=%d",
        _mode_names.get(_cudagraph_mode, str(_cudagraph_mode)),
        batch_desc,
        num_tokens_padded,
        num_reqs_padded,
    )
    with forward_context_manager:

        def _forward() -> None:
            # Use the runner forward path instead of calling ``self.model``
            # directly. On Ascend this preserves the same graph/context
            # handling used by warmup/capture/dummy runs, including FULL graph
            # parameter updates and PIECEWISE compiled subgraph dispatch.
            if hasattr(self, "_model_forward"):
                self._model_forward(
                    num_tokens_padded,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            else:
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
        samples_ms: list[float] = []
        for _ in range(max(n_measure, 0)):
            start_time = time.perf_counter()
            _forward()
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - start_time) * 1000.0)
        if samples_ms:
            samples = np.asarray(samples_ms, dtype=np.float64)
            avg_ms = float(samples.mean())
            timing_stats = {
                "avg_ms": avg_ms,
                "median_ms": float(np.median(samples)),
                "min_ms": float(samples.min()),
                "max_ms": float(samples.max()),
                "std_ms": float(samples.std()),
                "samples_ms": samples_ms,
            }
        else:
            timing_stats = {
                "avg_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
                "samples_ms": [],
            }
    return (
        _mode_names.get(_cudagraph_mode, str(_cudagraph_mode)),
        avg_ms,
        int(num_tokens_padded),
        timing_stats,
    )


def _dcut_profile_runner_if_needed(runner, reason: str) -> None:
    if runner is None or not hasattr(runner, "profile_adaptive_cost"):
        logger.info("D-Cut %s hook reached but model_runner/profile_adaptive_cost is unavailable.", reason)
        return
    if getattr(runner, "_verify_adaptive_controller", None) is None:
        logger.info("D-Cut %s hook reached but adaptive controller is not enabled on runner.", reason)
        return
    if getattr(runner, "_dcut_cost_profile_done", False):
        logger.info("D-Cut %s hook reached; cost table already profiled.", reason)
        return
    if getattr(runner, "_dcut_cost_profile_failed", False):
        logger.info("D-Cut %s hook reached; previous cost profiling attempt failed.", reason)
        return
    logger.info("D-Cut cost profiling LAZY START from %s: startup warmup hook is generating the cost table.", reason)
    try:
        runner.profile_adaptive_cost()
    except Exception as e:
        logger.error(
            "D-Cut: cost profiling failed during %s; execute_model fallback will not retry until restart: %s", reason, e
        )
        ctrl = getattr(runner, "_verify_adaptive_controller", None)
        if ctrl is not None:
            ctrl._cost_table.clear()
            ctrl._sorted_bs.clear()
            ctrl._sorted_sql_per_bs.clear()


def _patch_worker_class(worker_cls, class_label: str) -> None:
    if worker_cls.__dict__.get("_dcut_worker_hooks_patched", False):
        return

    if hasattr(worker_cls, "compile_or_warm_up_model"):
        original_warmup = worker_cls.compile_or_warm_up_model

        def compile_or_warm_up_model(self, *a, **k):
            logger.info("D-Cut worker warmup hook reached: %s.compile_or_warm_up_model", class_label)
            ret = original_warmup(self, *a, **k)
            _dcut_profile_runner_if_needed(
                getattr(self, "model_runner", None), f"{class_label}.compile_or_warm_up_model"
            )
            return ret

        worker_cls.compile_or_warm_up_model = compile_or_warm_up_model

    if hasattr(worker_cls, "execute_model"):
        original_worker_execute = worker_cls.execute_model

        def execute_model(self, scheduler_output, *args, **kwargs):
            _dcut_profile_runner_if_needed(getattr(self, "model_runner", None), f"{class_label}.execute_model")
            return original_worker_execute(self, scheduler_output, *args, **kwargs)

        worker_cls.execute_model = execute_model

    worker_cls._dcut_worker_hooks_patched = True
    logger.info("D-Cut patched worker hooks for %s", class_label)


def _watch_ascend_worker_module(module) -> None:
    if getattr(module, "_dcut_worker_module_watch_installed", False):
        return

    base_cls = module.__class__

    class DcutWorkerModuleWatch(base_cls):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "Worker":
                _patch_worker_class(value, "vllm_ascend.worker.worker.Worker")

    if issubclass(base_cls, types.ModuleType):
        module.__class__ = DcutWorkerModuleWatch
        module._dcut_worker_module_watch_installed = True
        logger.info("D-Cut Ascend worker module watch installed; waiting for Worker class definition.")


def _patch_loaded_ascend_worker() -> None:
    module = sys.modules.get("vllm_ascend.worker.worker")
    if module is None:
        logger.info("D-Cut Ascend worker hooks pending: vllm_ascend.worker.worker is not loaded yet.")
        return
    worker_cls = getattr(module, "Worker", None)
    if worker_cls is None:
        logger.info("D-Cut Ascend worker hooks pending: Worker class is not available yet.")
        _watch_ascend_worker_module(module)
        return
    _patch_worker_class(worker_cls, "vllm_ascend.worker.worker.Worker")


def _dcut_prepare_execute_model(self, scheduler_output):
    if getattr(self, "_verify_adaptive_controller", None) is None:
        return scheduler_output
    if not getattr(self, "_dcut_cost_profile_done", False) and not getattr(self, "_dcut_cost_profile_failed", False):
        logger.info(
            "D-Cut cost profiling LAZY START from execute_model: warmup hook "
            "did not produce a cost table before this batch."
        )
        # profile_adaptive_cost already logs the stack and marks failure.
        with suppress(Exception):
            self.profile_adaptive_cost()
    return _dcut_truncate(self, scheduler_output)


def _dcut_patch_execute_model(cls, class_label: str) -> None:
    if cls.__dict__.get("_dcut_execute_model_patched", False):
        return
    original_execute_model = cls.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        scheduler_output = _dcut_prepare_execute_model(self, scheduler_output)
        return original_execute_model(self, scheduler_output, *args, **kwargs)

    cls.execute_model = execute_model
    cls._dcut_execute_model_patched = True
    logger.info("D-Cut patched execute_model for %s", class_label)


def _patch_proposer() -> None:
    import vllm.v1.spec_decode.llm_base_proposer as m

    P = m.SpecDecodeBaseProposer
    if getattr(P, "_dcut_patched", False):
        return
    P.needs_draft_probs = False
    P._last_selected_probs = None

    @staticmethod
    def _gather_selected_probs(logits, token_ids, full_probs):
        idx = token_ids.long().unsqueeze(-1)
        if full_probs is not None:
            return full_probs.gather(-1, idx).squeeze(-1)
        chosen = logits.gather(-1, idx).squeeze(-1)
        return (chosen - logits.logsumexp(dim=-1)).exp()

    def take_last_selected_probs(self):
        return getattr(self, "_last_selected_probs", None)

    _orig_sample = P._sample_draft_tokens

    def _sample_draft_tokens(self, hidden_states, sampling_metadata):
        self._last_selected_probs = None
        out = _orig_sample(self, hidden_states, sampling_metadata)
        if getattr(self, "needs_draft_probs", False) and getattr(self, "parallel_drafting", False):
            token_ids = out[0]
            full_probs = out[1] if len(out) > 1 else None
            try:
                logits = None if full_probs is not None else self.model.compute_logits(hidden_states)
                sel = P._gather_selected_probs(logits, token_ids, full_probs)
                self._last_selected_probs = sel.view(-1, self.num_speculative_tokens).contiguous()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("D-Cut: gather selected probs failed: %s", e)
                self._last_selected_probs = None
        return out

    P._gather_selected_probs = _gather_selected_probs
    P.take_last_selected_probs = take_last_selected_probs
    P._sample_draft_tokens = _sample_draft_tokens
    P._dcut_patched = True


def _patch_runner() -> None:
    import vllm.v1.worker.gpu_model_runner as m

    R = m.GPUModelRunner
    if getattr(R, "_dcut_patched", False):
        return
    _orig_init = R.__init__

    def __init__(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            _dcut_init_controller(self)
            # For Ascend, the actual request path is usually an NPUModelRunner
            # subclass that overrides GPUModelRunner.execute_model. Patch the
            # concrete class lazily here, after vllm_ascend imports have
            # completed, to avoid circular imports during plugin installation.
            class_label = f"{self.__class__.__module__}.{self.__class__.__name__}"
            logger.info("D-Cut runner init concrete class: %s", class_label)
            if self.__class__ is not R:
                _dcut_patch_execute_model(self.__class__, class_label)
            _patch_loaded_ascend_worker()
        except Exception as e:
            logger.error("D-Cut init failed; running vanilla: %s", e)
            self._verify_adaptive_controller = None

    _orig_sample_tokens = R.sample_tokens

    def sample_tokens(self, *a, **k):
        out = _orig_sample_tokens(self, *a, **k)
        if getattr(self, "_adaptive_probs_pending", False):
            try:
                _maybe_process_adaptive_probs(self)
            except Exception as e:
                logger.warning("D-Cut: process probs failed: %s", e)
                self._adaptive_probs_pending = False
        return out

    _orig_copy = R._copy_draft_token_ids_to_cpu

    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        _orig_copy(self, scheduler_output, zeros_only)
        if getattr(self, "_verify_adaptive_controller", None) is not None:
            try:
                _dcut_queue_probs(self, zeros_only)
            except Exception as e:
                logger.warning("D-Cut: queue probs failed: %s", e)

    _orig_update = R._update_states

    def _update_states(self, scheduler_output):
        ret = _orig_update(self, scheduler_output)
        ctrl = getattr(self, "_verify_adaptive_controller", None)
        if ctrl is not None:
            for rid in scheduler_output.finished_req_ids:
                ctrl.invalidate(rid)
        return ret

    R.__init__ = __init__
    _dcut_patch_execute_model(R, "vllm.v1.worker.gpu_model_runner.GPUModelRunner")
    R.sample_tokens = sample_tokens
    R._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    R._update_states = _update_states
    R._adaptive_profile_run = _adaptive_profile_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_patched = True


def _patch_worker() -> None:
    import vllm.v1.worker.gpu_worker as m

    _patch_worker_class(m.Worker, "vllm.v1.worker.gpu_worker.Worker")
    _patch_loaded_ascend_worker()


def install(*args, **kwargs) -> None:
    """vLLM general-plugin entrypoint. Idempotent; safe to call per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        logger.info(
            "D-Cut install requested: VLLM_DCUT_CONFIG=%s "
            "VLLM_DCUT_COST_TABLE_OUT=%s VLLM_DCUT_TRIM_STATS_OUT=%s "
            "VLLM_DCUT_STAT_EVERY=%s VLLM_DCUT_PROFILE_FORCE_EAGER=%s "
            "VLLM_PLUGINS=%s",
            os.getenv(ENV_CONFIG),
            os.getenv("VLLM_DCUT_COST_TABLE_OUT"),
            os.getenv(ENV_TRIM_STATS_OUT),
            os.getenv(ENV_STAT_EVERY),
            os.getenv(ENV_PROFILE_FORCE_EAGER),
            os.getenv("VLLM_PLUGINS"),
        )
        _patch_proposer()
        _patch_runner()
        _patch_worker()
        _INSTALLED = True
        logger.info(
            "D-Cut adaptive-verify monkey patch installed "
            "(active only if VLLM_DCUT_CONFIG is set + method is dflash/PARD)."
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut install failed (vLLM continues normally): %s", e)

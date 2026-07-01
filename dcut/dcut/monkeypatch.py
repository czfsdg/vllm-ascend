# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextvars
import importlib.abc
import importlib.machinery
import inspect
import sys
import time
from copy import copy
from dataclasses import replace
from functools import wraps
from types import ModuleType

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from dcut.controller import VerifyAdaptiveController, dcut_enabled

_PATCHED_MODULES: set[str] = set()
_HOOK_INSTALLED = False
_TARGET_MODULES = {
    "vllm_ascend.attention.attention_v1",
    "vllm_ascend.spec_decode.llm_base_proposer",
    "vllm_ascend.worker.model_runner_v1",
    "vllm_ascend.worker.worker",
}
_ATTENTION_TIMING_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "dcut_attention_timing_context", default=None
)
_VERIFIER_BREAKDOWN_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "dcut_verifier_breakdown_context", default=None
)


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
    if module_name == "vllm_ascend.attention.attention_v1":
        _patch_attention_backend(module)
    elif module_name == "vllm_ascend.spec_decode.llm_base_proposer":
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
            self._last_selected_probs = (
                (chosen - logits.logsumexp(dim=-1)).exp().view(-1, self.num_speculative_tokens).contiguous()
            )
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


def _patch_attention_backend(module: ModuleType) -> None:
    for class_name in ("AscendAttentionBackendImpl", "AscendC8AttentionBackendImpl"):
        cls = getattr(module, class_name, None)
        if cls is None or getattr(cls, "_dcut_attention_timing_patched", False):
            continue
        original_forward = cls.forward

        @wraps(original_forward)
        def forward(self, *args, __dcut_original=original_forward, __dcut_class_name=class_name, **kwargs):
            return _dcut_time_attention_forward(__dcut_original, __dcut_class_name, self, args, kwargs)

        cls.forward = forward
        cls._dcut_attention_timing_patched = True


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
    original_build_attention_metadata = getattr(cls, "_build_attention_metadata", None)
    original_determine_batch_execution_and_padding = getattr(cls, "_determine_batch_execution_and_padding", None)
    original_model_forward = getattr(cls, "_model_forward", None)
    original_prepare_inputs = getattr(cls, "_prepare_inputs", None)
    original_preprocess = getattr(cls, "_preprocess", None)
    original_sanitize_placeholder_input_ids = getattr(cls, "_sanitize_placeholder_input_ids_for_forward", None)
    original_update_states = getattr(cls, "_update_states", None)
    original_update_full_graph_params = getattr(cls, "_update_full_graph_params_if_needed", None)
    original_all_gather_hidden_states_and_aux = getattr(cls, "_all_gather_hidden_states_and_aux", None)
    original_dummy_run = getattr(cls, "_dummy_run", None)
    supports_profile_num_scheduled_tokens = False
    if original_dummy_run is not None:
        original_dummy_params = inspect.signature(original_dummy_run).parameters
        supports_profile_num_scheduled_tokens = "profile_num_scheduled_tokens" in original_dummy_params

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
        self._dcut_verifier_timing_stats = {}
        self._dcut_model_forward_timing_stats = {}
        self._dcut_profile_num_scheduled_tokens_supported = supports_profile_num_scheduled_tokens
        self._dcut_missing_probs_warnings = 0
        self._dcut_fallback_probs_warnings = 0
        self._dcut_accepted_tokens_clamp_warnings = 0
        self._dcut_mixed_prefill_skip_warnings = 0
        self._dcut_high_concurrency_skip_warnings = 0
        _dcut_init_controller(self)
        _dcut_patch_model_compute_logits(self)
        _dcut_patch_model_forward_call(self)
        _dcut_patch_model_forward_modules(self)

    @wraps(original_execute_model)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        _dcut_patch_model_compute_logits(self)
        _dcut_patch_model_forward_call(self)
        _dcut_patch_model_forward_modules(self)
        scheduler_output = _dcut_truncate_scheduler_output(self, scheduler_output)
        time_verifier = _dcut_should_time_verifier(self, scheduler_output)
        breakdown_token = _dcut_start_verifier_breakdown(self, scheduler_output)
        attention_timing_token = _dcut_start_attention_timing(self, scheduler_output)
        if time_verifier or breakdown_token is not None:
            torch.npu.synchronize()
            start = time.perf_counter()
        try:
            result = original_execute_model(self, scheduler_output, intermediate_tensors)
        finally:
            if time_verifier or breakdown_token is not None:
                torch.npu.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if time_verifier:
                    _dcut_log_verifier_timing(self, scheduler_output, elapsed_ms)
                _dcut_finish_verifier_breakdown(scheduler_output, breakdown_token, elapsed_ms)
            _dcut_finish_attention_timing(scheduler_output, attention_timing_token)
        return result

    @wraps(original_sample_tokens)
    def sample_tokens(self, grammar_output):
        _dcut_maybe_process_probs(self)
        return original_sample_tokens(self, grammar_output)

    @wraps(original_copy_draft)
    def _copy_draft_token_ids_to_cpu(self, scheduler_output, zeros_only=False):
        original_copy_draft(self, scheduler_output, zeros_only=zeros_only)
        _dcut_queue_probs(self, zeros_only)

    if original_build_attention_metadata is not None:

        @wraps(original_build_attention_metadata)
        def _build_attention_metadata(self, *args, **kwargs):
            _dcut_log_attention_query_shape(self, args, kwargs)
            return _dcut_time_runner_phase(
                "build_attention_metadata", original_build_attention_metadata, self, args, kwargs
            )

    if original_determine_batch_execution_and_padding is not None:

        @wraps(original_determine_batch_execution_and_padding)
        def _determine_batch_execution_and_padding(self, *args, **kwargs):
            return _dcut_time_runner_phase(
                "determine_batch_execution", original_determine_batch_execution_and_padding, self, args, kwargs
            )

    if original_model_forward is not None:

        @wraps(original_model_forward)
        def _model_forward(self, *args, **kwargs):
            return _dcut_time_runner_phase("model_forward", original_model_forward, self, args, kwargs)

    if original_prepare_inputs is not None:

        @wraps(original_prepare_inputs)
        def _prepare_inputs(self, *args, **kwargs):
            return _dcut_time_runner_phase("prepare_inputs", original_prepare_inputs, self, args, kwargs)

    if original_preprocess is not None:

        @wraps(original_preprocess)
        def _preprocess(self, *args, **kwargs):
            return _dcut_time_runner_phase("preprocess", original_preprocess, self, args, kwargs)

    if original_sanitize_placeholder_input_ids is not None:

        @wraps(original_sanitize_placeholder_input_ids)
        def _sanitize_placeholder_input_ids_for_forward(self, *args, **kwargs):
            return _dcut_time_runner_phase(
                "sanitize_placeholder_input_ids", original_sanitize_placeholder_input_ids, self, args, kwargs
            )

    if original_update_states is not None:

        @wraps(original_update_states)
        def _update_states(self, *args, **kwargs):
            return _dcut_time_runner_phase("update_states", original_update_states, self, args, kwargs)

    if original_update_full_graph_params is not None:

        @wraps(original_update_full_graph_params)
        def _update_full_graph_params_if_needed(self, *args, **kwargs):
            return _dcut_time_runner_phase(
                "model_forward.update_full_graph_params", original_update_full_graph_params, self, args, kwargs
            )

    if original_all_gather_hidden_states_and_aux is not None:

        @wraps(original_all_gather_hidden_states_and_aux)
        def _all_gather_hidden_states_and_aux(hidden_states):
            return _dcut_time_callable_phase(
                "model_forward.all_gather_hidden_states",
                original_all_gather_hidden_states_and_aux,
                (hidden_states,),
                {},
            )

    if original_dummy_run is not None:

        @wraps(original_dummy_run)
        def _dummy_run(self, *args, skip_drafter=False, **kwargs):
            profile_num_scheduled_tokens = kwargs.get("profile_num_scheduled_tokens")
            if profile_num_scheduled_tokens is not None and not supports_profile_num_scheduled_tokens:
                raise RuntimeError(
                    "D-Cut cost profiling requires NPUModelRunner._dummy_run(profile_num_scheduled_tokens=...) "
                    "to preserve the intended batch/query shape. The installed vllm-ascend runtime does not "
                    "support that keyword, so profiling with default uniform_decode shape inference would produce "
                    "a misleading flat cost table. Update vllm-ascend or disable D-Cut cost profiling."
                )
            if not skip_drafter:
                return original_dummy_run(self, *args, **kwargs)
            drafter = getattr(self, "drafter", None)
            self.drafter = None
            try:
                return original_dummy_run(self, *args, **kwargs)
            finally:
                self.drafter = drafter

    cls.__init__ = __init__
    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    cls._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    if original_build_attention_metadata is not None:
        cls._build_attention_metadata = _build_attention_metadata
    if original_determine_batch_execution_and_padding is not None:
        cls._determine_batch_execution_and_padding = _determine_batch_execution_and_padding
    if original_model_forward is not None:
        cls._model_forward = _model_forward
    if original_prepare_inputs is not None:
        cls._prepare_inputs = _prepare_inputs
    if original_preprocess is not None:
        cls._preprocess = _preprocess
    if original_sanitize_placeholder_input_ids is not None:
        cls._sanitize_placeholder_input_ids_for_forward = _sanitize_placeholder_input_ids_for_forward
    if original_update_states is not None:
        cls._update_states = _update_states
    if original_update_full_graph_params is not None:
        cls._update_full_graph_params_if_needed = _update_full_graph_params_if_needed
    if original_all_gather_hidden_states_and_aux is not None:
        cls._all_gather_hidden_states_and_aux = staticmethod(_all_gather_hidden_states_and_aux)
    if original_dummy_run is not None:
        cls._dummy_run = _dummy_run
    cls.profile_dcut_cost = _dcut_profile_cost
    cls._dcut_patched = True


def _dcut_get_call_value(args, kwargs, index: int, name: str, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _dcut_to_int_list(values) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _dcut_patch_model_compute_logits(runner) -> None:
    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "compute_logits") or getattr(model, "_dcut_compute_logits_patched", False):
        return
    original_compute_logits = model.compute_logits

    @wraps(original_compute_logits)
    def compute_logits(*args, **kwargs):
        return _dcut_time_callable_phase("compute_logits", original_compute_logits, args, kwargs)

    model.compute_logits = compute_logits
    model._dcut_compute_logits_patched = True


def _dcut_patch_model_forward_call(runner) -> None:
    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "forward") or getattr(model, "_dcut_forward_call_patched", False):
        return
    original_forward = model.forward

    @wraps(original_forward)
    def forward(*args, **kwargs):
        return _dcut_time_callable_phase("model_forward.model_call", original_forward, args, kwargs)

    model.forward = forward
    model._dcut_forward_call_patched = True


def _dcut_patch_model_forward_modules(runner) -> None:
    controller = getattr(runner, "_dcut_controller", None)
    model = getattr(runner, "model", None)
    if (
        controller is None
        or model is None
        or not controller.config.log_model_forward_module_breakdown
        or getattr(model, "_dcut_module_timing_patched", False)
    ):
        return
    handles = []
    leaf_count = 0
    layer_block_count = 0
    gdn_count = 0
    for name, module in model.named_modules():
        if not name:
            continue
        layer_block_name = _dcut_layer_block_display_name(name)
        if layer_block_name is not None:
            handles.append(module.register_forward_pre_hook(_dcut_make_layer_block_pre_hook(layer_block_name)))
            handles.append(module.register_forward_hook(_dcut_make_layer_block_post_hook(layer_block_name)))
            layer_block_count += 1
        if _dcut_should_trace_gdn_module(name, module):
            handles.append(module.register_forward_pre_hook(_dcut_make_gdn_pre_hook(name), with_kwargs=True))
            handles.append(module.register_forward_hook(_dcut_make_gdn_post_hook(name), with_kwargs=True))
            gdn_count += 1
        if not _dcut_should_time_module(module):
            continue
        handles.append(module.register_forward_pre_hook(_dcut_make_module_pre_hook(name)))
        handles.append(module.register_forward_hook(_dcut_make_module_post_hook(name)))
        leaf_count += 1
    model._dcut_module_timing_handles = handles
    model._dcut_module_timing_patched = True
    logger.info(
        "D-Cut: installed model-forward timing hooks leaf_modules=%d layer_blocks=%d gdn_modules=%d",
        leaf_count,
        layer_block_count,
        gdn_count,
    )


def _dcut_layer_block_display_name(name: str) -> str | None:
    parts = name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and parts[idx + 1].isdigit():
            if idx + 2 == len(parts):
                return f"layers.{parts[idx + 1]}"
            return None
    return None


def _dcut_should_time_module(module) -> bool:
    try:
        next(module.children())
        return False
    except StopIteration:
        return True


def _dcut_should_trace_gdn_module(name: str, module) -> bool:
    class_name = module.__class__.__name__.lower()
    return "gateddeltanet" in class_name or name.endswith(".linear_attn")


def _dcut_make_gdn_pre_hook(name: str):
    def pre_hook(module, inputs, kwargs):
        context = _VERIFIER_BREAKDOWN_CONTEXT.get()
        if context is None:
            return
        torch.npu.synchronize()
        context["gdn_stack"].setdefault(id(module), []).append(
            {
                "start": time.perf_counter(),
                "name": name,
                "class": module.__class__.__name__,
                "inputs": _dcut_summarize_gdn_inputs(inputs, kwargs),
                "metadata": _dcut_summarize_gdn_metadata(module),
            }
        )

    return pre_hook


def _dcut_make_gdn_post_hook(name: str):
    def post_hook(module, _inputs, _kwargs, output):
        context = _VERIFIER_BREAKDOWN_CONTEXT.get()
        if context is None:
            return
        stack = context["gdn_stack"].get(id(module))
        if not stack:
            return
        torch.npu.synchronize()
        record = stack.pop()
        elapsed_ms = (time.perf_counter() - record["start"]) * 1000.0
        context["gdn_modules"].append(
            {
                "name": name,
                "class": record["class"],
                "elapsed_ms": elapsed_ms,
                "inputs": record["inputs"],
                "output": _dcut_summarize_value(output),
                "metadata": record["metadata"],
            }
        )

    return post_hook


def _dcut_summarize_gdn_inputs(inputs, kwargs) -> dict:
    hidden_states = kwargs.get("hidden_states") if kwargs else None
    output = kwargs.get("output") if kwargs else None
    if hidden_states is None and inputs:
        hidden_states = inputs[0]
    if output is None and len(inputs) > 1:
        output = inputs[1]
    return {
        "hidden_states": _dcut_summarize_value(hidden_states),
        "output": _dcut_summarize_value(output),
    }


def _dcut_summarize_gdn_metadata(module) -> dict:
    forward_context = get_forward_context()
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if attn_metadata is None:
        return {}
    if isinstance(attn_metadata, dict):
        prefix = getattr(module, "prefix", None)
        if prefix in attn_metadata:
            attn_metadata = attn_metadata[prefix]
        else:
            return {
                "type": "dict",
                "len": len(attn_metadata),
                "prefix": prefix,
                "keys": list(attn_metadata.keys())[:4],
            }
    return {
        key: _dcut_summarize_metadata_value(getattr(attn_metadata, key, None))
        for key in (
            "num_actual_tokens",
            "num_input_tokens",
            "num_spec_decodes",
            "num_prefills",
            "num_decodes",
            "max_query_len",
            "actual_seq_lengths_q",
            "query_start_loc",
            "spec_query_start_loc",
            "spec_state_indices_tensor",
            "spec_sequence_masks",
            "num_accepted_tokens",
        )
        if hasattr(attn_metadata, key)
    }


def _dcut_summarize_metadata_value(value):
    summary = _dcut_summarize_value(value)
    if isinstance(value, torch.Tensor):
        return summary
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "first": value[0] if value else None,
            "last": value[-1] if value else None,
        }
    return summary


def _dcut_top_gdn_records(records: list[dict], top_k: int) -> list[dict]:
    if not records or top_k < 1:
        return []
    top_records = sorted(records, key=lambda item: item["elapsed_ms"], reverse=True)[:top_k]
    return [
        {
            "name": record["name"],
            "class": record["class"],
            "elapsed_ms": round(record["elapsed_ms"], 3),
            "inputs": record["inputs"],
            "output": record["output"],
            "metadata": record["metadata"],
        }
        for record in top_records
    ]


def _dcut_make_layer_block_pre_hook(name: str):
    def pre_hook(module, _inputs):
        context = _VERIFIER_BREAKDOWN_CONTEXT.get()
        if context is None:
            return
        torch.npu.synchronize()
        context["layer_block_stack"].setdefault(id(module), []).append(time.perf_counter())

    return pre_hook


def _dcut_make_layer_block_post_hook(name: str):
    def post_hook(module, _inputs, _output):
        context = _VERIFIER_BREAKDOWN_CONTEXT.get()
        if context is None:
            return
        stack = context["layer_block_stack"].get(id(module))
        if not stack:
            return
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - stack.pop()) * 1000.0
        context["layer_blocks"][name] = context["layer_blocks"].get(name, 0.0) + elapsed_ms

    return post_hook


def _dcut_make_module_pre_hook(name: str):
    def pre_hook(module, _inputs):
        context = _VERIFIER_BREAKDOWN_CONTEXT.get()
        if context is None:
            return
        torch.npu.synchronize()
        context["module_stack"].setdefault(id(module), []).append(time.perf_counter())

    return pre_hook


def _dcut_make_module_post_hook(name: str):
    def post_hook(module, _inputs, _output):
        context = _VERIFIER_BREAKDOWN_CONTEXT.get()
        if context is None:
            return
        stack = context["module_stack"].get(id(module))
        if not stack:
            return
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - stack.pop()) * 1000.0
        class_name = module.__class__.__name__
        module_name = f"{name}:{class_name}"
        context["module_classes"][class_name] = context["module_classes"].get(class_name, 0.0) + elapsed_ms
        context["module_names"][module_name] = context["module_names"].get(module_name, 0.0) + elapsed_ms

    return post_hook


def _dcut_verifier_breakdown_top_k() -> int:
    context = _VERIFIER_BREAKDOWN_CONTEXT.get()
    if context is None:
        return 0
    return context["module_top_k"]


def _dcut_top_level_phase_sum(phases: dict[str, float]) -> float:
    return sum(value for name, value in phases.items() if "." not in name)


def _dcut_top_timing_items(values: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    if not values or top_k < 1:
        return []
    top_items = sorted(values.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(name, round(value, 3)) for name, value in top_items]


def _dcut_start_verifier_breakdown(runner, scheduler_output):
    controller = getattr(runner, "_dcut_controller", None)
    if (
        controller is None
        or not scheduler_output.scheduled_spec_decode_tokens
        or not controller.should_log_verifier_breakdown()
    ):
        return None
    spec_lens = [len(tokens) for tokens in scheduler_output.scheduled_spec_decode_tokens.values()]
    context = {
        "module_classes": {},
        "module_names": {},
        "module_stack": {},
        "layer_blocks": {},
        "layer_block_stack": {},
        "gdn_modules": [],
        "gdn_stack": {},
        "phases": {},
        "module_top_k": controller.config.log_model_forward_module_top_k,
        "log_input_shapes": controller.config.log_function_input_shapes,
        "input_shapes_max_items": controller.config.log_function_input_shapes_max_items,
        "spec_tokens": sum(spec_lens),
        "spec_reqs": len(spec_lens),
        "runner": runner,
    }
    return _VERIFIER_BREAKDOWN_CONTEXT.set(context)


def _dcut_finish_verifier_breakdown(scheduler_output, token, total_elapsed_ms: float) -> None:
    if token is None:
        return
    context = _VERIFIER_BREAKDOWN_CONTEXT.get()
    _VERIFIER_BREAKDOWN_CONTEXT.reset(token)
    if context is None:
        return
    phases = context["phases"]
    phase_sum_ms = _dcut_top_level_phase_sum(phases)
    untracked_ms = total_elapsed_ms - phase_sum_ms
    spec_reqs = context["spec_reqs"]
    avg_spec_len = context["spec_tokens"] / spec_reqs if spec_reqs else 0.0
    top_k = context["module_top_k"]
    model_forward_stats = _dcut_update_model_forward_stats(
        context.get("runner"), scheduler_output, context["spec_tokens"], phases.get("model_forward", 0.0)
    )
    logger.info(
        "D-Cut verifier breakdown: total_ms=%.3f tracked_ms=%.3f untracked_ms=%.3f "
        "total_tokens=%d spec_reqs=%d spec_tokens=%d avg_spec_len=%.2f num_reqs=%d "
        "phases=%s model_forward_shape_stats=%s layer_blocks=%s gdn_modules=%s module_classes=%s module_names=%s",
        total_elapsed_ms,
        phase_sum_ms,
        untracked_ms,
        scheduler_output.total_num_scheduled_tokens,
        spec_reqs,
        context["spec_tokens"],
        avg_spec_len,
        len(scheduler_output.num_scheduled_tokens),
        {name: round(value, 3) for name, value in sorted(phases.items())},
        model_forward_stats,
        _dcut_top_timing_items(context["layer_blocks"], top_k),
        _dcut_top_gdn_records(context["gdn_modules"], top_k),
        _dcut_top_timing_items(context["module_classes"], top_k),
        _dcut_top_timing_items(context["module_names"], top_k),
    )


def _dcut_update_model_forward_stats(
    runner, scheduler_output, spec_tokens: int, elapsed_ms: float
) -> dict[str, float | int]:
    total_tokens = scheduler_output.total_num_scheduled_tokens
    query_lens = _dcut_scheduler_query_lens(scheduler_output)
    shape_key = (len(query_lens), total_tokens, spec_tokens, max(query_lens, default=0))
    stats = getattr(runner, "_dcut_model_forward_timing_stats", None) if runner is not None else None
    if stats is None:
        return {
            "elapsed_ms": round(elapsed_ms, 3),
            "ms_per_token": round(elapsed_ms / total_tokens, 4) if total_tokens else 0.0,
        }
    count, total_ms, min_ms, max_ms = stats.get(shape_key, (0, 0.0, float("inf"), 0.0))
    count += 1
    total_ms += elapsed_ms
    min_ms = min(min_ms, elapsed_ms)
    max_ms = max(max_ms, elapsed_ms)
    stats[shape_key] = (count, total_ms, min_ms, max_ms)
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "ms_per_token": round(elapsed_ms / total_tokens, 4) if total_tokens else 0.0,
        "shape_count": count,
        "shape_avg_ms": round(total_ms / count, 3),
        "shape_min_ms": round(min_ms, 3),
        "shape_max_ms": round(max_ms, 3),
    }


def _dcut_log_function_input_shapes(phase_name: str, args, kwargs, context: dict) -> None:
    if not context.get("log_input_shapes", False):
        return
    max_items = context.get("input_shapes_max_items", 8)
    logger.info(
        "D-Cut function input shapes: phase=%s args=%s kwargs=%s",
        phase_name,
        _dcut_summarize_call_values(args, max_items),
        _dcut_summarize_call_values(kwargs, max_items),
    )


def _dcut_summarize_call_values(values, max_items: int):
    if isinstance(values, dict):
        return {str(key): _dcut_summarize_value(value) for key, value in list(values.items())[:max_items]}
    return [_dcut_summarize_value(value) for value in list(values)[:max_items]]


def _dcut_summarize_value(value):
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_dcut_summarize_value(item) for item in list(value)[:4]]
        return {"type": type(value).__name__, "len": len(value), "items": items}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "len": len(value),
            "items": {str(key): _dcut_summarize_value(item) for key, item in list(value.items())[:4]},
        }
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": value.__class__.__name__, "shape": tuple(shape)}
    if hasattr(value, "__len__"):
        try:
            return {"type": value.__class__.__name__, "len": len(value)}
        except TypeError:
            pass
    return {"type": value.__class__.__name__}


def _dcut_time_callable_phase(phase_name: str, original_callable, args, kwargs):
    context = _VERIFIER_BREAKDOWN_CONTEXT.get()
    if context is None:
        return original_callable(*args, **kwargs)
    _dcut_log_function_input_shapes(phase_name, args, kwargs, context)
    torch.npu.synchronize()
    start = time.perf_counter()
    result = original_callable(*args, **kwargs)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    phases = context["phases"]
    phases[phase_name] = phases.get(phase_name, 0.0) + elapsed_ms
    return result


def _dcut_time_runner_phase(phase_name: str, original_method, runner, args, kwargs):
    context = _VERIFIER_BREAKDOWN_CONTEXT.get()
    if context is None:
        return original_method(runner, *args, **kwargs)
    _dcut_log_function_input_shapes(phase_name, args, kwargs, context)
    torch.npu.synchronize()
    start = time.perf_counter()
    result = original_method(runner, *args, **kwargs)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    phases = context["phases"]
    phases[phase_name] = phases.get(phase_name, 0.0) + elapsed_ms
    return result


def _dcut_start_attention_timing(runner, scheduler_output):
    controller = getattr(runner, "_dcut_controller", None)
    if (
        controller is None
        or not scheduler_output.scheduled_spec_decode_tokens
        or not controller.should_log_attention_timing()
    ):
        return None
    spec_lens = [len(tokens) for tokens in scheduler_output.scheduled_spec_decode_tokens.values()]
    context = {
        "elapsed_ms": 0.0,
        "calls": 0,
        "query_tokens": 0,
        "max_query_tokens": 0,
        "impls": {},
        "spec_tokens": sum(spec_lens),
        "spec_reqs": len(spec_lens),
    }
    return _ATTENTION_TIMING_CONTEXT.set(context)


def _dcut_finish_attention_timing(scheduler_output, token) -> None:
    if token is None:
        return
    context = _ATTENTION_TIMING_CONTEXT.get()
    _ATTENTION_TIMING_CONTEXT.reset(token)
    if context is None:
        return
    spec_reqs = context["spec_reqs"]
    avg_spec_len = context["spec_tokens"] / spec_reqs if spec_reqs else 0.0
    logger.info(
        "D-Cut attention timing: elapsed_ms=%.3f calls=%d query_tokens=%d "
        "max_query_tokens=%d total_tokens=%d spec_reqs=%d spec_tokens=%d "
        "avg_spec_len=%.2f num_reqs=%d impls=%s",
        context["elapsed_ms"],
        context["calls"],
        context["query_tokens"],
        context["max_query_tokens"],
        scheduler_output.total_num_scheduled_tokens,
        spec_reqs,
        context["spec_tokens"],
        avg_spec_len,
        len(scheduler_output.num_scheduled_tokens),
        context["impls"],
    )


def _dcut_time_attention_forward(original_forward, class_name: str, attention_impl, args, kwargs):
    context = _ATTENTION_TIMING_CONTEXT.get()
    if context is None:
        return original_forward(attention_impl, *args, **kwargs)
    query = _dcut_get_call_value(args, kwargs, 1, "query")
    query_tokens = int(query.shape[0]) if query is not None and hasattr(query, "shape") else 0
    torch.npu.synchronize()
    start = time.perf_counter()
    result = original_forward(attention_impl, *args, **kwargs)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    context["elapsed_ms"] += elapsed_ms
    context["calls"] += 1
    context["query_tokens"] += query_tokens
    context["max_query_tokens"] = max(context["max_query_tokens"], query_tokens)
    context["impls"][class_name] = context["impls"].get(class_name, 0) + 1
    return result


def _dcut_log_attention_query_shape(runner, args, kwargs) -> None:
    use_spec_decode = bool(_dcut_get_call_value(args, kwargs, 7, "use_spec_decode", False))
    controller = getattr(runner, "_dcut_controller", None)
    if controller is None or not use_spec_decode or not controller.should_log_attention_query_shape():
        return
    num_tokens = _dcut_get_call_value(args, kwargs, 0, "num_tokens")
    num_reqs = _dcut_get_call_value(args, kwargs, 1, "num_reqs")
    max_query_len = _dcut_get_call_value(args, kwargs, 2, "max_query_len")
    num_tokens_padded = _dcut_get_call_value(args, kwargs, 3, "num_tokens_padded", num_tokens)
    num_reqs_padded = _dcut_get_call_value(args, kwargs, 4, "num_reqs_padded", num_reqs)
    query_lens = _dcut_to_int_list(_dcut_get_call_value(args, kwargs, 10, "num_scheduled_tokens_np"))
    max_records = controller.config.log_decision_max_records
    logger.info(
        "D-Cut attention query shape: num_tokens=%s num_tokens_padded=%s num_reqs=%s "
        "num_reqs_padded=%s max_query_len=%s query_lens_sum=%d query_lens_max=%d "
        "query_lens=%s",
        num_tokens,
        num_tokens_padded,
        num_reqs,
        num_reqs_padded,
        max_query_len,
        sum(query_lens),
        max(query_lens, default=0),
        query_lens[:max_records],
    )


def _dcut_should_time_verifier(runner, scheduler_output) -> bool:
    controller = getattr(runner, "_dcut_controller", None)
    if controller is None or not scheduler_output.scheduled_spec_decode_tokens:
        return False
    return controller.should_log_verifier_timing()


def _dcut_log_verifier_timing(runner, scheduler_output, elapsed_ms: float) -> None:
    spec_lens = [len(tokens) for tokens in scheduler_output.scheduled_spec_decode_tokens.values()]
    spec_tokens = sum(spec_lens)
    spec_reqs = len(spec_lens)
    avg_spec_len = spec_tokens / spec_reqs if spec_reqs else 0.0
    query_lens = _dcut_scheduler_query_lens(scheduler_output)
    num_reqs = len(query_lens)
    shape_key = (num_reqs, scheduler_output.total_num_scheduled_tokens, spec_tokens, max(spec_lens, default=0))
    stats = getattr(runner, "_dcut_verifier_timing_stats", None)
    if stats is None:
        stats = {}
        runner._dcut_verifier_timing_stats = stats
    count, total_ms, min_ms, max_ms = stats.get(shape_key, (0, 0.0, float("inf"), 0.0))
    count += 1
    total_ms += elapsed_ms
    min_ms = min(min_ms, elapsed_ms)
    max_ms = max(max_ms, elapsed_ms)
    stats[shape_key] = (count, total_ms, min_ms, max_ms)
    avg_ms = total_ms / count
    total_tokens = scheduler_output.total_num_scheduled_tokens
    ms_per_token = elapsed_ms / total_tokens if total_tokens else 0.0
    ms_per_spec_token = elapsed_ms / spec_tokens if spec_tokens else 0.0
    max_records = getattr(getattr(runner, "_dcut_controller", None), "config", None)
    max_records = getattr(max_records, "log_decision_max_records", 8)
    logger.info(
        "D-Cut verifier timing: elapsed_ms=%.3f total_tokens=%d spec_reqs=%d "
        "spec_tokens=%d avg_spec_len=%.2f max_spec_len=%d num_reqs=%d "
        "query_lens_sum=%d query_lens_max=%d query_lens=%s ms_per_token=%.4f "
        "ms_per_spec_token=%.4f shape_count=%d shape_avg_ms=%.3f shape_min_ms=%.3f "
        "shape_max_ms=%.3f timing_mode=execute_model_wall",
        elapsed_ms,
        scheduler_output.total_num_scheduled_tokens,
        spec_reqs,
        spec_tokens,
        avg_spec_len,
        max(spec_lens, default=0),
        num_reqs,
        sum(query_lens),
        max(query_lens, default=0),
        query_lens[:max_records],
        ms_per_token,
        ms_per_spec_token,
        count,
        avg_ms,
        min_ms,
        max_ms,
    )


def _dcut_scheduler_query_lens(scheduler_output) -> list[int]:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    if hasattr(num_scheduled_tokens, "values"):
        return [int(value) for value in num_scheduled_tokens.values()]
    return _dcut_to_int_list(num_scheduled_tokens)


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
        (runner.max_num_reqs, num_spec), dtype=torch.float32, device="cpu", pin_memory=runner.pin_memory
    )
    if getattr(runner.speculative_config, "method", None) == "dflash" and not getattr(
        getattr(runner, "ascend_config", None), "enable_reduce_sample", False
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
    if controller is None:
        return
    if not getattr(runner, "_dcut_profile_num_scheduled_tokens_supported", True):
        logger.warning(
            "D-Cut: skip cost profiling because installed NPUModelRunner._dummy_run does not accept "
            "profile_num_scheduled_tokens. Profiling without explicit scheduled-token shapes is misleading "
            "because lower-Q rows can be measured as fewer near-full requests. Update vllm-ascend to use "
            "batch-shape-preserving profiling."
        )
        return
    controller.profile_cost_table(runner)


def _dcut_truncate_scheduler_output(runner, scheduler_output):
    controller = getattr(runner, "_dcut_controller", None)
    if controller is None or not scheduler_output.scheduled_spec_decode_tokens:
        return scheduler_output
    if _dcut_has_long_non_spec_query(controller, scheduler_output):
        _dcut_log_skip_mixed_prefill_cut(runner, scheduler_output)
        _dcut_clear_scheduler_plans(controller, scheduler_output)
        return scheduler_output
    if not controller.config.apply_runtime_cuts:
        _dcut_clear_scheduler_plans(controller, scheduler_output)
        return scheduler_output
    max_runtime_cut_reqs = getattr(controller.config, "max_runtime_cut_reqs", 1024)
    if len(scheduler_output.scheduled_spec_decode_tokens) > max_runtime_cut_reqs:
        _dcut_log_skip_high_concurrency_cut(runner, scheduler_output, max_runtime_cut_reqs)
        _dcut_clear_scheduler_plans(controller, scheduler_output)
        return scheduler_output
    new_spec = scheduler_output.scheduled_spec_decode_tokens.copy()
    new_num_sched = scheduler_output.num_scheduled_tokens.copy()
    tokens_delta = 0
    tokens_delta_by_req: dict[str, int] = {}
    for req_id, draft_toks in list(new_spec.items()):
        adaptive_len = controller.get_adaptive_draft_len(req_id)
        if adaptive_len is not None:
            controller.invalidate(req_id)
        if adaptive_len is not None and adaptive_len < len(draft_toks):
            min_draft_len = _dcut_min_safe_draft_len(runner, req_id)
            if adaptive_len < min_draft_len:
                warnings = getattr(runner, "_dcut_accepted_tokens_clamp_warnings", 0)
                if warnings < 5:
                    logger.warning(
                        "D-Cut: clamping draft cut for req_id=%s from %d to %d "
                        "to keep the verifier segment compatible with already-accepted tokens.",
                        req_id,
                        adaptive_len,
                        min_draft_len,
                    )
                    runner._dcut_accepted_tokens_clamp_warnings = warnings + 1
                adaptive_len = min(min_draft_len, len(draft_toks))
            if adaptive_len >= len(draft_toks):
                continue
            diff = len(draft_toks) - adaptive_len
            tokens_delta += diff
            tokens_delta_by_req[req_id] = diff
            new_num_sched[req_id] -= diff
            if adaptive_len == 0:
                del new_spec[req_id]
            else:
                new_spec[req_id] = draft_toks[:adaptive_len]
    if tokens_delta <= 0:
        return scheduler_output
    tokens_after = scheduler_output.total_num_scheduled_tokens - tokens_delta
    logger.info(
        "D-Cut: cut scheduled speculative tokens reqs=%d tokens_before=%d tokens_after=%d delta=%d",
        len(scheduler_output.scheduled_spec_decode_tokens),
        scheduler_output.total_num_scheduled_tokens,
        tokens_after,
        tokens_delta,
    )
    if controller.config.log_decision_details:
        max_records = controller.config.log_decision_max_records
        spec_lens = {req_id: len(tokens) for req_id, tokens in new_spec.items()}
        logger.info(
            "D-Cut verifier input check: scheduled_spec_lens=%s total_num_scheduled_tokens=%d",
            dict(list(spec_lens.items())[:max_records]),
            tokens_after,
        )
    new_cached_reqs = _dcut_adjust_scheduled_cached_reqs(scheduler_output.scheduled_cached_reqs, tokens_delta_by_req)
    _dcut_log_runtime_cut_debug(
        controller,
        scheduler_output,
        new_spec,
        new_num_sched,
        tokens_delta_by_req,
        new_cached_reqs,
    )
    return replace(
        scheduler_output,
        scheduled_spec_decode_tokens=new_spec,
        num_scheduled_tokens=new_num_sched,
        total_num_scheduled_tokens=tokens_after,
        scheduled_cached_reqs=new_cached_reqs,
    )


def _dcut_log_runtime_cut_debug(
    controller,
    scheduler_output,
    new_spec: dict,
    new_num_sched: dict,
    tokens_delta_by_req: dict[str, int],
    new_cached_reqs,
) -> None:
    if not getattr(controller.config, "log_runtime_cut_debug", False):
        return
    max_records = controller.config.log_decision_max_records
    old_cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    logger.info(
        "D-Cut runtime cut debug: spec_reqs=%d tokens_before=%d tokens_after=%d "
        "delta_by_req=%s old_spec_lens=%s new_spec_lens=%s old_num_sched=%s "
        "new_num_sched=%s cached_req_ids=%s cached_num_computed_before=%s "
        "cached_num_computed_after=%s",
        len(scheduler_output.scheduled_spec_decode_tokens),
        scheduler_output.total_num_scheduled_tokens,
        sum(int(v) for v in new_num_sched.values()) if hasattr(new_num_sched, "values") else "unknown",
        dict(list(tokens_delta_by_req.items())[:max_records]),
        _dcut_lens_summary(scheduler_output.scheduled_spec_decode_tokens, max_records),
        _dcut_lens_summary(new_spec, max_records),
        _dcut_mapping_summary(scheduler_output.num_scheduled_tokens, max_records),
        _dcut_mapping_summary(new_num_sched, max_records),
        list(getattr(old_cached_reqs, "req_ids", [])[:max_records]) if old_cached_reqs is not None else None,
        _dcut_seq_summary(getattr(old_cached_reqs, "num_computed_tokens", None), max_records),
        _dcut_seq_summary(getattr(new_cached_reqs, "num_computed_tokens", None), max_records),
    )


def _dcut_lens_summary(values: dict, max_records: int) -> dict:
    return {req_id: len(tokens) for req_id, tokens in list(values.items())[:max_records]}


def _dcut_mapping_summary(values, max_records: int):
    if hasattr(values, "items"):
        return dict(list(values.items())[:max_records])
    return _dcut_seq_summary(values, max_records)


def _dcut_seq_summary(values, max_records: int):
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)[:max_records]


def _dcut_adjust_scheduled_cached_reqs(cached_reqs, tokens_delta_by_req: dict[str, int]):
    if not tokens_delta_by_req or cached_reqs is None:
        return cached_reqs
    req_ids = getattr(cached_reqs, "req_ids", None)
    num_computed_tokens = getattr(cached_reqs, "num_computed_tokens", None)
    if req_ids is None or num_computed_tokens is None:
        return cached_reqs
    if hasattr(num_computed_tokens, "copy"):
        adjusted = num_computed_tokens.copy()
    elif hasattr(num_computed_tokens, "clone"):
        adjusted = num_computed_tokens.clone()
    else:
        adjusted = list(num_computed_tokens)
    changed = False
    for index, req_id in enumerate(req_ids):
        diff = tokens_delta_by_req.get(req_id)
        if diff:
            adjusted[index] -= diff
            changed = True
    if not changed:
        return cached_reqs
    try:
        return replace(cached_reqs, num_computed_tokens=adjusted)
    except TypeError:
        new_cached_reqs = copy(cached_reqs)
        new_cached_reqs.num_computed_tokens = adjusted
        return new_cached_reqs


def _dcut_log_skip_high_concurrency_cut(runner, scheduler_output, max_runtime_cut_reqs: int) -> None:
    warnings = getattr(runner, "_dcut_high_concurrency_skip_warnings", 0)
    if warnings >= 5:
        return
    logger.warning(
        "D-Cut: skip runtime cut for high-concurrency batch because spec_reqs=%d exceeds "
        "max_runtime_cut_reqs=%d. Keeping verifier inputs unmodified to preserve accuracy.",
        len(scheduler_output.scheduled_spec_decode_tokens),
        max_runtime_cut_reqs,
    )
    runner._dcut_high_concurrency_skip_warnings = warnings + 1


def _dcut_clear_scheduler_plans(controller, scheduler_output) -> None:
    for req_id in scheduler_output.scheduled_spec_decode_tokens:
        if controller.get_adaptive_draft_len(req_id) is not None:
            controller.invalidate(req_id)


def _dcut_has_long_non_spec_query(controller, scheduler_output) -> bool:
    query_lens = _dcut_scheduler_query_lens(scheduler_output)
    return any(query_len > controller.max_query_len_per_req for query_len in query_lens)


def _dcut_log_skip_mixed_prefill_cut(runner, scheduler_output) -> None:
    warnings = getattr(runner, "_dcut_mixed_prefill_skip_warnings", 0)
    if warnings >= 5:
        return
    query_lens = _dcut_scheduler_query_lens(scheduler_output)
    logger.warning(
        "D-Cut: skip runtime cut for mixed prefill/decode batch because query_lens_max=%d "
        "exceeds speculative max query length; total_num_scheduled_tokens=%d spec_reqs=%d num_reqs=%d",
        max(query_lens, default=0),
        scheduler_output.total_num_scheduled_tokens,
        len(scheduler_output.scheduled_spec_decode_tokens),
        len(query_lens),
    )
    runner._dcut_mixed_prefill_skip_warnings = warnings + 1


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
    runner._dcut_active = {
        runner.input_batch.req_ids[i]
        for i in range(num_reqs)
        if runner.input_batch.num_computed_tokens_cpu[i] >= runner.input_batch.num_prompt_tokens[i]
    }
    runner._dcut_probs_pinned[:num_reqs].copy_(probs, non_blocking=True)
    if getattr(runner._dcut_controller.config, "log_runtime_cut_debug", False):
        logger.info(
            "D-Cut probs queued debug: num_reqs=%d req_ids=%s active_req_ids=%s probs_shape=%s",
            num_reqs,
            runner._dcut_req_ids[: runner._dcut_controller.config.log_decision_max_records],
            list(runner._dcut_active)[: runner._dcut_controller.config.log_decision_max_records],
            tuple(probs.shape),
        )
    runner._dcut_probs_event.record()


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
        if getattr(runner._dcut_controller.config, "log_runtime_cut_debug", False):
            logger.info(
                "D-Cut probs process debug: num_reqs=%d req_ids=%s active_req_ids=%s",
                runner._dcut_num_reqs,
                runner._dcut_req_ids[: runner._dcut_controller.config.log_decision_max_records],
                list(runner._dcut_active)[: runner._dcut_controller.config.log_decision_max_records],
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

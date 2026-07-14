
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch installer for D-Cut adaptive verifier step-length on **vLLM-Ascend / NPU**.

Ported from the CUDA plugin in ``Bensong0506/vllm`` branch
``feat/dcut-adaptive-verify`` (itself a port of the closed, unmerged vLLM
PR #44885) to run on Huawei Ascend NPU via vllm-ascend (vLLM v0.22.1 base).
Self-contained vLLM *general plugin* — **no vLLM / vllm-ascend source files are
edited**.

Algorithm is unchanged (see AngelSlim D-Cut,
https://angelslim.readthedocs.io/zh-cn/latest/dcut.html): the drafter still
proposes ``num_speculative_tokens`` every step, but the verifier only checks a
batch-adaptive subset chosen by a hardware-profiled ITL cost table +
draft-confidence prefix-product scores + batch-wide global top-K.

Only active for parallel speculative methods: ``method=dflash``, or
``method=draft_model`` with ``parallel_drafting=true`` (PARD).

------------------------------------------------------------------------------
GPU -> NPU deltas (there are **no operator / kernel changes** — this is a pure
spec-decode control-loop plugin; the delta is entirely *where* we patch and
*which device API* we call):

  1. Patch targets: ``NPUModelRunner`` (vllm_ascend.worker.model_runner_v1) /
     ``NPUWorker`` (vllm_ascend.worker.worker) / the Ascend spec-decode
     proposer — NOT the vLLM GPU classes.  This is mandatory: NPUModelRunner
     *overrides* ``execute_model``, ``sample_tokens``,
     ``_copy_draft_token_ids_to_cpu``, ``_update_states`` and ``__init__``, so
     patching ``GPUModelRunner`` would be shadowed by the NPU overrides and the
     plugin would silently no-op.  Likewise ``NPUWorker`` subclasses
     ``WorkerBase`` directly, not ``gpu_worker.Worker``.

  2. Device API: ``torch.cuda.Event`` / ``torch.cuda.synchronize`` ->
     ``torch.npu.Event`` / ``torch.npu.synchronize``.

  3. ``_adaptive_profile_run`` is rebuilt on the NPU forward path
     (``set_ascend_forward_context`` + ``self._model_forward`` + the NPU
     signature of ``_build_attention_metadata``).  The CUDA version relied on
     ``_get_slot_mappings`` / ``_init_model_kwargs`` / a ``slot_mappings=``
     kwarg that **do not exist** on NPUModelRunner, so that path is replaced
     with the NPU ``_dummy_run``-style plumbing.  Profiling is forced eager
     (no ACLGraph capture) — the cost table only needs *relative* ITL.

Enable: ``pip install -e .`` (this dir) + set ``VLLM_DCUT_CONFIG=/path/to.json``
+ ``VLLM_PLUGINS=dcut_adaptive_verify``.  See RUN.md.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from types import MethodType

import numpy as np
import torch

try:  # torch_npu registers the ``torch.npu`` namespace; already imported in a
    # real vllm-ascend worker process, but keep the plugin importable stand-alone.
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .verify_adaptive_config import VerifyAdaptiveConfig
from .verify_adaptive_controller import VerifyAdaptiveController

logger = init_logger(__name__)

_INSTALLED = False  # WorkerBase deferred-trigger armed (per process)
_PATCHED = False  # real monkey patches applied (per process)
ENV_CONFIG = "VLLM_DCUT_CONFIG"
ENV_TRIM_STATS_OUT = "VLLM_DCUT_TRIM_STATS_OUT"
ENV_PROFILE_FORCE_EAGER = "VLLM_DCUT_PROFILE_FORCE_EAGER"

# Phase 2A master switch: when True, the recurrent GDN attention core
# is captured directly in the main PIECEWISE graph (no splitting op,
# no graph_task_update for recurrent metadata). Static ASL/NAT/SSI
# buffers are filled graph-externally by the builder each step; the
# kernel inside the graph reads them. Conv1D still uses
# graph_task_update (host tuple args).
ENABLE_GDN_MAIN_PIECEWISE_GRAPH = True

# ── Static GDN buffers for PIECEWISE graph replay ──────────────────
# Pre-allocated ASL/SSI/NAT buffers with stable data_ptr.
# Filled graph-externally by _dcut_update_gdn_static() in _model_forward
# before each replay.  The GDN op inside the captured graph reads these
# buffers at replay time (proven by harness test_varlen_gdn_graph_replay.py).
#
# Key: (prefix, num_tokens, "spec"|"nonspec")
_dcut_gdn_static = {}


def _dcut_alloc_gdn_spec_bufs(prefix, num_tokens, spec_state_indices_tensor, device):
    """Allocate pre-allocated ASL/SSI/NAT buffers for spec decode path.
    Called once at capture time (inside _forward_core, _EXTRA_CTX.capturing)."""
    key = (prefix, num_tokens, "spec")
    if key not in _dcut_gdn_static:
        b_cap = spec_state_indices_tensor.size(0)
        nsp1 = spec_state_indices_tensor.size(1)  # num_spec + 1
        t_cap = b_cap * nsp1
        _dcut_gdn_static[key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "ssi": torch.full((t_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "nat": torch.zeros(b_cap, dtype=torch.int32, device=device),
            "b_cap": b_cap,
            "nsp1": nsp1,
            "t_cap": t_cap,
        }
        logger.warning(
            "D-Cut: alloc GDN spec static bufs prefix=%s num_tokens=%d "
            "b_cap=%d t_cap=%d", prefix, num_tokens, b_cap, t_cap)
    return _dcut_gdn_static[key]


def _dcut_fill_gdn_spec_bufs(prefix, num_tokens, spec_query_start_loc,
                              spec_state_indices_tensor, num_accepted_tokens,
                              num_spec_decodes, device):
    """Fill ASL/SSI/NAT buffers in-place with runtime values + b_cap padding.
    Called from _model_forward (outside captured graph) before each replay."""
    bufs = _dcut_alloc_gdn_spec_bufs(
        prefix, num_tokens, spec_state_indices_tensor, device)
    asl, ssi, nat = bufs["asl"], bufs["ssi"], bufs["nat"]

    # ASL: [0, per_seq_lens..., 0, 0, ...] padded to b_cap+1
    # Format: leading 0, then per-seq lengths, then 0 for dummy seqs
    asl.zero_()
    if num_spec_decodes > 0:
        cu = spec_query_start_loc[:num_spec_decodes + 1]
        asl[1:num_spec_decodes + 1].copy_(cu[1:] - cu[:-1])

    # SSI: compact real indices (only per-seq-length entries), pad with PAD_SLOT_ID
    # The GDN kernel uses ASL to accumulate global token positions and reads
    # SSI[global_token_pos]. We must compact SSI to match: only copy the first
    # per_seq_len entries from each sequence (boolean mask, like eager mode).
    # Copying all nsp1 entries per sequence causes SSI layout mismatch when
    # ASL < nsp1 (D-Cut truncation) -> wrong state indices for seq 1+.
    ssi.fill_(PAD_SLOT_ID)
    if num_spec_decodes > 0:
        per_seq_lens = (spec_query_start_loc[1:num_spec_decodes + 1] -
                        spec_query_start_loc[:num_spec_decodes])
        max_tokens = spec_state_indices_tensor.size(1)
        col_idx = torch.arange(max_tokens, device=spec_state_indices_tensor.device)
        mask = col_idx.unsqueeze(0) < per_seq_lens.unsqueeze(1)
        real = spec_state_indices_tensor[:num_spec_decodes][mask]
        ssi[:real.size(0)].copy_(real)

    # NAT: real values clamped to per-seq length (like eager mode), pad with 0
    # Without clamping, NAT can exceed ASL when D-Cut truncates the draft
    # after num_accepted_tokens was set. The kernel aborts (return) if
    # NAT > seqLen, causing ALL subsequent sequences to be unprocessed.
    nat.zero_()
    if num_spec_decodes > 0:
        per_seq_lens_nat = (spec_query_start_loc[1:num_spec_decodes + 1] -
                            spec_query_start_loc[:num_spec_decodes])
        clamped = torch.minimum(
            num_accepted_tokens[:num_spec_decodes].to(torch.int32),
            per_seq_lens_nat.to(torch.int32)
        )
        nat[:num_spec_decodes].copy_(clamped)

    logger.warning("D-Cut: FILL spec bufs prefix=%s nt=%d nsd=%d asl=%s ssi[:16]=%s nat=%s",
        prefix, num_tokens, num_spec_decodes,
        asl[:min(num_spec_decodes+2, asl.size(0))].tolist(),
        ssi[:min(16, ssi.size(0))].tolist(),
        nat[:min(num_spec_decodes+1, nat.size(0))].tolist())

    return bufs


def _dcut_alloc_gdn_nonspec_bufs(prefix, num_tokens,
                                  non_spec_state_indices_tensor, device):
    """Allocate pre-allocated ASL/SSI buffers for non-spec decode path."""
    key = (prefix, num_tokens, "nonspec")
    if key not in _dcut_gdn_static:
        b_cap = non_spec_state_indices_tensor.size(0)
        _dcut_gdn_static[key] = {
            "asl": torch.zeros(b_cap + 1, dtype=torch.int32, device=device),
            "ssi": torch.full((b_cap,), PAD_SLOT_ID, dtype=torch.int32, device=device),
            "b_cap": b_cap,
        }
        logger.warning(
            "D-Cut: alloc GDN nonspec static bufs prefix=%s num_tokens=%d "
            "b_cap=%d", prefix, num_tokens, b_cap)
    return _dcut_gdn_static[key]


def _dcut_fill_gdn_nonspec_bufs(prefix, num_tokens, non_spec_query_start_loc,
                                  non_spec_state_indices_tensor, num_decodes,
                                  device):
    """Fill ASL/SSI buffers in-place for non-spec decode path."""
    bufs = _dcut_alloc_gdn_nonspec_bufs(
        prefix, num_tokens, non_spec_state_indices_tensor, device)
    asl, ssi = bufs["asl"], bufs["ssi"]

    asl.zero_()
    if num_decodes > 0:
        cu = non_spec_query_start_loc[:num_decodes + 1]
        asl[1:num_decodes + 1].copy_(cu[1:] - cu[:-1])

    ssi.fill_(PAD_SLOT_ID)
    if num_decodes > 0:
        ssi[:num_decodes].copy_(non_spec_state_indices_tensor[:num_decodes])

    return bufs


def _dcut_update_gdn_static(forward_context, num_tokens, GDNAttentionMetadata):
    """Update GDN static buffers from forward context's attn_metadata.
    Called from patched _model_forward before _orig_model_forward (i.e. before
    graph replay).  Runs eagerly — NOT inside the captured graph piece."""
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return
    for prefix, meta in attn_metadata.items():
        if not isinstance(meta, GDNAttentionMetadata):
            continue
        if meta.spec_sequence_masks is not None and meta.num_spec_decodes > 0:
            _dcut_fill_gdn_spec_bufs(
                prefix, num_tokens,
                meta.spec_query_start_loc,
                meta.spec_state_indices_tensor,
                meta.num_accepted_tokens,
                meta.num_spec_decodes,
                meta.spec_query_start_loc.device,
            )
        elif meta.num_decodes > 0:
            _dcut_fill_gdn_nonspec_bufs(
                prefix, num_tokens,
                meta.non_spec_query_start_loc,
                meta.non_spec_state_indices_tensor,
                meta.num_decodes,
                meta.non_spec_query_start_loc.device,
            )


def _npu_event(enable_timing: bool = False):
    """torch.npu.Event, mirroring torch.cuda.Event on the CUDA plugin."""
    return torch.npu.Event(enable_timing=enable_timing)


def _supports_adaptive_verify(spec_cfg) -> bool:
    """Mirror of SpeculativeConfig.supports_adaptive_verify (which 0.22.x lacks)."""
    if spec_cfg is None:
        return False
    method = getattr(spec_cfg, "method", None)
    parallel = getattr(spec_cfg, "parallel_drafting", False)
    return method == "dflash" or (method == "draft_model" and parallel)


def _dcut_greedy_sample_with_selected_probs(logits):
    tp_group = get_tp_group()
    _, v_local = logits.shape
    rank = tp_group.rank_in_group

    local_max_logits, local_max_indices = logits.max(dim=-1)
    local_global_idx = local_max_indices + rank * v_local

    gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
    gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)
    global_max_rank = gathered_logits.argmax(dim=-1)
    next_token = gathered_global_idx.gather(
        dim=-1, index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1)
    selected_logits = gathered_logits.gather(
        dim=-1, index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1)

    local_lse = logits.logsumexp(dim=-1)
    gathered_lse = tp_group.all_gather(local_lse.unsqueeze(-1), dim=-1)
    global_lse = gathered_lse.logsumexp(dim=-1)
    selected_probs = (selected_logits - global_lse).exp()
    return next_token, selected_probs


def _dcut_patch_drafter_instance(drafter) -> None:
    """Patch the live Ascend drafter instance; robust to MRO/load order quirks."""
    if not hasattr(drafter, "take_last_selected_probs"):
        drafter.take_last_selected_probs = lambda: getattr(
            drafter, "_last_selected_probs", None
        )

    model = getattr(drafter, "model", None)
    if (
        model is not None
        and hasattr(model, "compute_logits")
        and not getattr(model, "_dcut_compute_logits_patched", False)
    ):
        orig_compute_logits = model.compute_logits

        def compute_logits(self_model, hidden_states, *args, **kwargs):
            logits = orig_compute_logits(hidden_states, *args, **kwargs)
            if getattr(drafter, "needs_draft_probs", False) and logits is not None:
                try:
                    token_ids = logits.argmax(dim=-1)
                    chosen = logits.gather(-1, token_ids.long().unsqueeze(-1))
                    selected_probs = (
                        chosen.squeeze(-1) - logits.logsumexp(dim=-1)
                    ).exp()
                    drafter._last_selected_probs = (
                        selected_probs.float().contiguous()
                    )
                    if not getattr(
                        drafter, "_dcut_logged_compute_logits_probs", False
                    ):
                        logger.warning(
                            "D-Cut: captured selected draft probs from "
                            "compute_logits on %s (logits_shape=%s).",
                            type(drafter).__name__,
                            tuple(logits.shape),
                        )
                        drafter._dcut_logged_compute_logits_probs = True
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "D-Cut: gather selected probs from compute_logits "
                        "failed: %s",
                        e,
                    )
                    drafter._last_selected_probs = None
            return logits

        model.compute_logits = MethodType(compute_logits, model)
        model._dcut_compute_logits_patched = True

    if (
        not hasattr(drafter, "compute_draft_token_ids")
        or getattr(drafter, "_dcut_instance_compute_patched", False)
    ):
        return

    orig_compute = drafter.compute_draft_token_ids

    def compute_draft_token_ids(self, hidden_states):
        self._last_selected_probs = None
        if not getattr(self, "needs_draft_probs", False):
            return orig_compute(hidden_states)
        try:
            logits = self.model.logits_processor(self.model.lm_head, hidden_states)
            logits = logits.contiguous()
            next_token, selected_probs = _dcut_greedy_sample_with_selected_probs(
                logits
            )
            self._last_selected_probs = selected_probs.float().contiguous()

            draft_map = getattr(self.model, "draft_id_to_target_id", None)
            if draft_map is None:
                return next_token
            bias = torch.index_select(
                draft_map, dim=0, index=next_token.view(-1)
            ).view(next_token.shape)
            return next_token + bias
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "D-Cut: gather selected probs in live drafter failed: %s", e
            )
            self._last_selected_probs = None
            return orig_compute(hidden_states)

    drafter.compute_draft_token_ids = MethodType(compute_draft_token_ids, drafter)
    drafter._dcut_instance_compute_patched = True


# ---------------------------------------------------------------------------
# Runner-side helpers (installed as methods or used by the wrappers).
# Device-agnostic except where noted; identical to the CUDA plugin.
# ---------------------------------------------------------------------------

def _dcut_init_controller(self) -> None:
    """Build the controller + async-probs buffers on an NPUModelRunner instance.

    Enabled iff ``VLLM_DCUT_CONFIG`` points to a JSON config AND the speculative
    method is parallel (dflash / PARD).  Otherwise leaves the runner untouched.
    """
    self._verify_adaptive_controller = None
    self._adaptive_probs_event = None
    self._adaptive_probs_pinned = None
    self._adaptive_probs_pending = False
    self._adaptive_num_reqs = 0
    self._adaptive_req_ids = []
    self._adaptive_active = set()
    # Verify-reduction stats (how much D-Cut trimmed); logged every N steps.
    self._dcut_stat_full = 0
    self._dcut_stat_trimmed = 0
    self._dcut_stat_reqs = 0
    self._dcut_stat_steps = 0
    self._dcut_stat_log_every = int(os.environ.get("VLLM_DCUT_STAT_EVERY", "200") or 0)
    self._dcut_trim_stats_out = os.environ.get(ENV_TRIM_STATS_OUT) or None
    self._dcut_missing_probs_steps = 0
    self._dcut_logged_drafter_probs = False

    cfg_path = os.environ.get(ENV_CONFIG) or None
    if not cfg_path:
        return

    spec_cfg = getattr(self, "speculative_config", None)
    if not _supports_adaptive_verify(spec_cfg):
        logger.warning(
            "VLLM_DCUT_CONFIG is set but the speculative method does not support "
            "adaptive verifier step-length (requires dflash or "
            "draft_model+parallel_drafting); D-Cut disabled."
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
    # NPU: torch.npu.Event instead of torch.cuda.Event.
    self._adaptive_probs_event = _npu_event()
    self._adaptive_probs_pinned = torch.empty(
        (self.max_num_reqs, num_spec),
        dtype=torch.float32,
        device="cpu",
        pin_memory=self.pin_memory,
    )
    _dcut_enable_drafter_probs(self)
    logger.info("D-Cut adaptive verify ENABLED on NPU (config=%s).", cfg_path)


def _dcut_enable_drafter_probs(self) -> None:
    """Enable draft-prob collection once the Ascend drafter object exists."""
    if getattr(self, "_verify_adaptive_controller", None) is None:
        return
    drafter = getattr(self, "drafter", None)
    if drafter is None:
        return
    if not hasattr(drafter, "needs_draft_probs"):
        if not getattr(self, "_dcut_logged_drafter_probs", False):
            logger.warning(
                "D-Cut: drafter %s has no needs_draft_probs flag.",
                type(drafter).__name__,
            )
            self._dcut_logged_drafter_probs = True
        return
    _dcut_patch_drafter_instance(drafter)
    was_enabled = bool(getattr(drafter, "needs_draft_probs", False))
    drafter.needs_draft_probs = True
    if not was_enabled or not getattr(self, "_dcut_logged_drafter_probs", False):
        logger.warning(
            "D-Cut: enabled selected draft probs on drafter %s "
            "(method=%s parallel=%s instance_compute_patched=%s).",
            type(drafter).__name__,
            getattr(drafter, "method", None),
            getattr(drafter, "parallel_drafting", None),
            getattr(drafter, "_dcut_instance_compute_patched", False),
        )
        self._dcut_logged_drafter_probs = True


def _dcut_truncate(self, scheduler_output):
    """Apply per-request draft_len caps cached by the previous step.

    Device-agnostic (pure scheduler-output bookkeeping); identical to CUDA.
    """
    ctrl = self._verify_adaptive_controller
    _spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)

    if ctrl is None or not _spec:
        return scheduler_output

    orig_spec = scheduler_output.scheduled_spec_decode_tokens
    # Draft positions that WOULD be verified this step without D-Cut, and the
    # number of spec-decode requests — captured before we mutate anything.
    full_draft = sum(len(t) for t in orig_spec.values())
    n_spec_reqs = len(orig_spec)

    new_spec = orig_spec.copy()
    new_num_sched = scheduler_output.num_scheduled_tokens.copy()
    tokens_delta = 0
    import random as _random
    _cut_log = []
    for req_id, draft_toks in list(new_spec.items()):
        # HARDCODED RANDOM CUT: randomly pick draft_len in [2, len(draft_toks)]
        max_dl = len(draft_toks)
        min_cut = 2
        if max_dl > min_cut:
            adaptive_len = _random.randint(min_cut, max_dl)
        else:
            adaptive_len = max_dl
        _cut_log.append((req_id[:8], max_dl, adaptive_len))
        if adaptive_len < max_dl:
            diff = max_dl - adaptive_len
            tokens_delta += diff
            new_num_sched[req_id] -= diff
            new_spec[req_id] = draft_toks[:adaptive_len]

    _dcut_record_trim(self, full_draft, tokens_delta, n_spec_reqs)

    if tokens_delta > 0:
        scheduler_output = replace(
            scheduler_output,
            scheduled_spec_decode_tokens=new_spec,
            num_scheduled_tokens=new_num_sched,
            total_num_scheduled_tokens=(
                scheduler_output.total_num_scheduled_tokens - tokens_delta
            ),
        )
    return scheduler_output


def _dcut_record_trim(self, full_draft: int, trimmed: int, n_spec_reqs: int) -> None:
    """Accumulate verify-reduction stats and log them every N steps (rank 0).

    Answers "how much verify did D-Cut save": trimmed vs full draft positions
    (the verifier checks one position per draft token).  Cadence is controlled
    by ``VLLM_DCUT_STAT_EVERY`` (steps; 0 disables).  Cumulative totals, so the
    running percentage is stable.
    """
    self._dcut_stat_full += full_draft
    self._dcut_stat_trimmed += trimmed
    self._dcut_stat_reqs += n_spec_reqs
    self._dcut_stat_steps += 1
    every = self._dcut_stat_log_every
    if not every or (self._dcut_stat_steps % every) != 0:
        return
    if get_tp_group().rank_in_group != 0 or not get_pp_group().is_first_rank:
        return
    full = self._dcut_stat_full
    trimmed_tot = self._dcut_stat_trimmed
    pct = 100.0 * trimmed_tot / full if full else 0.0
    kept = full - trimmed_tot
    reqs = max(self._dcut_stat_reqs, 1)
    _dcut_dump_trim_stats(
        self,
        full=full,
        trimmed_tot=trimmed_tot,
        kept=kept,
        pct=pct,
        reqs=reqs,
        last_full=full_draft,
        last_trimmed=trimmed,
        last_reqs=n_spec_reqs,
    )
    logger.info(
        "D-Cut verify trim: cut %d/%d draft positions (%.1f%% fewer verifies) "
        "over %d steps; avg %.2f->%.2f verified tok/spec-req",
        trimmed_tot, full, pct, self._dcut_stat_steps,
        full / reqs, kept / reqs,
    )


def _dcut_dump_trim_stats(
    self,
    *,
    full: int,
    trimmed_tot: int,
    kept: int,
    pct: float,
    reqs: int,
    last_full: int,
    last_trimmed: int,
    last_reqs: int,
) -> None:
    """Append rank-0 trim stats to JSONL for scripts that cannot see worker logs."""
    path = getattr(self, "_dcut_trim_stats_out", None)
    if not path:
        return
    row = {
        "time_unix": time.time(),
        "steps": self._dcut_stat_steps,
        "spec_reqs": self._dcut_stat_reqs,
        "full_draft_positions": full,
        "trimmed_draft_positions": trimmed_tot,
        "kept_draft_positions": kept,
        "trim_pct": pct,
        "avg_full_per_spec_req": full / reqs,
        "avg_kept_per_spec_req": kept / reqs,
        "last_full_draft_positions": last_full,
        "last_trimmed_draft_positions": last_trimmed,
        "last_spec_reqs": last_reqs,
    }
    try:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover - observability must not affect serving
        logger.debug("D-Cut: failed to write trim stats: %s", e)


def _dcut_queue_probs(self, zeros_only: bool) -> None:
    """Queue this step's selected_probs D2H (non-blocking) for next-step use.

    Device-agnostic apart from the async D2H copy + event record, which work
    the same on NPU (torch_npu supports non_blocking copies + npu.Event).
    """
    if (
        zeros_only
        or self._adaptive_probs_pending
        or self._adaptive_probs_pinned is None
        or self._adaptive_probs_event is None
    ):
        return
    _dcut_enable_drafter_probs(self)
    drafter = getattr(self, "drafter", None)
    if drafter is None or not hasattr(drafter, "take_last_selected_probs"):
        cnt = getattr(self, "_dcut_missing_probs_steps", 0) + 1
        self._dcut_missing_probs_steps = cnt
        if cnt <= 3 or cnt % 200 == 0:
            logger.warning(
                "D-Cut: drafter has no selected-probs hook; decision stats "
                "will not update (count=%s).",
                cnt,
            )
        return
    probs = drafter.take_last_selected_probs()
    if probs is None:
        cnt = getattr(self, "_dcut_missing_probs_steps", 0) + 1
        self._dcut_missing_probs_steps = cnt
        if cnt <= 3 or cnt % 200 == 0:
            logger.warning(
                "D-Cut: drafter did not expose selected draft probs; decision "
                "stats will not update (count=%s).",
                cnt,
            )
        return
    num_reqs = self.input_batch.num_reqs
    num_spec = self.num_spec_tokens
    if probs.dim() == 1:
        needed = num_reqs * num_spec
        if probs.numel() < needed:
            logger.warning(
                "D-Cut: selected draft probs too short: got=%s need=%s",
                probs.numel(),
                needed,
            )
            return
        probs = probs[:needed].view(num_reqs, num_spec)
    else:
        probs = probs[:num_reqs]
        if probs.shape[-1] != num_spec:
            logger.warning(
                "D-Cut: selected draft probs shape mismatch: shape=%s num_spec=%s",
                tuple(probs.shape),
                num_spec,
            )
            return
    self._adaptive_probs_pending = True
    self._adaptive_num_reqs = num_reqs
    self._adaptive_req_ids = self.input_batch.req_ids.copy()
    self._adaptive_active = {
        self.input_batch.req_ids[i]
        for i in range(num_reqs)
        if (
            self.input_batch.num_computed_tokens_cpu[i]
            >= self.input_batch.num_prompt_tokens[i]
        )
    }
    # Non-blocking D2H on the default stream (the drafter runs there too); the
    # event lets _maybe_process_adaptive_probs verify completion cheaply.
    self._adaptive_probs_pinned[:num_reqs].copy_(probs.contiguous(), non_blocking=True)
    self._adaptive_probs_event.record()


def _maybe_process_adaptive_probs(self) -> None:
    """Consume step-N probs and update the controller's draft_len cache.

    Device-agnostic; npu.Event exposes the same query()/synchronize() API.
    """
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
    """Profile verifier ITL after warmup (called from NPUWorker)."""
    if getattr(self, "_verify_adaptive_controller", None) is not None:
        self._verify_adaptive_controller.profile_cost_table(self)


@torch.inference_mode()
def _adaptive_profile_run(
    self,
    scheduled_tokens: "list[int]",
    seq_lens: int = 1024,
    n_warmup: int = 3,
    n_measure: int = 5,
):
    """Profile verifier forward latency for one (batch_size, query_len) shape.

    **NPU port** of PR #44885's GPUModelRunner._adaptive_profile_run.  Modelled
    on vllm-ascend's ``NPUModelRunner._dummy_run``: it builds attention metadata
    with the NPU signature of ``_build_attention_metadata`` (no ``slot_mappings``
    kwarg, no ``_get_slot_mappings`` pre-step), then times ``self._model_forward``
    inside ``set_ascend_forward_context`` using ``torch.npu.Event``.  Profiling
    is forced eager by default for safety; set
    ``VLLM_DCUT_PROFILE_FORCE_EAGER=0`` to let the dispatcher pick the same
    graph/eager runtime mode as serving.

    Returns (runtime_mode, avg_forward_ms, num_tokens_padded).
    """
    # Import here so a stand-alone `import dcut` never hard-requires vllm-ascend.
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    num_scheduled_tokens = np.array(scheduled_tokens, dtype=np.int32)
    num_reqs = len(scheduled_tokens)
    num_tokens_unpadded = int(num_scheduled_tokens.sum())
    max_query_len = int(num_scheduled_tokens.max())

    assert num_tokens_unpadded <= self.max_num_tokens, (
        f"adaptive profile: num_tokens={num_tokens_unpadded} > "
        f"max_num_tokens={self.max_num_tokens}"
    )

    profile_force_eager = (
        os.environ.get(ENV_PROFILE_FORCE_EAGER, "1").lower()
        not in ("0", "false", "no")
    )
    # Same dispatcher entrypoint as _dummy_run.  By default the profile run is
    # eager because manual graph replay on NPU is more fragile; the env override
    # is useful when comparing eager vs graph cost curves.
    _cudagraph_mode, batch_desc, _should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=False,
            force_eager=profile_force_eager,
            force_uniform_decode=True,
        )
    )

    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = (
        batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    )
    # allow_microbatching=False -> no ubatching for the profile run.
    ubatch_slices = None

    with self.synchronize_input_prep():
        # seq_lens / query_start_loc are needed by attention backends in all
        # modes.  Use the configured warmup_seq_lens so attention cost reflects
        # realistic long-context inference rather than trivial seq_len=1.
        self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

        cum_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
        self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
            cum_num_tokens[-1]
        )
        # PIECEWISE graph replay requires query_start_loc[-1] == num_tokens_padded
        # (the cuSeqlen the graph was captured with).  Without this, conv1d
        # validation fails: "queryStartLoc[last] must equal cuSeqlen".
        self.query_start_loc.np[num_reqs_padded] = num_tokens_padded
        self.query_start_loc.copy_to_gpu()
        # Also update gdn_query_start_loc: when self._has_gdn is True,
        # _build_attention_metadata overrides cm.query_start_loc_cpu to use
        # gdn_query_start_loc instead of query_start_loc.  Without this update,
        # the GDN builder sees stale values from the previous profile point,
        # causing aclnnCausalConv1d tiling failures (EZ9999).
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
                cum_num_tokens[-1]
            )
            # Same fix as query_start_loc: last entry must == num_tokens_padded
            self.gdn_query_start_loc.np[num_reqs_padded] = num_tokens_padded
            self.gdn_query_start_loc.copy_to_gpu()
        self.input_batch.block_table.commit_block_table(num_reqs_padded)

        # Mark requests as decode (num_computed_tokens > 0) so the model
        # treats them as decode-phase, not prefill.  Required for the
        # dispatcher to select PIECEWISE cudagraph during profiling.
        for i in range(num_reqs):
            self.input_batch.num_computed_tokens_cpu[i] = seq_lens
        self.input_batch.num_computed_tokens_cpu[num_reqs:].fill(0)

        # Mark every sequence as a spec-decode so hybrid GDN/Mamba backends
        # take the cheap incremental spec-decode path instead of the expensive
        # prefill chunk-scan.  Per-request draft_len must respect the
        # spec_state_indices_tensor width (num_spec + 1) to avoid OOB in
        # npu_causal_conv1d_custom / npu_recurrent_gated_delta_rule.
        if self.speculative_config is not None:
            num_spec = self.num_spec_tokens
            if hasattr(self, "num_decode_draft_tokens"):
                for i in range(num_reqs):
                    self.num_decode_draft_tokens.np[i] = min(
                        int(num_scheduled_tokens[i]) - 1, num_spec
                    )
                self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
                self.num_decode_draft_tokens.copy_to_gpu()
            if hasattr(self, "num_accepted_tokens"):
                # num_accepted_tokens must be <= num_spec + 1 (the width of
                # spec_state_indices_tensor) to prevent OOB state access in
                # the Mamba/GDN conv1d sliding-window update.
                for i in range(num_reqs):
                    self.num_accepted_tokens.gpu[i] = min(
                        int(num_scheduled_tokens[i]), num_spec + 1
                    )
                self.num_accepted_tokens.gpu[num_reqs:].fill_(1)

        # NPU _build_attention_metadata: no `slot_mappings` kwarg; takes
        # `num_scheduled_tokens_np` and `num_reqs_padded` instead.
        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            for_cudagraph_capture=not profile_force_eager,
            use_spec_decode=self.speculative_config is not None,
            num_scheduled_tokens_np=num_scheduled_tokens,
        )

    # Inputs — identical construction to _dummy_run so model kwargs (e.g. aux
    # hidden states for DFlash/Eagle3) are always provided.  Real verifier decode
    # steps are text-only, so we skip the vision encoder: mm-wrapped models route
    # through inputs_embeds, so we still supply a dummy embeds buffer for them —
    # just without the mm kwargs that would trigger the encoder.
    use_embeds = self.enable_prompt_embeds or (
        self.supports_mm_inputs and not self.model_config.is_encoder_decoder
    )
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
        from vllm.v1.outputs import IntermediateTensors  # lazy: PP>1 only
        if self.intermediate_tensors is None:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )
        intermediate_tensors = IntermediateTensors(
            {k: v[:num_tokens_padded] for k, v in self.intermediate_tensors.items()}
        )

    _mode_names = {
        CUDAGraphMode.FULL: "FCG",
        CUDAGraphMode.PIECEWISE: "PCG",
        CUDAGraphMode.NONE: "eager",
    }
    avg_ms = 0.0
    with set_ascend_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        in_profile_run=True,
        num_actual_tokens=num_tokens_padded,
        aclgraph_runtime_mode=_cudagraph_mode,
        batch_descriptor=batch_desc,
        model_instance=self.model,
        has_sinks=self._has_sinks,
        input_ids=input_ids,
    ):

        def _forward() -> None:
            self._model_forward(
                num_tokens_padded,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        for _ in range(max(n_warmup, 0)):
            _forward()
        torch.npu.synchronize()

        if n_measure > 0:
            start_ev = _npu_event(enable_timing=True)
            end_ev = _npu_event(enable_timing=True)
            start_ev.record()
            for _ in range(n_measure):
                _forward()
            end_ev.record()
            torch.npu.synchronize()
            avg_ms = start_ev.elapsed_time(end_ev) / n_measure

    mode_str = _mode_names.get(_cudagraph_mode, str(_cudagraph_mode))
    return mode_str, avg_ms, int(num_tokens_padded)


# ---------------------------------------------------------------------------
# Patch installers (idempotent, per class).  Targets are the *NPU* classes.
# ---------------------------------------------------------------------------

def _patch_proposer() -> None:
    """Patch the Ascend spec-decode proposer to expose selected draft probs.

    The Ascend dflash/PARD proposers inherit ``_sample_draft_tokens`` through a
    multi-level MRO (AscendDflashProposer -> AscendEagleProposer ->
    EagleProposer / AscendSpecDecodeBaseProposer -> SpecDecodeBaseProposer).
    We resolve, per proposer, the class in the MRO that actually *defines*
    ``_sample_draft_tokens`` and patch that class, so the wrapper is not
    shadowed by a subclass override.
    """
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    # Collect the concrete Ascend proposers that can run D-Cut (dflash / PARD).
    proposer_classes = []
    try:
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
        proposer_classes.append(AscendDflashProposer)
    except Exception as e:  # pragma: no cover
        logger.warning("D-Cut: could not import AscendDflashProposer: %s", e)
    try:
        from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
        proposer_classes.append(AscendDraftModelProposer)
    except Exception:  # PARD path is optional
        pass

    # Helper functions live on the shared base so every proposer can call them.
    if not getattr(SpecDecodeBaseProposer, "_dcut_helpers", False):
        @staticmethod
        def _should_collect_draft_probs(self):
            return getattr(self, "needs_draft_probs", False) and (
                getattr(self, "parallel_drafting", False)
                or getattr(self, "method", None) == "dflash"
            )

        @staticmethod
        def _gather_selected_probs(logits, token_ids, full_probs):
            idx = token_ids.long().unsqueeze(-1)
            if full_probs is not None:
                return full_probs.gather(-1, idx).squeeze(-1)
            chosen = logits.gather(-1, idx).squeeze(-1)
            return (chosen - logits.logsumexp(dim=-1)).exp()

        @staticmethod
        def _greedy_sample_with_selected_probs(logits):
            return _dcut_greedy_sample_with_selected_probs(logits)

        def take_last_selected_probs(self):
            return getattr(self, "_last_selected_probs", None)

        SpecDecodeBaseProposer.needs_draft_probs = False
        SpecDecodeBaseProposer._last_selected_probs = None
        SpecDecodeBaseProposer._should_collect_draft_probs = (
            _should_collect_draft_probs
        )
        SpecDecodeBaseProposer._gather_selected_probs = _gather_selected_probs
        SpecDecodeBaseProposer._greedy_sample_with_selected_probs = (
            _greedy_sample_with_selected_probs
        )
        SpecDecodeBaseProposer.take_last_selected_probs = take_last_selected_probs
        SpecDecodeBaseProposer._dcut_helpers = True

    compute_owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "compute_draft_token_ids" in klass.__dict__:
                if klass not in compute_owners:
                    compute_owners.append(klass)
                break

    for owner in compute_owners:
        if getattr(owner, "_dcut_compute_patched", False):
            continue
        _orig_compute = owner.compute_draft_token_ids
      
        def _make_compute_wrapper(orig):
            def compute_draft_token_ids(self, hidden_states):
                self._last_selected_probs = None
                if not type(self)._should_collect_draft_probs(self):
                    return orig(self, hidden_states)
                try:
                    logits = self.model.logits_processor(
                        self.model.lm_head, hidden_states
                    )
                    logits = logits.contiguous()
                    next_token, selected_probs = (
                        type(self)._greedy_sample_with_selected_probs(logits)
                    )
                    # Keep this flat here. Ascend may pad sample_hidden_states for
                    # lmhead TP; the runner slices and reshapes using real batch size.
                    self._last_selected_probs = selected_probs.float().contiguous()

                    draft_map = getattr(self.model, "draft_id_to_target_id", None)
                    if draft_map is None:
                        return next_token
                    bias = torch.index_select(
                        draft_map, dim=0, index=next_token.view(-1)
                    ).view(next_token.shape)
                    return next_token + bias
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "D-Cut: gather selected probs in compute_draft_token_ids "
                        "failed: %s",
                        e,
                    )
                    self._last_selected_probs = None
                    return orig(self, hidden_states)

            return compute_draft_token_ids

        owner.compute_draft_token_ids = _make_compute_wrapper(_orig_compute)
        owner._dcut_compute_patched = True
        logger.info(
            "D-Cut: patched compute_draft_token_ids on %s.", owner.__name__
        )

    # Find the distinct owner classes of _sample_draft_tokens across our
    # concrete proposers (usually a single class) and wrap each once.
    owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "_sample_draft_tokens" in klass.__dict__:
                if klass not in owners:
                    owners.append(klass)
                break

    for owner in owners:
        if getattr(owner, "_dcut_patched", False):
            continue
        _orig_sample = owner._sample_draft_tokens

        def _make_wrapper(orig):
            def _sample_draft_tokens(self, hidden_states, sampling_metadata):
                self._last_selected_probs = None
                out = orig(self, hidden_states, sampling_metadata)
                # D-Cut only targets parallel drafting (DFlash / PARD), where the
                # whole block is sampled in this single call -> selected_probs is
                # [B*T] which reshapes to [B, T].
                if type(self)._should_collect_draft_probs(self):
                    if isinstance(out, tuple):
                        token_ids = out[0]
                        full_probs = out[1] if len(out) > 1 else None
                    else:
                        token_ids = out
                        full_probs = None
                    try:
                        logits = (
                            None
                            if full_probs is not None
                            else self.model.compute_logits(hidden_states)
                        )
                        sel = type(self)._gather_selected_probs(
                            logits, token_ids, full_probs
                        )
                        self._last_selected_probs = sel.view(
                            -1, self.num_speculative_tokens
                        ).contiguous()
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning(
                            "D-Cut: gather selected probs failed: %s", e
                        )
                        self._last_selected_probs = None
                return out
            return _sample_draft_tokens

        owner._sample_draft_tokens = _make_wrapper(_orig_sample)
        owner._dcut_patched = True
        logger.info(
            "D-Cut: patched _sample_draft_tokens on %s.", owner.__name__
        )


def _patch_runner() -> None:
    import vllm_ascend.worker.model_runner_v1 as m

    # Class-level removal: modify CompilationConfig.splitting_ops at import
    # time so the config parser uses the modified list. The __init__ patch
    # alone is too late if the compiler reads splitting_ops before __init__.
    if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
        try:
            from vllm.config import CompilationConfig as _CC
            _gdn_op = "vllm::qwen_gdn_attention_core"
            if _CC.splitting_ops and _gdn_op in _CC.splitting_ops:
                _CC.splitting_ops = [op for op in _CC.splitting_ops if op != _gdn_op]
                logger.warning("D-Cut: removed %s from CompilationConfig.splitting_ops (class-level)", _gdn_op)
            if hasattr(_CC, '_attention_ops') and _gdn_op in _CC._attention_ops:
                _CC._attention_ops = [op for op in _CC._attention_ops if op != _gdn_op]
                logger.warning("D-Cut: removed %s from CompilationConfig._attention_ops (class-level)", _gdn_op)
        except Exception as e:
            logger.warning("D-Cut: class-level removal failed: %s", e)

    R = m.NPUModelRunner
    if getattr(R, "_dcut_patched", False):
        return

    _orig_init = R.__init__

    def __init__(self, *a, **k):
        # Re-enabled: Remove GDN from splitting_ops so attention core is
        # captured in the PIECEWISE graph. The attention core's metadata
        # (actual_seq_lengths, ssm_state_indices, num_accepted_tokens) is
        # updated at replay time via graph_task_update (see
        # update_gdn_attn_graph_params in gdn.py).
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            try:
                _vc = k.get("vllm_config") or (a[0] if a else None)
                if _vc is not None:
                    _cc = _vc.compilation_config
                    _gdn_op = "vllm::qwen_gdn_attention_core"
                    if _cc.splitting_ops and _gdn_op in _cc.splitting_ops:
                        _cc.splitting_ops.remove(_gdn_op)
                        logger.warning("D-Cut: REMOVED %s from instance splitting_ops, remaining=%d ops", _gdn_op, len(_cc.splitting_ops))
                    else:
                        logger.warning("D-Cut: %s NOT in splitting_ops (len=%d, type=%s)", _gdn_op, len(_cc.splitting_ops) if _cc.splitting_ops else -1, type(_cc.splitting_ops))
                    _cls = type(_cc)
                    if hasattr(_cls, '_attention_ops') and _gdn_op in _cls._attention_ops:
                        _cls._attention_ops = [op for op in _cls._attention_ops if op != _gdn_op]
                        logger.warning("D-Cut: REMOVED %s from class _attention_ops", _gdn_op)
                    logger.warning("D-Cut: GDN removed from splitting_ops and _attention_ops (Phase 2: true入图)")
            except Exception as e:
                logger.warning("D-Cut: failed to remove GDN in __init__: %s", e)
        _orig_init(self, *a, **k)
        try:
            _dcut_init_controller(self)
        except Exception as e:
            logger.error("D-Cut init failed; running vanilla: %s", e)
            self._verify_adaptive_controller = None

    _orig_exec = R.execute_model

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        # DEBUG: print to stdout to confirm wrapper is called
        _ctrl = getattr(self, "_verify_adaptive_controller", None)
        _has_spec = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
        if _ctrl is not None:
            _dcut_enable_drafter_probs(self)
            import os as _os_dcut
            if not _os_dcut.environ.get("VLLM_DCUT_DISABLE"):
                scheduler_output = _dcut_truncate(self, scheduler_output)
        # DEBUG: log every N calls to avoid flooding
        if not hasattr(self, "_dcut_exec_count"):
            self._dcut_exec_count = 0
        self._dcut_exec_count += 1
        if self._dcut_exec_count % 50 == 1:
            logger.info("DCUT_EXEC[%d]: ctrl=%s, has_spec=%s",
                        self._dcut_exec_count,
                        "None" if _ctrl is None else "set",
                        _has_spec)
        return _orig_exec(self, scheduler_output, intermediate_tensors)

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
    R.execute_model = execute_model
    R.sample_tokens = sample_tokens
    R._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    R._update_states = _update_states
    R._adaptive_profile_run = _adaptive_profile_run
    R.profile_adaptive_cost = profile_adaptive_cost
    R._maybe_process_adaptive_probs = _maybe_process_adaptive_probs
    R._dcut_enable_drafter_probs = _dcut_enable_drafter_probs
    R._dcut_patched = True

    # --- Force attn_metadata build during PIECEWISE graph capture ---
    # By default, _dummy_run only builds attn_metadata when
    # force_attention=True or cudagraph_runtime_mode==FULL.  In PIECEWISE
    # mode, force_attention defaults to False, so attn_metadata is None.
    # GDN's _forward_core returns early when attn_metadata is None, making
    # GDN a no-op in the captured graph -> garbled output.  Force
    # force_attention=True for PIECEWISE so attn_metadata is built during
    # capture, allowing GDN's recurrent attention capturing branch to run.
    _orig_dummy_run = R._dummy_run

    def _dummy_run(self, num_tokens, cudagraph_runtime_mode=None,
                   force_attention=False, **kwargs):
        # Force attention metadata build for PIECEWISE mode (both warmup AND capture).
        # During warmup, cudagraph_runtime_mode=NONE but we still need attn_metadata
        # so the GDN recurrent path is reached and v8b instance buffers are
        # pre-allocated in the regular memory pool (not the graph private pool).
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH and (
                cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE
                    and self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE)):
            # Only force attention during REAL warmup (not profile_cudagraph_memory,
            # which uses a minimal KV cache and hits GDN's build_for_cudagraph_capture
            # assertion when num_tokens > decode_cudagraph_max_bs).
            if getattr(self, '_dcut_in_real_warmup', False):
                force_attention = True
        self._dcut_in_dummy_run = True
        try:
            return _orig_dummy_run(
                self, num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention, **kwargs)
        finally:
            self._dcut_in_dummy_run = False

    R._dummy_run = _dummy_run
    logger.warning(
        "D-Cut: patched _dummy_run to force force_attention=True "
        "for PIECEWISE mode (GDN needs attn_metadata during capture)."
    )

    # --- Force use_spec_decode=True during _dummy_run for GDN recurrent path ---
    # During _dummy_run (warmup + capture), use_spec_decode defaults to False.
    # During capture, build_for_cudagraph_capture derives num_accepted_tokens
    # from query_start_loc, so spec_sequence_masks IS set and the recurrent
    # path is reached.  But during WARMUP, for_cudagraph_capture=False and
    # use_spec_decode=False, so extra_attn_metadata_args is empty,
    # num_accepted_tokens=None, spec_sequence_masks=None, and the recurrent
    # path is SKIPPED.  This means v8b instance buffers are never
    # pre-allocated during warmup -> during capture they fall back to the
    # graph's private pool -> overwritten during replay -> garbled output.
    #
    # Fix: when _dcut_in_dummy_run is True and speculative_config is set,
    # force use_spec_decode=True and set up num_decode_draft_tokens /
    # num_accepted_tokens so the GDN builder creates spec_sequence_masks
    # during warmup.  Buffers allocated during warmup (eager, not capturing)
    # go to the REGULAR memory pool, surviving into capture.
#     _orig_build_attn = R._build_attention_metadata
# 
#     def _build_attention_metadata(self, *args, **kwargs):
#         if (getattr(self, '_dcut_in_real_warmup', False)
#                 and self.speculative_config is not None
#                 and not kwargs.get('use_spec_decode', False)):
#             _nst_np = kwargs.get('num_scheduled_tokens_np')
#             _nr = kwargs.get('num_reqs')
#             if _nst_np is not None and _nr is not None and _nr > 0:
#                 num_spec = self.num_spec_tokens
#                 _nrp = kwargs.get('num_reqs_padded') or _nr
#                 # Set up num_decode_draft_tokens (>= 0 marks spec-decode reqs)
#                 if hasattr(self, 'num_decode_draft_tokens'):
#                     for i in range(_nr):
#                         self.num_decode_draft_tokens.np[i] = min(
#                             int(_nst_np[i]) - 1, num_spec
#                         )
#                     self.num_decode_draft_tokens.np[_nr:_nrp].fill(-1)
#                     self.num_decode_draft_tokens.copy_to_gpu()
#                 # Set up num_accepted_tokens
#                 if hasattr(self, 'num_accepted_tokens'):
#                     for i in range(_nr):
#                         self.num_accepted_tokens.gpu[i] = min(
#                             int(_nst_np[i]), num_spec + 1
#                         )
#                     self.num_accepted_tokens.gpu[_nr:_nrp].fill_(1)
#                 kwargs['use_spec_decode'] = True
#         return _orig_build_attn(self, *args, **kwargs)
# 
#     R._build_attention_metadata = _build_attention_metadata
#     logger.warning(
#         "D-Cut: patched _build_attention_metadata to force use_spec_decode=True "
#         "during _dummy_run (GDN recurrent path needs spec_sequence_masks)."
#     )

    # --- PIECEWISE conv1d subtask update ---
    # update_full_graph_params (which calls update_conv1d_graph_params) is
    # only invoked for FULL mode in the upstream model_runner.  Now that GDN
    # is inside the PIECEWISE graph, conv1d subtask host args must be updated
    # before each PIECEWISE replay too.  Without this, the events created
    # during capture are never re-recorded, and the graph replay hangs
    # waiting for them.
    # NOTE: Only apply when ENABLE_GDN_MAIN_PIECEWISE_GRAPH=True. When False,
    # GDN is a splitting op (eager), so conv1d is also eager and does not
    # need graph param updates. Calling update_conv1d_graph_params in that
    # case causes silent EngineCore crashes under high concurrency.
    _orig_update_full = R._update_full_graph_params_if_needed

    def _update_full_graph_params_if_needed(
        self, forward_context, num_tokens_padded, positions
    ):
        # When GDN is a splitting op (flag=False), conv1d is eager and does
        # not need graph param updates. Calling update_conv1d_graph_params
        # in that case causes silent EngineCore crashes under high concurrency.
        if not ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            _orig_update_full(self, forward_context, num_tokens_padded, positions)
            return
        # For prefill steps, force eager mode. The GDN graph only captured
        # the spec (decode) branch, which can't handle prefill (needs
        # chunk_gated_delta_rule).
        _am = getattr(forward_context, 'attn_metadata', None)
        # Detect spec decode: check if any metadata in the dict has
        # spec_sequence_masks is not None. If none do, this is a
        # prefill or regular decode step — force eager to avoid
        # graph replay hang (events never re-recorded when
        # spec_sequence_masks is None).
        # Also check for mixed batches (prefill + spec decode): the conv1d
        # host arg padding can't handle mixed batches (EZ9999 tiling failure).
        # Force eager for any batch with prefills.
        _has_spec = False
        _has_prefill = False
        if isinstance(_am, dict):
            for _v in _am.values():
                if getattr(_v, 'spec_sequence_masks', None) is not None:
                    _has_spec = True
                if getattr(_v, 'num_prefills', 0) > 0:
                    _has_prefill = True
        else:
            if getattr(_am, 'spec_sequence_masks', None) is not None:
                _has_spec = True
            if getattr(_am, 'num_prefills', 0) > 0:
                _has_prefill = True
        if not _has_spec or _has_prefill:
            if (forward_context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    and not getattr(self, '_dcut_in_dummy_run', False)):
                forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        _orig_update_full(self, forward_context, num_tokens_padded, positions)
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            and not forward_context.capturing
            and not self.use_sparse
            and not self.use_compress
        ):
            if not hasattr(self, 'update_stream'):
                import torch_npu as _tn
                self.update_stream = _tn.npu.Stream()
            from vllm_ascend.ops.gdn import update_conv1d_graph_params
            from vllm_ascend.compilation.acl_graph import _EXTRA_CTX as _ec
            update_conv1d_graph_params(
                self.update_stream,
                forward_context,
                num_tokens_padded,
                self.vllm_config,
                _ec.is_draft_model,
                None,
            )

    R._update_full_graph_params_if_needed = _update_full_graph_params_if_needed
    logger.warning(
        'D-Cut: patched _update_full_graph_params_if_needed for PIECEWISE conv1d update.'
    )

    # Patch _model_forward to call correct_conv1d_state() after graph replay.
    # With ENPU enabled, the flow is:
    #   _update_full_graph_params_if_needed() -> saves correction data (state before replay)
    #   run_model() -> graph replay (corrupts last segment state via padding)
    #   correct_conv1d_state() -> recomputes last segment state from saved state + real data
    _orig_model_forward = R._model_forward

    def _model_forward(self, *args, **kwargs):
        # Fill GDN static buffers before graph replay (outside captured graph).
        # At capture time: fills buffers so _forward_core's capture path can
        # pass them to the GDN op (stable data_ptr baked into graph).
        # At replay time: fills buffers with runtime values so the GDN op
        # inside the replayed graph reads updated ASL/SSI/NAT.
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            try:
                from vllm.forward_context import get_forward_context
                from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
                fc = get_forward_context()
                if fc is not None and fc.attn_metadata is not None:
                    # num_tokens is NOT passed to _model_forward as an arg.
                    # It's stored in fc.batch_descriptor.num_tokens by
                    # set_forward_context(num_tokens=num_tokens_padded).
                    _nt = fc.batch_descriptor.num_tokens if fc.batch_descriptor else None
                    if _nt is not None:
                        _dcut_update_gdn_static(fc, _nt, GDNAttentionMetadata)
                        logger.warning("D-Cut: _model_forward filled GDN static bufs, num_tokens=%d", _nt)
            except Exception as _e:
                logger.debug("D-Cut: GDN static buf update skipped: %s", _e)
        result = _orig_model_forward(self, *args, **kwargs)
        # DISABLED: state correction causes accuracy regression
        # if getattr(self, 'enable_enpu', False):
        #     from vllm_ascend.ops.gdn import correct_conv1d_state
        #     correct_conv1d_state()
        return result

    R._model_forward = _model_forward
    logger.warning(
        'D-Cut: patched _model_forward for post-replay conv1d state correction.'
    )


    logger.warning("D-Cut: patched NPUModelRunner.")


def _patch_worker() -> None:
    import vllm_ascend.worker.worker as m

    W = m.NPUWorker
    if getattr(W, "_dcut_patched", False):
        return

    _orig = W.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *a, **k):
        runner = getattr(self, "model_runner", None)
        if runner is not None and hasattr(runner, "_dcut_enable_drafter_probs"):
            try:
                runner._dcut_enable_drafter_probs()
            except Exception as e:
                logger.warning("D-Cut: enabling draft probs before warmup failed: %s", e)
        # Mark that we're in the REAL warmup (not profile_cudagraph_memory).
        # _build_attention_metadata patch checks this to force use_spec_decode=True
        # only during real warmup, not during profile_cudagraph_memory (which uses
        # a minimal KV cache that can't support spec-decode conv1d).
        if runner is not None:
            runner._dcut_in_real_warmup = True
        try:
            ret = _orig(self, *a, **k)
            if runner is not None and hasattr(runner, "profile_adaptive_cost"):
                try:
                    runner.profile_adaptive_cost()
                except Exception as e:
                    # Empty cost table => controller no-ops => graceful fall back to
                    # vanilla DFlash (full-length verify).
                    import traceback
                    logger.error("D-Cut: cost profiling failed; falling back: %s", e)
                    logger.error("D-Cut: full traceback: %s", traceback.format_exc())
                    ctrl = getattr(runner, "_verify_adaptive_controller", None)
                    if ctrl is not None:
                        ctrl._cost_table.clear()
                        ctrl._sorted_bs.clear()
                        ctrl._sorted_sql_per_bs.clear()
        finally:
            if runner is not None:
                runner._dcut_in_real_warmup = False
        return ret

    W.compile_or_warm_up_model = compile_or_warm_up_model
    W._dcut_patched = True
    logger.info("D-Cut: patched NPUWorker.")


def _patch_attention() -> None:
    """Patch full-attention (FIA) to skip the capturing branch in PIECEWISE mode.

    In PIECEWISE mode, full-attention ops are splitting ops — they run eagerly
    between graph pieces.  But ``_EXTRA_CTX.capturing`` is True during the
    entire capture process, so ``forward_fused_infer_attention`` enters its
    capturing branch and calls ``full_graph_fia`` → ``graph_task_group_begin``,
    which fails because the stream is not in capture status.

    Fix: temporarily set ``capturing=False`` for the duration of the full-
    attention forward pass when in PIECEWISE mode, so it uses the eager code
    path (correct for splitting ops).
    """
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        from vllm_ascend.ascend_forward_context import (
            _EXTRA_CTX,
            get_forward_context,
        )
    except Exception as e:
        logger.warning("D-Cut: cannot import AscendAttentionBackendImpl: %s", e)
        return

    if getattr(AscendAttentionBackendImpl, "_dcut_patched", False):
        return

    _orig_ffia = AscendAttentionBackendImpl.forward_fused_infer_attention

    def _forward_fused_infer_attention(
        self, query, key, value, attn_metadata, output, kv_cache=None
    ):
        if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
            ctx = get_forward_context()
            mode = getattr(ctx, "cudagraph_runtime_mode", None)
            if mode == CUDAGraphMode.PIECEWISE:
                # Full attention is a splitting op in PIECEWISE mode.
                # Temporarily disable capturing so the original method
                # uses the eager code path instead of full_graph_fia.
                orig_capturing = ctx.capturing
                ctx.capturing = False
                try:
                    return _orig_ffia(
                        self, query, key, value, attn_metadata, output, kv_cache
                    )
                finally:
                    ctx.capturing = orig_capturing
        return _orig_ffia(
            self, query, key, value, attn_metadata, output, kv_cache
        )

    AscendAttentionBackendImpl.forward_fused_infer_attention = (
        _forward_fused_infer_attention
    )
    AscendAttentionBackendImpl._dcut_patched = True
    logger.warning(
        "D-Cut: patched forward_fused_infer_attention to skip capturing "
        "branch in PIECEWISE mode (full attention is a splitting op)."
    )


def _apply_patches_once() -> None:
    """Apply the real monkey patches.  Runs once per process, deferred to the
    first worker construction (see ``install``) so that importing the NPU
    worker/runner/proposer modules is safe."""
    global _PATCHED
    if _PATCHED:
        return
    # Mark done up-front so a failure (e.g. non-Ascend platform) is not retried
    # on every subsequent worker construction and does not spam the log.
    _PATCHED = True
    try:
        import sys as _dbg
        _patch_proposer()
        _patch_runner()
        _patch_worker()
        _patch_attention()
        _patch_gdn_dcut()
        logger.info(
            "D-Cut adaptive-verify patches applied for NPU "
            "(active only if VLLM_DCUT_CONFIG is set + method is dflash/PARD)."
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut patching failed (vLLM continues normally): %s", e)


def install(*args, **kwargs) -> None:
    """vLLM general-plugin entrypoint.  Idempotent; safe to call per process.

    IMPORTANT — deferred by design.  ``install`` runs during *general-plugin
    load*, which happens BEFORE vllm-ascend has finished importing its own
    ``ops/fused_moe`` / ``device`` graph.  Eagerly importing the NPU
    worker/runner/proposer modules here re-enters that partially-initialised
    graph and raises a circular ``ImportError`` that poisons ``sys.modules`` —
    which then breaks vllm-ascend's *own* later imports (e.g.
    ``pre_register_and_update`` -> ``select_experts``), taking down even vanilla
    serving.  So here we only *arm* a deferred trigger on the vLLM-core
    ``WorkerBase`` (safe to import at this point) and apply the real patches on
    the first worker construction, by which time vllm-ascend is fully imported.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    try:
        # GDN removal from _attention_ops — putting GDN inside the PIECEWISE
        # graph so its compute cost is captured in the cost table.
        # This runs in install() which is BEFORE set_splitting_ops_for_v1()
        # copies _attention_ops into splitting_ops. Without this,
        # set_splitting_ops_for_v1() re-adds GDN to splitting_ops even
        # after the __init__ patch removes it.
        if ENABLE_GDN_MAIN_PIECEWISE_GRAPH:
            try:
                from vllm.config import CompilationConfig
                ops = CompilationConfig._attention_ops
                if isinstance(ops, list) and "vllm::qwen_gdn_attention_core" in ops:
                    CompilationConfig._attention_ops = [
                        op for op in ops if op != "vllm::qwen_gdn_attention_core"
                    ]
                    logger.warning(
                        "D-Cut: removed qwen_gdn_attention_core from _attention_ops "
                        "(class-level, in install())"
                    )
            except Exception as e:
                logger.warning(
                    "D-Cut: failed to remove GDN from _attention_ops in install(): %s", e
                )

        # Patch GDNAttentionMetadataBuilder.build_for_cudagraph_capture to
        # skip the decode-only assertion.  The assertion checks
        # num_actual_tokens <= decode_cudagraph_max_bs (256), which is only
        # valid for FULL cudagraph mode (decode-only).  In PIECEWISE mode,
        # GDN is inside the graph and needs attn_metadata built during
        # capture with larger token counts (up to max capture size).
        # The method body after the assertion is just self.build(...), so
        # skipping the assertion is safe.
        try:
            from vllm.v1.attention.backends.gdn_attn import (
                GDNAttentionMetadataBuilder,
            )

            _orig_build_cg = GDNAttentionMetadataBuilder.build_for_cudagraph_capture

            def _build_for_cudagraph_capture(self, common_attn_metadata):
                m = common_attn_metadata
                num_accepted_tokens = torch.diff(m.query_start_loc)
                num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
                return self.build(
                    0, m, num_accepted_tokens, num_decode_draft_tokens_cpu
                )

            GDNAttentionMetadataBuilder.build_for_cudagraph_capture = (
                _build_for_cudagraph_capture
            )
            logger.warning(
                "D-Cut: patched GDNAttentionMetadataBuilder."
                "build_for_cudagraph_capture to skip decode-only assertion "
                "(needed for PIECEWISE mode with GDN in graph)."
            )
        except Exception as e:
            logger.warning(
                "D-Cut: failed to patch GDN build_for_cudagraph_capture: %s", e
            )

        from vllm.v1.worker.worker_base import WorkerBase

        if getattr(WorkerBase, "_dcut_defer_armed", False):
            return
        _orig_wb_init = WorkerBase.__init__

        def __init__(self, *a, **k):
            # NPUWorker.__init__ calls super().__init__() (this) early, before it
            # builds the model runner — so patching here lands before any
            # NPUModelRunner / proposer instance exists.
            _apply_patches_once()
            return _orig_wb_init(self, *a, **k)

        WorkerBase.__init__ = __init__
        WorkerBase._dcut_defer_armed = True
        logger.info(
            "D-Cut deferred installer armed on WorkerBase "
            "(patches apply on first worker init to avoid a vllm-ascend "
            "circular import)."
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut install (arm) failed (vLLM continues normally): %s", e)
# === GDN D-Cut monkeypatch additions ===
# Appended to monkeypatch.py. These changes were previously applied directly
# to /vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py in vllm_dcut.
# Moved here so vllm-ascend stays at vllm_src baseline.
#
# Changes:
# 1. _conv1d_spec_varlen_eager — per-request F.conv1d fallback for variable
#    query_len (D-Cut truncation on hybrid Mamba/GDN)
# 2. _patch_gdn_dcut — patches AscendGatedDeltaNetAttention._forward_core to:
#    a. Use _conv1d_spec_varlen_eager in the spec Conv1D eager path
#    b. Align ssm_state_indices with actual token positions (boolean mask)
#    c. Clamp num_accepted_tokens to actual seq lengths

from torch.nn import functional as _F


def _conv1d_spec_varlen_eager(
    output_spec,
    mixed_qkv_spec,
    conv_weights,
    conv_state,
    bias,
    activation,
    num_spec,
    spec_query_start_loc,
    spec_state_indices_tensor,
    num_accepted_tokens,
    num_spec_decodes,
):
    """Per-request conv1d for variable query_len spec-decode (D-Cut).

    When D-Cut truncates draft tokens, spec-decode requests have variable
    query_len.  The CANN operator npu_causal_conv1d_custom with run_mode=1
    requires uniform q_per_seq = num_spec + 1 and crashes on variable-length
    input.  This fallback processes each request independently using F.conv1d
    and updates the conv_state following the kernel's spec-decode state-update
    semantics (shift=1, offset from num_accepted_tokens).

    conv_state layout is SD: (num_cache_lines, state_len, dim).
    """
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    width = conv_weights.size(1)  # conv_kernel_size
    state_len = width - 1 + num_spec  # conv_kernel_size - 1 + num_spec
    dim = conv_weights.size(0)
    # Depthwise conv weight: (dim, 1, width)
    dw_weight = conv_weights.unsqueeze(1)

    spec_total = mixed_qkv_spec.size(0)
    out_total = output_spec.size(0)
    for b in range(num_spec_decodes):
        qs = int(spec_query_start_loc[b])
        qe = int(spec_query_start_loc[b + 1])
        # Clamp to actual tensor size — D-Cut RANDOM_CUT may trim tokens,
        # making spec_query_start_loc offsets exceed the tensor length.
        qe = min(qe, spec_total, out_total)
        ql = qe - qs
        ci = int(spec_state_indices_tensor[b, 0])
        nat_b = min(int(num_accepted_tokens[b]), ql)

        if ci == PAD_SLOT_ID or ql <= 0:
            continue

        offset = nat_b - 1  # conv_state_token_offset in kernel

        # --- Conv1d computation ---
        initial_state = conv_state[ci, offset:offset + width - 1, :].t()
        x_b = mixed_qkv_spec[qs:qe].t()
        x_concat = torch.cat([initial_state, x_b], dim=1)
        out = _F.conv1d(
            x_concat.unsqueeze(0),
            dw_weight,
            bias,
            padding=0,
            groups=dim,
        )

        if activation:
            out = _F.silu(out)

        output_spec[qs:qe] = out.squeeze(0).t()

        # --- Conv-state update (kernel spec-decode semantics) ---
        state_len_run = width - 2 + ql
        keep = state_len_run - ql  # = width - 2
        old_state = conv_state[ci, offset:offset + state_len_run, :].clone()
        if keep > 0:
            conv_state[ci, 0:keep, :] = old_state[1:1 + keep, :]
        conv_state[ci, keep:keep + ql, :] = x_b.t()


def _patch_gdn_dcut() -> None:
    """Patch AscendGatedDeltaNetAttention._forward_core for D-Cut.

    Two changes:
    1. Spec Conv1D eager path: use _conv1d_spec_varlen_eager fallback instead
       of CANN op (which crashes on variable query_len from D-Cut truncation).
    2. Recurrent GDN spec kernel call: align ssm_state_indices with actual
       token positions (boolean mask) and clamp num_accepted_tokens to actual
       seq lengths.
    """
    try:
        import torch
        import torch_npu
        from einops import rearrange
        from vllm.distributed import get_pcp_group
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
        from vllm.v1.attention.backend import AttentionMetadata
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
        from vllm_ascend.compilation.acl_graph import (
            get_draft_graph_params,
            get_graph_params,
        )
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        from vllm_ascend.ops.gdn import (
            AscendGatedDeltaNetAttention,
            to_int64_tuple,
            get_non_spec_causal_conv1d_host_args,
            get_causal_conv1d_update_host_args,
            get_spec_causal_conv1d_update_host_args,
            get_non_spec_chunked_prefill_meta,
        )
        from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
        from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
        from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn
        from vllm_ascend.utils import weak_ref_tensors
    except Exception as e:
        import sys as _dbg2
        logger.warning("D-Cut: cannot import GDN ops for patching: %s", e)
        return


    if getattr(QwenGatedDeltaNetAttention, "_dcut_gdn_patched", False):
        return

    # Monkeypatch _pad_conv1d_host_args_to_capture to handle pad_tokens < q_per_seq.
    # When D-Cut truncation reduces spec tokens, the padding falls short.
    from vllm_ascend.ops.gdn import _pad_conv1d_host_args_to_capture as _orig_pad
    if not getattr(_orig_pad, '_dcut_patched', False):
        def _dcut_pad_conv1d_host_args(qsl_host, cidx_host, num_accepted_host,
                                        cap_x_dim0, q_per_seq, with_num_accepted):
            result = _orig_pad(qsl_host, cidx_host, num_accepted_host,
                               cap_x_dim0, q_per_seq, with_num_accepted)
            qsl, cidx, nat = result
            # If still short (pad_tokens < q_per_seq case), add one final dummy
            if qsl and int(qsl[-1]) != cap_x_dim0:
                qsl = tuple(qsl) + (int(cap_x_dim0),)
                cidx = tuple(cidx) + (PAD_SLOT_ID,)
                if with_num_accepted:
                    nat = tuple(nat) + (1,)
            # Clamp num_accepted_tokens to not exceed segment lengths.
            # D-Cut truncation can make nat[i] > (qsl[i+1] - qsl[i]),
            # causing EZ9999: "numAcceptedTokens[i]=X exceeds varlen segment length=Y".
            if with_num_accepted and nat:
                clamped = []
                for i in range(len(nat)):
                    if i + 1 < len(qsl):
                        seg_len = int(qsl[i + 1]) - int(qsl[i])
                        clamped.append(min(int(nat[i]), seg_len))
                    else:
                        clamped.append(int(nat[i]))
                nat = tuple(clamped)
            return qsl, cidx, nat
        _dcut_pad_conv1d_host_args._dcut_patched = True
        # Patch in the gdn module so update_conv1d_graph_params uses the fixed version
        import vllm_ascend.ops.gdn as _gdn_mod
        _gdn_mod._pad_conv1d_host_args_to_capture = _dcut_pad_conv1d_host_args
        logger.warning("D-Cut: patched _pad_conv1d_host_args_to_capture for sub-q_per_seq padding")


    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core attention computation (called by custom op). D-Cut patched."""
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            (spec_qsl_host, spec_ci_host, spec_nat_host) = get_spec_causal_conv1d_update_host_args(attn_metadata)
            if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
                graph_params.conv1d_events[num_actual_tokens].append(event)

                output_spec = torch.empty_like(mixed_qkv_spec)
                spec_q_per_seq = int(attn_metadata.spec_state_indices_tensor.size(-1))
                graph_params.conv1d_params[num_actual_tokens].append(
                    (
                        weak_ref_tensors(output_spec),
                        weak_ref_tensors(mixed_qkv_spec),
                        weak_ref_tensors(conv_weights_T),
                        weak_ref_tensors(self_kv_cache[0]),
                        self.conv1d.bias,
                        activation_num,
                        PAD_SLOT_ID,
                        1,
                        "spec",
                        self.prefix,
                        spec_qsl_host,
                        spec_ci_host,
                        spec_nat_host,
                        spec_q_per_seq,
                    )
                )

                torch.npu.graph_task_group_begin(stream)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    output_spec,
                    mixed_qkv_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=spec_qsl_host,
                    cache_indices_opt=spec_ci_host,
                    initial_state_mode_opt=(),
                    num_accepted_tokens_opt=spec_nat_host,
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=1,
                )
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.conv1d_handles[num_actual_tokens].append(handle)
                mixed_qkv_spec = output_spec
            else:
                # D-Cut: per-request F.conv1d fallback for variable query_len.
                num_spec_decodes = attn_metadata.num_spec_decodes
                use_cann = False  # CANN op crashes in eager mode; always use fallback

                if use_cann:
                    output_spec = torch.empty_like(mixed_qkv_spec)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        output_spec,
                        mixed_qkv_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=spec_qsl_host,
                        cache_indices_opt=spec_ci_host,
                        initial_state_mode_opt=(),
                        num_accepted_tokens_opt=spec_nat_host,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=1,
                    )
                    mixed_qkv_spec = output_spec
                else:
                    output_spec = torch.empty_like(mixed_qkv_spec)
                    _conv1d_spec_varlen_eager(
                        output_spec,
                        mixed_qkv_spec,
                        conv_weights,
                        self_kv_cache[0],
                        self.conv1d.bias,
                        self.activation,
                        self.num_spec,
                        spec_query_start_loc,
                        spec_state_indices_tensor,
                        num_accepted_tokens,
                        num_spec_decodes,
                    )
                    mixed_qkv_spec = output_spec

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                if get_pcp_group().world_size > 1:
                    mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                    has_initial_state = attn_metadata.has_initial_state
                    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
                    conv_state = self_kv_cache[0].transpose(-1, -2)
                    mixed_qkv_non_spec = causal_conv1d_fn(
                        mixed_qkv_non_spec_T,
                        conv_weights,
                        self.conv1d.bias,
                        activation=self.activation,
                        conv_states=conv_state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=attn_metadata,
                    ).transpose(0, 1)
                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    (
                        query_start_loc_opt,
                        cache_indices_opt,
                        initial_state_mode_opt,
                    ) = get_non_spec_causal_conv1d_host_args(attn_metadata)
                    mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec_output,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=[],
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
        elif attn_metadata.num_decodes > 0:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            non_spec_qsl_host, non_spec_ci_host = get_causal_conv1d_update_host_args(attn_metadata)
            if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params = get_graph_params() if not _EXTRA_CTX.is_draft_model else get_draft_graph_params()
                graph_params.conv1d_events[num_actual_tokens].append(event)

                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                non_spec_q_per_seq = 1
                graph_params.conv1d_params[num_actual_tokens].append(
                    (
                        weak_ref_tensors(output_non_spec),
                        weak_ref_tensors(mixed_qkv_non_spec),
                        weak_ref_tensors(conv_weights_T),
                        weak_ref_tensors(self_kv_cache[0]),
                        self.conv1d.bias,
                        activation_num,
                        PAD_SLOT_ID,
                        1,
                        "non_spec_decode",
                        self.prefix,
                        non_spec_qsl_host,
                        non_spec_ci_host,
                        [],
                        non_spec_q_per_seq,
                    )
                )

                torch.npu.graph_task_group_begin(stream)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    output_non_spec,
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=non_spec_qsl_host,
                    cache_indices_opt=non_spec_ci_host,
                    initial_state_mode_opt=(),
                    num_accepted_tokens_opt=[],
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=1,
                )
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.conv1d_handles[num_actual_tokens].append(handle)
                mixed_qkv_non_spec = output_non_spec
            else:
                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    output_non_spec,
                    mixed_qkv_non_spec,
                    conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias_opt=self.conv1d.bias,
                    query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc[: num_actual_tokens + 1]),
                    cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor[:num_actual_tokens]),
                    initial_state_mode_opt=[],
                    num_accepted_tokens_opt=[],
                    activation_mode=activation_num,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=1,
                )
                mixed_qkv_non_spec = output_non_spec
        else:
            mixed_qkv_non_spec = None

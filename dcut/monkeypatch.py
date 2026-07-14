
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

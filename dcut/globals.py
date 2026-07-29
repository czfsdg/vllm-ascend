# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401, SIM105
"""Monkey-patch installer for D-Cut adaptive verifier step-length on **vLLM-Ascend / NPU**.

Ported from the CUDA plugin in ``Bensong0506/vllm`` branch
``feat/dcut-adaptive-verify`` (itself a port of the closed, unmerged vLLM
PR #44885) to run on Huawei Ascend NPU via vllm-ascend (vLLM v0.23.0 base).
The adaptive controller remains a vLLM *general plugin*. This fork also adds
a self-contained 0.23 GDN core plus two D-Cut AscendC state operators.

Algorithm is unchanged (see AngelSlim D-Cut,
https://angelslim.readthedocs.io/zh-cn/latest/dcut.html): the drafter still
proposes ``num_speculative_tokens`` every step, but the verifier only checks a
batch-adaptive subset chosen by a hardware-profiled ITL cost table +
draft-confidence prefix-product scores + batch-wide global top-K.

Only active for parallel speculative methods: ``method=dflash``, or
``method=draft_model`` with ``parallel_drafting=true`` (PARD).

------------------------------------------------------------------------------
GPU -> NPU deltas include two D-Cut state-aware custom operators for GDN
spec-decode. The Python control loop still patches only the NPU runner and the
``_forward_core`` invoked by the GDN custom op. By default the plugin preserves
that custom op in PIECEWISE ``splitting_ops``, so GDN remains an eager boundary.
Set ``VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE=1`` to remove the boundary and
capture GDN with its surrounding graph piece:

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
ENV_FULL_DECODE_ONLY = "VLLM_DCUT_FULL_DECODE_ONLY"
ENV_GDN_SHARED_STATIC = "VLLM_DCUT_GDN_SHARED_STATIC"

# vLLM 0.23 owns GDN graph inputs through GDNSpecDecodeMetadata. The removed
# 0.22 graph-task host-argument APIs must not be patched.

# ── Static GDN buffers for PIECEWISE graph replay ──────────────────
# Pre-allocated ASL/SSI/NAT buffers with stable data_ptr.
# Filled graph-externally by _dcut_update_gdn_static() in _model_forward
# before each replay. The GDN op inside the captured graph reads these buffers
# at replay time.
#
# PIECEWISE aliases every layer prefix in one attention-metadata group to the
# same buffer dict, matching vLLM's group-shared attention metadata. Keys remain
# prefix-addressable so the compiled GDN forward can look them up unchanged.
_dcut_gdn_static = {}

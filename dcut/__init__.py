# SPDX-License-Identifier: Apache-2.0
"""D-Cut adaptive verifier step-length for DFlash/PARD speculative decoding on NPU.

Self-contained monkey-patch plugin (no vLLM / vllm-ascend source edits).  Ported
from the CUDA plugin in Bensong0506/vllm branch ``feat/dcut-adaptive-verify``
(itself a port of the closed, unmerged vLLM PR #44885) to run on Huawei Ascend
NPU via vllm-ascend (vLLM v0.22.1 base).  ``install`` is wired as a
``vllm.general_plugins`` entry point so it is applied automatically in every
engine/worker process (including TP workers).

See RUN.md for enabling and caveats.
"""
from .monkeypatch import install

__all__ = ["install"]

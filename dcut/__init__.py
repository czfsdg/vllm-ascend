# SPDX-License-Identifier: Apache-2.0
"""D-Cut adaptive verifier step-length for vLLM DFlash/PARD speculative decoding.

Self-contained monkey-patch plugin (no vLLM source edits). Ported from the
(closed, unmerged) vLLM PR #44885 to run on vLLM 0.22.x. ``install`` is wired
as a ``vllm.general_plugins`` entry point so it is applied automatically in
every engine/worker process (including TP workers).

See RUN.md for enabling and caveats.
"""

from .monkeypatch import install

__all__ = ["install"]

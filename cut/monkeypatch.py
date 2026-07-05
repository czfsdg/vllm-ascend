# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch installer for D-Cut adaptive verifier step-length.

Copied from the GPU dcut plugin layout; Ascend-specific integration is expected
in follow-up changes.
"""
from __future__ import annotations

import os

ENV_CONFIG = "VLLM_DCUT_CONFIG"
_INSTALLED = False


def install(*args, **kwargs) -> None:
    """vLLM general-plugin entrypoint.

    This initial copy keeps the entrypoint idempotent and non-invasive in the
    Ascend repository while the GPU implementation is reviewed for porting.
    """
    del args, kwargs
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if os.environ.get(ENV_CONFIG):
        # Keep import side effects explicit; full runner patching is not enabled
        # until the GPU-specific code is adapted for Ascend workers.
        return

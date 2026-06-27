# SPDX-License-Identifier: Apache-2.0
"""D-Cut plugin entrypoint for vLLM Ascend.

Usage after installing this repository editable::

    pip install -e .
    VLLM_ASCEND_ENABLE_DCUT=1 vllm serve ...

Set ``VLLM_ASCEND_DCUT_CONFIG=/path/to/config.json`` to override defaults. If your
environment restricts vLLM general plugins with ``VLLM_PLUGINS``, append ``dcut``
without dropping the existing Ascend plugin entries.
"""

from __future__ import annotations

import os

from vllm.logger import logger


def register() -> None:
    """Register the D-Cut plugin with vLLM's general plugin loader.

    The implementation hooks live in ``vllm_ascend`` so they are available to
    the Ascend worker. This plugin entrypoint provides the installable/opt-in
    UX: install editable, then set ``VLLM_ASCEND_ENABLE_DCUT=1``.
    """
    if os.getenv("VLLM_ASCEND_ENABLE_DCUT", "0") == "1":
        logger.info(
            "D-Cut plugin is enabled. Config: %s",
            os.getenv("VLLM_ASCEND_DCUT_CONFIG", "<built-in defaults>"),
        )
    else:
        logger.info(
            "D-Cut plugin is installed but disabled. Set VLLM_ASCEND_ENABLE_DCUT=1 to enable it."
        )

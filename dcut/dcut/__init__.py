# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def register() -> None:
    """vLLM general-plugin entrypoint."""
    from vllm.logger import logger

    from dcut.monkeypatch import apply_patch

    apply_patch()
    logger.info("D-Cut plugin registered.")

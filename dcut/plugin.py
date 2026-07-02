# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin entry point for D-Cut planning."""

from __future__ import annotations

import logging
import os
from functools import wraps

from dcut.cost_table import DcutCostTable
from dcut.policy import DcutPolicy

logger = logging.getLogger("dcut")
_PATCHED = False


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _batch_size_from_runner(runner) -> int:
    input_batch = getattr(runner, "input_batch", None)
    return int(getattr(input_batch, "num_reqs", 1) or 1)


def _num_speculative_tokens(runner) -> int:
    speculative_config = getattr(runner, "speculative_config", None)
    return int(getattr(speculative_config, "num_speculative_tokens", 1) or 1)


def _install_runner_patch() -> None:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_init = NPUModelRunner.__init__
    original_propose = NPUModelRunner.propose_draft_token_ids

    @wraps(original_init)
    def init_with_dcut(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not _env_flag("DCUT_ENABLE", "1"):
            return
        max_verify_len = _num_speculative_tokens(self)
        self.dcut_cost_table = DcutCostTable(max_verify_len=max_verify_len)
        self.dcut_policy = DcutPolicy(self.dcut_cost_table)
        logger.info(
            "[dcut] cost-table initialized: enabled=%s max_verify_len=%d table=[%s]",
            True,
            max_verify_len,
            self.dcut_cost_table.summary(),
        )

    @wraps(original_propose)
    def propose_with_dcut(self, *args, **kwargs):
        policy = getattr(self, "dcut_policy", None)
        if policy is not None and _env_flag("DCUT_ENABLE", "1"):
            requested_len = _num_speculative_tokens(self)
            decision = policy.decide(
                requested_len=requested_len,
                batch_size=_batch_size_from_runner(self),
            )
            self.dcut_last_decision = decision
            logger.info(
                "[dcut] cut-policy decision: requested_len=%d selected_len=%d "
                "batch_size=%d acceptance_rate=%.3f score=%.6f reason=%s "
                "mode=plan_only",
                decision.requested_len,
                decision.selected_len,
                decision.batch_size,
                decision.acceptance_rate,
                decision.score,
                decision.reason,
            )
        return original_propose(self, *args, **kwargs)

    NPUModelRunner.__init__ = init_with_dcut
    NPUModelRunner.propose_draft_token_ids = propose_with_dcut


def register() -> None:
    """Register D-Cut as a vLLM general plugin.

    This patch only builds/logs the cost table and policy decision. It does not
    change speculative token tensors yet; the actual cut execution is reserved
    for the next implementation step.
    """

    global _PATCHED
    if _PATCHED:
        return
    _install_runner_patch()
    _PATCHED = True
    logger.info("[dcut] plugin registered: phase=cost_table_and_cut_policy")

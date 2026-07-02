# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin entry point for D-Cut planning."""

from __future__ import annotations

import builtins
import logging
import os
import sys
from functools import wraps
from types import ModuleType

from dcut.config import load_dcut_config
from dcut.cost_table import DcutCostTable
from dcut.policy import DcutPolicy

logger = logging.getLogger("dcut")
_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_PATCHED = False
_IMPORT_HOOK_INSTALLED = False
_ORIGINAL_IMPORT = builtins.__import__


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _batch_size_from_runner(runner) -> int:
    input_batch = getattr(runner, "input_batch", None)
    return int(getattr(input_batch, "num_reqs", 1) or 1)


def _num_speculative_tokens(runner) -> int:
    speculative_config = getattr(runner, "speculative_config", None)
    return int(getattr(speculative_config, "num_speculative_tokens", 1) or 1)


def _spec_method(runner) -> str:
    speculative_config = getattr(runner, "speculative_config", None)
    return str(getattr(speculative_config, "method", "") or "")


def _should_use_target_only(runner) -> bool:
    if not getattr(runner, "dcut_accuracy_safe_mode", True):
        return False
    target_only_methods = getattr(runner, "dcut_target_only_methods", ("dflash",))
    return _spec_method(runner) in target_only_methods


def _patch_runner_class(npu_model_runner: type) -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_init = npu_model_runner.__init__
    original_propose = npu_model_runner.propose_draft_token_ids

    @wraps(original_init)
    def init_with_dcut(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not _env_flag("DCUT_ENABLE", "1"):
            return
        config, config_source = load_dcut_config()
        max_verify_len = _num_speculative_tokens(self)
        self.dcut_cost_table = DcutCostTable(
            max_verify_len=max_verify_len,
            target_base_cost=config.target_base_cost,
            target_token_cost=config.target_token_cost,
            draft_token_cost=config.draft_token_cost,
        )
        self.dcut_policy = DcutPolicy(
            self.dcut_cost_table,
            default_acceptance_rate=config.default_acceptance_rate,
            high_concurrency_batch=config.high_concurrency_batch,
        )
        self.dcut_accuracy_safe_mode = config.accuracy_safe_mode
        self.dcut_target_only_methods = config.target_only_methods
        logger.info(
            "[dcut] cost-table initialized: enabled=%s config=%s max_verify_len=%d "
            "accuracy_safe_mode=%s target_only_methods=%s table=[%s]",
            True,
            config_source or "<defaults>",
            max_verify_len,
            config.accuracy_safe_mode,
            config.target_only_methods,
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
            if _should_use_target_only(self):
                logger.warning(
                    "[dcut] accuracy-safe target-only fallback is active for "
                    "speculative method=%s. Returning no draft tokens to preserve "
                    "target-model output quality. Set DCUT_ACCURACY_SAFE_MODE=0 "
                    "after DFlash acceptance/accuracy is verified.",
                    _spec_method(self),
                )
                return None
        return original_propose(self, *args, **kwargs)

    npu_model_runner.__init__ = init_with_dcut
    npu_model_runner.propose_draft_token_ids = propose_with_dcut
    _PATCHED = True
    logger.info("[dcut] NPUModelRunner patch installed")


def _maybe_patch_loaded_runner(module: ModuleType | None = None) -> bool:
    runner_module = module or sys.modules.get(_RUNNER_MODULE)
    npu_model_runner = getattr(runner_module, "NPUModelRunner", None)
    if npu_model_runner is None:
        return False
    _patch_runner_class(npu_model_runner)
    return True


def _install_import_hook() -> None:
    global _IMPORT_HOOK_INSTALLED
    if _IMPORT_HOOK_INSTALLED:
        return

    def dcut_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        if name == _RUNNER_MODULE or name.startswith(f"{_RUNNER_MODULE}."):
            _maybe_patch_loaded_runner(sys.modules.get(_RUNNER_MODULE))
        return module

    builtins.__import__ = dcut_import
    _IMPORT_HOOK_INSTALLED = True


def register() -> None:
    """Register D-Cut as a vLLM general plugin.

    vLLM loads general plugins while building CLI arguments, before importing
    the Ascend model runner is safe. Therefore registration only installs an
    import hook and patches ``NPUModelRunner`` after vLLM imports it normally.
    """

    if not _env_flag("DCUT_ENABLE", "1"):
        logger.info("[dcut] plugin disabled by DCUT_ENABLE=0")
        return
    if not _maybe_patch_loaded_runner():
        _install_import_hook()
        logger.info("[dcut] plugin registered: waiting_for=%s", _RUNNER_MODULE)

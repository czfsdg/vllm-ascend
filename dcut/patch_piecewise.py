# SPDX-License-Identifier: Apache-2.0
"""Keep GDN as a PIECEWISE boundary for hybrid local graph replay.

The outer model stays in PIECEWISE mode for every batch composition. The
``vllm::qwen_gdn_attention_core`` boundary selects between a local pure-spec
GDN ACLGraph and the original eager prefill/mixed implementation.
"""

from __future__ import annotations

import os

_TARGET_OP = "vllm::qwen_gdn_attention_core"
ENV_DCUT_CONFIG = "VLLM_DCUT_CONFIG"
ENV_GDN_PIECEWISE = "VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE"
LEGACY_ENV_GDN_PIECEWISE = "VLLM_DCUT_GDN_PIECEWISE"


def _ensure_gdn_splitting_op(ops):
    """Return a fresh splitting-op list that contains the GDN boundary."""
    result = list(ops or ())
    if _TARGET_OP not in result:
        result.append(_TARGET_OP)
    return result


def _env_flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _is_enabled() -> bool:
    """Return whether GDN may be captured by PIECEWISE ACLGraph.

    The registered vllm-ascend variable is authoritative. The earlier
    D-Cut-only spelling remains accepted so existing launch scripts do not
    silently lose the optimization.
    """
    if not os.environ.get(ENV_DCUT_CONFIG):
        return False

    legacy = os.environ.get(LEGACY_ENV_GDN_PIECEWISE)
    if legacy is not None:
        return _env_flag(legacy)

    try:
        from vllm_ascend import envs

        return bool(envs.VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE)
    except (AttributeError, ImportError, ValueError):
        return _env_flag(os.environ.get(ENV_GDN_PIECEWISE, "0"))


def _arm_gdn_piecewise_splitting_patch():
    """Ensure the GDN boundary is present in every PIECEWISE process.
    Must be called **before** vllm-ascend platform code invokes
    ``set_splitting_ops_for_v1`` (i.e. during ``install()``, not during the
    deferred ``WorkerBase.__init__`` trigger).  Only vLLM-core symbols are
    imported here — no vllm_ascend.worker / model_runner — so there is no
    circular-import risk.

    Using ``print(flush=True)`` for all diagnostics because ``logger.*`` calls
    are silently swallowed in the dcut vLLM service process.
    """
    if not _is_enabled():
        print(
            "[D-Cut] GDN local graph patch SKIPPED "
            f"(requires {ENV_DCUT_CONFIG} and "
            f"{ENV_GDN_PIECEWISE}=1).",
            flush=True,
        )
        return

    try:
        from vllm.config import CompilationConfig
    except Exception as exc:  # pragma: no cover - vLLM not installed
        print(
            f"[D-Cut] cannot import CompilationConfig, "
            f"GDN local graph patch NOT armed: {exc}",
            flush=True,
        )
        return

    if getattr(
        CompilationConfig.set_splitting_ops_for_v1,
        "_dcut_piecewise_patched",
        False,
    ):
        print("[D-Cut] GDN local graph patch already armed (skip).", flush=True)
        return

    _ORIG_SET_SPLITTING_OPS = CompilationConfig.set_splitting_ops_for_v1

    def _patched_set_splitting_ops_for_v1(self, *args, **kwargs):
        _ORIG_SET_SPLITTING_OPS(self, *args, **kwargs)

        before = list(self.splitting_ops or ())
        after = _ensure_gdn_splitting_op(self.splitting_ops)
        self.splitting_ops = after

        if _TARGET_OP not in before:
            print(
                "[D-Cut] added the GDN local-graph boundary to splitting_ops "
                f"(before={len(before)} ops, after={len(after)} ops).",
                flush=True,
            )
            print(f"[D-Cut] splitting_ops before={before}", flush=True)
            print(f"[D-Cut] splitting_ops after ={after}", flush=True)

    _patched_set_splitting_ops_for_v1._dcut_piecewise_patched = True  # type: ignore[attr-defined]
    CompilationConfig.set_splitting_ops_for_v1 = (  # type: ignore[assignment]
        _patched_set_splitting_ops_for_v1
    )
    print(
        "[D-Cut] GDN boundary preserved for local PIECEWISE graph replay.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Module-level arming — runs at import time so the patch is applied in EVERY
# process that imports the dcut package (EngineCore + Worker), not just where
# ``install()`` is called.  The vLLM general-plugin ``install()`` entrypoint
# is only invoked in Worker processes; the EngineCore process (where
# ``set_splitting_ops_for_v1`` actually runs during config creation) never
# calls ``install()``.  Arming at import time closes that gap.
# ---------------------------------------------------------------------------
print("[D-Cut] patch_piecewise module imported — arming patch.", flush=True)
try:
    _arm_gdn_piecewise_splitting_patch()
except Exception as _e:  # pragma: no cover - never break import
    print(f"[D-Cut] patch_piecewise module-level arm failed: {_e}", flush=True)

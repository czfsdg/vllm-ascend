# SPDX-License-Identifier: Apache-2.0
"""Remove ``vllm::qwen_gdn_attention_core`` from PIECEWISE splitting_ops so the
native GDN op (and the D-Cut conv/recurrent kernels behind it) are captured
into the same ACLGraph segment instead of being a graph split boundary.

Only the splitting_ops list is touched.  The GDN forward itself
(``torch.ops.vllm.qwen_gdn_attention_core`` / ``_forward_core``) is left
unchanged — the op still exists as a custom op in the graph, it simply no
longer forces a piecewise boundary.
"""

from __future__ import annotations

import os

_TARGET_OP = "vllm::qwen_gdn_attention_core"
ENV_DCUT_CONFIG = "VLLM_DCUT_CONFIG"
ENV_GDN_PIECEWISE = "VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE"
LEGACY_ENV_GDN_PIECEWISE = "VLLM_DCUT_GDN_PIECEWISE"


def _filter_splitting_ops(ops):
    """Return a new list with *only* ``_TARGET_OP`` removed.

    A fresh list is returned (never mutates the input in place) so the caller
    can safely reassign ``self.splitting_ops``.
    """
    if ops is None:
        return None
    return [op for op in ops if op != _TARGET_OP]


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
    """Wrap ``CompilationConfig.set_splitting_ops_for_v1`` and
    ``splitting_ops_contain_attention`` once per process.

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
            "[D-Cut] GDN PIECEWISE split patch SKIPPED "
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
            f"GDN PIECEWISE split patch NOT armed: {exc}",
            flush=True,
        )
        return

    if getattr(
        CompilationConfig.set_splitting_ops_for_v1,
        "_dcut_piecewise_patched",
        False,
    ):
        print("[D-Cut] GDN PIECEWISE split patch already armed (skip).", flush=True)
        return

    # -------------------------------------------------------------------------
    # Patch 1: Remove _TARGET_OP from splitting_ops
    # -------------------------------------------------------------------------
    _ORIG_SET_SPLITTING_OPS = CompilationConfig.set_splitting_ops_for_v1

    def _patched_set_splitting_ops_for_v1(self, *args, **kwargs):
        _ORIG_SET_SPLITTING_OPS(self, *args, **kwargs)

        before = list(self.splitting_ops or [])
        after = _filter_splitting_ops(self.splitting_ops)
        self.splitting_ops = after

        removed = [op for op in before if after is None or op not in after]
        if removed:
            print(
                f"[D-Cut] GDN PIECEWISE split patch — removed {removed} "
                f"from splitting_ops (before={len(before)} ops, "
                f"after={len(after) if after else 0} ops).",
                flush=True,
            )
            print(f"[D-Cut] splitting_ops before={before}", flush=True)
            print(f"[D-Cut] splitting_ops after ={after}", flush=True)

    _patched_set_splitting_ops_for_v1._dcut_piecewise_patched = True  # type: ignore[attr-defined]
    CompilationConfig.set_splitting_ops_for_v1 = (  # type: ignore[assignment]
        _patched_set_splitting_ops_for_v1
    )
    print("[D-Cut] Patched set_splitting_ops_for_v1.", flush=True)

    # -------------------------------------------------------------------------
    # Patch 2: Make splitting_ops_contain_attention() exclude _TARGET_OP
    # This is needed because vLLM's assertion in CudagraphDispatcher.__init__
    # requires all _attention_ops to be in splitting_ops, but we removed
    # _TARGET_OP from splitting_ops.
    # -------------------------------------------------------------------------
    _ORIG_SPLITTING_OPS_CONTAIN_ATTENTION = CompilationConfig.splitting_ops_contain_attention

    def _patched_splitting_ops_contain_attention(self):
        # Filter out _TARGET_OP from _attention_ops for this check
        attention_ops_to_check = [op for op in self._attention_ops if op != _TARGET_OP]
        return self.splitting_ops is not None and all(
            op in self.splitting_ops for op in attention_ops_to_check
        )

    CompilationConfig.splitting_ops_contain_attention = (  # type: ignore[assignment]
        _patched_splitting_ops_contain_attention
    )
    print("[D-Cut] Patched splitting_ops_contain_attention to exclude _TARGET_OP.", flush=True)

    print("[D-Cut] GDN PIECEWISE split patch ARMED.", flush=True)


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

# SPDX-License-Identifier: Apache-2.0
"""Bootstrap the repository-local D-Cut custom-operator installation."""

from __future__ import annotations

import os
from pathlib import Path

_LOCAL_DCUT_VENDOR_PATH = Path(__file__).resolve().parent / "kernel" / "build" / "custom_ops" / "vendors" / "dcut"


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    entries = [entry for entry in os.environ.get(name, "").split(os.pathsep) if entry]
    if value not in entries:
        os.environ[name] = os.pathsep.join((value, *entries))


def bootstrap_dcut_custom_op_env() -> Path | None:
    """Expose the local OPP before the Torch registration library is loaded."""
    if not _LOCAL_DCUT_VENDOR_PATH.is_dir():
        return None
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", _LOCAL_DCUT_VENDOR_PATH)
    return _LOCAL_DCUT_VENDOR_PATH

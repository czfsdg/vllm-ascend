# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _load_kernel_runtime():
    module_path = Path(__file__).resolve().parents[1] / "kernel_runtime.py"
    spec = importlib.util.spec_from_file_location("dcut_kernel_runtime_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_custom_op_vendor_is_prepended_once() -> None:
    module = _load_kernel_runtime()

    with TemporaryDirectory() as tmpdir:
        vendor_path = Path(tmpdir) / "vendors" / "dcut"
        vendor_path.mkdir(parents=True)
        module._LOCAL_DCUT_VENDOR_PATH = vendor_path

        with patch.dict(
            os.environ,
            {"ASCEND_CUSTOM_OPP_PATH": "/existing/vendor"},
            clear=False,
        ):
            assert module.bootstrap_dcut_custom_op_env() == vendor_path
            module.bootstrap_dcut_custom_op_env()
            assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [
                str(vendor_path),
                "/existing/vendor",
            ]


def test_missing_local_vendor_keeps_environment_unchanged() -> None:
    module = _load_kernel_runtime()

    with TemporaryDirectory() as tmpdir:
        module._LOCAL_DCUT_VENDOR_PATH = Path(tmpdir) / "missing"
        with patch.dict(
            os.environ,
            {"ASCEND_CUSTOM_OPP_PATH": "/existing/vendor"},
            clear=False,
        ):
            assert module.bootstrap_dcut_custom_op_env() is None
            assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/existing/vendor"

# SPDX-License-Identifier: Apache-2.0
"""Editable installer for the standalone D-Cut adaptive verify plugin.

This file intentionally lives inside ``dcut/`` so users can run
``pip install -e .`` from that directory when the base vLLM Ascend image is
already fixed and only this plugin directory is mounted or copied in.
"""

from setuptools import setup

setup(
    name="dcut-adaptive-verify",
    version="0.1.0",
    description="D-Cut adaptive speculative verification plugin for vLLM Ascend",
    packages=["dcut"],
    package_dir={"dcut": "."},
    package_data={
        "dcut": [
            "serve_dcut_adaptive_verify.sh",
            "verify_adaptive_config.example.json",
        ]
    },
    python_requires=">=3.10",
    entry_points={
        "vllm.general_plugins": [
            "dcut_adaptive_verify = dcut:register",
            "dcut = dcut:register",
        ],
    },
)

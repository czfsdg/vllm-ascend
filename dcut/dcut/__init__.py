# SPDX-License-Identifier: Apache-2.0
"""D-Cut adaptive verifier plugin for vLLM Ascend."""


def install() -> None:
    from dcut.monkeypatch import install as _install

    _install()


__all__ = ["install"]

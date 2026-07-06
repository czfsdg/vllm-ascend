# SPDX-License-Identifier: Apache-2.0
"""D-Cut adaptive verifier plugin entrypoint."""

from .monkeypatch import install

# Some deployed/editable environments may still have an older entry point that
# resolves to ``dcut:register``. Keep it as a compatibility alias so plugin
# loading remains stable after upgrading the package in place.
register = install

__all__ = ["install", "register"]

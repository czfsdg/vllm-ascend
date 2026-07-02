# SPDX-License-Identifier: Apache-2.0
"""Configuration loading for the D-Cut adaptive verify plugin."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DcutConfig:
    target_base_cost: float = 1.0
    target_token_cost: float = 0.18
    draft_token_cost: float = 0.08
    default_acceptance_rate: float = 0.75
    high_concurrency_batch: int = 16


def load_dcut_config() -> tuple[DcutConfig, str | None]:
    """Load ``DCUT_CONFIG`` when present.

    The config file is optional so the plugin can still be enabled with only
    ``DCUT_ENABLE=1``. Returning the source path makes server logs explicit
    about which configuration is active.
    """

    config_path = os.getenv("DCUT_CONFIG")
    if not config_path:
        return DcutConfig(), None

    path = Path(config_path).expanduser()
    with path.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)

    return _parse_config(raw_config), str(path)


def _parse_config(raw_config: dict[str, Any]) -> DcutConfig:
    cost_table = raw_config.get("cost_table", {})
    policy = raw_config.get("policy", {})
    return DcutConfig(
        target_base_cost=float(cost_table.get("target_base_cost", DcutConfig.target_base_cost)),
        target_token_cost=float(cost_table.get("target_token_cost", DcutConfig.target_token_cost)),
        draft_token_cost=float(cost_table.get("draft_token_cost", DcutConfig.draft_token_cost)),
        default_acceptance_rate=float(policy.get("default_acceptance_rate", DcutConfig.default_acceptance_rate)),
        high_concurrency_batch=int(policy.get("high_concurrency_batch", DcutConfig.high_concurrency_batch)),
    )

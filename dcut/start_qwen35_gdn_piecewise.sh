#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Backward-compatible entrypoint for the requested NPU Qwen3.5-9B D-Cut launch.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_npu_qwen35_9b_dcut.sh" "$@"

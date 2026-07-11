#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Example launcher for Qwen3.5 GDN + PIECEWISE graph + D-Cut cost table dump.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL:?Set MODEL=/path/to/Qwen3.5-GDN model}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL=/path/to/Qwen3.5-GDN draft model}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-gdn}"
TP="${TP:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SPEC_TOKENS="${SPEC_TOKENS:-15}"
DCUT_CONFIG="${DCUT_CONFIG:-$SCRIPT_DIR/qwen35_gdn_piecewise_config.example.json}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/qwen35_gdn_piecewise_results/$(date +%Y%m%d_%H%M%S)}"
COST_TABLE_JSON="${COST_TABLE_JSON:-$OUT_DIR/qwen35_gdn_piecewise_cost_table.json}"
COST_TABLE_MD="${COST_TABLE_MD:-$OUT_DIR/qwen35_gdn_piecewise_cost_table.md}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
mkdir -p "$OUT_DIR"

spec_config="$(python3 - "$DRAFT_MODEL" "$SPEC_TOKENS" <<'PY'
import json
import sys
print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))
PY
)"

cmd=(
  vllm serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --speculative-config "$spec_config"
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
  --default-chat-template-kwargs '{"enable_thinking": false}'
  --no-async-scheduling
)
if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=( $EXTRA_SERVE_ARGS )
  cmd+=("${extra[@]}")
fi

cat <<EOF2
[start_qwen35_gdn_piecewise]
  model:              $MODEL
  draft_model:        $DRAFT_MODEL
  served_model_name:  $SERVED_MODEL_NAME
  host/port:          $HOST:$PORT
  tp:                 $TP
  max_num_seqs:       $MAX_NUM_SEQS
  max_model_len:      $MAX_MODEL_LEN
  spec_tokens:        $SPEC_TOKENS
  dcut_config:        $DCUT_CONFIG
  cost_table_json:    $COST_TABLE_JSON
  cost_table_md:      $COST_TABLE_MD
  compilation_config: {"cudagraph_mode":"PIECEWISE"}
EOF2

export VLLM_DCUT_CONFIG="$DCUT_CONFIG"
export VLLM_DCUT_COST_TABLE_OUT="$COST_TABLE_JSON"
export VLLM_DCUT_COST_TABLE_MD_OUT="$COST_TABLE_MD"
export VLLM_PLUGINS="${VLLM_PLUGINS:-dcut_adaptive_verify}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

exec "${cmd[@]}"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch vanilla DFlash and D-Cut vLLM servers, then benchmark both with
# ALLaVA-style text prompts through the OpenAI chat API.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

: "${MODEL:?Set MODEL=/path/to/Qwen3.5 target model}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL=/path/to/DFlash draft model}"
: "${DATASET:?Set DATASET=/path/to/allava.jsonl or .json}"

PORT="${PORT:-8100}"
HOST="${HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35}"
TP="${TP:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
SPEC_TOKENS="${SPEC_TOKENS:-15}"
NUM="${NUM:-256}"
WARMUP="${WARMUP:-8}"
TAIL_FRAC="${TAIL_FRAC:-0.1}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1,16,32,64}"
MAXTOK="${MAXTOK:-128}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
DCUT_CONFIG="${DCUT_CONFIG:-$SCRIPT_DIR/verify_adaptive_config.example.json}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/qwen35_dcut_results/$(date +%Y%m%d_%H%M%S)}"
DCUT_COST_TABLE_OUT="${DCUT_COST_TABLE_OUT:-$OUT_DIR/dcut_cost_table.json}"
VARIANTS="${VARIANTS:-vanilla,dcut}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cleanup] stopping vLLM server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
mkdir -p "$OUT_DIR"

make_spec_config() {
  python3 - "$DRAFT_MODEL" "$SPEC_TOKENS" <<'PY'
import json
import sys
print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
}))
PY
}

wait_health() {
  local log_file="$1"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  local url="http://${HOST}:${PORT}/health"
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[serve] server exited early; last 120 log lines:" >&2
      tail -120 "$log_file" >&2 || true
      exit 1
    fi
    if python3 - "$url" <<'PY'
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as resp:
        raise SystemExit(0 if resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "[serve] healthy: $url"
      return 0
    fi
    sleep 5
  done
  echo "[serve] timeout waiting for $url; last 120 log lines:" >&2
  tail -120 "$log_file" >&2 || true
  exit 1
}

start_server() {
  local variant="$1"
  local log_file="$OUT_DIR/server_${variant}.log"
  local spec_config
  spec_config="$(make_spec_config)"
  cleanup
  SERVER_PID=""
  local -a cmd=(
    vllm serve "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-num-seqs "$MAX_NUM_SEQS"
    --speculative-config "$spec_config"
    --default-chat-template-kwargs '{"enable_thinking": false}'
  )
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ -n "$EXTRA_SERVE_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $EXTRA_SERVE_ARGS )
    cmd+=("${extra[@]}")
  fi
  echo "[serve] variant=$variant log=$log_file"
  if [[ "$variant" == "dcut" ]]; then
    env \
      VLLM_DCUT_CONFIG="$DCUT_CONFIG" \
      VLLM_DCUT_COST_TABLE_OUT="$DCUT_COST_TABLE_OUT" \
      VLLM_PLUGINS="${VLLM_PLUGINS:-dcut_adaptive_verify}" \
      VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
      no_proxy="${no_proxy:-localhost,127.0.0.1}" \
      "${cmd[@]}" >"$log_file" 2>&1 &
  else
    env \
      VLLM_DCUT_CONFIG="" \
      VLLM_DCUT_COST_TABLE_OUT="" \
      VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
      no_proxy="${no_proxy:-localhost,127.0.0.1}" \
      "${cmd[@]}" >"$log_file" 2>&1 &
  fi
  SERVER_PID="$!"
  echo "[serve] pid=$SERVER_PID"
  wait_health "$log_file"
}

run_bench() {
  local variant="$1"
  local conc="$2"
  local tag="${variant}_c${conc}"
  local out_jsonl="$OUT_DIR/${tag}.jsonl"
  local summary_json="$OUT_DIR/${tag}.summary.json"
  echo "[bench] $tag"
  # shellcheck disable=SC2086
  python3 "$SCRIPT_DIR/bench_qwen35_dcut.py" \
    --dataset "$DATASET" \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "$SERVED_MODEL_NAME" \
    --endpoint chat \
    --num "$NUM" \
    --warmup "$WARMUP" \
    --tail-frac "$TAIL_FRAC" \
    --concurrency "$conc" \
    --max-tokens "$MAXTOK" \
    --tag "$tag" \
    --out "$out_jsonl" \
    --summary-out "$summary_json" \
    $BENCH_EXTRA_ARGS
}

print_table() {
  python3 - "$OUT_DIR" <<'PY'
import glob
import json
import os
import re
import sys

rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.summary.json"))):
    with open(path, encoding="utf-8") as f:
        rows.append(json.load(f))
if not rows:
    raise SystemExit(0)
print("\n=== summary ===")
print("tag ok/req req/s out tok/s total tok/s ttft p50 ttft p90 lat p50 lat p90")
for r in rows:
    def f(key, nd=2):
        val = r.get(key)
        if val is None:
            return "NA"
        return f"{val:.{nd}f}" if isinstance(val, float) else str(val)
    print(
        f"{r['tag']:<19} "
        f"{r.get('succeeded', 0):>3}/{r.get('requests', 0):<3} "
        f"{f('request_per_s'):>7} "
        f"{f('output_tok_per_s', 1):>10} "
        f"{f('total_tok_per_s', 1):>11} "
        f"{f('ttft_p50_s'):>8} "
        f"{f('ttft_p90_s'):>8} "
        f"{f('latency_p50_s'):>7} "
        f"{f('latency_p90_s'):>7}")
by_conc = {}
for r in rows:
    m = re.match(r"^(?P<variant>vanilla|dcut)_c(?P<conc>\d+)$", r.get("tag", ""))
    if not m:
        continue
    by_conc.setdefault(int(m.group("conc")), {})[m.group("variant")] = r
pairs = [(c, a["vanilla"], a["dcut"]) for c, a in sorted(by_conc.items())
         if "vanilla" in a and "dcut" in a]
if pairs:
    def pct(new, old):
        if new is None or old in (None, 0):
            return "NA"
        return f"{(new / old - 1.0) * 100:+.1f}%"
    print("\n=== D-Cut delta vs vanilla DFlash ===")
    print("(throughput: + is better; latency: - is better)")
    print("conc req/s out tok/s total tok/s ttft p50 lat p50")
    for conc, vanilla, dcut in pairs:
        print(
            f"{conc:<5} "
            f"{pct(dcut.get('request_per_s'), vanilla.get('request_per_s')):>7} "
            f"{pct(dcut.get('output_tok_per_s'), vanilla.get('output_tok_per_s')):>10} "
            f"{pct(dcut.get('total_tok_per_s'), vanilla.get('total_tok_per_s')):>11} "
            f"{pct(dcut.get('ttft_p50_s'), vanilla.get('ttft_p50_s')):>8} "
            f"{pct(dcut.get('latency_p50_s'), vanilla.get('latency_p50_s')):>7}")
PY
}

print_dcut_server_check() {
  local log_file="$OUT_DIR/server_dcut.log"
  if [[ ! -f "$log_file" ]]; then
    return 0
  fi
  echo
  echo "=== D-Cut server check ==="
  if ! grep -E "D-Cut adaptive|VerifyAdaptiveController|profile bs=|cost table|falling back|D-Cut:" \
      "$log_file" | tail -120; then
    echo "[warn] no D-Cut activation/profiling lines found in $log_file"
  fi
  if [[ -f "$DCUT_COST_TABLE_OUT" ]]; then
    echo "[cost-table] $DCUT_COST_TABLE_OUT"
  else
    echo "[warn] no exported D-Cut cost table found at $DCUT_COST_TABLE_OUT"
  fi
}

echo "[config] repo=$REPO_DIR"
echo "[config] model=$MODEL"
echo "[config] draft=$DRAFT_MODEL"
echo "[config] dataset=$DATASET"
echo "[config] out=$OUT_DIR"
echo "[config] dcut_cost_table_out=$DCUT_COST_TABLE_OUT"
echo "[config] variants=$VARIANTS concurrency=$CONCURRENCY_LIST num=$NUM warmup=$WARMUP tail_frac=$TAIL_FRAC"

IFS=',' read -r -a variants <<< "$VARIANTS"
IFS=',' read -r -a concs <<< "$CONCURRENCY_LIST"
for variant in "${variants[@]}"; do
  start_server "$variant"
  for conc in "${concs[@]}"; do
    run_bench "$variant" "$conc"
  done
  if [[ "$variant" == "dcut" ]]; then
    print_dcut_server_check
  fi
done
cleanup
SERVER_PID=""
print_table
echo "[done] results in $OUT_DIR"

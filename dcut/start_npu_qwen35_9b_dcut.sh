#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NPU launch script matching the requested Qwen3.5-9B D-Cut setup.
# It is intentionally runnable without editing: just `bash dcut/start_npu_qwen35_9b_dcut.sh`.

set -euo pipefail


export VLLM_PLUGINS=ascend,dcut_adaptive_verify
export VLLM_TARGET_DEVICE=ascend
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-11}"
export VLLM_DCUT_CONFIG="${VLLM_DCUT_CONFIG:-/data/c00954457/codex_vllm/vllm-ascend/dcut/verify_adaptive_config.example.json}"
export VLLM_DCUT_COST_TABLE_OUT="${VLLM_DCUT_COST_TABLE_OUT:-/data/c00954457/codex_vllm/vllm-ascend/dcut/cost_table.json}"
export VLLM_DCUT_TRIM_STATS_OUT="${VLLM_DCUT_TRIM_STATS_OUT:-/data/c00954457/codex_vllm/vllm-ascend/dcut/trim_stats.txt}"
export VLLM_DCUT_STAT_EVERY="${VLLM_DCUT_STAT_EVERY:-1}"
export VLLM_DCUT_PROFILE_FORCE_EAGER="${VLLM_DCUT_PROFILE_FORCE_EAGER:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_MODEL_PLUGIN="${VLLM_ASCEND_MODEL_PLUGIN:-vllm_ascend.patch_qwen3_5}"

DCUT_HOST="${DCUT_HOST:-127.0.0.1}"
DCUT_PORT="${DCUT_PORT:-8305}"
DCUT_PROFILE_AFTER_READY="${DCUT_PROFILE_AFTER_READY:-1}"
DCUT_READY_TIMEOUT_S="${DCUT_READY_TIMEOUT_S:-900}"
DCUT_COST_TABLE_TIMEOUT_S="${DCUT_COST_TABLE_TIMEOUT_S:-900}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "[dcut-start] VLLM_PLUGINS=${VLLM_PLUGINS}"
echo "[dcut-start] VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}"
echo "[dcut-start] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[dcut-start] VLLM_DCUT_CONFIG=${VLLM_DCUT_CONFIG}"
echo "[dcut-start] VLLM_DCUT_COST_TABLE_OUT=${VLLM_DCUT_COST_TABLE_OUT}"
echo "[dcut-start] VLLM_DCUT_TRIM_STATS_OUT=${VLLM_DCUT_TRIM_STATS_OUT}"
echo "[dcut-start] VLLM_DCUT_STAT_EVERY=${VLLM_DCUT_STAT_EVERY}"
echo "[dcut-start] VLLM_DCUT_PROFILE_FORCE_EAGER=${VLLM_DCUT_PROFILE_FORCE_EAGER}"
echo "[dcut-start] VLLM_USE_V1=${VLLM_USE_V1}"
echo "[dcut-start] VLLM_ASCEND_MODEL_PLUGIN=${VLLM_ASCEND_MODEL_PLUGIN}"
echo "[dcut-start] served_model_names=qwen35 qwen3.5-9b port=${DCUT_PORT}"
echo "[dcut-start] service-ready profiling=${DCUT_PROFILE_AFTER_READY}"
echo "[dcut-start] If cost table is missing, check for: D-Cut adaptive verify ENABLED, D-Cut cost profiling START, profile row runtime_mode=FCG/PCG, and dumped JSON cost table."

rm -f "${VLLM_DCUT_COST_TABLE_OUT}"

python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.5-9B \
  --served-model-name qwen35 qwen3.5-9b \
  --port "${DCUT_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --allowed-local-media-path /data \
  --gpu-memory-utilization 0.50 \
  -cc '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512]}' \
  --speculative-config '{"method":"dflash","model":"/data/models/Qwen3.5-9B-DFlash","num_speculative_tokens":7,"enforce_eager":true}' &
SERVER_PID=$!
trap cleanup INT TERM

if [[ "${DCUT_PROFILE_AFTER_READY}" == "1" ]]; then
  echo "[dcut-start] waiting for service health before cost-table probe..."
  deadline=$((SECONDS + DCUT_READY_TIMEOUT_S))
  until curl -fsS "http://${DCUT_HOST}:${DCUT_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}"
      exit $?
    fi
    if (( SECONDS >= deadline )); then
      echo "[dcut-start] timed out waiting for /health" >&2
      exit 1
    fi
    sleep 2
  done

  echo "[dcut-start] service is healthy; triggering D-Cut cost-table probe request..."
  curl -fsS "http://${DCUT_HOST}:${DCUT_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen35","messages":[{"role":"user","content":"D-Cut cost table warmup probe."}],"max_tokens":1,"temperature":0}' \
    >/dev/null || echo "[dcut-start] warning: probe request failed; inspect server logs" >&2

  echo "[dcut-start] waiting for cost table: ${VLLM_DCUT_COST_TABLE_OUT}"
  deadline=$((SECONDS + DCUT_COST_TABLE_TIMEOUT_S))
  until [[ -s "${VLLM_DCUT_COST_TABLE_OUT}" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}"
      exit $?
    fi
    if (( SECONDS >= deadline )); then
      echo "[dcut-start] timed out waiting for cost table; continuing with server logs for diagnosis" >&2
      break
    fi
    sleep 2
  done
  if [[ -s "${VLLM_DCUT_COST_TABLE_OUT}" ]]; then
    echo "[dcut-start] D-Cut cost table is ready: ${VLLM_DCUT_COST_TABLE_OUT}"
  fi
fi

wait "${SERVER_PID}"

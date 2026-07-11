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
echo "[dcut-start] served_model_names=qwen35 qwen3.5-9b port=8305"
echo "[dcut-start] If cost table is missing, check for: D-Cut adaptive verify ENABLED, D-Cut cost profiling START, and dumped JSON cost table."

python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.5-9B \
  --served-model-name qwen35 qwen3.5-9b \
  --port 8305 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --allowed-local-media-path /data \
  --gpu-memory-utilization 0.50 \
  -cc '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512]}' \
  --speculative-config '{"method":"dflash","model":"/data/models/Qwen3.5-9B-DFlash","num_speculative_tokens":7,"enforce_eager":true}'

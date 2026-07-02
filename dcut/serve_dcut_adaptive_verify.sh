#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-ascend}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"

# 必须包含 dcut_adaptive_verify；只写 ascend 不会加载 D-Cut。
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,dcut_adaptive_verify}"

# 使用 DCUT_*，避免 vLLM unknown env warning。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DCUT_ENABLE="${DCUT_ENABLE:-1}"
export DCUT_CONFIG="${DCUT_CONFIG:-${SCRIPT_DIR}/verify_adaptive_config.example.json}"
# 默认先保护精度：DFlash 接收率/精度确认正常后，可设置 DCUT_ACCURACY_SAFE_MODE=0。
export DCUT_ACCURACY_SAFE_MODE="${DCUT_ACCURACY_SAFE_MODE:-1}"

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/data/models/Qwen3.5-9B/}"
DFLASH_DRAFT_PATH="${DFLASH_DRAFT_PATH:-/data/models/Qwen3.5-9B-DFlash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
VLLM_PORT="${VLLM_PORT:-8304}"

SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"dflash\",\"model\":\"${DFLASH_DRAFT_PATH}\",\"num_speculative_tokens\":15}}"

vllm serve "${TARGET_MODEL_PATH}" \
  --served-model-name="${SERVED_MODEL_NAME}" \
  --port="${VLLM_PORT}" \
  --tensor-parallel-size=1 \
  --gpu-memory-utilization=0.8 \
  --block-size=128 \
  --max-num-seqs=256 \
  --no-async-scheduling \
  --max-model-len=32768 \
  --allowed-local-media-path /data \
  --max-num-batched-tokens=32768 \
  --trust-remote-code \
  --speculative-config "${SPEC_CONFIG}" \
  --enforce-eager \
  --additional-config '{"enable_reduce_sample": true}'

# 如需 full-decode graph，可按需增加：
# --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

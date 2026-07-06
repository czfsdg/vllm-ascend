#!/usr/bin/env bash
set -euo pipefail

# Ascend runtime knobs.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-ascend}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"

# Important: D-Cut is a standalone vLLM general plugin.  Keep both plugins here;
# using only "ascend" will not load the D-Cut monkey patches.
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,dcut_adaptive_verify}"

# Prefer DCUT_* to avoid vLLM unknown-environment warnings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DCUT_ENABLE="${DCUT_ENABLE:-1}"
export DCUT_CONFIG="${DCUT_CONFIG:-${SCRIPT_DIR}/verify_adaptive_config.example.json}"

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/data/models/Qwen3.5-9B}"
DFLASH_DRAFT_PATH="${DFLASH_DRAFT_PATH:-/data/models/Qwen3.5-9B-DFlash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
PORT="${PORT:-8304}"
TP_SIZE="${TP_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/data}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"

SPEC_CONFIG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s}' \
  "${DFLASH_DRAFT_PATH}" "${NUM_SPECULATIVE_TOKENS}")
ADDITIONAL_CONFIG='{"enable_reduce_sample": true}'

cat <<EOF
[D-Cut] launching vLLM with:
  TARGET_MODEL_PATH=${TARGET_MODEL_PATH}
  DFLASH_DRAFT_PATH=${DFLASH_DRAFT_PATH}
  DCUT_ENABLE=${DCUT_ENABLE}
  DCUT_CONFIG=${DCUT_CONFIG}
  VLLM_PLUGINS=${VLLM_PLUGINS}
  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}
EOF

vllm serve "${TARGET_MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --block-size "${BLOCK_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --no-async-scheduling \
  --max-model-len "${MAX_MODEL_LEN}" \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --trust-remote-code \
  --speculative-config "${SPEC_CONFIG}" \
  --enforce-eager \
  --additional-config "${ADDITIONAL_CONFIG}"

# If you want to test compiled decode instead of eager, remove --enforce-eager
# above and add e.g.:
#   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

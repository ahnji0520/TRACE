#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUPPLEMENTARY_ROOT="${SUPPLEMENTARY_ROOT:-$SCRIPT_DIR}"
while [ ! -d "${SUPPLEMENTARY_ROOT}/data" ] || [ ! -d "${SUPPLEMENTARY_ROOT}/codes" ]; do
  NEXT_ROOT=$(dirname "${SUPPLEMENTARY_ROOT}")
  if [ "$NEXT_ROOT" = "${SUPPLEMENTARY_ROOT}" ]; then
    echo "[ERROR] Could not locate supplementary-materials root from $SCRIPT_DIR" >&2
    exit 1
  fi
  SUPPLEMENTARY_ROOT="$NEXT_ROOT"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/response_generator/without_persona/qwen2.5-7b-instruct/2026-05-23-01-05-16/checkpoint-16400}"
INPUT_FILE="${INPUT_FILE:-${SUPPLEMENTARY_ROOT}/data/evocrs/response_generator/target_response/without_persona/response_generation_test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/without_persona/qwen2.5-7b-instruct/checkpoint-16400}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
NUM_BEAMS="${NUM_BEAMS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RESUME="${RESUME:-false}"

IFS=, read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_IDS[@]}}"

mkdir -p "${OUTPUT_DIR}" logs

COMMON_ARGS=(
  --base_model "${BACKBONE_PATH}"
  --lora_path "${LORA_PATH}"
  --input_file "${INPUT_FILE}"
  --num_shards "${WORLD_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --max_input_length "${MAX_INPUT_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --num_beams "${NUM_BEAMS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
)

if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

for ((SHARD_IDX = 0; SHARD_IDX < WORLD_SIZE; SHARD_IDX++)); do
  GPU_ID="${GPU_IDS[$((SHARD_IDX % ${#GPU_IDS[@]}))]}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python response_generator_inference.py \
    "${COMMON_ARGS[@]}" \
    --shard_idx "${SHARD_IDX}" \
    --output_file "${OUTPUT_DIR}/generated_responses.shard${SHARD_IDX}.jsonl" \
    > "logs/response_generator_without_persona_qwen25_shard${SHARD_IDX}.log" 2>&1 &
done

wait

cat "${OUTPUT_DIR}"/generated_responses.shard*.jsonl \
  > "${OUTPUT_DIR}/generated_responses.all.jsonl"

echo "Wrote ${OUTPUT_DIR}/generated_responses.all.jsonl"

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
PERSONA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PERSONA_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"

BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Llama-3.1-8B-Instruct}"
LORA_PATH="${LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/behavioral_persona_extract/llama-3.1-8b-instruct/2026-03-16-10-17-36/checkpoint-16000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSONA_DIR}/output/llama-3.1-8B/bp}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_BEAMS="${NUM_BEAMS:-1}"
RESUME="${RESUME:-false}"

IFS=, read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_IDS[@]}}"

mkdir -p logs/bp "${OUTPUT_DIR}"

COMMON_ARGS=(
  --backbone llama3.1
  --backbone_path "${BACKBONE_PATH}"
  --lora_path "${LORA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_shards "${WORLD_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --max_retries "${MAX_RETRIES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --num_beams "${NUM_BEAMS}"
)

if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

for ((SHARD_IDX = 0; SHARD_IDX < WORLD_SIZE; SHARD_IDX++)); do
  GPU_ID="${GPU_IDS[$((SHARD_IDX % ${#GPU_IDS[@]}))]}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PERSONA_DIR}/src/extract_bp.py" \
    "${COMMON_ARGS[@]}" \
    --shard_idx "${SHARD_IDX}" \
    > "logs/bp/log_extract_bp_llama_shard${SHARD_IDX}.log" 2>&1 &
done

wait

cat "${OUTPUT_DIR}"/extracted_behavior_only.shard*.jsonl \
  > "${OUTPUT_DIR}"/extracted_behavior_only.llama.all.jsonl

python - "${OUTPUT_DIR}" <<PY
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rows = []
for path in sorted(output_dir.glob("extracted_behavior_only.shard*.jsonl")):
    with path.open("r", encoding="utf-8") as f:
        rows.extend(json.loads(line) for line in f if line.strip())
ids = [row["id"] for row in rows]
print(f"[INFO] merged rows: {len(rows)}")
print(f"[INFO] unique ids: {len(set(ids))}")
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate ids detected in merged llama behavior output")
PY

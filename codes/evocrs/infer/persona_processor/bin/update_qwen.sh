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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

BACKBONE="${BACKBONE:-qwen2.5-7b-instruct}"
BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/persona_update/qwen2.5-7b-instruct/2026-05-19-20-16-38/checkpoint-10500}"

DEFAULT_BEHAVIOR_FINAL="${PERSONA_DIR}/output/qwen2.5-7b-instruct/bp/extracted_behavior_only.qwen.final.jsonl"
DEFAULT_BEHAVIOR_ALL="${PERSONA_DIR}/output/qwen2.5-7b-instruct/bp/extracted_behavior_only.qwen.all.jsonl"
if [[ -z "${BEHAVIOR_PATH:-}" ]]; then
  if [[ -s "${DEFAULT_BEHAVIOR_FINAL}" ]]; then
    BEHAVIOR_PATH="${DEFAULT_BEHAVIOR_FINAL}"
  else
    BEHAVIOR_PATH="${DEFAULT_BEHAVIOR_ALL}"
  fi
fi

DEFAULT_DIALOGUE_FINAL="${PERSONA_DIR}/output/qwen2.5-7b-instruct/dp/extracted_dialogue_only.qwen.final.jsonl"
DEFAULT_DIALOGUE_ALL="${PERSONA_DIR}/output/qwen2.5-7b-instruct/dp/extracted_dialogue_only.qwen.all.jsonl"
if [[ -z "${DIALOGUE_PATH:-}" ]]; then
  if [[ -s "${DEFAULT_DIALOGUE_FINAL}" ]]; then
    DIALOGUE_PATH="${DEFAULT_DIALOGUE_FINAL}"
  else
    DIALOGUE_PATH="${DEFAULT_DIALOGUE_ALL}"
  fi
fi

EXTRACTED_DATA_PATH="${EXTRACTED_DATA_PATH:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/final/extracted_personas_only.qwen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/upd}"
UPDATE_DATA_PATH="${UPDATE_DATA_PATH:-${SUPPLEMENTARY_ROOT}/data/evocrs/persona_processor/persona_update/persona_update_test.json}"
FILTER_DATA_PATH="${FILTER_DATA_PATH:-${SUPPLEMENTARY_ROOT}/data/evocrs/persona_processor/filter_reficr.json}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
DET_RETRIES="${DET_RETRIES:-1}"
SAMPLE_RETRIES="${SAMPLE_RETRIES:-1}"
DET_NUM_BEAMS="${DET_NUM_BEAMS:-1}"
THINKING="${THINKING:-false}"
RESUME="${RESUME:-true}"
RESUME_FROM_ALL="${RESUME_FROM_ALL:-true}"
REBUILD_RESUME_SHARDS="${REBUILD_RESUME_SHARDS:-false}"
CLEAN_RESUME_SHARDS="${CLEAN_RESUME_SHARDS:-true}"
MERGED_UPDATE_NAME="${MERGED_UPDATE_NAME:-updated_personas_logs.qwen.all.jsonl}"

IFS=, read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_IDS[@]}}"

mkdir -p logs/update "${OUTPUT_DIR}" "$(dirname "${EXTRACTED_DATA_PATH}")"
export BEHAVIOR_PATH DIALOGUE_PATH EXTRACTED_DATA_PATH

python - <<'PYMERGE'
import json
import os
from pathlib import Path

behavior_path = Path(os.environ["BEHAVIOR_PATH"])
dialogue_path = Path(os.environ["DIALOGUE_PATH"])
merged_path = Path(os.environ["EXTRACTED_DATA_PATH"])

if not behavior_path.exists() or behavior_path.stat().st_size == 0:
    raise SystemExit(f"Missing behavior extraction output: {behavior_path}")
if not dialogue_path.exists() or dialogue_path.stat().st_size == 0:
    raise SystemExit(f"Missing dialogue extraction output: {dialogue_path}")

def load_jsonl_as_map(path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data[row["id"]] = row
    return data

behavior_map = load_jsonl_as_map(behavior_path)
dialogue_map = load_jsonl_as_map(dialogue_path)
common_ids = sorted(set(behavior_map) & set(dialogue_map))

merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", encoding="utf-8") as out:
    for sample_id in common_ids:
        b = behavior_map[sample_id]
        d = dialogue_map[sample_id]
        out.write(json.dumps({
            "id": sample_id,
            "user_id": b["user_id"],
            "session_num": b["session_num"],
            "extracted_behavior": b["extracted_behavior"],
            "extracted_dialogue": d["extracted_dialogue"],
        }, ensure_ascii=False) + "\n")

seed_count = sum(1 for sample_id in common_ids if sample_id.endswith("_1"))
print(f"Behavior rows: {len(behavior_map)}")
print(f"Dialogue rows: {len(dialogue_map)}")
print(f"Merged rows: {len(common_ids)}")
print(f"Merged session-1 rows: {seed_count}")
print(f"Saved merged personas to: {merged_path}")
PYMERGE

COMMON_ARGS=(
  --backbone "${BACKBONE}"
  --backbone_path "${BACKBONE_PATH}"
  --lora_path "${LORA_PATH}"
  --extracted_data_path "${EXTRACTED_DATA_PATH}"
  --update_data_path "${UPDATE_DATA_PATH}"
  --filter_data_path "${FILTER_DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_gpus 1
  --num_shards "${WORLD_SIZE}"
  --batch_per_device "${BATCH_PER_DEVICE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --det_retries "${DET_RETRIES}"
  --sample_retries "${SAMPLE_RETRIES}"
  --det_num_beams "${DET_NUM_BEAMS}"
)

if [[ "${THINKING}" == "true" || "${THINKING}" == "1" ]]; then
  COMMON_ARGS+=(--thinking)
fi
if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  if [[ "${REBUILD_RESUME_SHARDS}" == "true" || "${REBUILD_RESUME_SHARDS}" == "1" ]]; then
    python "${PERSONA_DIR}/tools/rebuild_update_resume_shards.py" \
      --output-dir "${OUTPUT_DIR}" \
      --filter-data-path "${FILTER_DATA_PATH}" \
      --num-shards "${WORLD_SIZE}" \
      --merged-name "${MERGED_UPDATE_NAME}"
  elif [[ ("${RESUME_FROM_ALL}" == "true" || "${RESUME_FROM_ALL}" == "1") && -s "${OUTPUT_DIR}/${MERGED_UPDATE_NAME}" ]]; then
    python "${PERSONA_DIR}/tools/rebuild_update_resume_shards.py" \
      --output-dir "${OUTPUT_DIR}" \
      --filter-data-path "${FILTER_DATA_PATH}" \
      --num-shards "${WORLD_SIZE}" \
      --merged-name "${MERGED_UPDATE_NAME}" \
      --from-merged \
      --no-backup
  fi
fi

for ((SHARD_IDX = 0; SHARD_IDX < WORLD_SIZE; SHARD_IDX++)); do
  GPU_ID="${GPU_IDS[$((SHARD_IDX % ${#GPU_IDS[@]}))]}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PERSONA_DIR}/src/update.py" \
    "${COMMON_ARGS[@]}" \
    --shard_idx "${SHARD_IDX}" \
    > "logs/update/log_update_qwen_shard${SHARD_IDX}.log" 2>&1 &
done

wait

cat "${OUTPUT_DIR}"/updated_personas_logs.shard*.jsonl \
  > "${OUTPUT_DIR}/${MERGED_UPDATE_NAME}"

python - "${OUTPUT_DIR}" <<'PYCHECK'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rows = []
failed = []
for path in sorted(output_dir.glob("updated_personas_logs.shard*.jsonl")):
    with path.open("r", encoding="utf-8") as f:
        rows.extend(json.loads(line) for line in f if line.strip())
for path in sorted(output_dir.glob("failed_update_inputs.shard*.json")):
    if path.exists():
        failed.extend(json.loads(path.read_text(encoding="utf-8")))
ids = [row["id"] for row in rows]
print(f"[INFO] merged update rows: {len(rows)}")
print(f"[INFO] unique update ids: {len(set(ids))}")
print(f"[INFO] failed update records: {len(failed)}")
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate ids detected in merged qwen update output")
PYCHECK

if [[ "${CLEAN_RESUME_SHARDS}" == "true" || "${CLEAN_RESUME_SHARDS}" == "1" ]]; then
  python "${PERSONA_DIR}/tools/rebuild_update_resume_shards.py" \
    --output-dir "${OUTPUT_DIR}" \
    --filter-data-path "${FILTER_DATA_PATH}" \
    --num-shards "${WORLD_SIZE}" \
    --merged-name "${MERGED_UPDATE_NAME}" \
    --from-merged \
    --cleanup-shards \
    --no-backup
fi

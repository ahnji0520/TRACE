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

MODEL_FAMILY="${MODEL_FAMILY:-llama}"

if [[ "${MODEL_FAMILY}" == "llama" ]]; then
  BACKBONE="${BACKBONE:-llama3.1}"
  BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Llama-3.1-8B-Instruct}"
  LORA_PATH="${LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/dialogue_persona_extract/llama-3.1-8b-instruct/2026-03-19-11-52-26/checkpoint-6800}"
  OUTPUT_DIR="${OUTPUT_DIR:-${PERSONA_DIR}/output/llama-3.1-8B/dp}"
  MERGED_NAME="${MERGED_NAME:-extracted_dialogue_only.llama.all.jsonl}"
  FINAL_NAME="${FINAL_NAME:-extracted_dialogue_only.llama.final.jsonl}"
  LOG_PREFIX="${LOG_PREFIX:-llama}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
elif [[ "${MODEL_FAMILY}" == "qwen" ]]; then
  BACKBONE="${BACKBONE:-qwen2.5-7b-instruct}"
  BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Qwen2.5-7B-Instruct}"
  LORA_PATH="${LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/dialogue_persona_extract/qwen2.5-7b-instruct/2026-05-19-20-16-36/checkpoint-13900}"
  OUTPUT_DIR="${OUTPUT_DIR:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/dp}"
  MERGED_NAME="${MERGED_NAME:-extracted_dialogue_only.qwen.all.jsonl}"
  FINAL_NAME="${FINAL_NAME:-extracted_dialogue_only.qwen.final.jsonl}"
  LOG_PREFIX="${LOG_PREFIX:-qwen}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
else
  echo "MODEL_FAMILY must be llama or qwen; got ${MODEL_FAMILY}" >&2
  exit 2
fi

mkdir -p logs/dp "${OUTPUT_DIR}"

RETRY_ROUND="${RETRY_ROUND:-auto}"
if [[ "${RETRY_ROUND}" == "auto" ]]; then
  RETRY_ROUND="$(python - "${OUTPUT_DIR}" "${MODEL_FAMILY}" <<'PYAUTO'
import re
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
model = sys.argv[2]
pat = re.compile(rf"^extracted_dialogue_only\.{re.escape(model)}\.retry(\d+)\.jsonl$")
rounds = []
for path in out_dir.glob(f"extracted_dialogue_only.{model}.retry*.jsonl"):
    m = pat.match(path.name)
    if m:
        rounds.append(int(m.group(1)))
print((max(rounds) + 1) if rounds else 1)
PYAUTO
)"
fi

if ! [[ "${RETRY_ROUND}" =~ ^[0-9]+$ ]] || [[ "${RETRY_ROUND}" -lt 1 ]]; then
  echo "RETRY_ROUND must be a positive integer or auto; got ${RETRY_ROUND}" >&2
  exit 2
fi

ROUND_TAG="retry${RETRY_ROUND}"
PREV_ROUND=$((RETRY_ROUND - 1))
RETRY_PATH="${OUTPUT_DIR}/extracted_dialogue_only.${MODEL_FAMILY}.${ROUND_TAG}.jsonl"
FAILED_RETRY_PATH="${FAILED_RETRY_PATH:-${OUTPUT_DIR}/failed_dialogue_inputs.${MODEL_FAMILY}.${ROUND_TAG}.json}"
INPUT_OVERRIDE="${INPUT_OVERRIDE:-${OUTPUT_DIR}/dialogue_${ROUND_TAG}_list.json}"
WORK_ROOT="${WORK_ROOT:-${PERSONA_DIR}/.retry_work/dp_${MODEL_FAMILY}/${ROUND_TAG}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_BEAMS="${NUM_BEAMS:-1}"
DO_SAMPLE="${DO_SAMPLE:-true}"
RESUME="${RESUME:-false}"
THINKING="${THINKING:-false}"

if [[ -z "${FAILED_GLOB:-}" ]]; then
  if [[ "${RETRY_ROUND}" -eq 1 ]]; then
    FAILED_GLOB="${OUTPUT_DIR}/failed_dialogue_inputs.shard*.json"
  else
    FAILED_GLOB="${OUTPUT_DIR}/failed_dialogue_inputs.${MODEL_FAMILY}.retry${PREV_ROUND}.json"
  fi
fi

IFS=, read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_IDS[@]}}"
mkdir -p "${WORK_ROOT}"

python - "${INPUT_OVERRIDE}" ${FAILED_GLOB} <<'PYMAKE'
import json
import sys
from pathlib import Path
out_path = Path(sys.argv[1])
failed_paths = [Path(p) for p in sys.argv[2:]]
items = []
seen = set()
for path in failed_paths:
    if not path.exists():
        continue
    for item in json.loads(path.read_text(encoding='utf-8')):
        sample_id = item.get('id')
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            items.append(item)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'[INFO] retry input records: {len(items)} -> {out_path}')
PYMAKE

if [[ ! -s "${INPUT_OVERRIDE}" ]] || [[ "$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${INPUT_OVERRIDE}")" == "0" ]]; then
  echo "[INFO] no failed dialogue inputs to retry for ${ROUND_TAG}"
  python "${PERSONA_DIR}/tools/merge_extraction_retries.py" --output-dir "${OUTPUT_DIR}" --kind dialogue --model-family "${MODEL_FAMILY}" --base-name "${MERGED_NAME}" --final-name "${FINAL_NAME}"
  python "${PERSONA_DIR}/tools/cleanup_empty_failures.py" "${OUTPUT_DIR}" "${WORK_ROOT}"
  exit 0
fi

COMMON_ARGS=(
  --backbone "${BACKBONE}"
  --backbone_path "${BACKBONE_PATH}"
  --lora_path "${LORA_PATH}"
  --input_override "${INPUT_OVERRIDE}"
  --num_shards "${WORLD_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --max_retries "${MAX_RETRIES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --num_beams "${NUM_BEAMS}"
)

if [[ "${DO_SAMPLE}" == "true" || "${DO_SAMPLE}" == "1" ]]; then
  COMMON_ARGS+=(--do_sample)
fi
if [[ "${THINKING}" == "true" || "${THINKING}" == "1" ]]; then
  COMMON_ARGS+=(--thinking)
fi
if [[ "${RESUME}" == "true" || "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

for ((SHARD_IDX = 0; SHARD_IDX < WORLD_SIZE; SHARD_IDX++)); do
  GPU_ID="${GPU_IDS[$((SHARD_IDX % ${#GPU_IDS[@]}))]}"
  SHARD_WORK_DIR="${WORK_ROOT}/shard${SHARD_IDX}"
  rm -rf "${SHARD_WORK_DIR}/output"
  mkdir -p "${SHARD_WORK_DIR}/output"
  (
    cd "${SHARD_WORK_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PERSONA_DIR}/src/extract_dp.py" \
      "${COMMON_ARGS[@]}" \
      --output_dir "${SHARD_WORK_DIR}/output" \
      --shard_idx "${SHARD_IDX}"
  ) > "logs/dp/log_extract_dp_${LOG_PREFIX}_${ROUND_TAG}_shard${SHARD_IDX}.log" 2>&1 &
done

wait

shopt -s nullglob
retry_parts=("${WORK_ROOT}"/shard*/output/extracted_dialogue_only.retry.jsonl)
if [[ "${#retry_parts[@]}" -gt 0 ]]; then
  cat "${retry_parts[@]}" > "${RETRY_PATH}"
else
  : > "${RETRY_PATH}"
fi
shopt -u nullglob

python - "${WORK_ROOT}" "${FAILED_RETRY_PATH}" <<'PYFAILED'
import json
import sys
from pathlib import Path
work_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
failed_inputs = []
seen = set()
for path in sorted(work_root.glob('shard*/output/failed_dialogue_inputs.retry.json')):
    if not path.exists():
        continue
    for item in json.loads(path.read_text(encoding='utf-8')):
        sample_id = item.get('id')
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            failed_inputs.append(item)
if failed_inputs:
    output_path.write_text(json.dumps(failed_inputs, ensure_ascii=False, indent=2), encoding='utf-8')
else:
    if output_path.exists():
        output_path.unlink()
print(f'[INFO] still failed after retry: {len(failed_inputs)} -> {output_path}')
PYFAILED

python "${PERSONA_DIR}/tools/merge_extraction_retries.py" --output-dir "${OUTPUT_DIR}" --kind dialogue --model-family "${MODEL_FAMILY}" --base-name "${MERGED_NAME}" --final-name "${FINAL_NAME}"
python "${PERSONA_DIR}/tools/cleanup_empty_failures.py" "${OUTPUT_DIR}" "${WORK_ROOT}"

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


MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
BACKBONE="${BACKBONE:-qwen2.5-7b-instruct}"
BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Qwen2.5-7B-Instruct}"
DP_LORA_PATH="${DP_LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/dialogue_persona_extract/qwen2.5-7b-instruct/2026-05-19-20-16-36/checkpoint-13900}"
BP_LORA_PATH="${BP_LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/behavioral_persona_extract/qwen2.5-7b-instruct/2026-05-19-20-16-37/checkpoint-11200}"
UPDATE_LORA_PATH="${UPDATE_LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/persona_update/qwen2.5-7b-instruct/2026-05-19-20-16-38/checkpoint-10400}"

MAX_EXTRACT_RETRY_ROUNDS="${MAX_EXTRACT_RETRY_ROUNDS:-2}"
MAX_UPDATE_CYCLES="${MAX_UPDATE_CYCLES:-4}"
RUN_DP="${RUN_DP:-true}"
RUN_BP="${RUN_BP:-true}"
RUN_UPDATE="${RUN_UPDATE:-true}"
ALLOW_EXTRACT_PENDING_FAILURES="${ALLOW_EXTRACT_PENDING_FAILURES:-false}"

BP_DIR="${BP_DIR:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/bp}"
DP_DIR="${DP_DIR:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/dp}"
UPD_DIR="${UPD_DIR:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/upd}"
FINAL_DIR="${FINAL_DIR:-${PERSONA_DIR}/output/qwen2.5-7b-instruct/final}"
UPDATE_CUDA_VISIBLE_DEVICES="${UPDATE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
IFS=, read -ra UPDATE_GPU_IDS <<< "${UPDATE_CUDA_VISIBLE_DEVICES}"
UPDATE_WORLD_SIZE="${UPDATE_WORLD_SIZE:-${WORLD_SIZE:-${#UPDATE_GPU_IDS[@]}}}"

count_json_items() {
  local pattern="$1"
  python - "${pattern}" <<'PYCOUNT'
import glob
import json
import sys
count = 0
for name in glob.glob(sys.argv[1]):
    try:
        with open(name, encoding='utf-8') as f:
            count += len(json.load(f))
    except FileNotFoundError:
        pass
print(count)
PYCOUNT
}

count_extract_pending() {
  local output_dir="$1"
  local kind="$2"
  python - "${output_dir}" "${kind}" "${MODEL_FAMILY}" <<'PYPENDING'
import glob
import json
import re
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
kind = sys.argv[2]
model = sys.argv[3]
out_pat = re.compile(rf'^extracted_{re.escape(kind)}_only\.{re.escape(model)}\.retry(\d+)\.jsonl$')
rounds = []
for path in out_dir.glob(f'extracted_{kind}_only.{model}.retry*.jsonl'):
    m = out_pat.match(path.name)
    if m:
        rounds.append(int(m.group(1)))
if rounds:
    latest = max(rounds)
    failed_path = out_dir / f'failed_{kind}_inputs.{model}.retry{latest}.json'
    if failed_path.exists():
        print(len(json.loads(failed_path.read_text(encoding='utf-8'))))
    else:
        print(0)
    raise SystemExit
count = 0
for name in glob.glob(str(out_dir / f'failed_{kind}_inputs.shard*.json')):
    with open(name, encoding='utf-8') as f:
        count += len(json.load(f))
print(count)
PYPENDING
}

merge_behavior_final() {
  local extra_args=()
  if [[ "${ALLOW_EXTRACT_PENDING_FAILURES}" == "true" || "${ALLOW_EXTRACT_PENDING_FAILURES}" == "1" ]]; then
    extra_args+=(--allow-pending-failures)
  fi
  python "${PERSONA_DIR}/tools/merge_extraction_retries.py" \
    --output-dir "${BP_DIR}" \
    --kind behavior \
    --model-family "${MODEL_FAMILY}" \
    --base-name "extracted_behavior_only.qwen.all.jsonl" \
    --final-name "extracted_behavior_only.qwen.final.jsonl" \
    "${extra_args[@]}"
}

merge_dialogue_final() {
  local extra_args=()
  if [[ "${ALLOW_EXTRACT_PENDING_FAILURES}" == "true" || "${ALLOW_EXTRACT_PENDING_FAILURES}" == "1" ]]; then
    extra_args+=(--allow-pending-failures)
  fi
  python "${PERSONA_DIR}/tools/merge_extraction_retries.py" \
    --output-dir "${DP_DIR}" \
    --kind dialogue \
    --model-family "${MODEL_FAMILY}" \
    --base-name "extracted_dialogue_only.qwen.all.jsonl" \
    --final-name "extracted_dialogue_only.qwen.final.jsonl" \
    "${extra_args[@]}"
}

if [[ "${RUN_DP}" == "true" || "${RUN_DP}" == "1" ]]; then
  echo "[PIPELINE] 1/3 qwen2.5 dialogue persona extraction"
  BACKBONE="${BACKBONE}" BACKBONE_PATH="${BACKBONE_PATH}" LORA_PATH="${DP_LORA_PATH}" \
  OUTPUT_DIR="${DP_DIR}" RESUME="${DP_RESUME:-false}" "${SCRIPT_DIR}/extract_dp_qwen.sh"
  for ((round = 1; round <= MAX_EXTRACT_RETRY_ROUNDS; round++)); do
    failed_count="$(count_extract_pending "${DP_DIR}" dialogue)"
    [[ "${failed_count}" == "0" ]] && break
    echo "[PIPELINE] dialogue retry round ${round} | failed=${failed_count}"
    OUTPUT_DIR="${DP_DIR}" MODEL_FAMILY="${MODEL_FAMILY}" BACKBONE="${BACKBONE}" BACKBONE_PATH="${BACKBONE_PATH}" \
    LORA_PATH="${DP_LORA_PATH}" RETRY_ROUND="${round}" "${SCRIPT_DIR}/retry_extract_dp.sh"
  done
  merge_dialogue_final
fi

if [[ "${RUN_BP}" == "true" || "${RUN_BP}" == "1" ]]; then
  echo "[PIPELINE] 2/3 qwen2.5 behavioral persona extraction"
  BACKBONE="${BACKBONE}" BACKBONE_PATH="${BACKBONE_PATH}" LORA_PATH="${BP_LORA_PATH}" \
  OUTPUT_DIR="${BP_DIR}" RESUME="${BP_RESUME:-false}" "${SCRIPT_DIR}/extract_bp_qwen.sh"
  for ((round = 1; round <= MAX_EXTRACT_RETRY_ROUNDS; round++)); do
    failed_count="$(count_extract_pending "${BP_DIR}" behavior)"
    [[ "${failed_count}" == "0" ]] && break
    echo "[PIPELINE] behavior retry round ${round} | failed=${failed_count}"
    OUTPUT_DIR="${BP_DIR}" MODEL_FAMILY="${MODEL_FAMILY}" BACKBONE="${BACKBONE}" BACKBONE_PATH="${BACKBONE_PATH}" \
    LORA_PATH="${BP_LORA_PATH}" RETRY_ROUND="${round}" "${SCRIPT_DIR}/retry_extract_bp.sh"
  done
  merge_behavior_final
fi

if [[ "${RUN_UPDATE}" == "true" || "${RUN_UPDATE}" == "1" ]]; then
  echo "[PIPELINE] 3/3 qwen2.5 persona update with retry/apply/resume loop"
  for ((cycle = 1; cycle <= MAX_UPDATE_CYCLES; cycle++)); do
    echo "[PIPELINE] update cycle ${cycle}/${MAX_UPDATE_CYCLES}"
    CUDA_VISIBLE_DEVICES="${UPDATE_CUDA_VISIBLE_DEVICES}" \
    WORLD_SIZE="${UPDATE_WORLD_SIZE}" \
    BACKBONE="${BACKBONE}" \
    BACKBONE_PATH="${BACKBONE_PATH}" \
    LORA_PATH="${UPDATE_LORA_PATH}" \
    OUTPUT_DIR="${UPD_DIR}" \
    EXTRACTED_DATA_PATH="${FINAL_DIR}/extracted_personas_only.qwen.jsonl" \
    BEHAVIOR_PATH="${BP_DIR}/extracted_behavior_only.qwen.final.jsonl" \
    DIALOGUE_PATH="${DP_DIR}/extracted_dialogue_only.qwen.final.jsonl" \
    RESUME=true \
    "${SCRIPT_DIR}/update_qwen.sh"

    failed_count="$(count_json_items "${UPD_DIR}/failed_update_inputs.shard*.json")"
    if [[ "${failed_count}" == "0" ]]; then
      echo "[PIPELINE] update has no shard failures"
      break
    fi

    echo "[PIPELINE] update retry cycle ${cycle} | failed=${failed_count}"
    CUDA_VISIBLE_DEVICES="${UPDATE_CUDA_VISIBLE_DEVICES}" \
    BACKBONE="${BACKBONE}" \
    BACKBONE_PATH="${BACKBONE_PATH}" \
    LORA_PATH="${UPDATE_LORA_PATH}" \
    OUTPUT_DIR="${UPD_DIR}" \
    RETRY_ROUND="auto" \
    "${SCRIPT_DIR}/update_qwen_retry.sh"
    python "${PERSONA_DIR}/tools/rebuild_update_resume_shards.py" \
      --output-dir "${UPD_DIR}" \
      --filter-data-path "${FILTER_DATA_PATH:-${SUPPLEMENTARY_ROOT}/data/evocrs/persona_processor/filter_reficr.json}" \
      --num-shards "${UPDATE_WORLD_SIZE}" \
      --merged-name "updated_personas_logs.qwen.all.jsonl" \
      --from-merged \
      --no-backup
  done
fi

python "${PERSONA_DIR}/tools/cleanup_empty_failures.py" "${BP_DIR}" "${DP_DIR}" "${UPD_DIR}"
echo "[PIPELINE] done"

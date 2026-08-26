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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

BACKBONE_PATH="${BACKBONE_PATH:-${HF_MODEL_DIR}/Llama-3.1-8B-Instruct}"
LORA_PATH="${LORA_PATH:-${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/persona_update/llama-3.1-8b-instruct/2026-03-14-22-29-26/checkpoint-11000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSONA_DIR}/output/llama-3.1-8B/upd}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
DET_RETRIES="${DET_RETRIES:-0}"
SAMPLE_RETRIES="${SAMPLE_RETRIES:-3}"
DET_NUM_BEAMS="${DET_NUM_BEAMS:-1}"

mkdir -p logs/update "${OUTPUT_DIR}"

RETRY_ROUND="${RETRY_ROUND:-auto}"
if [[ "${RETRY_ROUND}" == "auto" ]]; then
  RETRY_ROUND="$(python - "${OUTPUT_DIR}" <<'PYAUTO'
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
pat = re.compile(r"^updated_personas_logs\.llama\.retry(\d+)\.jsonl$")
rounds = []
for path in out_dir.glob("updated_personas_logs.llama.retry*.jsonl"):
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
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/updated_personas_logs.llama.${ROUND_TAG}.jsonl}"
LOG_PATH="${LOG_PATH:-${SCRIPT_DIR}/logs/update/log_update_llama_${ROUND_TAG}.log}"

if [[ -z "${FAILED_INPUT_PATH:-}" ]]; then
  if [[ "${RETRY_ROUND}" -eq 1 ]]; then
    FAILED_GLOB="${FAILED_GLOB:-${OUTPUT_DIR}/failed_update_inputs.shard*.json}"
  else
    FAILED_INPUT_PATH="${OUTPUT_DIR}/updated_personas_logs.llama.retry${PREV_ROUND}.failed.json"
  fi
fi

if [[ -z "${FAILED_INPUT_PATH:-}" ]]; then
  FAILED_INPUT_PATH="${OUTPUT_DIR}/failed_update_inputs.llama.${ROUND_TAG}.json"
  python - "${FAILED_INPUT_PATH}" ${FAILED_GLOB} <<'PYMAKE'
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
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        sample_id = item.get("id")
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            items.append(item)
out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] retry input records: {len(items)} -> {out_path}")
PYMAKE
fi

if [[ ! -s "${FAILED_INPUT_PATH}" ]]; then
  echo "[INFO] no failed update inputs found: ${FAILED_INPUT_PATH}"
  exit 0
fi

FAILED_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${FAILED_INPUT_PATH}")"
if [[ "${FAILED_COUNT}" == "0" ]]; then
  echo "[INFO] no failed update inputs to retry for ${ROUND_TAG}: ${FAILED_INPUT_PATH}"
  exit 0
fi

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"

COMMON_ARGS=(
  --backbone llama3.1
  --backbone_path "${BACKBONE_PATH}"
  --lora_path "${LORA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_gpus "${NUM_GPUS}"
  --batch_per_device "${BATCH_PER_DEVICE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --det_retries "${DET_RETRIES}"
  --sample_retries "${SAMPLE_RETRIES}"
  --det_num_beams "${DET_NUM_BEAMS}"
  --failed_input_path "${FAILED_INPUT_PATH}"
  --output_path "${OUTPUT_PATH}"
)

echo "[INFO] round: ${ROUND_TAG}"
echo "[INFO] retry input: ${FAILED_INPUT_PATH} (${FAILED_COUNT} records)"
echo "[INFO] retry output: ${OUTPUT_PATH}"
echo "[INFO] retry failures: ${OUTPUT_PATH%.jsonl}.failed.json"
echo "[INFO] retry log: ${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${PERSONA_DIR}/src/update.py" \
  "${COMMON_ARGS[@]}" \
  > "${LOG_PATH}" 2>&1

python - "${OUTPUT_PATH}" "${OUTPUT_PATH%.jsonl}.failed.json" <<'PYCHECK'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
failed_path = Path(sys.argv[2])
rows = []
if output_path.exists():
    with output_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
failed = json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else []
ids = [row["id"] for row in rows]
print(f"[INFO] retry success rows: {len(rows)}")
print(f"[INFO] retry unique ids: {len(set(ids))}")
print(f"[INFO] still failed records: {len(failed)}")
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate ids detected in llama retry output")
PYCHECK

APPLY_RETRY="${APPLY_RETRY:-true}"
if [[ "${APPLY_RETRY}" == "true" || "${APPLY_RETRY}" == "1" ]]; then
  python "${PERSONA_DIR}/tools/apply_update_retries.py" \
    --output-dir "${OUTPUT_DIR}" \
    --merged-name "${MERGED_UPDATE_NAME:-updated_personas_logs.llama.all.jsonl}"
fi

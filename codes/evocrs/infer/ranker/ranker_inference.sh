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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${BASE_MODEL:?Set BASE_MODEL to the base model path}"
: "${LORA_PATH:?Set LORA_PATH to the ranker adapter path}"
: "${TEST_FILE:?Set TEST_FILE to the ranker eval JSON path}"
: "${VERSION:?Set VERSION to one of: v13, v14, v15, v16}"

EXTRACTED_IDS="${EXTRACTED_IDS:-${SUPPLEMENTARY_ROOT}/data/test_session_ids_without_first_session.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SUPPLEMENTARY_ROOT}/codes/evocrs/infer/ranker/output/${VERSION}}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "Starting Ranker Inference"
echo "Version       : $VERSION"
echo "Base Model    : $BASE_MODEL"
echo "LoRA Path     : $LORA_PATH"
echo "Test File     : $TEST_FILE"
echo "Session IDs   : $EXTRACTED_IDS"
echo "Output Dir    : $OUTPUT_DIR"
echo "=========================================================="

python "${SCRIPT_DIR}/ranker_inference_50.py" \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --test_file "$TEST_FILE" \
  --extracted_ids "$EXTRACTED_IDS" \
  --output_dir "$OUTPUT_DIR" \
  --version "$VERSION" \
  --max_new_tokens 512

echo "Inference for $VERSION completed successfully."

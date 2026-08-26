#!/usr/bin/env bash
# Usage:
#   ./build.sh
#   ./build.sh listwise_ranking_v13 listwise_ranking_v14 listwise_ranking_v15 listwise_ranking_v16

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUPPLEMENTARY_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

if [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=("listwise_ranking_v13" "listwise_ranking_v14" "listwise_ranking_v15" "listwise_ranking_v16")
fi

OUTPUT_ROOT="${SUPPLEMENTARY_ROOT}/data/evocrs/ranker/train"
ID2ITEM_PATH="${SUPPLEMENTARY_ROOT}/data/trace/mts-kion_processed/id2items_in_dialogue.json"
CANDIDATE_PATH="${SUPPLEMENTARY_ROOT}/data/candidate_curator/top-k_candidates/candidates.json"

for TASK in "${TASKS[@]}"; do
  case "$TASK" in
    behavioral_persona_extract)
      JSON_PATH="${SUPPLEMENTARY_ROOT}/data/trace/gpt-4o-generated/behavioral_persona_extract_data.json"
      ;;
    persona_update)
      JSON_PATH="${SUPPLEMENTARY_ROOT}/data/trace/gpt-4o-generated/persona_update_9_components.json"
      ;;
    dialogue_persona_extract|listwise_ranking*)
      JSON_PATH="${SUPPLEMENTARY_ROOT}/data/trace/gpt-4o-generated/final_crs_dataset.json"
      ;;
    *)
      echo "[ERROR] Unknown task: $TASK"
      exit 1
      ;;
  esac

  OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}"
  mkdir -p "$OUTPUT_DIR"

  echo "Task: $TASK"
  echo "Output directory: $OUTPUT_DIR"

  python "${SUPPLEMENTARY_ROOT}/codes/evocrs/preprocess/training/build.py"     --json_path "$JSON_PATH"     --task "$TASK"     --output_dir "$OUTPUT_DIR"     --candidate_path "$CANDIDATE_PATH"     --id2item_path "$ID2ITEM_PATH"
done

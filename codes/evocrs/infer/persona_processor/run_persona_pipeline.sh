#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SELECTOR="${1:-${PERSONA_MODEL:-${MODEL_FAMILY:-qwen}}}"
MODEL_SELECTOR="$(printf '%s' "${MODEL_SELECTOR}" | tr '[:upper:]' '[:lower:]')"

usage() {
  cat <<'USAGE'
Usage:
  ./run_persona_pipeline.sh qwen
  ./run_persona_pipeline.sh llama

You can also set PERSONA_MODEL=qwen or PERSONA_MODEL=llama.
Other pipeline options such as RUN_DP, RUN_BP, RUN_UPDATE,
CUDA_VISIBLE_DEVICES, MAX_EXTRACT_RETRY_ROUNDS, and MAX_UPDATE_CYCLES are
passed through to the selected model-specific pipeline.
USAGE
}

case "${MODEL_SELECTOR}" in
  qwen|qwen2.5|qwen2.5-7b|qwen2.5-7b-instruct)
    MODEL_FAMILY=qwen exec "${SCRIPT_DIR}/bin/_run_qwen25_persona_pipeline.sh"
    ;;
  llama|llama3.1|llama-3.1|llama-3.1-8b|llama-3.1-8b-instruct)
    MODEL_FAMILY=llama exec "${SCRIPT_DIR}/bin/_run_llama_persona_pipeline.sh"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown persona pipeline model: ${MODEL_SELECTOR}" >&2
    usage >&2
    exit 2
    ;;
esac

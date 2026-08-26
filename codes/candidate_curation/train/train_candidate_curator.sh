#!/bin/bash

# ==========================================================
# System and GPU settings
# ==========================================================
export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM=false

# ==========================================================
# Timestamped run directory settings
# ==========================================================
# Store the current time in YYYY-MM-DD-HH-MM-SS format, e.g. 2026-02-27-15-51-15
CURRENT_TIME=$(date "+%Y-%m-%d-%H-%M-%S")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUPPLEMENTARY_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# ==========================================================
# Hyperparameter and path settings
# ==========================================================
MODEL_PATH="/home/work/huggingface/Qwen3-Embedding-8B"
TRAIN_DATA="${SUPPLEMENTARY_ROOT}/data/candidate_curator/train/train.json"
VALID_DATA="${SUPPLEMENTARY_ROOT}/data/candidate_curator/train/valid.json"

# Create a timestamped subdirectory under the base output directory
BASE_OUTPUT_DIR="${SUPPLEMENTARY_ROOT}/output/candidate_curator"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CURRENT_TIME}"

# Create the run directory before writing logs
mkdir -p "$RUN_OUTPUT_DIR"

BATCH_SIZE=8
EPOCHS=3
LR=2e-4
TEMP=0.05

echo "=========================================================="
echo "🔥 Qwen3-Embedding-8B Retriever In-batch Contrastive Learning"
echo "=========================================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Data: $TRAIN_DATA"
echo "Output directory: $RUN_OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE | Epochs: $EPOCHS | LR: $LR | Temp: $TEMP"
echo "Training has started. Progress is written to the log file."
echo "Check logs: tail -f ${RUN_OUTPUT_DIR}/train.log"
echo "=========================================================="

# ==========================================================
# Launch distributed training and write output to the log file
# ==========================================================
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision="bf16" \
    "${SCRIPT_DIR}/train_candidate_curator.py" \
    --model_name_or_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --valid_data_path "$VALID_DATA" \
    --output_dir "$RUN_OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --temperature $TEMP \
    --logging_steps 1 \
    --eval_save_steps 100 > "$RUN_OUTPUT_DIR/train.log" 2>&1

echo "=========================================================="
echo "Training script finished."
echo "=========================================================="
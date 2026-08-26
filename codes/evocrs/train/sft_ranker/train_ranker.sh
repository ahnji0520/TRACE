#!/bin/bash

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
TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")
EXP_NAME="qwen25_v13_c"
BASE_OUTPUT_DIR="${SUPPLEMENTARY_ROOT}/checkpoints/ranker/ranker_sft_reficr_top50/qwen2.5-7b-instruct/${EXP_NAME}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"

mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/train.log"

# --- [FIX 1] Set environment variables to avoid encoding and debug-log issues ---
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export TORCH_DISTRIBUTED_DEBUG=OFF  # Prevent the c10d logger from failing while reading binary data as text

# --- Configuration ---
MASTER_PORT=29522

# MODEL_PATH="${HF_MODEL_DIR}/Llama-3.1-8B-Instruct"
MODEL_PATH="${HF_MODEL_DIR}/Qwen2.5-7B-Instruct"
TRAIN_DATA="${SUPPLEMENTARY_ROOT}/data/evocrs/ranker/train/listwise_ranking_v13/listwise_ranking_v13_train.json"
VAL_DATA="${SUPPLEMENTARY_ROOT}/data/evocrs/ranker/train/listwise_ranking_v13/listwise_ranking_v13_val.json"

WANDB_API_KEY="__WANDB_API_KEY__"

WANDB_PROJECT="KT_CRS_ranker_sft"
WANDB_RUN_NAME="${EXP_NAME}-Run-$(date "+%Y%m%d-%H%M")"

# --- Run Training ---
{
    echo "=========================================================="
    echo "Starting DeepSpeed Training..."
    echo "Timestamp: $TIMESTAMP"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Log File: $LOG_FILE"
    echo "=========================================================="
} | tee -a "$LOG_FILE"

# Run DeepSpeed
deepspeed --include localhost:0,1 --master_port=$MASTER_PORT \
    ${SUPPLEMENTARY_ROOT}/codes/evocrs/train/sft_ranker/train_ranker.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_DATA \
    --validation_file $VAL_DATA \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --max_seq_len 20000 \
    --save_steps 100 \
    --eval_steps 100 \
    --use_lora \
    --gradient_checkpointing \
    --output_dir $OUTPUT_DIR \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    >> "$LOG_FILE" 2>&1

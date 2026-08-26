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

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export NCCL_SOCKET_IFNAME=eth0
export NUMEXPR_MAX_THREADS=64

TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")
BASE_OUTPUT_DIR="${SUPPLEMENTARY_ROOT}/checkpoints/persona_processor/dialogue_persona_extract/qwen2.5-7b-instruct"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"

mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/train.log"

MASTER_PORT=$(( RANDOM % 45001 + 20000 ))

TRAIN_DATA="${SUPPLEMENTARY_ROOT}/data/evocrs/persona_processor/dialogue_persona_extract/dialogue_persona_extract_train.json"
VAL_DATA="${SUPPLEMENTARY_ROOT}/data/evocrs/persona_processor/dialogue_persona_extract/dialogue_persona_extract_val.json"

WANDB_API_KEY="__WANDB_API_KEY__"

WANDB_RUN_NAME="DP-Run-$(date "+%Y%m%d-%H%M")"


NUM_GPUS=2
WANDB_PROJECT="KT_CRS_Persona"
TASK='SFT' ## SFT, response_generation. 
# MODEL_PATH="${HF_MODEL_DIR}/Llama-3.1-8B-Instruct" ## Use "Checkpoint" when training from a checkpoint. 
MODEL_PATH="${HF_MODEL_DIR}/Qwen2.5-7B-Instruct"


{
    echo "=========================================================="
    echo "Starting Training..."
    echo "Timestamp: $TIMESTAMP"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Log File: $LOG_FILE"
    echo "Backbone: $MODEL_PATH"
    echo "=========================================================="
} | tee -a "$LOG_FILE"

####### Inputs to update!!!

echo "Training running... Logs are being saved to $LOG_FILE only."

deepspeed --include localhost:4,5 --master_port=$MASTER_PORT train.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_DATA \
    --validation_file $VAL_DATA \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --max_seq_len 13507 \
    --zero_stage 2 \
    --save_steps 100 \
    --eval_steps 100 \
    --use_4bit \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir $OUTPUT_DIR \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --task $TASK \
    > "$LOG_FILE" 2>&1
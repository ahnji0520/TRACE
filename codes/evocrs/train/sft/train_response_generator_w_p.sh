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
BASE_OUTPUT_DIR="${SUPPLEMENTARY_ROOT}/checkpoints/response_generator/with_persona/qwen2.5-7b-instruct"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/train.log"
MASTER_PORT=$(( RANDOM % 45001 + 20000 ))

DATA_DIR="${SUPPLEMENTARY_ROOT}/data/evocrs/response_generator/target_response/with_persona"
TRAIN_DATA="${DATA_DIR}/response_generation_train.json"
VAL_DATA="${DATA_DIR}/response_generation_valid.json"

WANDB_API_KEY="__WANDB_API_KEY__"
WANDB_PROJECT="KT_CRS_response_generator"
WANDB_RUN_NAME="response_generator_w_p_qwen25_7b-$(date "+%Y%m%d-%H%M")"

MODEL_PATH="${HF_MODEL_DIR}/Qwen2.5-7B-Instruct"

# Keep this as SFT. utils.get_train_model() adds [REC]/[CHAT]/[QUE]/[ANS]
# only when task contains "response_generation", which we do not want here.
TASK="SFT"

{
    echo "=========================================================="
    echo "Starting Response Generator Training (with persona)..."
    echo "Timestamp: $TIMESTAMP"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Log File: $LOG_FILE"
    echo "Backbone: $MODEL_PATH"
    echo "Train Data: $TRAIN_DATA"
    echo "Val Data: $VAL_DATA"
    echo "GPUs: 4,5"
    echo "Task: $TASK"
    echo "=========================================================="
} | tee -a "$LOG_FILE"

echo "Training running... Logs are being saved to $LOG_FILE only."

deepspeed --include localhost:4,5 --master_port="$MASTER_PORT" ${SUPPLEMENTARY_ROOT}/codes/evocrs/train/sft/train.py     --model_name_or_path "$MODEL_PATH"     --train_file "$TRAIN_DATA"     --validation_file "$VAL_DATA"     --per_device_train_batch_size 2     --per_device_eval_batch_size 2     --gradient_accumulation_steps 8     --num_train_epochs 10     --learning_rate 2e-5     --max_seq_len 8192     --zero_stage 2     --save_steps 200     --eval_steps 200     --use_4bit     --use_lora     --lora_r 16     --lora_alpha 32     --gradient_checkpointing     --output_dir "$OUTPUT_DIR"     --wandb_api_key "$WANDB_API_KEY"     --wandb_project "$WANDB_PROJECT"     --wandb_run_name "$WANDB_RUN_NAME"     --task "$TASK"     > "$LOG_FILE" 2>&1

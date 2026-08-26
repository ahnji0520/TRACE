RUN_NAME=2026-01-22-main
MAX_WORKERS_EXTRACT=50
MAX_WORKERS_UPDATE=50
CONTEXT_K=3
LIMIT_USERS=1200
WRITE_HISTORY_JSONL=true


python src/dataset_generation/dialogue_generation/behavioral_persona_updater.py \
    --run_name $RUN_NAME \
    --max_workers_extract $MAX_WORKERS_EXTRACT \
    --max_workers_update $MAX_WORKERS_UPDATE \
    --context_k $CONTEXT_K \
    --limit_users $LIMIT_USERS \
    $( [ "$WRITE_HISTORY_JSONL" = true ] && echo "--write_history_jsonl" )
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from model_utils import get_inference_model
SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[5]

from format_verifier import validate_dialogue_record


# Input data paths
behavioral_persona_data_path = str(SUPPLEMENTARY_ROOT / "data/evocrs/persona_processor/behavioral_persona_extract/behavioral_persona_extract_test.json")
dialogue_persona_data_path = str(SUPPLEMENTARY_ROOT / "data/evocrs/persona_processor/dialogue_persona_extract/dialogue_persona_extract_test.json")
filter_data_path = str(SUPPLEMENTARY_ROOT / "data/evocrs/persona_processor/filter_reficr.json")

# Default inference settings
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_NUM_BEAMS = 3


def parse_args():
    """Parse command-line arguments for sharded dialogue extraction."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["auto", "llama3.1", "qwen", "qwen2.5", "qwen2.5-7b-instruct", "qwen3", "qwen3-8b"],
        default="auto",
        help="Backbone family. auto infers it from --backbone_path.",
    )
    parser.add_argument("--backbone_path", type=str, default="${HF_MODEL_DIR}/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--shard_idx", "--rank", dest="shard_idx", type=int, default=None)
    parser.add_argument("--num_shards", "--world_size", dest="num_shards", type=int, default=None)
    parser.add_argument("--input_override", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen thinking mode in chat templates. Default is disabled.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="single",
        choices=["single", "auto"],
        help="single keeps each shard on its visible GPU; auto lets transformers shard one model.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
    args = parser.parse_args()

    args.shard_idx = (
        args.shard_idx
        if args.shard_idx is not None
        else int(os.environ.get("RANK", 0))
    )
    args.num_shards = (
        args.num_shards
        if args.num_shards is not None
        else int(os.environ.get("WORLD_SIZE", 1))
    )
    args.backbone = resolve_backbone(args.backbone, args.backbone_path)

    if args.thinking and not is_qwen_backbone(args.backbone):
        raise ValueError("--thinking is only supported for Qwen backbones.")

    return args

def resolve_backbone(backbone: str, backbone_path: str) -> str:
    if backbone != "auto":
        return backbone

    normalized_path = backbone_path.lower()
    if "qwen2.5" in normalized_path:
        return "qwen2.5"
    if "qwen3" in normalized_path:
        return "qwen3"
    if "qwen" in normalized_path:
        return "qwen"
    if "llama" in normalized_path:
        return "llama3.1"

    raise ValueError(
        "Could not infer backbone from --backbone_path. "
        "Please pass --backbone llama3.1 or a qwen backbone."
    )


def is_qwen_backbone(backbone: str) -> bool:
    return backbone.startswith("qwen")


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, Any]],
    backbone: str,
    thinking: bool,
) -> str:
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
    }

    if is_qwen_backbone(backbone):
        kwargs["enable_thinking"] = thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise

        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def get_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_json(path: str) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_processed_ids(path: str) -> set[str]:
    """Load already processed sample IDs from an existing JSONL file."""
    processed_ids = set()

    if not os.path.exists(path):
        return processed_ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                if "id" in row:
                    processed_ids.add(row["id"])
            except json.JSONDecodeError:
                continue

    return processed_ids


def build_items_to_process() -> list[dict[str, Any]]:
    """
    Build dialogue extraction inputs.

    For each behavioral persona sample at session t, pair it with the
    dialogue sample from session t-1 and use the dialogue history up to
    that point.
    """
    behavioral_persona = load_json(behavioral_persona_data_path)
    dialogue_persona = load_json(dialogue_persona_data_path)
    filter_data = load_json(filter_data_path)

    behavioral_persona_map = {item["id"]: item for item in behavioral_persona}
    dialogue_persona_map = {item["id"]: item for item in dialogue_persona}

    items_to_process = []

    for sample_id in sorted(behavioral_persona_map.keys()):
        user_id, session_str = sample_id.split("_")
        session_num = int(session_str)

        if session_num == 1:
            continue

        current_behavioral_id = sample_id
        previous_dialogue_id = f"{user_id}_{session_num - 1}"

        behavioral_sample = behavioral_persona_map.get(current_behavioral_id)
        dialogue_sample = dialogue_persona_map.get(previous_dialogue_id)

        if behavioral_sample is None or dialogue_sample is None:
            print(
                f"[WARN] missing sample: "
                f"b={current_behavioral_id in behavioral_persona_map}, "
                f"d={previous_dialogue_id in dialogue_persona_map}"
            )
            continue

        if user_id in filter_data and filter_data[user_id]:
            max_needed = max(filter_data[user_id]) - 1

            if session_num - 1 <= max_needed:
                items_to_process.append(
                    {
                        "id": previous_dialogue_id,
                        "user_id": user_id,
                        "session_num": session_num - 1,
                        "dialogue_messages": dialogue_sample["messages"][:-1],
                    }
                )

    return items_to_process


def generate_batch(
    tokenizer,
    model,
    batch: list[dict[str, Any]],
    do_sample: bool,
    backbone: str,
    thinking: bool,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    """Generate dialogue summaries for a batch."""
    prompts = [
        apply_chat_template(
            tokenizer=tokenizer,
            messages=item["dialogue_messages"],
            backbone=backbone,
            thinking=thinking,
        )
        for item in batch
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(get_input_device(model))

    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_beams": num_beams,
    }


    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    responses = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return responses


def main():
    args = parse_args()

    print(
        f"[INFO] shard_idx={args.shard_idx}, num_shards={args.num_shards}, "
        f"backbone={args.backbone}, thinking={args.thinking}"
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.input_override is not None:
        output_save_path = os.path.join(output_dir, "extracted_dialogue_only.retry.jsonl")
        failed_input_save_path = os.path.join(output_dir, "failed_dialogue_inputs.retry.json")
    else:
        output_save_path = os.path.join(output_dir, f"extracted_dialogue_only.shard{args.shard_idx}.jsonl")
        failed_input_save_path = os.path.join(output_dir, f"failed_dialogue_inputs.shard{args.shard_idx}.json")

    processed_ids = load_processed_ids(output_save_path) if args.resume else set()
    if processed_ids:
        print(f"[INFO] loaded {len(processed_ids)} processed items from {output_save_path}")

    if args.input_override is not None:
        items_to_process = load_json(args.input_override)
        print(f"[INFO] loaded override inputs: {len(items_to_process)} from {args.input_override}")
        items_to_process = items_to_process[args.shard_idx::args.num_shards]
        print(
            f"[INFO] override shard {args.shard_idx}/{args.num_shards} -> "
            f"{len(items_to_process)} items"
        )
    else:
        items_to_process = build_items_to_process()
        print(f"[INFO] total dialogue sessions to process: {len(items_to_process)}")
        items_to_process = [
            item
            for idx, item in enumerate(items_to_process)
            if idx % args.num_shards == args.shard_idx
        ]
        print(
            f"[INFO] shard {args.shard_idx}/{args.num_shards} -> "
            f"{len(items_to_process)} items"
        )

    if processed_ids:
        before = len(items_to_process)
        items_to_process = [item for item in items_to_process if item["id"] not in processed_ids]
        print(f"[INFO] resume skip: {before - len(items_to_process)} already processed items")

    if args.device_map == "auto":
        device_map = "auto"
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = {"": "cpu"}
    tokenizer, model = get_inference_model(
        backbone_path=args.backbone_path,
        lora_path=args.lora_path,
        device_map=device_map,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.resume:
        with open(output_save_path, "w", encoding="utf-8") as f:
            pass

    failed_inputs = []

    for i in tqdm(
        range(0, len(items_to_process), args.batch_size),
        desc="Extracting Dialogue Personas",
    ):
        batch = items_to_process[i:i + args.batch_size]
        pending_items = batch[:]
        final_results = {}

        for attempt in range(1, args.max_retries + 1):
            if not pending_items:
                break

            responses = generate_batch(
                tokenizer=tokenizer,
                model=model,
                batch=pending_items,
                do_sample=args.do_sample,
                backbone=args.backbone,
                thinking=args.thinking,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            next_pending_items = []

            for idx, item in enumerate(pending_items):
                result_data = {
                    "id": item["id"],
                    "user_id": item["user_id"],
                    "session_num": item["session_num"],
                    "extracted_dialogue": responses[idx],
                }

                is_valid, errors, _ = validate_dialogue_record(result_data)

                if is_valid:
                    final_results[item["id"]] = result_data
                else:
                    if args.do_sample and attempt < args.max_retries:
                        next_pending_items.append(item)
                    else:
                        print(
                            f"[WARN] validation failed after {attempt} attempts: "
                            f"id={item['id']}, errors={errors}"
                        )
                        final_results[item["id"]] = result_data
                        failed_inputs.append(item)

            if next_pending_items:
                print(
                    f"[INFO] batch {i // args.batch_size + 1}, attempt {attempt}: "
                    f"{len(next_pending_items)} samples failed validation and will be retried"
                )

            pending_items = next_pending_items

        with open(output_save_path, "a", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(final_results[item["id"]], ensure_ascii=False) + "\n")

    if failed_inputs:
        with open(failed_input_save_path, "w", encoding="utf-8") as f:
            json.dump(failed_inputs, f, ensure_ascii=False, indent=2)
        print(f"[INFO] saved {len(failed_inputs)} failed inputs to {failed_input_save_path}")
    else:
        if os.path.exists(failed_input_save_path):
            os.remove(failed_input_save_path)
        print("[INFO] no failed dialogue inputs")


if __name__ == "__main__":
    main()

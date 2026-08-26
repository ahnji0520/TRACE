import os
import re
import json
import sys
from pathlib import Path
from typing import Any
import argparse
import torch
import concurrent.futures
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from model_utils import get_inference_model
SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[5]

from format_verifier import validate_updated_memory_record


update_model_path = str(SUPPLEMENTARY_ROOT / "checkpoints/persona_processor/persona_update/qwen3-8b/2026-05-12-18-03-38/checkpoint-7200")

extracted_data_path = str(SUPPLEMENTARY_ROOT / "codes/evocrs/infer/persona_processor/output/extracted_personas_only.qwen.jsonl")
update_data_path = str(SUPPLEMENTARY_ROOT / "data/evocrs/persona_processor/persona_update/persona_update_test.json")

filter_data_path = str(SUPPLEMENTARY_ROOT / "data/evocrs/persona_processor/filter_reficr.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["auto", "llama3.1", "qwen", "qwen2.5", "qwen2.5-7b-instruct", "qwen3", "qwen3-8b"],
        default="auto",
        help="Backbone family. auto infers it from --backbone_path.",
    )
    parser.add_argument("--backbone_path", type=str, default="${HF_MODEL_DIR}/Qwen3-8B")
    parser.add_argument("--lora_path", type=str, default=update_model_path)
    parser.add_argument("--extracted_data_path", type=str, default=extracted_data_path)
    parser.add_argument("--update_data_path", type=str, default=update_data_path)
    parser.add_argument("--filter_data_path", type=str, default=filter_data_path)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_per_device", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--det_retries", type=int, default=1)
    parser.add_argument("--sample_retries", type=int, default=0)
    parser.add_argument("--det_num_beams", type=int, default=1)
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--sample_top_p", type=float, default=0.95)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_shards", "--world_size", dest="num_shards", type=int, default=None)
    parser.add_argument("--shard_idx", "--rank", dest="shard_idx", type=int, default=None)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen thinking mode in chat templates. Default is disabled.",
    )
    parser.add_argument(
        "--failed_input_path",
        type=str,
        default=None,
        help="Path to failed_update_inputs.shard*.json for rerun-only mode."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output JSONL path. Useful for rerun-only mode."
    )

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
    if args.det_retries < 0 or args.sample_retries < 0:
        raise ValueError("--det_retries and --sample_retries must be non-negative.")
    if args.det_retries + args.sample_retries < 1:
        raise ValueError("At least one generation attempt is required.")

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
        "Please pass --backbone llama3.1 or --backbone qwen3."
    )


def is_qwen_backbone(backbone: str) -> bool:
    return backbone.startswith("qwen")


def apply_chat_template(tokenizer, messages, backbone: str, thinking: bool) -> str:
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


def load_update_models(args):
    print(f"Loading update models on {args.num_gpus} GPUs...")
    update_tokenizer, first_model = get_inference_model(
        backbone_path=args.backbone_path,
        lora_path=args.lora_path,
        device_map={"": 0} if torch.cuda.is_available() else {"": "cpu"},
    )
    update_models = [first_model]

    for gpu_id in range(1, args.num_gpus):
        _, model = get_inference_model(
            backbone_path=args.backbone_path,
            lora_path=args.lora_path,
            device_map={"": gpu_id},
        )
        update_models.append(model)

    update_tokenizer.padding_side = 'left'
    if update_tokenizer.pad_token is None:
        update_tokenizer.pad_token = update_tokenizer.eos_token

    return update_tokenizer, update_models


## Jsonl/Json Utils
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, item: dict) -> None:
    """Append one JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def build_seed_state(uid, extracted_data):
    """
    Build the initial session-1 state from extracted personas.
    This seed will also be saved to output jsonl for stable resume behavior.
    """
    ext_item = extracted_data[(uid, 1)]
    behavior_obj = json.loads(ext_item["extracted_behavior"])
    dialogue_obj = json.loads(ext_item["extracted_dialogue"])

    init_memory = {
        "behavior_persona": behavior_obj,
        "dialogue_persona": dialogue_obj.get("dialogue_persona"),
        "memory": {
            "session_events": [],
            "recommendation_history": []
        }
    }

    return {
        "id": ext_item["id"],
        "user_id": uid,
        "session_num": 1,
        "extracted_behavior": ext_item["extracted_behavior"],
        "extracted_dialogue": ext_item["extracted_dialogue"],
        "updated_memory": init_memory,
    }

###=====###

## Build input
def extract_persona_strings(past_memory):
    """
    Extract only behavioral/dialogue personas from the previous state.
    Memory is intentionally not reused in the prompt and will stay empty.
    """
    if isinstance(past_memory, dict):
        past_b = json.dumps(past_memory.get("behavior_persona", {}), ensure_ascii=False, separators=(",", ":"))
        past_d = json.dumps(past_memory.get("dialogue_persona", []), ensure_ascii=False, separators=(",", ":"))
        return past_b, past_d

    try:
        parsed = json.loads(past_memory)
        past_b = json.dumps(parsed.get("behavior_persona", {}), ensure_ascii=False, separators=(",", ":"))
        past_d = json.dumps(parsed.get("dialogue_persona", []), ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        past_b, past_d = str(past_memory), "[]"

    return past_b, past_d


def build_prompt(system_prompt, ext_item, past_memory, tokenizer, backbone: str, thinking: bool):
    """
    Build the update prompt.
    Memory is intentionally fixed to an empty structure for both old/new session blocks.
    """
    past_b, past_d = extract_persona_strings(past_memory)
    new_b = json.dumps(json.loads(ext_item["extracted_behavior"]), ensure_ascii=False, separators=(",", ":"))
    new_d = json.dumps(json.loads(ext_item["extracted_dialogue"]), ensure_ascii=False, separators=(",", ":"))
 

    new_content = f"""[Current Long-term State]
- Behavioral Persona: {past_b}
- Dialogue Persona: {past_d}
- Memory: {{"session_events": [], "recommendation_history": []}}

[New Session Data]
- New Behavioral Persona: {new_b}
- New Dialogue Persona: {new_d}
- New Session Memory: {{"session_events": [], "recommendation_history": []}}

Now, based on the given information, generate the response."""

    update_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": new_content},
    ]
    return apply_chat_template(
        tokenizer=tokenizer,
        messages=update_input,
        backbone=backbone,
        thinking=thinking,
    )
###=====###

def get_generation_config(attempt: int, args):
    """Build generation config for the current retry attempt."""
    if attempt <= args.det_retries:
        return {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.det_num_beams,
            "do_sample": False,
        }

    return {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "do_sample": True,
        "temperature": args.sample_temperature,
        "top_p": args.sample_top_p,
    }

## Verification

def normalize_updated_memory_schema(obj: dict) -> dict:
    """
    Normalize known key variants into the schema expected by the validator.
    This keeps repair logic separate from generation logic.
    """
    if not isinstance(obj, dict):
        return obj

    updated_memory = obj.get("updated_memory")
    if not isinstance(updated_memory, dict):
        updated_memory = {}
        obj["updated_memory"] = updated_memory

    # Normalize episodic_memory -> session_events
    if "episodic_memory" in updated_memory and "session_events" not in updated_memory:
        updated_memory["session_events"] = updated_memory.pop("episodic_memory")

    # Normalize recommendation_outcomes -> recommendation_history
    if "recommendation_outcomes" in updated_memory and "recommendation_history" not in updated_memory:
        updated_memory["recommendation_history"] = updated_memory.pop("recommendation_outcomes")

    # Move misplaced top-level recommendation history under updated_memory
    if "updated_recommendation_history" in obj:
        if "recommendation_history" not in updated_memory:
            updated_memory["recommendation_history"] = obj.pop("updated_recommendation_history")
        else:
            obj.pop("updated_recommendation_history")

    # Move misplaced top-level session event list under updated_memory if it ever appears
    if "episodic_memory" in obj and "session_events" not in updated_memory:
        updated_memory["session_events"] = obj.pop("episodic_memory")

    updated_memory.setdefault("session_events", [])
    updated_memory.setdefault("recommendation_history", [])

    return obj


def _as_dict(value, default=None):
    """Return a dict from either a dict or a JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else (default or {})
        except json.JSONDecodeError:
            return default or {}
    return default or {}


def _dedupe_preserve_order(items):
    """Deduplicate string/list entries without changing their first-seen order."""
    deduped = []
    seen = set()
    for item in items or []:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else item
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_dialogue_append_fallback_memory(meta: dict) -> dict:
    """
    Build a valid updated_memory object when rerun generation keeps producing
    malformed JSON. It preserves past state, replaces behavior with the new
    extracted behavior, appends new dialogue persona lines, and carries over
    dialogue memory entries from the new extracted dialogue.
    """
    ext = meta.get("ext", {})
    past_memory = _as_dict(meta.get("past_memory"), default={})
    extracted_behavior = _as_dict(ext.get("extracted_behavior"), default={})
    extracted_dialogue = _as_dict(ext.get("extracted_dialogue"), default={})

    past_dialogue = past_memory.get("dialogue_persona", [])
    if not isinstance(past_dialogue, list):
        past_dialogue = []
    new_dialogue = extracted_dialogue.get("dialogue_persona", [])
    if not isinstance(new_dialogue, list):
        new_dialogue = []

    past_nested_memory = past_memory.get("memory", {})
    if not isinstance(past_nested_memory, dict):
        past_nested_memory = {}
    new_nested_memory = extracted_dialogue.get("memory", {})
    if not isinstance(new_nested_memory, dict):
        new_nested_memory = {}

    session_events = []
    session_events.extend(past_nested_memory.get("session_events", []))
    session_events.extend(past_nested_memory.get("episodic_memory", []))
    session_events.extend(new_nested_memory.get("session_events", []))
    session_events.extend(new_nested_memory.get("episodic_memory", []))

    recommendation_history = []
    recommendation_history.extend(past_nested_memory.get("recommendation_history", []))
    recommendation_history.extend(past_nested_memory.get("recommendation_outcomes", []))
    recommendation_history.extend(new_nested_memory.get("recommendation_history", []))
    recommendation_history.extend(new_nested_memory.get("recommendation_outcomes", []))

    return {
        "behavior_persona": past_memory.get("behavior_persona", extracted_behavior),
        "dialogue_persona": _dedupe_preserve_order(past_dialogue + new_dialogue),
        "memory": {
            "session_events": _dedupe_preserve_order(session_events),
            "recommendation_history": _dedupe_preserve_order(recommendation_history),
        },
    }



def try_repair_updated_memory_response(response: str) -> str:
    """
    Best-effort repair for malformed JSON patterns observed in failed shards.

    Repair order:
    1) unwrap a JSON-encoded outer string if present
    2) rename updated_* keys to the schema expected by the validator path
    3) fix malformed boundary before recommendation_history
    4) cut trailing natural-language text after the last closing brace
    5) add one final missing closing brace if needed
    6) parse + normalize schema + return compact JSON string

    If repair fails, return the original response unchanged.
    """
    if not isinstance(response, str):
        return response

    original = response
    text = response.strip()

    # Step 0: unwrap once if the whole response is a JSON-encoded string
    try:
        unwrapped = json.loads(text)
        if isinstance(unwrapped, str):
            text = unwrapped.strip()
        return text
    except json.JSONDecodeError:
        pass

    ### >>>
    # # Step 1: rename keys
    # replacements = {
    #     "updated_behavioral_persona": "behavioral_persona",
    #     "updated_dialogue_persona": "dialogue_persona",
    #     "updated_memory": "memory",
    #     "updated_recommendation_history": "recommendation_history",
    #     "episodic_memory": "session_events"
    # }
    # for old, new in replacements.items():
    #     text = text.replace(f'"{old}"', f'"{new}"')

    # # Step 2: fix malformed boundary before recommendation_history
    # # Example:
    # # ..."memory":{"episodic_memory":["..."},"recommendation_history":[...]}
    # # -> ..."memory":{"episodic_memory":["..."],"recommendation_history":[...]}
    # text = re.sub(
    #     r'\}\s*,\s*"recommendation_history"\s*:',
    #     r'],"recommendation_history":',
    #     text
    # )
    ### <<<


    ### >>>
    # Step 2: fix malformed boundary before recommendation_history
    # Example:
    # ..."memory":{"episodic_memory":["..."},"recommendation_history":[...]}
    # -> ..."memory":{"episodic_memory":["..."],"recommendation_history":[...]}
    text = re.sub(
        r'\}\s*,\s*"updated_recommendation_history"\s*:',
        r'],"updated_recommendation_history":',
        text
    )
    ### <<<


    # Step 3: remove trailing commentary after the last closing brace
    last_close = text.rfind("}")
    if last_close != -1:
        text = text[:last_close + 1].strip()

    # Step 4: add one final missing top-level closing brace if needed
    # Observed repaired pattern sometimes ends with "}]}",
    # which means the outermost object is still missing one "}".
    if text.endswith("}]}"):
        text += "}"

    # Step 5: parse repaired candidate
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return original

    return original


## Retry Funciton
def rerun_failed_inputs(
    failed_records,
    update_models,
    update_tokenizer,
    args,
    executor,
    gen_fn,
    output_save_path,
    ):
    """
    Rerun only the failed records saved in failed_update_inputs.shard*.json.
    This mode is independent from the normal session-by-session update pipeline.
    """
    import math
    import time

    if not failed_records:
        print("[INFO] no failed records to rerun")
        return

    BATCH_SIZE = args.batch_per_device * args.num_gpus
    final_results = []
    still_failed = []

    pending_meta_list = []
    unique_users = set()

    for rec in failed_records:
        pending_meta_list.append({
            "uid": rec["user_id"],
            "session_num": int(rec["session_num"]),
            "ext": {
                "id": rec["id"],
                "extracted_behavior": rec["extracted_behavior"],
                "extracted_dialogue": rec["extracted_dialogue"],
            },
            "prompt": rec["prompt"],
            "past_memory": rec.get("past_memory"),
        })
        unique_users.add(rec["user_id"])

    print(
        f"[INFO] rerun mode started | failed records: {len(pending_meta_list)} "
        f"| unique users: {len(unique_users)} | batch size: {BATCH_SIZE}"
    )

    total_retries = args.det_retries + args.sample_retries
    for attempt in range(1, total_retries + 1):
        if not pending_meta_list:
            break

        attempt_start_time = time.time()
        next_pending_meta_list = []

        num_batches = math.ceil(len(pending_meta_list) / BATCH_SIZE)
        decode_mode = "deterministic" if attempt <= args.det_retries else "sampling"

        print(
            f"\n[INFO] rerun attempt {attempt}/{total_retries} ({decode_mode}) "
            f"| pending samples: {len(pending_meta_list)} | total batches: {num_batches}"
        )

        for batch_idx, i in enumerate(
            tqdm(
                range(0, len(pending_meta_list), BATCH_SIZE),
                desc=f"Rerun Attempt {attempt}",
                leave=True
            ),
            start=1
        ):
            batch_meta = pending_meta_list[i:i + BATCH_SIZE]
            batch_ids = [meta["ext"]["id"] for meta in batch_meta]
            batch_sessions = [meta["session_num"] for meta in batch_meta]

            print(
                f"[INFO] attempt {attempt} | batch {batch_idx}/{num_batches} "
                f"| ids={batch_ids} | sessions={batch_sessions}"
            )

            prompt_chunks = [
                batch_meta[j:j + args.batch_per_device]
                for j in range(0, len(batch_meta), args.batch_per_device)
            ]

            batch_start_time = time.time()

            with torch.inference_mode():
                futures = []
                for chunk_idx, chunk in enumerate(prompt_chunks):
                    model = update_models[chunk_idx % len(update_models)]
                    futures.append(executor.submit(gen_fn, model, chunk, attempt))

                all_outputs = []
                for fut in futures:
                    all_outputs.extend(fut.result())

            for out in all_outputs:
                meta = out["meta"]
                raw_response = out["response"]
                repaired_response = try_repair_updated_memory_response(raw_response)
                if raw_response != repaired_response:
                    print(f"[REPAIR_APPLIED] id={meta['ext']['id']} | session={meta['session_num']}")
                else:
                    print(f"[REPAIR_NO_CHANGE] id={meta['ext']['id']} | session={meta['session_num']}")

                # print(f"[RAW_TAIL] {raw_response[-300:]}")
                # print(f"[REPAIRED_TAIL] {repaired_response[-300:]}")
                
                res_obj = {
                    "id": meta["ext"]["id"],
                    "user_id": meta["uid"],
                    "session_num": meta["session_num"],
                    "extracted_behavior": meta["ext"]["extracted_behavior"],
                    "extracted_dialogue": meta["ext"]["extracted_dialogue"],
                    "updated_memory": repaired_response,
                }

                is_valid, errors, parsed = validate_updated_memory_record(res_obj)

                if is_valid:
                    res_obj["updated_memory"] = parsed["updated_memory"]
                    final_results.append(res_obj)

                    if attempt == 1:
                        print(
                            f"[OK] id={meta['ext']['id']} | user_id={meta['uid']} "
                            f"| session={meta['session_num']} | success on first attempt"
                        )
                    else:
                        print(
                            f"[OK_AFTER_RETRY] id={meta['ext']['id']} | user_id={meta['uid']} "
                            f"| session={meta['session_num']} | success on attempt {attempt}"
                        )
                else:
                    if attempt < total_retries:
                        next_pending_meta_list.append(meta)
                        print(
                            f"[RETRY] id={meta['ext']['id']} | user_id={meta['uid']} "
                            f"| session={meta['session_num']} | attempt={attempt} "
                            f"| errors={errors}"
                        )
                    else:
                        fallback_memory = build_dialogue_append_fallback_memory(meta)
                        fallback_obj = {
                            "id": meta["ext"]["id"],
                            "user_id": meta["uid"],
                            "session_num": meta["session_num"],
                            "extracted_behavior": meta["ext"]["extracted_behavior"],
                            "extracted_dialogue": meta["ext"]["extracted_dialogue"],
                            "updated_memory": json.dumps(fallback_memory, ensure_ascii=False),
                        }
                        fallback_valid, fallback_errors, fallback_parsed = validate_updated_memory_record(fallback_obj)
                        if fallback_valid:
                            fallback_obj["updated_memory"] = fallback_parsed["updated_memory"]
                            final_results.append(fallback_obj)
                            print(
                                f"[FALLBACK_OK] id={meta['ext']['id']} | user_id={meta['uid']} "
                                f"| session={meta['session_num']} | source=dialogue_append"
                            )
                        else:
                            still_failed.append({
                                "id": meta["ext"]["id"],
                                "user_id": meta["uid"],
                                "session_num": meta["session_num"],
                                "prompt": meta["prompt"],
                                "past_memory": meta["past_memory"],
                                "last_response": raw_response,
                                "repaired_response": repaired_response,
                                "errors": errors,
                                "fallback_errors": fallback_errors,
                                "extracted_behavior": meta["ext"]["extracted_behavior"],
                                "extracted_dialogue": meta["ext"]["extracted_dialogue"],
                            })
                            print(
                                f"[FAIL] id={meta['ext']['id']} | user_id={meta['uid']} "
                                f"| session={meta['session_num']} | final attempt={attempt} "
                                f"| errors={errors} | fallback_errors={fallback_errors}"
                            )

            batch_elapsed = time.time() - batch_start_time
            print(
                f"[INFO] attempt {attempt} | batch {batch_idx}/{num_batches} done "
                f"| elapsed={batch_elapsed:.2f}s | successes_so_far={len(final_results)} "
                f"| retry_queue={len(next_pending_meta_list)}"
            )

        attempt_elapsed = time.time() - attempt_start_time
        print(
            f"[INFO] attempt {attempt} finished | elapsed={attempt_elapsed:.2f}s "
            f"| next_retry_count={len(next_pending_meta_list)}"
        )

        if next_pending_meta_list:
            print(
                f"[INFO] rerun attempt {attempt}/{total_retries} ({decode_mode}): "
                f"{len(next_pending_meta_list)} samples failed validation and will be retried"
            )

        pending_meta_list = next_pending_meta_list

    with open(output_save_path, "w", encoding="utf-8") as f_out:
        for item in final_results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    failed_save_path = output_save_path.replace(".jsonl", ".failed.json")
    if still_failed:
        with open(failed_save_path, "w", encoding="utf-8") as f:
            json.dump(still_failed, f, ensure_ascii=False, indent=2)
    elif os.path.exists(failed_save_path):
        os.remove(failed_save_path)

    print(f"[INFO] rerun success: {len(final_results)}")
    print(f"[INFO] rerun still failed: {len(still_failed)}")
    print(f"[INFO] saved rerun outputs to {output_save_path}")
    if still_failed:
        print(f"[INFO] saved rerun failures to {failed_save_path}")
    else:
        print("[INFO] no rerun failures")


## Main Function
def main():
    args = parse_args()
    BATCH_SIZE = args.batch_per_device * args.num_gpus
    os.makedirs(args.output_dir, exist_ok=True)
    output_save_path = args.output_path or os.path.join(
        args.output_dir,
        f'updated_personas_logs.shard{args.shard_idx}.jsonl',
    )
    failed_input_save_path = os.path.join(
        args.output_dir,
        f'failed_update_inputs.shard{args.shard_idx}.json',
    )

    failed_inputs = []

    # Rerun-only mode should not touch the normal dataset/seed pipeline.
    # It only needs the model, tokenizer, and failed input file.
    if args.failed_input_path is not None:
        if args.output_path is None:
            raise ValueError("--failed_input_path requires --output_path for safe rerun-only execution.")

        update_tokenizer, update_models = load_update_models(args)

        def gen(model, prompt_items, attempt):
            """
            Run generation for one chunk and return both meta info and decoded outputs.
            Keeping meta + output together is safer than relying on index alignment later.
            """
            if not prompt_items:
                return []

            prompts = [item["prompt"] for item in prompt_items]
            ins = update_tokenizer(prompts, return_tensors="pt", padding=True).to(get_input_device(model))

            gen_config = get_generation_config(attempt, args)
            outs = model.generate(
                input_ids=ins.input_ids,
                attention_mask=ins.attention_mask,
                **gen_config
            )

            decoded = update_tokenizer.batch_decode(
                outs[:, ins.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return [
                {
                    "meta": meta,
                    "response": response,
                }
                for meta, response in zip(prompt_items, decoded)
            ]

        failed_records = load_json(args.failed_input_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
            rerun_failed_inputs(
                failed_records=failed_records,
                update_models=update_models,
                update_tokenizer=update_tokenizer,
                args=args,
                executor=executor,
                gen_fn=gen,
                output_save_path=output_save_path,
            )
        return
    else:

        # 1. Load datasets
        print("Loading datasets...")
        filter_data = load_json(args.filter_data_path)
        update_data_raw = load_json(args.update_data_path)
        system_prompt = update_data_raw[0]["messages"][0]["content"]

        extracted_data = {}
        with open(args.extracted_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                extracted_data[(item['user_id'], int(item['session_num']))] = item

        # 2. Restore previous states if resume is enabled
        user_session_logs = {}
        if args.resume:
            if os.path.exists(output_save_path):
                print(f"Reading existing logs from {output_save_path} to resume...")
                with open(output_save_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        uid = item['user_id']
                        if uid not in user_session_logs or item['session_num'] > user_session_logs[uid]['session_num']:
                            user_session_logs[uid] = item
            else:
                print(f"[WARN] --resume was given but output file does not exist: {output_save_path}")
                print("[WARN] Starting from scratch instead.")
        else:
            # Reset output file when not resuming
            with open(output_save_path, "w", encoding="utf-8") as f:
                pass

        # 3. Decide the maximum session index to process for each user
        user_max_session = {}
        for uid in filter_data.keys():
            if filter_data[uid]:
                max_needed = max(filter_data[uid])
                # Users with only session 1 do not need persona update generation.
                if max_needed <= 1:
                    print(f"[user_id:{uid}] Users with only session 1 do not need persona update generation.")
                    continue
                user_max_session[uid] = max_needed



        all_user_ids = sorted(list(user_max_session.keys()))
        user_ids = [
            uid for i, uid in enumerate(all_user_ids)
            if i % args.num_shards == args.shard_idx
        ]
        user_max_session = {uid: user_max_session[uid] for uid in user_ids}

        # 4. Seed session-1 state if missing
        #    Also write it to output jsonl so resume can fully reconstruct the state.
        seeded_count = 0
        for uid in list(user_max_session.keys()):
            if (uid, 1) not in extracted_data:
                print(f"[WARN] missing seed extracted data: {(uid, 1)}")
                user_max_session.pop(uid, None)
                continue

            if uid not in user_session_logs:
                seed_obj = build_seed_state(uid, extracted_data)
                user_session_logs[uid] = seed_obj
                append_jsonl(output_save_path, seed_obj)
                seeded_count += 1

        # Rebuild user_ids after removing users with missing seed data.
        user_ids = sorted(list(user_max_session.keys()))

        if not user_ids:
            print(f"Nothing to process for shard {args.shard_idx}/{args.num_shards}")
            return
    


        done_sessions = [user_session_logs[uid]["session_num"] for uid in user_ids if uid in user_session_logs]
        if done_sessions:
            print(
                f"Loaded states for {len(done_sessions)} users | "
                f"min session={min(done_sessions)}, max session={max(done_sessions)}"
            )
        if seeded_count:
            print(f"Seeded and saved {seeded_count} users at session 1")

        # 5. Load update models across GPUs
        update_tokenizer, update_models = load_update_models(args)

        def gen(model, prompt_items, attempt):
            """
            Run generation for one chunk and return both meta info and decoded outputs.
            Keeping meta + output together is safer than relying on index alignment later.
            """
            if not prompt_items:
                return []

            prompts = [item["prompt"] for item in prompt_items]
            ins = update_tokenizer(prompts, return_tensors="pt", padding=True).to(get_input_device(model))

            gen_config = get_generation_config(attempt, args)
            outs = model.generate(
                input_ids=ins.input_ids,
                attention_mask=ins.attention_mask,
                **gen_config
            )

            decoded = update_tokenizer.batch_decode(
                outs[:, ins.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return [
                {
                    "meta": meta,
                    "response": response,
                }
                for meta, response in zip(prompt_items, decoded)
            ]

        overall_max = max(user_max_session.values())

        

        # 6. Process sessions from 2 to overall_max
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
            for current_session in range(2, overall_max + 1):
                active_uids = [
                    uid for uid in user_ids
                    if (
                        current_session <= user_max_session[uid]
                        and (uid, current_session) in extracted_data
                        and uid in user_session_logs
                        and user_session_logs[uid]["session_num"] == current_session - 1
                    )
                ]
                if not active_uids:
                    continue

                print(f"\n--- [Session {current_session}] processing users: {len(active_uids)} ---")

                for i in tqdm(range(0, len(active_uids), BATCH_SIZE), desc=f"S{current_session} Batch"):
                    batch_uids = active_uids[i:i + BATCH_SIZE]
                    pending_meta_list = []

                    # Build prompts for this batch
                    for uid in batch_uids:
                        ext_item = extracted_data[(uid, current_session)]
                        past_memory = user_session_logs[uid]["updated_memory"]

                        prompt = build_prompt(
                            system_prompt=system_prompt,
                            ext_item=ext_item,
                            past_memory=past_memory,
                            tokenizer=update_tokenizer,
                            backbone=args.backbone,
                            thinking=args.thinking,
                        )

                        pending_meta_list.append({
                            "uid": uid,
                            "ext": ext_item,
                            "prompt": prompt,
                            "past_memory": past_memory,
                        })

                    final_results = {}

                    # Retry up to total_retries times
                    total_retries = args.det_retries + args.sample_retries
                    for attempt in range(1, total_retries + 1):
                        if not pending_meta_list:
                            break

                        prompt_chunks = [
                            pending_meta_list[j:j + args.batch_per_device]
                            for j in range(0, len(pending_meta_list), args.batch_per_device)
                        ]

                        with torch.inference_mode():
                            futures = []
                            for chunk_idx, chunk in enumerate(prompt_chunks):
                                model = update_models[chunk_idx % len(update_models)]
                                futures.append(executor.submit(gen, model, chunk, attempt))

                            all_outputs = []
                            for fut in futures:
                                all_outputs.extend(fut.result())

                        next_pending_meta_list = []

                        for out in all_outputs:
                            meta = out["meta"]
                            raw_response = out["response"]
                            repaired_response = try_repair_updated_memory_response(raw_response)


                            res_obj = {
                                "id": meta["ext"]["id"],
                                "user_id": meta["uid"],
                                "session_num": current_session,
                                "extracted_behavior": meta["ext"]["extracted_behavior"],
                                "extracted_dialogue": meta["ext"]["extracted_dialogue"],
                                "updated_memory": repaired_response
                            }

                            is_valid, errors, parsed = validate_updated_memory_record(res_obj)

                            if is_valid:
                                res_obj["updated_memory"] = parsed["updated_memory"]
                                final_results[meta["uid"]] = res_obj
                            else:
                                if attempt < total_retries:
                                    next_pending_meta_list.append(meta)
                                else:
                                    print(
                                        f"[WARN] validation failed after {total_retries} attempts: "
                                        f"id={meta['ext']['id']}, errors={errors}"
                                    )
                                    failed_inputs.append({
                                        "id": meta["ext"]["id"],
                                        "user_id": meta["uid"],
                                        "session_num": current_session,
                                        "prompt": meta["prompt"],
                                        "past_memory": meta["past_memory"],
                                        "last_response": raw_response,
                                        "repaired_response": repaired_response,
                                        "errors": errors,
                                        "extracted_behavior": meta["ext"]["extracted_behavior"],
                                        "extracted_dialogue": meta["ext"]["extracted_dialogue"],
                                    })

                        if next_pending_meta_list:
                            decode_mode = "deterministic" if attempt <= args.det_retries else "sampling"
                            print(
                                f"[INFO] session {current_session}, batch {i // BATCH_SIZE + 1}, "
                                f"attempt {attempt}/{total_retries} ({decode_mode}): "
                                f"{len(next_pending_meta_list)} samples failed validation and will be retried"
                            )

                        pending_meta_list = next_pending_meta_list

                    # Save only successful results for this batch
                    with open(output_save_path, "a", encoding="utf-8") as f_out:
                        for uid in batch_uids:
                            if uid not in final_results:
                                continue
                            res_obj = final_results[uid]
                            user_session_logs[uid] = res_obj
                            f_out.write(json.dumps(res_obj, ensure_ascii=False) + "\n")
                        f_out.flush()

        if failed_inputs:
            with open(failed_input_save_path, "w", encoding="utf-8") as f:
                json.dump(failed_inputs, f, ensure_ascii=False, indent=2)
            print(f"[INFO] saved {len(failed_inputs)} failed inputs to {failed_input_save_path}")
        else:
            if os.path.exists(failed_input_save_path):
                os.remove(failed_input_save_path)
            print("[INFO] no failed update inputs")


if __name__ == '__main__':
    main()

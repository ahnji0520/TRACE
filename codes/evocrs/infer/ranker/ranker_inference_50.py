import os
import re
import sys
import json
import math
import string
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[4]

# Set the path to prompt.py
sys.path.append(str(SUPPLEMENTARY_ROOT / "codes/evocrs/preprocess/training"))
from prompt import *


DEFAULT_BASE_MODEL = "${HF_MODEL_DIR}/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_DIR = str(SUPPLEMENTARY_ROOT / "codes/evocrs/infer/ranker/output/reficr_candidates/train50_infer50")
# QWEN25_SYSTEM_PREFIX = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
QWEN25_SYSTEM_PREFIX = ""

def calculate_metrics(rank, k_list=[1, 5, 10, 20], max_rank=50):
    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = 1.0 if rank <= k else 0.0
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    metrics[f"mrr@{max_rank}"] = 1.0 / rank if rank <= max_rank else 0.0
    return metrics


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_to_str(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def get_prompt_map():
    return {
        "v13": (CRS_RANKING_LISTWISE_DEFAULT_SYSTEM, CRS_RANKING_LISTWISE_DEFAULT_USER),
        "v14": (CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_SYSTEM, CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_USER),
        "v15": (CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_SYSTEM, CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_USER),
        "v16": (CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_SYSTEM, CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_USER),
    }


def extract_session_idx(sample_id):
    return str(sample_id).split("_")[-1]


def parse_checkpoint_step(path_or_name):
    name = os.path.basename(os.path.normpath(path_or_name))
    match = re.search(r"checkpoint-(\d+)$", name)
    if not match:
        return None
    return int(match.group(1))


def is_qwen25_model(model_path):
    normalized = str(model_path).lower()
    return re.search(r"qwen[-_/]?(?:2\.5|25)", normalized) is not None


def apply_model_specific_system_prompt(sys_prompt, base_model_path):
    if not is_qwen25_model(base_model_path):
        return sys_prompt

    prefix = QWEN25_SYSTEM_PREFIX.strip()
    if sys_prompt.lstrip().startswith(prefix):
        return sys_prompt

    return f"{prefix}\n\n{sys_prompt}"


def load_adapter_config(lora_path):
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return {}
    return load_json(config_path)


def has_tokenizer_files(path):
    required_files = ["tokenizer_config.json", "tokenizer.json"]
    return all(os.path.exists(os.path.join(path, filename)) for filename in required_files)


def resolve_base_model_path(base_model_arg, lora_path):
    if base_model_arg and base_model_arg != "auto":
        return base_model_arg

    adapter_config = load_adapter_config(lora_path)
    base_model_path = adapter_config.get("base_model_name_or_path")
    if base_model_path:
        return base_model_path

    raise ValueError(
        "Could not resolve base model path. "
        "Pass --base_model explicitly or use a checkpoint with adapter_config.json."
    )


def resolve_tokenizer_path(base_model_path, lora_path, tokenizer_source):
    if tokenizer_source == "base_model":
        return base_model_path

    if tokenizer_source == "lora_path":
        if not has_tokenizer_files(lora_path):
            raise ValueError(
                f"--tokenizer_source=lora_path was requested, but tokenizer files were not found in: {lora_path}"
            )
        return lora_path

    if has_tokenizer_files(lora_path):
        return lora_path

    return base_model_path


def build_inference_jobs(args):
    if args.checkpoint_root:
        jobs = []
        checkpoint_root = os.path.abspath(args.checkpoint_root)

        if not os.path.isdir(checkpoint_root):
            raise ValueError(f"checkpoint_root does not exist: {checkpoint_root}")

        for name in os.listdir(checkpoint_root):
            full_path = os.path.join(checkpoint_root, name)
            if not os.path.isdir(full_path):
                continue

            step = parse_checkpoint_step(name)
            if step is None:
                continue

            if args.checkpoint_stride and step % args.checkpoint_stride != 0:
                continue
            if args.checkpoint_min is not None and step < args.checkpoint_min:
                continue
            if args.checkpoint_max is not None and step > args.checkpoint_max:
                continue

            jobs.append(
                {
                    "label": name,
                    "step": step,
                    "lora_path": full_path,
                }
            )

        jobs.sort(key=lambda job: job["step"])
        if not jobs:
            raise ValueError(
                "No checkpoints matched the given filters. "
                "Check --checkpoint_root / --checkpoint_stride / --checkpoint_min / --checkpoint_max."
            )
        return jobs

    if not args.lora_path:
        raise ValueError("Either --lora_path or --checkpoint_root must be provided.")

    lora_path = os.path.abspath(args.lora_path)
    return [
        {
            "label": os.path.basename(os.path.normpath(lora_path)),
            "step": parse_checkpoint_step(lora_path),
            "lora_path": lora_path,
        }
    ]


def enrich_job_with_paths(job, args):
    base_model_path = resolve_base_model_path(args.base_model, job["lora_path"])
    tokenizer_path = resolve_tokenizer_path(base_model_path, job["lora_path"], args.tokenizer_source)

    enriched = dict(job)
    enriched["base_model_path"] = base_model_path
    enriched["tokenizer_path"] = tokenizer_path
    return enriched


def get_letter_token_ids(tokenizer, num_letters=20, add_prefix_space_variants=True):
    letters = list(string.ascii_uppercase[:num_letters])
    token_map = {}

    for letter in letters:
        candidate_ids = []

        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) > 0:
            candidate_ids.append(ids[-1])

        if add_prefix_space_variants:
            for variant in [f" {letter}", f"\n{letter}", f"\n\n{letter}"]:
                v_ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(v_ids) > 0:
                    candidate_ids.append(v_ids[-1])

        candidate_ids = list(dict.fromkeys(candidate_ids))
        if len(candidate_ids) == 0:
            raise ValueError(f"Failed to tokenize letter: {letter}")

        token_map[letter] = candidate_ids

    return token_map


def score_items_with_logits(
    model,
    inputs,
    current_window,
    letter_token_map,
    alphabet_keys,
    return_trace=False,
):
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    scored_items = []
    letter_scores = []
    for i, item in enumerate(current_window):
        letter = alphabet_keys[i]
        candidate_token_ids = letter_token_map[letter]
        score = max(next_token_logits[token_id].item() for token_id in candidate_token_ids)
        scored_items.append((score, letter, item))
        letter_scores.append(
            {
                "letter": letter,
                "item_id": str(item.get("item_id")),
                "logit": score,
            }
        )

    scored_items = sorted(scored_items, key=lambda x: x[0], reverse=True)
    sorted_items = [x[2] for x in scored_items]

    if not return_trace:
        return sorted_items

    trace = {
        "letter_scores": letter_scores,
        "sorted_letters": [x[1] for x in scored_items],
        "sorted_item_ids": [str(x[2].get("item_id")) for x in scored_items],
        "sorted_logits": [x[0] for x in scored_items],
    }
    return sorted_items, trace


def rank_window_from_logits(next_token_logits, current_window, letter_token_map, alphabet_keys):
    scored_items = []
    letter_scores = []

    for i, item in enumerate(current_window):
        letter = alphabet_keys[i]
        candidate_token_ids = letter_token_map[letter]
        score = max(next_token_logits[token_id].item() for token_id in candidate_token_ids)
        scored_items.append((score, letter, item))
        letter_scores.append(
            {
                "letter": letter,
                "item_id": str(item.get("item_id")),
                "logit": score,
            }
        )

    scored_items = sorted(scored_items, key=lambda x: x[0], reverse=True)
    sorted_items = [x[2] for x in scored_items]
    trace = {
        "letter_scores": letter_scores,
        "sorted_letters": [x[1] for x in scored_items],
        "sorted_item_ids": [str(x[2].get("item_id")) for x in scored_items],
        "sorted_logits": [x[0] for x in scored_items],
    }
    return sorted_items, trace


def score_windows_with_logits_batch(
    model,
    tokenizer,
    device,
    prompt_texts,
    current_windows,
    letter_token_map,
    alphabet_keys,
):
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]

    return [
        rank_window_from_logits(
            next_token_logits=next_token_logits[row_idx],
            current_window=current_windows[row_idx],
            letter_token_map=letter_token_map,
            alphabet_keys=alphabet_keys,
        )
        for row_idx in range(len(current_windows))
    ]


def generate_raw_outputs_batch(
    model,
    tokenizer,
    inputs,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    stopping_criteria=None,
):
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(generated[row_idx, prompt_len:], skip_special_tokens=False).strip()
        for row_idx in range(generated.shape[0])
    ]


def parse_answer_tag(raw_text, valid_letters):
    text = raw_text.strip()
    predicted_letter = None
    answer_start_idx = None

    answer_match = re.search(
        r"<answer>\s*([A-Za-z])\s*</answer>",
        text,
        flags=re.IGNORECASE,
    )

    if answer_match:
        candidate = answer_match.group(1).upper()
        if candidate in valid_letters:
            predicted_letter = candidate
            answer_start_idx = answer_match.start(1)

    return {
        "raw_text": text,
        "predicted_letter": predicted_letter,
        "answer_start_idx": answer_start_idx,
    }


def parse_plain_answer(raw_text, valid_letters):
    text = raw_text.strip()
    search_text = text
    offset = 0

    think_end = re.search(r"</think>", text, flags=re.IGNORECASE)
    if think_end:
        offset = think_end.end()
        search_text = text[offset:]

    patterns = [
        r"(?:^|[^A-Za-z])(?:answer|index|final answer)\s*[:\-]?\s*([A-Za-z])(?=$|[^A-Za-z])",
        r"(?:^|[^A-Za-z])([A-Za-z])(?=$|[^A-Za-z])",
    ]

    matches = []
    for pattern in patterns:
        matches = list(re.finditer(pattern, search_text, flags=re.IGNORECASE))
        matches = [m for m in matches if m.group(1).upper() in valid_letters]
        if matches:
            break

    if not matches:
        return {
            "raw_text": text,
            "predicted_letter": None,
            "answer_start_idx": None,
        }

    match = matches[-1]
    return {
        "raw_text": text,
        "predicted_letter": match.group(1).upper(),
        "answer_start_idx": offset + match.start(1),
    }


class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_sequences):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            return False

        generated_ids = input_ids[0].tolist()
        for seq in self.stop_sequences:
            if len(generated_ids) >= len(seq) and generated_ids[-len(seq):] == seq:
                return True
        return False


def build_answer_tag_stopping_criteria(tokenizer):
    stop_text_variants = ["</answer>", " </answer>", "\n</answer>"]
    stop_sequences = []

    for text in stop_text_variants:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            stop_sequences.append(token_ids)

    if not stop_sequences:
        return None

    return StoppingCriteriaList([StopOnSubsequence(stop_sequences)])


def generate_raw_output(
    model,
    tokenizer,
    inputs,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    stopping_criteria=None,
):
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = generated[0, prompt_len:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    return gen_text.strip()


def score_items_at_answer_position(
    model,
    tokenizer,
    device,
    prompt_text,
    raw_text,
    answer_start_idx,
    current_window,
    letter_token_map,
    alphabet_keys,
    return_trace=False,
):
    answer_prefix_text = prompt_text + raw_text[:answer_start_idx]
    answer_inputs = tokenizer(answer_prefix_text, return_tensors="pt").to(device)
    return score_items_with_logits(
        model=model,
        inputs=answer_inputs,
        current_window=current_window,
        letter_token_map=letter_token_map,
        alphabet_keys=alphabet_keys,
        return_trace=return_trace,
    )


def compute_rank_from_finalized(target_id, finalized_lower_items, pending_higher_count):
    finalized_ids = [str(item.get("item_id")) for item in finalized_lower_items]
    if target_id not in finalized_ids:
        return None
    return pending_higher_count + finalized_ids.index(target_id) + 1


def build_compact_step_trace(step, current_window, sorted_items, trace, target_id, alphabet_keys, total_steps):
    window_item_ids = [str(item.get("item_id")) for item in current_window]
    sorted_item_ids = [str(item.get("item_id")) for item in sorted_items]

    target_info = {
        "in_window": False,
        "letter": None,
        "window_position": None,
        "sorted_rank": None,
        "logit": None,
        "survived_to_next_step": None,
    }

    if target_id in window_item_ids:
        target_position = window_item_ids.index(target_id)
        target_info["in_window"] = True
        target_info["letter"] = alphabet_keys[target_position]
        target_info["window_position"] = target_position

        if target_id in sorted_item_ids:
            target_info["sorted_rank"] = sorted_item_ids.index(target_id) + 1

        for score_row in trace["letter_scores"]:
            if score_row["item_id"] == target_id:
                target_info["logit"] = score_row["logit"]
                break

        if step < total_steps - 1 and target_info["sorted_rank"] is not None:
            target_info["survived_to_next_step"] = target_info["sorted_rank"] <= 10

    return {
        "step": step,
        "window_item_ids": window_item_ids,
        "sorted_item_ids": sorted_item_ids,
        "target": target_info,
    }


def summarize_metrics(global_metrics, session_metrics, total_ref):
    session_avg_results = {}
    for session_idx, metric_dict in sorted(session_metrics.items(), key=lambda x: int(x[0])):
        session_avg_results[f"Session_{session_idx}"] = {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in metric_dict.items()
        }

    global_avg = {}
    for key, values in global_metrics.items():
        global_avg[key] = sum(values) / total_ref if total_ref else 0.0

    return global_avg, session_avg_results


def get_run_output_dir(args, checkpoint_job, multiple_checkpoints):
    run_output_dir = args.output_dir
    if multiple_checkpoints:
        run_output_dir = os.path.join(args.output_dir, checkpoint_job["label"])
    return run_output_dir


def save_partial_progress(
    args,
    checkpoint_job,
    multiple_checkpoints,
    worker_rank,
    assigned_samples,
    processed_samples,
    total_eval_samples,
    true_fail_count,
    all_results,
    global_metrics,
    session_metrics,
    progress_dict,
    last_sample_id,
):
    run_output_dir = get_run_output_dir(args, checkpoint_job, multiple_checkpoints)
    os.makedirs(run_output_dir, exist_ok=True)

    partial_path = os.path.join(
        run_output_dir,
        f"partial_progress_{args.version}_{checkpoint_job['label']}_worker{worker_rank}.json",
    )

    global_avg, session_avg_results = summarize_metrics(
        global_metrics=global_metrics,
        session_metrics=session_metrics,
        total_ref=processed_samples,
    )
    global_processed_samples = sum(progress_dict.values())

    save_data = {
        "snapshot_type": "partial_worker_progress",
        "is_final_result": False,
        "version": args.version,
        "checkpoint": checkpoint_job["lora_path"],
        "checkpoint_label": checkpoint_job["label"],
        "checkpoint_step": checkpoint_job["step"],
        "base_model": checkpoint_job["base_model_path"],
        "tokenizer_path": checkpoint_job["tokenizer_path"],
        "worker_rank": worker_rank,
        "processed_samples": processed_samples,
        "assigned_samples": assigned_samples,
        "global_processed_samples": global_processed_samples,
        "total_eval_samples": total_eval_samples,
        "partial_save_every": args.partial_save_every,
        "last_updated_sample_id": last_sample_id,
        "true_fail_count_so_far": true_fail_count,
        "global_metrics_so_far": global_avg,
        "session_wise_metrics_so_far": session_avg_results,
        "details_so_far": all_results,
    }

    temp_path = f"{partial_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    os.replace(temp_path, partial_path)

    print(
        f"[Worker {worker_rank}] partial saved: "
        f"worker={processed_samples}/{assigned_samples} | "
        f"global={global_processed_samples}/{total_eval_samples} | "
        f"path={partial_path}",
        flush=True,
    )


def get_worker_result_path(args, checkpoint_job, multiple_checkpoints, worker_rank):
    run_output_dir = get_run_output_dir(args, checkpoint_job, multiple_checkpoints)
    return os.path.join(
        run_output_dir,
        f"final_worker_result_{args.version}_{checkpoint_job['label']}_worker{worker_rank}.json",
    )


def save_worker_result(
    args,
    checkpoint_job,
    multiple_checkpoints,
    worker_rank,
    assigned_samples,
    total_eval_samples,
    worker_result,
):
    result_path = get_worker_result_path(args, checkpoint_job, multiple_checkpoints, worker_rank)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    save_data = {
        "snapshot_type": "final_worker_result",
        "is_final_result": True,
        "version": args.version,
        "checkpoint": checkpoint_job["lora_path"],
        "checkpoint_label": checkpoint_job["label"],
        "checkpoint_step": checkpoint_job["step"],
        "base_model": checkpoint_job["base_model_path"],
        "tokenizer_path": checkpoint_job["tokenizer_path"],
        "worker_rank": worker_rank,
        "processed_samples": len(worker_result["all_results"]),
        "assigned_samples": assigned_samples,
        "total_eval_samples": total_eval_samples,
        "worker_result": worker_result,
    }
    temp_path = f"{result_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    os.replace(temp_path, result_path)
    print(
        f"[Worker {worker_rank}] final saved: "
        f"worker={save_data['processed_samples']}/{assigned_samples} | path={result_path}",
        flush=True,
    )


def load_worker_result(args, checkpoint_job, multiple_checkpoints, worker_rank):
    result_path = get_worker_result_path(args, checkpoint_job, multiple_checkpoints, worker_rank)
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("snapshot_type") != "final_worker_result":
        raise ValueError(f"Unexpected worker result snapshot type in {result_path}")
    return data["worker_result"]


def evaluate_samples(
    model,
    tokenizer,
    device,
    args,
    checkpoint_job,
    multiple_checkpoints,
    worker_rank,
    my_target_ids,
    total_eval_samples,
    test_data_dict,
    progress_dict,
):
    letter_token_map = get_letter_token_ids(tokenizer, num_letters=20)
    alphabet_keys = list(string.ascii_uppercase[:20])
    answer_tag_stopping = build_answer_tag_stopping_criteria(tokenizer)
    prompt_map = get_prompt_map()


    if args.version not in prompt_map:
        available_versions = ", ".join(sorted(prompt_map.keys()))
        raise ValueError(
            f"Unknown --version: {args.version}. Available versions: {available_versions}"
        )
    sys_prompt, user_template = prompt_map[args.version]
    sys_prompt = apply_model_specific_system_prompt(
        sys_prompt, checkpoint_job["base_model_path"]
    )

    all_results = []
    global_metrics = defaultdict(list)
    session_metrics = defaultdict(lambda: defaultdict(list))
    true_matching_fail_count = 0
    assigned_samples = len(my_target_ids)
    batch_size = max(1, args.batch_size)

    def update_progress_and_maybe_save(last_sample_id):
        processed_samples = len(all_results)
        progress_dict[worker_rank] = processed_samples

        if args.partial_save_every > 0 and processed_samples % args.partial_save_every == 0:
            save_partial_progress(
                args=args,
                checkpoint_job=checkpoint_job,
                multiple_checkpoints=multiple_checkpoints,
                worker_rank=worker_rank,
                assigned_samples=assigned_samples,
                processed_samples=processed_samples,
                total_eval_samples=total_eval_samples,
                true_fail_count=true_matching_fail_count,
                all_results=all_results,
                global_metrics=global_metrics,
                session_metrics=session_metrics,
                progress_dict=progress_dict,
                last_sample_id=last_sample_id,
            )

    def record_result(state):
        nonlocal true_matching_fail_count

        ranked_item_ids = [str(item["item_id"]) for item in state["final_ranked_items"]]
        if state["rank_override"] is not None:
            rank_val = state["rank_override"]
        else:
            try:
                rank_val = ranked_item_ids.index(state["target_id"]) + 1
            except ValueError:
                true_matching_fail_count += 1
                rank_val = 51

        metrics = calculate_metrics(rank_val)
        for key, value in metrics.items():
            global_metrics[key].append(value)
            session_metrics[state["session_idx"]][key].append(value)

        all_results.append(
            {
                "id": state["sample_id"],
                "session_idx": state["session_idx"],
                "target_id": state["target_id"],
                "rank": rank_val,
                "raw_text": state["sample_raw_texts"] if state["sample_raw_texts"] else None,
                "step_traces": state["step_traces"],
                "num_steps": state["num_steps"],
                "early_stopped": state["early_stopped"],
                "batch_size": batch_size,
                "status": "SUCCESS" if rank_val <= 50 else "FAIL_LOGIC_ERROR",
            }
        )
        update_progress_and_maybe_save(state["sample_id"])

    def initialize_state(sample_id):
        nonlocal true_matching_fail_count

        session_idx = extract_session_idx(sample_id)
        if sample_id not in test_data_dict:
            true_matching_fail_count += 1
            rank_val = 51
            metrics = calculate_metrics(rank_val)
            for key, value in metrics.items():
                global_metrics[key].append(value)
                session_metrics[session_idx][key].append(value)
            all_results.append({"id": sample_id, "status": "FAIL_DATA_MISSING", "rank": 51, "raw_text": None})
            update_progress_and_maybe_save(sample_id)
            return None

        sample = test_data_dict[sample_id]
        target_id = str(sample["target_id"])
        top50_items = sample.get("top50_items", [])
        top50_ids = [str(item["item_id"]) for item in top50_items]

        if target_id not in top50_ids:
            rank_val = 51
            metrics = calculate_metrics(rank_val)
            for key, value in metrics.items():
                global_metrics[key].append(value)
                session_metrics[session_idx][key].append(value)
            all_results.append({"id": sample_id, "status": "IGNORED_TARGET_ABSENCE", "rank": 51, "raw_text": None})
            update_progress_and_maybe_save(sample_id)
            return None

        num_steps = max(1, min(args.num_steps, 4))
        initial_start = (num_steps - 1) * 10
        initial_end = min(initial_start + 20, len(top50_items))
        final_ranked_items = top50_items[initial_end:]
        rank_override = compute_rank_from_finalized(
            target_id=target_id,
            finalized_lower_items=final_ranked_items,
            pending_higher_count=initial_end,
        )
        early_stopped = False
        step_traces = []

        if args.stop_when_target_absent and rank_override is not None:
            early_stopped = True
            step_traces.append(
                {
                    "step": None,
                    "early_stop_reason": "TARGET_OUTSIDE_EVALUATED_WINDOWS",
                    "pending_higher_count": initial_end,
                    "rank_override": rank_override,
                }
            )

        return {
            "sample_id": sample_id,
            "session_idx": session_idx,
            "sample": sample,
            "target_id": target_id,
            "top50_items": top50_items,
            "dialogue_history": sample.get("dialogue_history", ""),
            "behavioral_persona": safe_to_str(sample.get("behavioral_persona", {})),
            "dialogue_persona": safe_to_str(sample.get("dialogue_persona", [])),
            "num_steps": num_steps,
            "initial_start": initial_start,
            "initial_end": initial_end,
            "final_ranked_items": final_ranked_items,
            "sample_raw_texts": [],
            "step_traces": step_traces,
            "current_window": top50_items[initial_start:initial_end],
            "early_stopped": early_stopped,
            "rank_override": rank_override,
        }

    def build_prompt_for_state(state):
        candidate_dict = {
            alphabet_keys[i]: {k: v for k, v in item.items() if k != "item_id"}
            for i, item in enumerate(state["current_window"])
        }
        user_content = user_template.format(
            Dialogue_history=state["dialogue_history"],
            Movie_list=candidate_dict,
            Behavioral_persona=state["behavioral_persona"],
            Dialogue_persona=state["dialogue_persona"],
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )

    progress_bar = tqdm(total=len(my_target_ids), desc=f"GPU {device.index}", position=device.index)
    try:
        for batch_start in range(0, len(my_target_ids), batch_size):
            batch_ids = my_target_ids[batch_start : batch_start + batch_size]
            states = []
            for sample_id in batch_ids:
                state = initialize_state(sample_id)
                if state is None:
                    progress_bar.update(1)
                elif state["early_stopped"]:
                    record_result(state)
                    progress_bar.update(1)
                else:
                    states.append(state)

            if not states:
                continue

            for step in range(max(state["num_steps"] for state in states)):
                active_states = [
                    state
                    for state in states
                    if not state["early_stopped"] and step < state["num_steps"]
                ]
                if not active_states:
                    break

                prompt_texts = [build_prompt_for_state(state) for state in active_states]
                current_windows = [state["current_window"] for state in active_states]
                ranked_outputs = score_windows_with_logits_batch(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_texts=prompt_texts,
                    current_windows=current_windows,
                    letter_token_map=letter_token_map,
                    alphabet_keys=alphabet_keys,
                )
                sorted_items_list = [item for item, _ in ranked_outputs]
                trace_list = [trace for _, trace in ranked_outputs]

                if args.version in {"cot", "cot_qwen"} and (args.use_answer_tag or args.score_after_generation):
                    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
                    raw_texts = generate_raw_outputs_batch(
                        model=model,
                        tokenizer=tokenizer,
                        inputs=inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stopping_criteria=answer_tag_stopping if args.use_answer_tag else None,
                    )

                    prefix_rows = []
                    for row_idx, state in enumerate(active_states):
                        raw_text = raw_texts[row_idx]
                        state["sample_raw_texts"].append(raw_text)
                        valid_letters = set(alphabet_keys[: len(state["current_window"])])
                        parsed = (
                            parse_answer_tag(raw_text, valid_letters=valid_letters)
                            if args.use_answer_tag
                            else parse_plain_answer(raw_text, valid_letters=valid_letters)
                        )
                        if parsed["answer_start_idx"] is not None:
                            prefix_rows.append(
                                {
                                    "row_idx": row_idx,
                                    "prefix_text": prompt_texts[row_idx] + raw_text[: parsed["answer_start_idx"]],
                                    "predicted_letter": parsed["predicted_letter"],
                                }
                            )
                        else:
                            trace_list[row_idx]["answer_logit_source"] = "prompt_next_token_fallback"

                    if prefix_rows:
                        prefix_texts = [row["prefix_text"] for row in prefix_rows]
                        prefix_windows = [active_states[row["row_idx"]]["current_window"] for row in prefix_rows]
                        rescored_outputs = score_windows_with_logits_batch(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            prompt_texts=prefix_texts,
                            current_windows=prefix_windows,
                            letter_token_map=letter_token_map,
                            alphabet_keys=alphabet_keys,
                        )
                        for rescored_idx, row in enumerate(prefix_rows):
                            row_idx = row["row_idx"]
                            sorted_items_list[row_idx], trace_list[row_idx] = rescored_outputs[rescored_idx]
                            trace_list[row_idx]["answer_logit_source"] = "generated_answer_position"
                            trace_list[row_idx]["predicted_letter"] = row["predicted_letter"]

                for row_idx, state in enumerate(active_states):
                    sorted_items = sorted_items_list[row_idx]
                    trace = trace_list[row_idx]
                    state["step_traces"].append(
                        build_compact_step_trace(
                            step=step,
                            current_window=state["current_window"],
                            sorted_items=sorted_items,
                            trace=trace,
                            target_id=state["target_id"],
                            alphabet_keys=alphabet_keys,
                            total_steps=state["num_steps"],
                        )
                    )

                    if step < state["num_steps"] - 1:
                        winners = sorted_items[:10]
                        losers = sorted_items[10:]
                        state["final_ranked_items"] = losers + state["final_ranked_items"]
                        next_start = state["initial_start"] - ((step + 1) * 10)
                        next_end = next_start + 10
                        pending_higher_count = next_start + len(winners)
                        state["rank_override"] = compute_rank_from_finalized(
                            target_id=state["target_id"],
                            finalized_lower_items=state["final_ranked_items"],
                            pending_higher_count=pending_higher_count,
                        )

                        if args.stop_when_target_absent and state["rank_override"] is not None:
                            state["early_stopped"] = True
                            state["step_traces"][-1]["early_stop_reason"] = "TARGET_ELIMINATED_BEFORE_NEXT_WINDOW"
                            state["step_traces"][-1]["pending_higher_count"] = pending_higher_count
                            state["step_traces"][-1]["rank_override"] = state["rank_override"]
                        else:
                            state["current_window"] = state["top50_items"][next_start:next_end] + winners
                    else:
                        state["final_ranked_items"] = sorted_items + state["final_ranked_items"]

            for state in states:
                record_result(state)
                progress_bar.update(1)
    finally:
        progress_bar.close()

    progress_dict[worker_rank] = len(all_results)
    if len(all_results) > 0:
        save_partial_progress(
            args=args,
            checkpoint_job=checkpoint_job,
            multiple_checkpoints=multiple_checkpoints,
            worker_rank=worker_rank,
            assigned_samples=assigned_samples,
            processed_samples=len(all_results),
            total_eval_samples=total_eval_samples,
            true_fail_count=true_matching_fail_count,
            all_results=all_results,
            global_metrics=global_metrics,
            session_metrics=session_metrics,
            progress_dict=progress_dict,
            last_sample_id=all_results[-1]["id"],
        )

    return {
        "global_metrics": dict(global_metrics),
        "session_metrics": {key: dict(value) for key, value in session_metrics.items()},
        "true_fail_count": true_matching_fail_count,
        "all_results": all_results,
    }

def inference_worker(
    rank,
    world_size,
    args,
    checkpoint_job,
    multiple_checkpoints,
    target_id_list,
    test_data_dict,
    progress_dict,
):
    chunk_size = math.ceil(len(target_id_list) / world_size)
    start_idx = rank * chunk_size
    end_idx = min((rank + 1) * chunk_size, len(target_id_list))
    my_target_ids = target_id_list[start_idx:end_idx]

    device = torch.device(f"cuda:{rank}")

    print(
        f"[Worker {rank}] checkpoint={checkpoint_job['label']} | "
        f"base_model={checkpoint_job['base_model_path']} | "
        f"tokenizer={checkpoint_job['tokenizer_path']}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_job["tokenizer_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_job["base_model_path"],
        torch_dtype=torch.bfloat16,
        device_map={"": rank},
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_job["lora_path"])
    model.eval()

    worker_result = evaluate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args,
        checkpoint_job=checkpoint_job,
        multiple_checkpoints=multiple_checkpoints,
        worker_rank=rank,
        my_target_ids=my_target_ids,
        total_eval_samples=len(target_id_list),
        test_data_dict=test_data_dict,
        progress_dict=progress_dict,
    )
    save_worker_result(
        args=args,
        checkpoint_job=checkpoint_job,
        multiple_checkpoints=multiple_checkpoints,
        worker_rank=rank,
        assigned_samples=len(my_target_ids),
        total_eval_samples=len(target_id_list),
        worker_result=worker_result,
    )
    del model
    del base_model
    torch.cuda.empty_cache()


def merge_worker_results(worker_results, total_ref):
    merged_results = []
    merged_global_metrics = defaultdict(list)
    merged_session_metrics = defaultdict(lambda: defaultdict(list))
    total_true_fails = 0

    for worker_result in worker_results:
        merged_results.extend(worker_result["all_results"])
        total_true_fails += worker_result["true_fail_count"]

        for key, values in worker_result["global_metrics"].items():
            merged_global_metrics[key].extend(values)

        for session_idx, metric_dict in worker_result["session_metrics"].items():
            for key, values in metric_dict.items():
                merged_session_metrics[session_idx][key].extend(values)

    global_avg, session_avg_results = summarize_metrics(
        global_metrics=merged_global_metrics,
        session_metrics=merged_session_metrics,
        total_ref=total_ref,
    )

    return {
        "global_metrics": global_avg,
        "session_wise_metrics": session_avg_results,
        "details": merged_results,
        "true_fail_count": total_true_fails,
    }


def merge_worker_result_files(args, checkpoint_job, multiple_checkpoints, world_size, total_ref):
    worker_results = [
        load_worker_result(args, checkpoint_job, multiple_checkpoints, rank)
        for rank in range(world_size)
    ]
    return merge_worker_results(worker_results, total_ref)


def print_report(args, checkpoint_job, merged_result):
    global_metrics = merged_result["global_metrics"]
    session_metrics = merged_result["session_wise_metrics"]

    print("\n" + "=" * 72)
    print(f"[Inference Report] version={args.version} | checkpoint={checkpoint_job['label']}")
    print("-" * 72)
    for session_id, scores in session_metrics.items():
        print(
            f"[{session_id}] "
            f"R@1={scores.get('recall@1', 0.0):.4f} | "
            f"R@10={scores.get('recall@10', 0.0):.4f} | "
            f"NDCG@10={scores.get('ndcg@10', 0.0):.4f} | "
            f"NDCG@20={scores.get('ndcg@20', 0.0):.4f}"
        )
    print("-" * 72)
    print(
        f"Global: "
        f"R@1={global_metrics.get('recall@1', 0.0):.4f} | "
        f"R@5={global_metrics.get('recall@5', 0.0):.4f} | "
        f"R@10={global_metrics.get('recall@10', 0.0):.4f} | "
        f"R@20={global_metrics.get('recall@20', 0.0):.4f}"
    )
    print(
        f"        "
        f"NDCG@5={global_metrics.get('ndcg@5', 0.0):.4f} | "
        f"NDCG@10={global_metrics.get('ndcg@10', 0.0):.4f} | "
        f"NDCG@20={global_metrics.get('ndcg@20', 0.0):.4f} | "
        f"MRR@50={global_metrics.get('mrr@50', 0.0):.4f}"
    )
    print(f"True matching fails: {merged_result['true_fail_count']}")
    print("=" * 72 + "\n")


def save_result(args, checkpoint_job, merged_result, multiple_checkpoints):
    run_output_dir = get_run_output_dir(args, checkpoint_job, multiple_checkpoints)
    os.makedirs(run_output_dir, exist_ok=True)

    output_json = os.path.join(
        run_output_dir,
        f"inference_results_{args.version}_{checkpoint_job['label']}.json",
    )
    save_data = {
        "version": args.version,
        "checkpoint": checkpoint_job["lora_path"],
        "checkpoint_label": checkpoint_job["label"],
        "checkpoint_step": checkpoint_job["step"],
        "base_model": checkpoint_job["base_model_path"],
        "tokenizer_path": checkpoint_job["tokenizer_path"],
        "global_metrics": merged_result["global_metrics"],
        "session_wise_metrics": merged_result["session_wise_metrics"],
        "true_fail_count": merged_result["true_fail_count"],
        "details": merged_result["details"],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    print(f"Saved result: {output_json}")
    return output_json


def save_summary(args, summary_rows):
    summary_path = os.path.join(args.output_dir, f"checkpoint_summary_{args.version}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=4)
    print(f"Saved summary: {summary_path}")


def run_single_checkpoint(args, checkpoint_job, multiple_checkpoints, target_id_list, test_data_dict, world_size):
    manager = mp.Manager()
    progress_dict = manager.dict({rank: 0 for rank in range(world_size)})
    for rank in range(world_size):
        stale_result_path = get_worker_result_path(args, checkpoint_job, multiple_checkpoints, rank)
        if os.path.exists(stale_result_path):
            os.remove(stale_result_path)

    processes = [
        mp.Process(
            target=inference_worker,
            args=(
                i,
                world_size,
                args,
                checkpoint_job,
                multiple_checkpoints,
                target_id_list,
                test_data_dict,
                progress_dict,
            ),
        )
        for i in range(world_size)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    missing_result_ranks = []
    for rank in range(world_size):
        if not os.path.exists(get_worker_result_path(args, checkpoint_job, multiple_checkpoints, rank)):
            missing_result_ranks.append(rank)

    bad_exitcodes = {
        rank: process.exitcode
        for rank, process in enumerate(processes)
        if process.exitcode != 0
    }
    if bad_exitcodes and not missing_result_ranks:
        print(
            f"[WARN] Worker process exitcode(s) after final result save: {bad_exitcodes}. "
            "Merging saved worker result files anyway.",
            flush=True,
        )
    elif bad_exitcodes:
        raise RuntimeError(
            f"Inference worker exited abnormally for checkpoint {checkpoint_job['label']} "
            f"(exitcodes={bad_exitcodes}; missing final worker results={missing_result_ranks})"
        )

    return merge_worker_result_files(
        args=args,
        checkpoint_job=checkpoint_job,
        multiple_checkpoints=multiple_checkpoints,
        world_size=world_size,
        total_ref=len(target_id_list),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument("--checkpoint_stride", type=int, default=None)
    parser.add_argument("--checkpoint_min", type=int, default=None)
    parser.add_argument("--checkpoint_max", type=int, default=None)
    parser.add_argument(
        "--tokenizer_source",
        type=str,
        default="auto",
        choices=["auto", "base_model", "lora_path"],
    )
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--extracted_ids", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", type=str, default="cot")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_answer_tag", action="store_true")
    parser.add_argument(
        "--score_after_generation",
        action="store_true",
        help="Generate CoT first, then compute candidate logits at the generated final-answer letter position.",
    )
    parser.add_argument("--num_steps", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--stop_when_target_absent",
        action="store_true",
        help="Evaluation-only oracle early stop once the known target cannot appear in the next window.",
    )
    parser.add_argument("--partial_save_every", type=int, default=50)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in chat templates. Default is disabled.",
    )
    args = parser.parse_args()
    if args.use_answer_tag and args.batch_size > 1:
        raise ValueError("--use_answer_tag currently supports only --batch_size 1.")

    os.makedirs(args.output_dir, exist_ok=True)

    target_id_list = load_json(args.extracted_ids)

    raw_test_data = load_json(args.test_file)
    test_data_dict = {str(item["id"]): item for item in raw_test_data}

    if args.max_samples is not None:
        target_id_list = target_id_list[: args.max_samples]

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices were detected.")

    checkpoint_jobs = [enrich_job_with_paths(job, args) for job in build_inference_jobs(args)]
    multiple_checkpoints = len(checkpoint_jobs) > 1

    print(f"[Main] version={args.version}")
    print(f"[Main] total_eval_samples={len(target_id_list)}")
    print(f"[Main] checkpoint_count={len(checkpoint_jobs)}")
    print(f"[Main] world_size={world_size}")

    mp.set_start_method("spawn", force=True)

    summary_rows = []
    for job in checkpoint_jobs:
        print(
            f"[Main] starting checkpoint={job['label']} | "
            f"step={job['step']} | "
            f"lora={job['lora_path']}",
            flush=True,
        )

        merged_result = run_single_checkpoint(
            args=args,
            checkpoint_job=job,
            multiple_checkpoints=multiple_checkpoints,
            target_id_list=target_id_list,
            test_data_dict=test_data_dict,
            world_size=world_size,
        )
        print_report(args, job, merged_result)
        output_json = save_result(args, job, merged_result, multiple_checkpoints)

        summary_rows.append(
            {
                "checkpoint": job["lora_path"],
                "checkpoint_label": job["label"],
                "checkpoint_step": job["step"],
                "base_model": job["base_model_path"],
                "tokenizer_path": job["tokenizer_path"],
                "output_json": output_json,
                **merged_result["global_metrics"],
            }
        )

    if multiple_checkpoints:
        save_summary(args, summary_rows)


if __name__ == "__main__":
    main()

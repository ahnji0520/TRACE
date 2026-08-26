#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_BASE_MODEL = "${HF_MODEL_DIR}/Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run response-generator LoRA inference on target-response test data.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_processed_ids(path: str) -> set[str]:
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
            except json.JSONDecodeError:
                continue
            if "id" in row:
                processed_ids.add(str(row["id"]))
    return processed_ids


def get_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_model_and_tokenizer(base_model: str, lora_path: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": {"": 0} if torch.cuda.is_available() else {"": "cpu"},
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return tokenizer, model


def split_example(example: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = example.get("messages", [])
    if not messages:
        raise ValueError(f"Example has no messages: {example.get('id')}")

    if messages[-1].get("role") == "assistant":
        return messages[:-1], str(messages[-1].get("content", "")).strip()

    prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
    return prompt_messages, ""


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(assistant|agent)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    tokenizer,
    model,
    batch: list[dict[str, Any]],
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
) -> list[str]:
    prompts = []
    for item in batch:
        prompt_messages, _ = split_example(item)
        prompts.append(build_prompt(tokenizer, prompt_messages))

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(get_input_device(model))

    do_sample = temperature > 0.0
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_beams": num_beams,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    response_ids = outputs[:, inputs.input_ids.shape[1]:]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    return [clean_generation(response) for response in responses]


def main() -> None:
    args = parse_args()

    data = load_json(args.input_file)
    shard = [item for idx, item in enumerate(data) if idx % args.num_shards == args.shard_idx]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    processed_ids = load_processed_ids(args.output_file) if args.resume else set()
    if not args.resume:
        with open(args.output_file, "w", encoding="utf-8"):
            pass
    if processed_ids:
        shard = [item for item in shard if str(item.get("id")) not in processed_ids]

    print(
        f"[INFO] shard={args.shard_idx}/{args.num_shards}, "
        f"items={len(shard)}, input={args.input_file}, output={args.output_file}"
    )
    print(f"[INFO] base_model={args.base_model}")
    print(f"[INFO] lora_path={args.lora_path}")

    tokenizer, model = load_model_and_tokenizer(
        base_model=args.base_model,
        lora_path=args.lora_path,
        load_in_4bit=not args.no_4bit,
    )

    with open(args.output_file, "a", encoding="utf-8") as out:
        for start in tqdm(range(0, len(shard), args.batch_size), desc="Generating responses"):
            batch = shard[start:start + args.batch_size]
            generations = generate_batch(
                tokenizer=tokenizer,
                model=model,
                batch=batch,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )

            for item, generated_response in zip(batch, generations):
                prompt_messages, gold_response = split_example(item)
                row = {
                    "id": item.get("id"),
                    "generated_response": generated_response,
                    "gold_response": gold_response,
                    "messages": prompt_messages,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    print(f"[INFO] wrote {args.output_file}")


if __name__ == "__main__":
    main()

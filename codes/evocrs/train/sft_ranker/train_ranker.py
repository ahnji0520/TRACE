import os
import argparse
import json
import torch
import torch.nn as nn
import string
import re
import wandb
from tqdm import tqdm

import deepspeed
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from utils import get_train_model, get_checkpoint_model


def _is_qwen3_tokenizer(tokenizer):
    tokenizer_name = getattr(tokenizer, "name_or_path", "") or ""
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "qwen3" in tokenizer_name.lower() or "enable_thinking" in chat_template


def apply_chat_template(tokenizer, messages, enable_thinking=False, **kwargs):
    template_kwargs = dict(kwargs)
    if _is_qwen3_tokenizer(tokenizer):
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **template_kwargs)

# ------------------------------------------------------------------
# 0. DeepSpeed Configuration
# ------------------------------------------------------------------
def get_ds_config(args, world_size):
    ds_config = {
        "train_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": { "enabled": True },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }
    return ds_config

# ------------------------------------------------------------------
# 1. Ranking loss (kept)
# ------------------------------------------------------------------
def calc_ranking_loss(candidate_ids_tensor, sliced_logits, gt_token_ids):
    if sliced_logits.size(0) == 0:
        return torch.tensor(0.0, device=sliced_logits.device, requires_grad=True)

    s_i = sliced_logits.gather(1, gt_token_ids.unsqueeze(1))
    candidate_logits = sliced_logits[:, candidate_ids_tensor]
    diff = candidate_logits - s_i 
    loss = torch.nn.functional.softplus(diff).mean()
    return loss

# ------------------------------------------------------------------
# 2. Dataset (updated version)
# ------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len, enable_thinking=False, chat_mode="multi_turn"):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        self.chat_mode = chat_mode
        self.eos_id = tokenizer.eos_token_id
        self.reasoning_marker_ids = tokenizer(
            "Based on the given information, generate the step-by-step reasoning.",
            add_special_tokens=False
        )["input_ids"]
        self.index_marker_ids = tokenizer(
            "Based on the step-by-step reasoning, generate the index.",
            add_special_tokens=False
        )["input_ids"]
        self.rank_candidate_token_ids = set(
            tokenizer.encode(c, add_special_tokens=False)[0]
            for c in string.ascii_uppercase[:20]
        )
        
    def __len__(self): 
        return len(self.data)

    @staticmethod
    def _find_subseq(seq, pattern):
        if not pattern or len(seq) < len(pattern):
            return -1
        for i in range(len(seq) - len(pattern) + 1):
            if seq[i:i+len(pattern)] == pattern:
                return i

        # Some tokenizers split the same text differently depending on the
        # preceding newline/space. Keep the old tolerant behavior as fallback.
        if len(pattern) >= 2:
            search_pattern = pattern[1:]
            for i in range(len(seq) - len(search_pattern) + 1):
                if seq[i:i+len(search_pattern)] == search_pattern:
                    return max(i - 1, 0)
        return -1

    @staticmethod
    def _strip_answer_block(answer_text):
        marker = "<answer>"
        end_marker = "</answer>"
        if marker not in answer_text:
            return answer_text

        start = answer_text.find(marker) + len(marker)
        end = answer_text.find(end_marker, start)
        if end == -1:
            return answer_text.replace(marker, "").strip()

        before = answer_text[:start - len(marker)].rstrip()
        answer = answer_text[start:end].strip()
        after = answer_text[end + len(end_marker):].strip()
        parts = [part for part in (before, answer, after) if part]
        return "\n\n".join(parts)

    def _build_single_turn_thinking_sample(self, sample):
        assistant_positions = [
            i for i, message in enumerate(sample)
            if message.get("role") == "assistant"
        ]

        if len(assistant_positions) >= 2:
            reasoning_pos = assistant_positions[0]
            answer_pos = assistant_positions[-1]
            prompt_messages = sample[:reasoning_pos]
            reasoning_text = (sample[reasoning_pos].get("content") or "").strip()
            answer_text = (sample[answer_pos].get("content") or "").strip()
        else:
            prompt_messages = sample[:-1]
            answer_text = (sample[-1].get("content") or "").strip()
            reasoning_text = ""

        if "<answer>" in answer_text:
            answer_text = self._strip_answer_block(answer_text)

        if "<think>" in answer_text:
            return prompt_messages, answer_text

        if reasoning_text:
            answer_text = f"<think>\n{reasoning_text}\n</think>\n\n{answer_text}"

        return prompt_messages, answer_text

    def _build_sample(self, sample):
        if self.chat_mode == "single_turn_thinking":
            return self._build_single_turn_thinking_sample(sample)

        answer_text = (sample[-1].get("content") or "").strip()
        prompt_messages = sample[:-1]
        return prompt_messages, answer_text

    def _find_candidate_pos(self, input_ids, start_idx):
        for pos in range(max(start_idx, 0), len(input_ids)):
            if input_ids[pos] in self.rank_candidate_token_ids:
                return pos
        return -1

    def _find_last_candidate_pos(self, input_ids, start_idx):
        for pos in range(len(input_ids) - 1, max(start_idx, 0) - 1, -1):
            if input_ids[pos] in self.rank_candidate_token_ids:
                return pos
        return -1

    def __getitem__(self, index):
        sample = self.data[index]
        prompt_messages, answer_text = self._build_sample(sample)
        
        prompt_ids = apply_chat_template(
            self.tokenizer,
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        ans_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        
        input_ids = prompt_ids + ans_ids + ([self.eos_id] if self.eos_id is not None else [])
        
        labels = [-100] * len(input_ids)
        rank_target_pos = -1

        reasoning_pos = self._find_subseq(input_ids, self.reasoning_marker_ids)
        index_pos = self._find_subseq(input_ids, self.index_marker_ids)

        if self.chat_mode == "single_turn_thinking":
            answer_start_idx = len(prompt_ids)
            labels[answer_start_idx:] = input_ids[answer_start_idx:]
            rank_target_pos = self._find_last_candidate_pos(input_ids, answer_start_idx)
        elif reasoning_pos != -1 and index_pos != -1:
            sft_start_idx = reasoning_pos + len(self.reasoning_marker_ids)
            labels[sft_start_idx:] = input_ids[sft_start_idx:]

            rank_target_pos = self._find_candidate_pos(
                input_ids,
                index_pos + len(self.index_marker_ids)
            )
        else:
            answer_start_idx = len(prompt_ids)
            rank_target_pos = answer_start_idx if ans_ids else -1
            labels[answer_start_idx:] = input_ids[answer_start_idx:]
         
        # Truncate when max length is exceeded
        if len(input_ids) > self.max_seq_len:
            overflow = len(input_ids) - self.max_seq_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
            if rank_target_pos != -1:
                rank_target_pos -= overflow
                if rank_target_pos <= 0 or rank_target_pos >= len(labels):
                    rank_target_pos = -1
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "rank_target_pos": rank_target_pos,
        }


def extract_rank_targets(batch, outputs, candidate_ids, device):
    sliced_logits_list = []
    gt_token_ids_list = []

    rank_target_pos = batch.get("rank_target_pos", None)

    for b in range(batch["labels"].size(0)):
        idx = -1

        if rank_target_pos is not None:
            pos = rank_target_pos[b].item()
            if pos > 0 and pos < batch["labels"].size(1):
                label_token = batch["labels"][b, pos].item()
                if label_token in candidate_ids.tolist():
                    idx = pos

        if idx == -1:
            mask = torch.isin(batch["labels"][b], candidate_ids)
            target_pos = torch.where(mask)[0]
            if len(target_pos) > 0:
                idx = target_pos[-1].item()

        if idx > 0:
            sliced_logits_list.append(outputs.logits[b, idx-1:idx, :])
            gt_token_ids_list.append(batch["labels"][b, idx])

    if len(sliced_logits_list) > 0:
        return torch.cat(sliced_logits_list, 0).squeeze(1), torch.stack(gt_token_ids_list)

    return None, None
# ------------------------------------------------------------------
# Validation function (kept)
# ------------------------------------------------------------------
def validate(model_engine, dataloader, device, candidate_ids, args):
    model_engine.eval()
    total_ce_loss = 0
    total_rank_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            ce_loss = outputs.loss
      
            sliced_logits, gt_token_ids = extract_rank_targets(batch, outputs, candidate_ids, device)

            if sliced_logits is not None:
                rank_loss = calc_ranking_loss(candidate_ids, sliced_logits, gt_token_ids)
            else:
                rank_loss = torch.tensor(0.0, device=device)

            total_ce_loss += ce_loss.detach()
            total_rank_loss += rank_loss.detach()
            total_steps += 1

    dist.all_reduce(total_ce_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_rank_loss, op=dist.ReduceOp.SUM)
    
    avg_ce = total_ce_loss.item() / (total_steps * dist.get_world_size())
    avg_rank = total_rank_loss.item() / (total_steps * dist.get_world_size())
    
    return avg_ce, avg_rank


def _checkpoint_step_from_tag(tag):
    if not tag:
        return 0
    match = re.search(r"checkpoint-(\d+)$", tag)
    return int(match.group(1)) if match else 0


def _resolve_deepspeed_checkpoint(checkpoint_path):
    checkpoint_path = os.path.abspath(checkpoint_path)
    tag = os.path.basename(os.path.normpath(checkpoint_path))
    if tag.startswith("checkpoint-"):
        return os.path.dirname(checkpoint_path), tag
    return checkpoint_path, None


def _load_training_checkpoint(model_engine, args, global_rank):
    if not args.resume_from_checkpoint:
        return 0, 0, -1

    load_dir, tag = _resolve_deepspeed_checkpoint(args.resume_from_checkpoint)
    try:
        load_result = model_engine.load_checkpoint(load_dir, tag=tag, load_module_strict=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DeepSpeed state from {args.resume_from_checkpoint}. "
            "Use --checkpoint_path for adapter-only warm-start checkpoints."
        ) from exc

    if load_result is None:
        raise RuntimeError(
            f"No DeepSpeed checkpoint state found at {args.resume_from_checkpoint}. "
            "Use --checkpoint_path for adapter-only warm-start checkpoints."
        )

    _, client_state = load_result
    client_state = client_state or {}
    total_step = int(client_state.get("global_step", _checkpoint_step_from_tag(tag)))
    start_epoch = int(client_state.get("epoch", 0))
    resume_step_in_epoch = int(client_state.get("step_in_epoch", -1))

    if global_rank == 0:
        print(
            f"[Resume] Loaded {args.resume_from_checkpoint}: "
            f"total_step={total_step}, epoch={start_epoch}, "
            f"step_in_epoch={resume_step_in_epoch}"
        )

    return total_step, start_epoch, resume_step_in_epoch


def _save_training_checkpoint(args, model_engine, tokenizer, tag, client_state, global_rank):
    if not args.output_dir:
        return

    checkpoint_dir = os.path.join(args.output_dir, tag)
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.use_lora:
            model_engine.module.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"--- Saved checkpoint at {checkpoint_dir} ---")

    if dist.is_initialized():
        dist.barrier()
    model_engine.save_checkpoint(args.output_dir, tag=tag, client_state=client_state)
    if dist.is_initialized():
        dist.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--ranking_loss_weight", type=float, default=10.0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--use_lora", action="store_true") 
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="TRACE-DS-SFT")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="DeepSpeed checkpoint dir/tag for exact resume")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in chat templates. Default is disabled."
    )
    parser.add_argument(
        "--chat_mode",
        choices=["auto", "multi_turn", "single_turn_thinking"],
        default="auto",
        help=(
            "SFT conversation format. auto uses single_turn_thinking when "
            "--thinking is set, otherwise keeps the original multi_turn format."
        )
    )

    args = parser.parse_args()
    if args.chat_mode == "auto":
        args.chat_mode = "single_turn_thinking" if args.thinking else "multi_turn"

    if args.local_rank == -1:
         args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Set CUDA device
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # Initialize the process group if doing so explicitly to get rank information before DeepSpeed handles it
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # WandB
    if global_rank == 0 and args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # 2. Model Load

    adapter_checkpoint = args.resume_from_checkpoint or args.checkpoint_path
    if adapter_checkpoint and args.use_lora:
        model, tokenizer = get_checkpoint_model(args.model_name_or_path, adapter_checkpoint, thinking=args.thinking)
    else:
        model, tokenizer = get_train_model(args.model_name_or_path, 'ranking', thinking=args.thinking)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False 
    model.enable_input_require_grads()

    # 3. Dataset
    raw_train = json.load(open(args.train_file, 'r'))
    raw_train = [i['messages'] for i in raw_train]
    train_dataset = SFTDataset(raw_train, tokenizer, args.max_seq_len, args.thinking, args.chat_mode)
    
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=DistributedSampler(train_dataset, shuffle=True),
        batch_size=args.per_device_train_batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)
    )

    if args.validation_file:
        raw_val = json.load(open(args.validation_file, 'r'))
        raw_val = [i['messages'] for i in raw_val]

        val_dataset = SFTDataset(raw_val, tokenizer, args.max_seq_len, args.thinking, args.chat_mode)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset, shuffle=False),
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)
        )
    else:
        val_dataloader = None

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_update_steps = (len(train_dataloader) * args.num_train_epochs) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_update_steps)

    ds_config = get_ds_config(args, world_size)
    
    # DeepSpeed Init
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config 
    )

    total_step, start_epoch, resume_step_in_epoch = _load_training_checkpoint(model_engine, args, global_rank)

    candidate_chars = list(string.ascii_uppercase)[:20]
    candidate_ids = torch.tensor([tokenizer.encode(c, add_special_tokens=False)[0] for c in candidate_chars], device=device)

    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_estimated_steps = num_update_steps_per_epoch * args.num_train_epochs
    
    if global_rank == 0:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {total_estimated_steps}")
        print(f"  Eval/Save steps = {args.eval_steps} / {args.save_steps}")    
        print(f"  Chat mode = {args.chat_mode}")


    if global_rank == 0:
        print(f"***** Training Start *****")
        print(f"  Total epochs: {args.num_train_epochs}")
        print(f"  Batch size per device: {args.per_device_train_batch_size}")
        print(f"  Save/Eval every {args.save_steps} / {args.eval_steps} MICRO steps (Raw Iterations)")

    for epoch in range(start_epoch, args.num_train_epochs):
        model_engine.train()
        train_dataloader.sampler.set_epoch(epoch)
        
        iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}") if global_rank == 0 else train_dataloader

        for step, batch in enumerate(iterator):
            if epoch == start_epoch and step <= resume_step_in_epoch:
                continue

            total_step += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            ce_loss = outputs.loss

            sliced_logits, gt_token_ids = extract_rank_targets(batch, outputs, candidate_ids, device)

            if sliced_logits is not None:
                avg_rank_loss = calc_ranking_loss(candidate_ids, sliced_logits, gt_token_ids)
            else:
                avg_rank_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            total_loss = ce_loss + (args.ranking_loss_weight * avg_rank_loss)

            model_engine.backward(total_loss)
            model_engine.step()
            
            if global_rank == 0:
                iterator.set_postfix({
                    "step": total_step,      
                    "total": f"{total_loss.item():.4f}",
                    "ce_loss": f"{ce_loss.item():.4f}",
                    "rank": f"{avg_rank_loss.item():.4f}"
                })

                # WandB logging; keep existing keys
                if args.wandb_api_key and total_step % 10 == 0: # Use total_step here as well
                     wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/rank_loss": avg_rank_loss.item(),
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "step": total_step 
                    })

            if args.validation_file and total_step > 0 and total_step % args.eval_steps == 0:
                val_ce, val_rank = validate(model_engine, val_dataloader, device, candidate_ids, args)
                model_engine.train()
                
                if global_rank == 0:
                    print(f"\n[Step {total_step}] Val CE: {val_ce:.4f}, Rank: {val_rank:.4f}")
                    if args.wandb_api_key:
                        wandb.log({
                            "val/ce_loss": val_ce,
                            "val/rank_loss": val_rank,
                            "step": total_step
                        })

            if total_step > 0 and total_step % args.save_steps == 0:
                tag = f"checkpoint-{total_step}"
                _save_training_checkpoint(
                    args,
                    model_engine,
                    tokenizer,
                    tag,
                    {
                        "global_step": total_step,
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "chat_mode": args.chat_mode,
                        "max_seq_len": args.max_seq_len,
                    },
                    global_rank,
                )

    # cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

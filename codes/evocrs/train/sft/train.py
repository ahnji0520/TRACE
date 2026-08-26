import os
import argparse
import json
import math
import re
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import DataCollatorForSeq2Seq
import wandb
import deepspeed
from tqdm import tqdm
from utils import get_train_model, get_checkpoint_model


import os
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["NCCL_SHM_DISABLE"] = "1"       # Disable shared-memory communication and force P2P communication
os.environ["NCCL_CUMEM_HOST_ENABLE"] = "0" # Prevent host-memory allocation errors as recommended by NCCL logs

def _is_qwen3_tokenizer(tokenizer):
    tokenizer_name = getattr(tokenizer, "name_or_path", "") or ""
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "qwen3" in tokenizer_name.lower() or "enable_thinking" in chat_template


def apply_chat_template(tokenizer, messages, enable_thinking=False, **kwargs):
    template_kwargs = dict(kwargs)
    if _is_qwen3_tokenizer(tokenizer):
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def scan_dataset_length(data_list, tokenizer, split_name="Train", enable_thinking=False):
    """
    Iterate over the full chat dataset, measure token lengths used for training, and return the maximum length.
    """
    max_len = 0
    
    # Configure only the main process (rank 0) to print logs in multi-GPU environments
    is_main_process = True
    if torch.distributed.is_initialized():
        is_main_process = (torch.distributed.get_rank() == 0)
    
    if is_main_process:
        print(f"[{split_name}] Scanning dataset for max sequence length...")
    
    # Show tqdm progress bars only on the main process
    iterator = tqdm(data_list, desc=f"Scanning {split_name}", disable=not is_main_process)
    
    # No gradient computation needed; prevents memory leaks
    with torch.no_grad():
        for conversation in iterator:
            # conversation is a list like [{"role": "system", "content": "..."}, ...]
            
            # apply_chat_template includes all special tokens from the model template
            # so the exact input length for the model can be measured.
            token_ids = apply_chat_template(
                tokenizer,
                conversation["messages"], # Updated here: extract only the messages list from the dictionary
                tokenize=True, 
                add_generation_prompt=False,
                enable_thinking=enable_thinking
            )
            
            current_len = len(token_ids)
            if current_len > max_len:
                max_len = current_len
                
    return max_len


def evaluate(model_engine, eval_dataloader, device):
    """
    Evaluate the validation dataset and return the average loss.
    Average loss across GPUs for multi-GPU training.
    """
    model_engine.eval() # Switch to evaluation mode and disable dropout
    total_loss = 0.0
    num_steps = 0
    
    iterator = tqdm(eval_dataloader, desc="Evaluating", disable=torch.distributed.get_rank() != 0)
    
    with torch.no_grad(): # Disable gradient computation to save memory
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward Pass
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            # Gather loss across GPUs; optional but recommended for accuracy
            # Collect the current batch loss from all GPUs and average it
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            
            total_loss += loss.item()
            num_steps += 1
            
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss

# --- [Dataset] Apply chat template and multi-turn masking ---
class ChatSFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len, enable_thinking=False):
        self.data = data_list 
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        conversation = self.data[index]["messages"] # Updated here: fetch only messages
        
        # 1. Tokenize the full conversation
        full_input_ids = apply_chat_template(
            self.tokenizer,
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
            return_tensors="pt"
        ).squeeze(0)

        # 2. Initialize labels for masking
        target_labels = full_input_ids.clone()
        target_labels[:] = -100 

        # 3. Find assistant turns and unmask them
        for i, msg in enumerate(conversation):
            if msg['role'] == 'assistant':
                conv_current = conversation[:i+1]
                ids_current = apply_chat_template(
                    self.tokenizer,
                    conv_current,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=self.enable_thinking
                )
                end_idx = len(ids_current)

                if i == 0:
                    start_idx = 0
                else:
                    conv_previous = conversation[:i]
                    ids_previous = apply_chat_template(
                        self.tokenizer,
                        conv_previous,
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=self.enable_thinking
                    )
                    start_idx = len(ids_previous)
                
                if end_idx <= len(full_input_ids):
                    target_labels[start_idx:end_idx] = full_input_ids[start_idx:end_idx]

        # 4. Padding & Truncation
        if len(full_input_ids) > self.max_seq_len:
            input_ids = full_input_ids[:self.max_seq_len]
            labels = target_labels[:self.max_seq_len]
            attention_mask = torch.ones_like(input_ids)
        else:
            pad_len = self.max_seq_len - len(full_input_ids)
            input_ids = torch.cat([full_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            labels = torch.cat([target_labels, torch.full((pad_len,), -100, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones_like(full_input_ids), torch.zeros((pad_len,), dtype=torch.long)])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def _load_training_checkpoint(model_engine, args):
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
    global_step = int(client_state.get("global_step", _checkpoint_step_from_tag(tag)))
    start_epoch = int(client_state.get("epoch", 0))
    resume_step_in_epoch = int(client_state.get("step_in_epoch", -1))

    if args.global_rank == 0:
        print(
            f"[Resume] Loaded {args.resume_from_checkpoint}: "
            f"global_step={global_step}, epoch={start_epoch}, "
            f"step_in_epoch={resume_step_in_epoch}"
        )

    return global_step, start_epoch, resume_step_in_epoch


def _save_training_checkpoint(args, model_engine, tokenizer, tag, client_state):
    if not args.output_dir:
        return

    checkpoint_dir = os.path.join(args.output_dir, tag)
    if args.global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.use_lora:
            if hasattr(model_engine.module, "save_pretrained"):
                model_engine.module.save_pretrained(checkpoint_dir)
            else:
                model_engine.module.base_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saving checkpoint to {checkpoint_dir}")

    if dist.is_initialized():
        dist.barrier()
    model_engine.save_checkpoint(args.output_dir, tag=tag, client_state=client_state)
    if dist.is_initialized():
        dist.barrier()

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Chat SFT Training")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="LoRA adapter checkpoint for warm-start training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="DeepSpeed checkpoint dir/tag for exact resume")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200, help="Step interval for validation")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="chat-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--task", type=str, default='nothing')
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in chat templates. Default is disabled for SFT."
    )
    
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def get_ds_config(args):
    ds_config = {
        "train_batch_size": args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": {"enabled": False}, 
        "bf16": {"enabled": True},  
        "zero_optimization": {"stage": args.zero_stage},
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.num_train_epochs * 1000, # Initial placeholder
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": 100 
            }
        }
    }
    return ds_config

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()
    torch.manual_seed(args.seed)

    if args.global_rank == 0 and args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    adapter_checkpoint = args.resume_from_checkpoint or args.checkpoint_path
    if adapter_checkpoint and args.use_lora:
        model, tokenizer = get_checkpoint_model(args.model_name_or_path, adapter_checkpoint)
    else:
        model, tokenizer = get_train_model(args.model_name_or_path, args.task)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if args.global_rank == 0:
        print(f"Tokenizer Pad Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        print("Loading Data...")
    
    raw_train_data = load_json(args.train_file)
    raw_val_data = load_json(args.validation_file) if args.validation_file else []

    if args.global_rank == 0:
        print("Scanning dataset for Chat Template Lengths...")
    
    train_max = scan_dataset_length(raw_train_data, tokenizer, "Train", args.enable_thinking)
    val_max = scan_dataset_length(raw_val_data, tokenizer, "Val", args.enable_thinking) if raw_val_data else 0
    
    found_max = max(train_max, val_max)
    final_seq_len = min(found_max, args.max_seq_len)
    
    if final_seq_len % 8 != 0:
        final_seq_len = ((final_seq_len // 8) + 1) * 8
        
    if args.global_rank == 0:
        print(f"\n[Result] Found Max Token Length: {found_max}")
        print(f"[Config] User Hard Limit: {args.max_seq_len}")
        print(f"[Final] Setting Max Sequence Length to: {final_seq_len}\n")
    
    args.max_seq_len = final_seq_len

    train_dataset = ChatSFTDataset(raw_train_data, tokenizer, args.max_seq_len, args.enable_thinking)
    val_dataset = ChatSFTDataset(raw_val_data, tokenizer, args.max_seq_len, args.enable_thinking) if raw_val_data else None

    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)

    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator)
    else:
        val_dataloader = None

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_train_epochs
    ds_config = get_ds_config(args)
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), config=ds_config
    )
    
    global_step, start_epoch, resume_step_in_epoch = _load_training_checkpoint(model_engine, args)
    if args.global_rank == 0:
        print("Starting Chat SFT Training...")

    for epoch in range(start_epoch, args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        
        iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}") if args.global_rank == 0 else train_dataloader
        
        for step, batch in enumerate(iterator):
            if epoch == start_epoch and step <= resume_step_in_epoch:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model_engine(**batch)
            loss = outputs.loss 
            
            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1
            
            if args.global_rank == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch + (step + 1) / steps_per_epoch,
                    "global_step": global_step
                })
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            if val_dataloader and global_step % args.eval_steps == 0:
                if args.global_rank == 0:
                    print(f"\n[Step {global_step}] Running Evaluation...")
                
                val_loss = evaluate(model_engine, val_dataloader, device)
                
                model_engine.train()
                
                if args.global_rank == 0:
                    try:
                        ppl = math.exp(val_loss)
                    except OverflowError:
                        ppl = float("inf")
                        
                    print(f"Validation Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
                    wandb.log({
                        "eval/loss": val_loss,
                        "eval/perplexity": ppl,
                        "global_step": global_step
                    })

            if global_step % args.save_steps == 0:
                tag = f"checkpoint-{global_step}"
                _save_training_checkpoint(
                    args,
                    model_engine,
                    tokenizer,
                    tag,
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "max_seq_len": args.max_seq_len,
                    },
                )

    if args.global_rank == 0:
        print("Training Finished.")
        wandb.finish()

if __name__ == "__main__":
    main()

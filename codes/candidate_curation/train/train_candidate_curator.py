import os
import json
import logging
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from tqdm import tqdm

# ==========================================
# Dataset class for split JSON files
# ==========================================
class QwenSplitDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return {"query": self.data[idx]['query'], "item": self.data[idx]['item']}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ==========================================
# 2. Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Embedding Contrastive Learning")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model path")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to train.json")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to valid.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for InfoNCE loss")
    
    # Step-level logging and save intervals
    parser.add_argument("--logging_steps", type=int, default=10, help="Log train loss every X steps")
    parser.add_argument("--eval_save_steps", type=int, default=100, help="Run validation and save checkpoint every X steps")
    return parser.parse_args()

# ==========================================
# Main function
# ==========================================
def main():
    args = parse_args()
    accelerator = Accelerator()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ------------------------------------------
    # Logging setup
    # ------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if accelerator.is_local_main_process:
        # Keep console output so the shell script can redirect it to a log file
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

    logger.info("="*50)
    logger.info("Initializing experiment settings...")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Batch Size (per GPU): {args.batch_size}")
    logger.info("="*50)

    # ------------------------------------------
    # Tokenizer & Dataloader
    # ------------------------------------------
    logger.info("Preparing dataloaders and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def collate_fn(batch):
        queries = [b['query'] for b in batch]
        items = [b['item'] for b in batch]
        q_enc = tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors="pt")
        i_enc = tokenizer(items, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return q_enc, i_enc

    train_dataset = QwenSplitDataset(args.train_data_path)
    valid_dataset = QwenSplitDataset(args.valid_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Data loaded: Train {len(train_dataset)} | Valid {len(valid_dataset)}")

    # ------------------------------------------
    # Load QLoRA model
    # ------------------------------------------
    logger.info("Loading model and configuring QLoRA adapters...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": accelerator.local_process_index}
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(model, peft_config)
    logger.info("Model loaded and adapters attached.")
    
    # ------------------------------------------
    # Optimizer & Accelerator Prepare
    # ------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    # ------------------------------------------
    # Training & Validation Loop
    # ------------------------------------------
    logger.info(f"Starting training for {args.epochs} epochs.")
    
    global_step = 0
    total_steps_per_epoch = len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        
        for step, (q_enc, i_enc) in enumerate(progress_bar):
            with accelerator.accumulate(model):
                q_out = model(**q_enc)
                i_out = model(**i_enc)
                
                q_emb = F.normalize(mean_pooling(q_out, q_enc['attention_mask']), p=2, dim=1)
                i_emb = F.normalize(mean_pooling(i_out, i_enc['attention_mask']), p=2, dim=1)
                
                logits = torch.matmul(q_emb, i_emb.T) / args.temperature
                labels = torch.arange(logits.size(0)).long().to(accelerator.device)
                
                loss = F.cross_entropy(logits, labels)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.detach().item()
                global_step += 1
                
                progress_bar.set_postfix({'step': global_step, 'ce_loss': f"{current_loss:.4f}"})

                # --------------------------------------
                # Train loss logging
                # --------------------------------------
                if global_step % args.logging_steps == 0 and accelerator.is_local_main_process:
                    pct = (step + 1) / total_steps_per_epoch * 100
                    logger.info(f"Epoch {epoch}: {pct:02.0f}% | {step+1}/{total_steps_per_epoch} [step={global_step}, ce_loss={current_loss:.4f}]")

                # --------------------------------------
                # Validation & Save
                # --------------------------------------
                if global_step % args.eval_save_steps == 0:
                    model.eval()
                    total_val_loss = 0
                    
                    with torch.no_grad():
                        for v_q_enc, v_i_enc in valid_loader:
                            v_q_out = model(**v_q_enc)
                            v_i_out = model(**v_i_enc)
                            
                            v_q_emb = F.normalize(mean_pooling(v_q_out, v_q_enc['attention_mask']), p=2, dim=1)
                            v_i_emb = F.normalize(mean_pooling(v_i_out, v_i_enc['attention_mask']), p=2, dim=1)
                            
                            v_logits = torch.matmul(v_q_emb, v_i_emb.T) / args.temperature
                            v_labels = torch.arange(v_logits.size(0)).long().to(accelerator.device)
                            
                            v_loss = F.cross_entropy(v_logits, v_labels)
                            total_val_loss += v_loss.detach().item()

                    avg_val_loss = total_val_loss / len(valid_loader)
                    
                    if accelerator.is_local_main_process:
                        logger.info(f"[Step {global_step}] Val CE: {avg_val_loss:.4f}")
                        
                        # Save adapter
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        
                        logger.info(f"--- Saved Adapter at Step {global_step} ---")

                    model.train()

    # ------------------------------------------
    # Final model save logic
    # ------------------------------------------
    if accelerator.is_local_main_process:
        logger.info("Saving final model...")
        final_dir = os.path.join(args.output_dir, "checkpoint-final")
        os.makedirs(final_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        logger.info(f"--- Saved Final Adapter at {final_dir} ---")

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
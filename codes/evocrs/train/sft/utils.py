import torch
import torch.nn as nn
from typing import *

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

CAUSAL_LM_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _get_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _resize_token_embeddings(model, tokenizer):
    target = model
    if not hasattr(target, "resize_token_embeddings") and hasattr(model, "language_model"):
        target = model.language_model
    if not hasattr(target, "resize_token_embeddings") and hasattr(model, "model"):
        target = model.model
    target.resize_token_embeddings(len(tokenizer))


def _get_modules_to_save(is_response_gen):
    if not is_response_gen:
        return None
    return ["embed_tokens", "lm_head"]

def get_checkpoint_model(base_model_path, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Configure 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading Base Model from: {base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 4. Prepare the 4-bit model for training, including gradient checkpointing settings
    model = prepare_model_for_kbit_training(model)

    print(f"Loading LoRA Adapter from: {checkpoint_path} (is_trainable=True)")
    # 5. Load the checkpoint (LoRA adapter) and make it trainable
    # is_trainable=True is required for gradients and continued training.
    model = PeftModel.from_pretrained(
        model, 
        checkpoint_path, 
        is_trainable=True
    )

    model.print_trainable_parameters()

    return model, tokenizer

def get_train_model(path, task, lora_path=None):

    tokenizer = _get_tokenizer(path)

    is_response_gen = 'response_generation' in task

    if is_response_gen:
        special_tokens_dict = {
            'additional_special_tokens': ["[REC]", "[CHAT]", "[QUE]", "[ANS]"]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens: {special_tokens_dict['additional_special_tokens']}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 consistently for training stability
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    if is_response_gen:
        _resize_token_embeddings(model, tokenizer)

    model = prepare_model_for_kbit_training(model)

    modules_to_save = _get_modules_to_save(is_response_gen)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=CAUSAL_LM_TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save
    )

    if lora_path is not None:
        print(f"Loading Phase 1 LoRA adapter from: {lora_path}")
        
        model = PeftModel.from_pretrained(
            model, 
            lora_path, 
            adapter_name="phase1", 
            is_trainable=False
        )
        
        print("Adding a new trainable LoRA adapter for Phase 2...")
        model.add_adapter("phase2", peft_config)
        
        model.set_adapter("phase2")
        
    else:
        print("Initializing a fresh LoRA adapter for Phase 1...")
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    
    return model, tokenizer


def get_inference_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval() # Explicitly set inference mode
    
    return model, tokenizer    


def get_gated_train_model(path, task):
    # NOTE: gated training loads only the base 4-bit model without LoRA
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Keep task-specific special-token additions 
    if 'response_generation' in task:
        special_tokens_dict = {
            'additional_special_tokens': ["[REC]", "[CHAT]", "[QUE]", "[ANS]"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
    # elif 'ranking' in task:
    #     m_tokens = [f"[M{i}]" for i in range(1, 41)]
    #     tokenizer.add_special_tokens({'additional_special_tokens': m_tokens})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Resize embeddings if special tokens were added
    model.resize_token_embeddings(len(tokenizer))

    # Prepare for 4-bit training; safe even when training only the gate
    model = prepare_model_for_kbit_training(model)

    # attention swap
    model = _swap_llama_attention_with_gated(model)
    return model, tokenizer

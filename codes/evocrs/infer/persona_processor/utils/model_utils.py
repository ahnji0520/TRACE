# import torch
# from transformers import (
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     BitsAndBytesConfig,
#     AutoConfig
# )
# from peft import PeftModel, PeftConfig

# import torch.nn as nn
# import math

# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model

# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from safetensors.torch import load_file

# import bitsandbytes as bnb 


# import json
# import os
# from dataclasses import asdict, dataclass, field
# from enum import Enum
# from typing import List, Optional, Union

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel

# def get_inference_model(path, task):
#     if 'baseline' in task:
#         print(f"Loading Baseline Model from: {path}")

# 1. Load tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(path)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "left"

# 4. Load model (base model)
#         model = AutoModelForCausalLM.from_pretrained(
#             path,
#             # quantization_config=bnb_config,
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             # device_map="auto"
#             device_map=device_map
#         )
        
#         model.eval()
#     else:
#         base_model_id = "${HF_MODEL_DIR}/Llama-3.1-8B-Instruct" 
#         peft_model_path = path

#         tokenizer = AutoTokenizer.from_pretrained(base_model_id)

#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
# Implementation note.
#             bnb_4bit_use_double_quant=True,
#         )

#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_id,
#             quantization_config=bnb_config,
#             # device_map="auto"
#             device_map=device_map
#         )
#         model = PeftModel.from_pretrained(base_model, peft_model_path)

#         model.eval()

#         return tokenizer, model

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_LEGACY_BACKBONE_PATH = "${HF_MODEL_DIR}/Llama-3.1-8B-Instruct"


def get_inference_model(
    backbone_path,
    lora_path=None,
    device_map="auto",
    load_in_4bit=True,
):
    if lora_path == "nothing":
        lora_path = backbone_path
        backbone_path = DEFAULT_LEGACY_BACKBONE_PATH

    if lora_path is None:
        print(f"Loading Baseline Model from: {backbone_path}")

        tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )
        
        model.eval()
    else:
        print(f"Loading LoRA Model from: {lora_path}")
        print(f"Backbone Model: {backbone_path}")

        tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
        }

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, lora_path)

    model.eval()
    return tokenizer, model

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig

from typing_extensions import Unpack
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs

from typing import *


def use_kernelized_func(module_names: list[Callable] | Callable):
    """
    This decorator attaches the target function as an attribute of the module.
    The function must already be decorated with @use_kernel_func_from_hub
    this decorator then wraps it as an nn.Module internally.
    When kernelize is later applied to the full model, the function can be accessed as a regular module attribute and kernelized just like any other layer.
    The kernelization is performed in place, modifying the module directly.
    """
    if isinstance(module_names, Callable):
        module_names = [module_names]

    def decorator(cls):
        orig_init = cls.__init__

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            for fn in module_names:
                # we hardcode the name of the function to "rotary_fn" for now
                setattr(self, "rotary_fn", fn)

        cls.__init__ = new_init
        return cls

    return decorator



# Our eager_attention_forward implementation
def eager_attention_forward_with_logit_bias(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    persona_mask: torch.Tensor | None=None,
    persona_gate: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
    ):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if persona_mask is not None and persona_gate is not None:
        # persona_mask: [B,Tk] -> [B,1,1,Tk]
        m = persona_mask.to(attn_weights.dtype).unsqueeze(1).unsqueeze(2)
        eps = 1e-6
        # persona_gate: [B,H] -> log-gate broadcast to [B,H,1,1]
        if persona_gate.dim() != 2:
            raise ValueError(f"persona_gate must be [B,H], got {tuple(persona_gate.shape)}")
        lg = torch.log(persona_gate.clamp_min(eps)).to(attn_weights.dtype).unsqueeze(2).unsqueeze(3)  # [B,H,1,1]
        attn_weights = attn_weights + lg * m

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Our Logit-gated attention (LlamaGatedAttention) Implementation
@use_kernelized_func(apply_rotary_pos_emb)
class LlamaGatedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # --- input-conditioned head-wise gate g in (0,1) ---
        gate_hidden = getattr(config, "persona_gate_hidden", 128)
        self.persona_gate_net = nn.Sequential(
            nn.Linear(config.hidden_size, gate_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(gate_hidden, config.num_attention_heads, bias=True),
        )
        # init: gate ~= 1 (no change) at start
        nn.init.zeros_(self.persona_gate_net[-1].weight)
        nn.init.constant_(self.persona_gate_net[-1].bias, 10.0)

        # --- segment id cache for inference with KV cache ---
        self._segment_cache = None  # [B, T_total] bool
        self._persona_gate_cache = None # [B, H]


    @classmethod
    def from_existing(cls, old_attn: nn.Module, layer_idx: int):
        # Create and copy pretrained projections
        new = cls(old_attn.config, layer_idx=layer_idx)
        new.to(old_attn.q_proj.weight.device)

        new.q_proj.load_state_dict(old_attn.q_proj.state_dict())
        new.k_proj.load_state_dict(old_attn.k_proj.state_dict())
        new.v_proj.load_state_dict(old_attn.v_proj.state_dict())
        new.o_proj.load_state_dict(old_attn.o_proj.state_dict())

        # copy misc attrs if needed
        new.attention_dropout = getattr(old_attn, "attention_dropout", new.attention_dropout)
        new.is_causal = getattr(old_attn, "is_causal", True)
        return new

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)


        # -------- persona gate (compute once in prefill; reuse in decode) --------
        if past_key_values is None or cache_position is None:
            gate_pos = kwargs.pop("gate_pos", None)  # [B] or scalar
            if gate_pos is None:
                pooled = hidden_states[:, -1, :]
            else:
                if gate_pos.dim() == 0:
                    gate_pos = gate_pos.view(1).expand(hidden_states.size(0))
                gate_pos = gate_pos.clamp_min(0).clamp_max(hidden_states.size(1) - 1)
                pooled = hidden_states[
                    torch.arange(hidden_states.size(0), device=hidden_states.device),
                    gate_pos,
                    :
                ]
            persona_gate = torch.sigmoid(self.persona_gate_net(pooled))  # [B,H]
            self._persona_gate_cache = persona_gate
        else:
            if self._persona_gate_cache is None:
                # fallback (shouldn't happen if prefill ran)
                pooled = hidden_states[:, -1, :]
                self._persona_gate_cache = torch.sigmoid(self.persona_gate_net(pooled))
            persona_gate = self._persona_gate_cache
       

        # -------- persona mask over KEY positions (must match Tk_total) --------
        # We accept two modes:
        # (1) segment_ids already full length [B,T_total] -> use directly
        # (2) segment_ids is incremental [B,q_len] (often 1 in decode) -> append to cache
        segment_ids = kwargs.pop("segment_ids", None)  # int tensor
        persona_mask = None
        if segment_ids is not None:
            seg_bool = (segment_ids == 1)  # persona tokens marked as 1
            if past_key_values is None or cache_position is None:
                # prefill: reset cache
                self._segment_cache = seg_bool
            else:
                # decode: append (segment_ids typically length 1)
                if self._segment_cache is None:
                    self._segment_cache = seg_bool
                else:
                    self._segment_cache = torch.cat([self._segment_cache, seg_bool], dim=1)
            persona_mask = self._segment_cache

        # safety: if persona_mask exists, ensure length matches key_states Tk_total
        if persona_mask is not None:
            Tk_total = key_states.size(2)
            if persona_mask.size(1) != Tk_total:
                raise ValueError(f"persona_mask length {persona_mask.size(1)} != Tk_total {Tk_total}. "
                                 f"Pass full-length segment_ids during decode or ensure appending matches KV cache.")
 

        attn_output, attn_weights = eager_attention_forward_with_logit_bias(
             self,
             query_states,
             key_states,
             value_states,
             attention_mask,
             dropout=0.0 if not self.training else self.attention_dropout,
             scaling=self.scaling,
             persona_mask=persona_mask,
             persona_gate=persona_gate,
             **kwargs,
         )




        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights





def _swap_llama_attention_with_gated(model):
    # Handles plain HF model and PEFT-wrapped model
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    llama = core.model if hasattr(core, "model") else core
    for i, layer in enumerate(llama.layers):
        old = layer.self_attn
        layer.self_attn = LlamaGatedAttention.from_existing(old, layer_idx=i)
    return model


# # HF eager_attention_forward implementation
# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: torch.Tensor | None,
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs: Unpack[TransformersKwargs],
#     ):
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)

#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
#     if attention_mask is not None:
#         attn_weights = attn_weights + attention_mask

#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
#     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()

#     return attn_output, attn_weights







# # HF LlamaAttention Implementation
# @use_kernelized_func(apply_rotary_pos_emb)
# class LlamaAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = self.head_dim**-0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = True

#         self.q_proj = nn.Linear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.v_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
#         attention_mask: torch.Tensor | None = None,
#         past_key_values: Cache | None = None,
#         cache_position: torch.LongTensor | None = None,
#         **kwargs: Unpack[TransformersKwargs],
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)

#         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_values is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
#             self.config._attn_implementation, eager_attention_forward
#         )

#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
import json
from types import SimpleNamespace

from einops import rearrange, repeat

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers.activations import ACT2FN

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

from transformers import AutoTokenizer 
logger = logging.get_logger(__name__)

_flash_supports_window_size = False
try:
    from flash_attn import flash_attn_func 
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except ImportError as error:
    logger.warning(
        f"`flash-attention` package not found, consider installing for better performance: {error}."
    )
    if not _flash_supports_window_size:
        logger.warning(
            "Current `flash-attention` does not support `window_size`. Either upgrade or use `attn_implementation='eager'`."
        )

class _Phi4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _Phi4RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class _Phi4LongRoPEScaledRotaryEmbedding(_Phi4RotaryEmbedding):
    def __init__(self, dim, config, device=None):
        super().__init__(int(dim * config.partial_rotary_factor), config.max_position_embeddings, config.rope_theta, device)
        self.short_factor = config.rope_scaling.short_factor
        self.long_factor = config.rope_scaling.long_factor
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.partial_rotary_factor = config.partial_rotary_factor
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class _Phi4MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)
        
# Copied from transformers.models.llama.modeling_llama.repeat_kv with llama->phi
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class _Phi4Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = _Phi4RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.type
            if scaling_type == "longrope":
                self.rotary_emb = _Phi4LongRoPEScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.warning_once("You are not running the flash-attention implementation, expect numerical differences.")
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value_states, position_ids) #, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_cache = torch.cat([past_key_value[0], key_states], dim=-2)
            value_cache = torch.cat([past_key_value[1], value_states], dim=-2)
        else:
            key_cache = key_states
            value_cache = value_states
            
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_cache, self.num_key_value_groups)
        value_states = repeat_kv(value_cache, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        #if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #    raise ValueError(
        #        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #        f" {attn_weights.size()}"
        #    )
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask < 0.1, float('-inf'))
            #attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)

        return attn_output, key_cache, value_cache


class _Phi4FlashAttention2(_Phi4Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inference_mode = False,
        max_generation = 0,
        cur_pos = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)#.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids) 
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = query_states.transpose(1,2).to(hidden_states.dtype)
        key_states = key_states.transpose(1,2).to(hidden_states.dtype)
        #value_states = value_states.transpose(1,2)
        
        if not inference_mode:
            key_cache = key_states
            value_cache = value_states
        elif inference_mode and past_key_value is None: 
            key_cache = torch.zeros(bsz, max_generation, self.num_key_value_heads, self.head_dim, dtype=hidden_states.dtype) 
            value_cache = torch.zeros(bsz, max_generation, self.num_key_value_heads, self.head_dim, dtype=hidden_states.dtype)
            key_cache[:, :q_len] = key_states
            value_cache[:, :q_len] = value_states
        else:
            key_cache, value_cache = past_key_value
            key_cache[:, cur_pos:cur_pos + q_len] = key_states
            value_cache[:, cur_pos:cur_pos + q_len] = value_states
        
        #if past_key_value is not None:
        #    key_cache = torch.cat([past_key_value[0], key_states], dim=-2)
        #    value_cache = torch.cat([past_key_value[1], value_states], dim=-2)
        #else:
        #    key_cache = key_states
        #    value_cache = value_states
            
        key_states = key_cache[:, :cur_pos + q_len]
        value_states = value_cache[:, :cur_pos + q_len]
        attn_dropout = self.attention_dropout if self.training else 0.0

        if query_states.dtype == torch.float32:
            target_dtype = torch.float16
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=attn_dropout,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, key_cache, value_cache

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        softmax_scale=None,
    ):
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=True)
        return attn_output


_PHI4_ATTENTION_CLASSES = {
    "flash_attention_2": _Phi4FlashAttention2
}


class _Phi4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = _Phi4FlashAttention2(config, layer_idx=layer_idx) 

        self.mlp = _Phi4MLP(config)
        self.input_layernorm = _Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = _Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inference_mode = False,
        max_generation = 0,
        cur_pos = 0
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        attn_outputs, key_cache, value_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            inference_mode = inference_mode,
            max_generation = max_generation,
            cur_pos = cur_pos
        )
        #inference_mode = False,
        #max_generation = 0,
        #cur_pos = 0,
        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        return hidden_states, key_cache, value_cache


class _Phi4PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model_prefix = "model"
        self.supports_gradient_checkpointing = True
        self._no_split_modules = ["Phi4DecoderLayer"]
        self._skip_keys_device_placement = "past_key_values"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = False
        self._supports_cache_class = True
        self._version = "0.0.5"
        self.config = config
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class _Phi4Model(_Phi4PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [_Phi4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = 'flash_attention_2' # config._attn_implementation
        self.norm = _Phi4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        inference_mode = False,
        max_generation = 0,
        cur_pos = 0,
    ):  
        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]

        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        #if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        
        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = None # attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=device))
            # Add Causal Mask
            if past_key_values is not None:
                attention_mask = torch.cat([torch.ones((seq_length, past_key_values_length), device=device), attention_mask], dim=-1) 
            #attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Expand for batch & heads
            # Apply the Mask
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = [] 
        for layer_idx, decoder_layer in enumerate(self.layers):
            kv_cache = None
            if past_key_values is not None:
                kv_cache = past_key_values[layer_idx]
                
            if self.gradient_checkpointing and self.training:
                #user_cp.CheckpointwithRngFunction.apply(layer_module.forward, len(args_tensors), *full_args)
                layer_outputs, nk_cache, nv_cache = checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kv_cache,
                    inference_mode,
                    max_generation,
                    cur_pos
                )
            else:
                layer_outputs, nk_cache, nv_cache = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=kv_cache,
                    inference_mode=inference_mode,
                    max_generation=max_generation,
                    cur_pos=cur_pos
                )

            hidden_states = layer_outputs
            next_decoder_cache.append((nk_cache, nv_cache))

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        # return lastlayer hidden states, and KV cache.
        return hidden_states, next_decoder_cache

import torch.nn.functional as F

def sample_top_p(probs, top_p=0.9):
    """
    Perform Top-p (Nucleus) Sampling.
    Args:
        logits (torch.Tensor): Model logits (batch_size, vocab_size).
        top_p (float): Probability threshold for top-p sampling.
        temperature (float): Sampling temperature (default: 1.0).
    Returns:
        sampled_token (torch.Tensor): Sampled token index.
    """
    # Sort tokens by probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Mask tokens where cumulative probability > top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    #sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Set logits of removed tokens to -inf
    for i in range(probs.size(0)):
        probs[i, sorted_indices[i, sorted_indices_to_remove[i]]] = 0.0

    normalized_probs = probs / probs.sum(dim=-1, keepdim=True)
    # Sample from the filtered distribution
    sampled_token = torch.multinomial(normalized_probs, num_samples=1)
    return sampled_token
    
# PP, TP, DP
# Causal Large Language Model 
class _Phi4ForCausalLM(_Phi4PreTrainedModel):
    
    @staticmethod
    def load_hfckpt(local_model_path):
        with open(f'{local_model_path}/config.json', 'r') as file:
            llm_config = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
        #vocab_size = llm_config.vocab_size 
        #eos_token_id = llm_config.eos_token_id #": 32000,

        # check for the ckpt. 
        safetensor_files = [
            f"{local_model_path}/model-00001-of-00002.safetensors",
            f"{local_model_path}/model-00002-of-00002.safetensors"
        ]
        
        model_state_dict = {}
        for file in safetensor_files:
            part_state_dict = load_file(file, device="cpu")  # Load each part
            model_state_dict.update(part_state_dict)  # Merge into one dictionary

        llm_model = _Phi4ForCausalLM(llm_config) 
    
        # Step 4: Apply the merged state_dict to the model
        missing_keys, unexpected_keys = llm_model.load_state_dict(model_state_dict, strict=False) 
        print('missing_keys: ', missing_keys)
        print('unexpected_keys: ', unexpected_keys)    
        if 'lm_head.weight' in missing_keys:
            llm_model.lm_head.weight = llm_model.model.embed_tokens.weight
            #self.lm_head.weight = self.embed_tokens.weight  # Share the weights
            #   self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)


        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True) 

        return llm_model, llm_config, tokenizer
        
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = _Phi4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # model archiecture: connect from intermedia layer possibily.
        self.critic_head = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.xavier_normal_(self.critic_head.weight)
        nn.init.constant_(self.critic_head.bias, -8.0)
        
        
        self._tied_weights_keys = ["lm_head.weight"]
        # Initialize weights and apply final processing
        #self.post_init()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        inference_mode = False,
        max_generation = 0,
        cur_pos = 0,
    ):
        hidden_states, next_decoder_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inference_mode = inference_mode,
            max_generation = max_generation,
            cur_pos = cur_pos,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        # suppose it is next token's Q value. 
        critics = self.critic_head(hidden_states[:, -num_logits_to_keep:, :])

        # Apply sigmoid
        critics = torch.sigmoid(critics)

        loss = None
        if labels is not None:
            #loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)
            logits_flat = logits.reshape(-1, self.vocab_size)    # Shape: (batch_size * seq_len, vocab_size)
            target_flat = labels.reshape(-1)            # Shape: (batch_size * seq_len)
            loss = criterion(logits, target_flat)
        
        return loss, logits, critics, next_decoder_cache 
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        early_stop = True,
        force_wait_tokens : List[int] = None,
    ) -> Tuple[ List[List[int]], List[List[float]] ]: # these are the actions[token index, critic score, prob] 
        
        bsz = len(prompt_tokens)
        #assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        #assert max_prompt_len <= params.max_seq_len
        total_len = max_gen_len #+ max_prompt_len)

        pad_id = self.config.eos_token_id # self.config.pad_token_id
        eos_id = self.config.eos_token_id # eos_token_id
        bos_id = self.config.bos_token_id
        
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        # position starts with 0.
        pos_ids = torch.full((bsz, total_len), 0, dtype=torch.long, device="cuda")
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
            
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        token_critics = torch.zeros_like(tokens, dtype=torch.float)
        #token_critics = 
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")

        force_wait = False
        if force_wait_tokens:
            force_wait_tokens = torch.tensor(force_wait_tokens, device="cuda")
            force_wait = True
        #force_wait_tokens = torch.tensor(force_wait_tokens, device="cuda")

        input_text_mask = tokens != pad_id
        
        past_kv = None
        pos = None
        for cur_pos in range(min_prompt_len, total_len):
            _, logits, critics, past_kv  = self.forward(tokens[:, prev_pos:cur_pos], position_ids = pos, past_key_values=past_kv, inference_mode=True, max_generation=max_gen_len, cur_pos = prev_pos)

            #print('logits.shape', logits.shape)
            #print('past_kv len', len(past_kv))
            #print('past_kv[0][0].shape', past_kv[0][0].shape)
            #print('past_kv[0][1].shape', past_kv[0][1].shape)
            
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            #print('next_token.shape', next_token.shape)
            
            next_token = next_token.reshape(-1)

            if not early_stop and force_wait:
                for bsz_idx, _token in enumerate(next_token.tolist()):
                    if _token == eos_id and cur_pos + force_wait_tokens.shape[0] < total_len:
                        tokens[bsz_idx, cur_pos: cur_pos + force_wait_tokens.shape[0]] = force_wait_tokens
                        input_text_mask[bsz_idx, cur_pos: cur_pos + force_wait_tokens.shape[0]] = True
                        
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            
            #print('tokens', tokens)
            #print(logits.shape)
            
            #if logprobs:
            #print('logits.shape', logits.shape)
            #print('target.shape', tokens[:, prev_pos + 1 : cur_pos + 1].shape)

            #print('prev_pos', prev_pos, 'cur_pos', cur_pos, 'logits.shape', logits.shape)
            
            token_logprobs[:, prev_pos: cur_pos] = -F.cross_entropy(
                input=logits.reshape(-1, self.vocab_size), #.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1].reshape(-1),
                reduction="none"
            ).view(bsz, -1)
            #print('critics.shape', critics.shape)
            #print('token_critics[:, prev_pos: cur_pos].shape', token_critics[:, prev_pos: cur_pos].shape)
            
            token_critics[:, prev_pos: cur_pos] = critics.squeeze(-1) #[: -1]
            
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == eos_id
            )
            prev_pos = cur_pos
            pos = torch.tensor([prev_pos+1] * bsz, dtype=torch.long, device='cuda')
            if early_stop and all(eos_reached):
                break

        #print('generation done...................')
        token_logprobs = token_logprobs.tolist()
        #print('token_logprobs', token_logprobs.shape, token_logprobs.tolist())
        token_critics = token_critics.tolist()
        
        out_tokens, out_logprobs, out_critics = [], [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = len(prompt_tokens[i])
            toks = toks[start :]
            #probs = None
            #if logprobs:
            probs = token_logprobs[i][start-1 :-1]
            # cut to eos tok if any
            critics = token_critics[i][start-1: -1]
            
            if eos_id in toks:
                eos_idx = toks.index(eos_id)
                toks = toks[:eos_idx+1] # include the last eos token. 
                probs = probs[:eos_idx+1] # if logprobs else None
                critics = critics[:eos_idx+1]
                
            out_tokens.append(toks)
            out_logprobs.append(probs)
            out_critics.append(critics)
        
        return out_tokens, out_logprobs, out_critics

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
from xlmlib.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

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

class _TformerConfig:
    @classmethod
    def from_name(cls, name):
        if name == '400m':
            return cls()  # Sum the list and initialize
        elif name == '1b':
            return cls(hidden_size = 2048, intermediate_size=5120, num_attention_heads=16, num_key_value_heads=16, num_hidden_layers=24)
            
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1536,
        intermediate_size=4096,
      
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,

        initializer_range=0.02,
        rms_norm_eps=1e-5,
      
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1,
        sliding_window=None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
      
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.sliding_window = sliding_window


class _RMSNorm(nn.Module):
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

class _RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=100000, device=None):
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class _MLP(nn.Module):
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
        

class _FlashAttention2(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config, layer_idx: Optional[int] = None):
        #super().__init__(*args, **kwargs)
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

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    def _init_rope(self):
          self.rotary_emb = _RotaryEmbedding(
              self.head_dim,
              max_position_embeddings=self.max_position_embeddings,
              base=self.rope_theta)

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
            key_cache = torch.zeros(bsz, max_generation, self.num_key_value_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype) 
            value_cache = torch.zeros(bsz, max_generation, self.num_key_value_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            key_cache[:, :q_len] = key_states
            value_cache[:, :q_len] = value_states
        else:
            key_cache, value_cache = past_key_value
            key_cache[:, cur_pos:cur_pos + q_len] = key_states
            value_cache[:, cur_pos:cur_pos + q_len] = value_states
        
            
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


_ATTENTION_CLASSES = {
    "flash_attention_2": _FlashAttention2
}

class _DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = _FlashAttention2(config, layer_idx=layer_idx) 

        self.mlp = _MLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class _PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model_prefix = "model"
        self.supports_gradient_checkpointing = True
        self._no_split_modules = ["_DecoderLayer"]
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


class _Model(_PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        #self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) #, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [_DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = 'flash_attention_2' # config._attn_implementation
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_embed : torch.FloatTensor = None,
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
        if input_embed is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = input_embed
        #else:
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

# PP, TP, DP
# Causal Large Language Model 
class _TformerForCausalLM(_PreTrainedModel):
    @staticmethod
    def create_model(name):
        _config = _TformerConfig.from_name(name)
        _model = _TformerForCausalLM(_config)
        _model.apply(_model._init_weights)
        return _model, _config
        
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = _Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # model archiecture: connect from intermedia layer possibily.
        # self.critic_head = nn.Linear(config.hidden_size, 1, bias=True)
        #nn.init.xavier_normal_(self.critic_head.weight)
        #nn.init.constant_(self.critic_head.bias, -8.0)
        #self._tied_weights_keys = ["lm_head.weight"]
        # Initialize weights and apply final processing
        #self.post_init()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_embed: torch.FloatTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        inference_mode = False,
        max_generation = 0,
        cur_pos = 0,
        fuse_loss = False
    ):
        hidden_states, next_decoder_cache = self.model(
            input_ids=input_ids,
            input_embed=input_embed,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inference_mode=inference_mode,
            max_generation=max_generation,
            cur_pos=cur_pos,
        )

        loss = None
        logits = None
        
        if not fuse_loss:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    
            if labels is not None:
                #loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)
                logits_flat = logits.reshape(-1, self.vocab_size)    # Shape: (batch_size * seq_len, vocab_size)
                target_flat = labels.reshape(-1)            # Shape: (batch_size * seq_len)
                loss = self.criterion(logits_flat, target_flat)
        else:
            states = hidden_states[:, -num_logits_to_keep:, :]
            loss = LigerFusedLinearCrossEntropyFunction.apply(states.view(-1, states.size(-1)), self.lm_head.weight, labels.reshape(-1), None, None, -100, 0.0, 0.0, 'none', None, False)
            loss = loss[0].reshape(input_ids.shape)
        return loss, logits, next_decoder_cache 
    
        

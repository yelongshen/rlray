
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.phi3.configuration_phi3 import Phi3Config

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

import checkpoint as user_checkpoint
from torch.utils.checkpoint import checkpoint

logger = logging.get_logger(__name__)

# Transformers scans dependencies in the modeling file, causing issues on conditional loading. The regex only ignores try/catch blocks, but not if statements
# if is_flash_attn_2_available():
_flash_supports_window_size = False
try:
    from flash_attn import flash_attn_func #, flash_attn_varlen_func
    #from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except ImportError as error:
    logger.warning(
        f"`flash-attention` package not found, consider installing for better performance: {error}."
    )
    if not _flash_supports_window_size:
        logger.warning(
            "Current `flash-attention` does not support `window_size`. Either upgrade or use `attn_implementation='eager'`."
        )

class _Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with gemma->phi3, Gemma->Phi3
class _Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class _Phi3LongRoPEScaledRotaryEmbedding(_Phi3RotaryEmbedding):
    def __init__(self, dim, config, device=None):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

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

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
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

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class _Phi3MLP(nn.Module):
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

class _Phi3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
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
            self.rotary_emb = _Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = _Phi3LongRoPEScaledRotaryEmbedding(self.head_dim, self.config)
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
        #kv_seq_len = key_states.shape[-2]
        #if past_key_value is not None:
        #    if self.layer_idx is None:
        #        raise ValueError(
        #            f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
        #            "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
        #            "with a layer index."
        #        )
        #    kv_seq_len +=  past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
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


class _Phi3FlashAttention2(_Phi3Attention):
    """
    Phi-3 flash attention module. This module inherits from `Phi3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Phi3FlashAttention2 attention does not support output_attentions

        #if not _flash_supports_window_size:
        #    logger.warning_once(
        #        "The current flash attention version does not support sliding window attention. Please use `attn_implementation='eager'` or upgrade flash-attn library."
        #    )
        #    raise ValueError("The current flash attention version does not support sliding window attention.")

        #output_attentions = False

        #if "padding_mask" in kwargs:
        #    warnings.warn(
        #        "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        #    )
        #    # overwrite attention_mask with padding_mask
        #    attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        #kv_seq_len = key_states.shape[-2]
        #if past_key_value is not None:
        #    if self.layer_idx is None:
        #        raise ValueError(
        #            f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
        #            "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
        #            "with a layer index."
        #        )
        #    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        #rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item() + 1)
        cos, sin = self.rotary_emb(value_states, position_ids) #, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # sliding_window by default is False.
        #use_sliding_windows = (
        #    _flash_supports_window_size
        #    and getattr(self.config, "sliding_window", None) is not None
        #    and kv_seq_len > self.config.sliding_window
        #)
        if past_key_value is not None:
            key_cache = torch.cat([past_key_value[0], key_states], dim=-2)
            value_cache = torch.cat([past_key_value[1], value_states], dim=-2)
        else:
            key_cache = key_states
            value_cache = value_states
            
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_cache, self.num_key_value_groups)
        value_states = repeat_kv(value_cache, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.
        if query_states.dtype == torch.float32:
            #if torch.is_autocast_enabled():
            #    target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            #elif hasattr(self.config, "_pre_quantization_dtype"):
            #    target_dtype = self.config._pre_quantization_dtype
            #else:
            #    target_dtype = self.qkv_proj.weight.dtype

            target_dtype = torch.float16
            
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, key_cache, value_cache

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=True)
        return attn_output

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )



_PHI3_ATTENTION_CLASSES = {
    "eager": _Phi3Attention,
    "flash_attention_2": _Phi3FlashAttention2
}


class _Phi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.config = config
        #print('config._attn_implementation', config._attn_implementation)
        self.self_attn = _Phi3FlashAttention2(config, layer_idx=layer_idx) 
        # _PHI3_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        self.mlp = _Phi3MLP(config)
        self.input_layernorm = _Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = _Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        #output_attentions: Optional[bool] = False,
        #use_cache: Optional[bool] = False,
        #**kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        #if "padding_mask" in kwargs:
        #    warnings.warn(
        #        "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        #    )
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, key_cache, value_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            #output_attentions=output_attentions,
            #use_cache=use_cache,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        #outputs = (hidden_states,)
        #if use_cache:
        #    outputs += (present_key_value,)
        return hidden_states, key_cache, value_cache


class _Phi3PreTrainedModel(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.config_class = Phi3Config
        self.base_model_prefix = "model"
        self.supports_gradient_checkpointing = True
        self._no_split_modules = ["Phi3DecoderLayer"]
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


class _Phi3Model(_Phi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]
    Args:
        config: Phi3Config
    """
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [_Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = _Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        #attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        #inputs_embeds: Optional[torch.FloatTensor] = None,
        #use_cache: Optional[bool] = None,
        #output_attentions: Optional[bool] = None,
        #output_hidden_states: Optional[bool] = None,
        #return_dict: Optional[bool] = None,
    ):  # -> Union[Tuple, BaseModelOutputWithPast]:
        #output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #output_hidden_states = (
        #    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        #)
        #use_cache = use_cache if use_cache is not None else self.config.use_cache
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        #if input_ids is not None and inputs_embeds is not None:
        #    raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        #elif input_ids is not None:
        #    batch_size, seq_length = input_ids.shape[:2]
        #elif inputs_embeds is not None:
        #    batch_size, seq_length = inputs_embeds.shape[:2]
        #else:
        #    raise ValueError("You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]

        #if past_key_values is None:
        #else:    
        #    past_key_values_length = past_key_values[0][0].shape[-2] #.get_usable_length(seq_length)
        
        #if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        logger.warning_once(
        #            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #        )
        #        use_cache = False
        #if use_cache:
        #    use_legacy_cache = not isinstance(past_key_values, Cache)
        #    if use_legacy_cache:
        #        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #    past_key_values_length = past_key_values.get_usable_length(seq_length)
        device = input_ids.device
        
        if position_ids is None:
             # if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        #if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        #if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        #    is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        #    if is_padding_right:
        #        raise ValueError(
        #            "You are attempting to perform batched generation with padding_side='right'"
        #            " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
        #            " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
        #        )
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
        #all_hidden_states = () if output_hidden_states else None
        #all_self_attns = () if output_attentions else None
        next_decoder_cache = [] #List[Tuple[torch.FloatTensor, torch.FloatTensor]]


        for layer_idx, decoder_layer in enumerate(self.layers):
            #if output_hidden_states:
            #    all_hidden_states += (hidden_states,)
            #print('layer_idx', layer_idx, 'hidden_state.dtype', hidden_states.dtype, 'decoder_layer.dtype', decoder_layer.self_attn.qkv_proj.weight.dtype)

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
                    kv_cache
                )
            else:
                layer_outputs, nk_cache, nv_cache = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=kv_cache,
                    #output_attentions=output_attentions,
                    #use_cache=use_cache,
                )

            hidden_states = layer_outputs
            next_decoder_cache.append((nk_cache, nv_cache))
            #if use_cache:
            #    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            #if output_attentions:
            #    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        #if output_hidden_states:
        #    all_hidden_states += (hidden_states,)
        #next_cache = None
        #if use_cache:
        #    next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        
        #if not return_dict:
        #    return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        #return BaseModelOutputWithPast(
        #    last_hidden_state=hidden_states,
        #    past_key_values=next_cache,
        #    hidden_states=all_hidden_states,
        #    attentions=all_self_attns,
        #)
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
    #print('shape1111', sorted_indices.shape, sorted_indices_to_remove.shape, sorted_indices[sorted_indices_to_remove].shape)

    #probs[sorted_indices[sorted_indices_to_remove]] = 0.0

    for i in range(probs.size(0)):
        probs[i, sorted_indices[i, sorted_indices_to_remove[i]]] = 0.0

    #print('shape2222', probs.shape)
    normalized_probs = probs / probs.sum(dim=-1, keepdim=True)
    # Sample from the filtered distribution
    sampled_token = torch.multinomial(normalized_probs, num_samples=1)
    return sampled_token
    
# PP, TP, DP
# Causal Large Language Model 
class _Phi3ForCausalLM(_Phi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = _Phi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # model archiecture: connect from intermedia layer possibily.
        self.critic_head = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.xavier_normal_(self.critic_head.weight)
        
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
        ##  attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        ## inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ## use_cache: Optional[bool] = None,
        ## output_attentions: Optional[bool] = None,
        ## output_hidden_states: Optional[bool] = None,
        ## return_dict: Optional[bool] = None,
        ## cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        ## **loss_kwargs,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Phi3ForCausalLM

        >>> model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```"""

        #output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #output_hidden_states = (
        #    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        #)
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states, next_decoder_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values
        )

        #hidden_states = outputs[0]
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
            # Calculate Loss
            loss = criterion(logits, target_flat)

        #if not return_dict:
        #    output = (logits,) + outputs[1:]
        #    return (loss,) + output if loss is not None else output

        return loss, logits, critics, next_decoder_cache 
        #return CausalLMOutputWithPast(
        #    loss=loss,
        #    logits=logits,
        #    past_key_values=outputs.past_key_values,
        #    hidden_states=outputs.hidden_states,
        #    attentions=outputs.attentions,
        #)
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[ List[List[int]], List[List[float]] ]: # these are the actions[token index, critic score, prob] 
        """
        Generate text sequences based on provided prompts using the language generation model.
        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        Returns:
            Tuple[ List[List[int]], List[List[float]], List[List[float]] ]:  A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.
        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        #params = self.model.params
        bsz = len(prompt_tokens)
        #assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        #assert max_prompt_len <= params.max_seq_len
        total_len = min(4096, max_gen_len + max_prompt_len)

        pad_id = self.config.pad_token_id
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
        input_text_mask = tokens != pad_id
        
        # decode the last tokens
        #if min_prompt_len == total_len:
        #    _, logits, critics, _ = self.forward(tokens)
        #    token_logprobs = -F.cross_entropy(
        #        input=logits.transpose(1, 2),
        #        target=tokens,
        #        reduction="none",
        #        ignore_index=pad_id,
        #    )
        #input_ids: torch.LongTensor = None,
        #position_ids: Optional[torch.LongTensor] = None,
        #past_key_values: Optional[List[torch.FloatTensor]] = None,
        #labels: Optional[torch.LongTensor] = None,

        past_kv = None
        pos = None
        for cur_pos in range(min_prompt_len, total_len):
            _, logits, critics, past_kv  = self.forward(tokens[:, prev_pos:cur_pos], position_ids = pos, past_key_values=past_kv)

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
            
            token_logprobs[:, prev_pos: cur_pos] = -F.cross_entropy(
                input=logits.reshape(-1, self.vocab_size), #.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1].reshape(-1),
                reduction="none",
                ignore_index=pad_id,
            )
            #print('critics.shape', critics.shape)
            #print('token_critics[:, prev_pos: cur_pos].shape', token_critics[:, prev_pos: cur_pos].shape)
            
            token_critics[:, prev_pos: cur_pos] = critics.squeeze(-1) #[: -1]
            
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == eos_id
            )
            prev_pos = cur_pos
            pos = torch.tensor([prev_pos+1] * bsz, dtype=torch.long, device='cuda')
            if all(eos_reached):
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




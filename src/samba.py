import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint
import checkpoint as user_checkpoint

from transformers.activations import ACT2FN
from transformers.utils import logging

#from transformers.cache_utils import Cache, DynamicCache
logger = logging.get_logger(__name__)

import selective_scan_cuda
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    #from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    if not _flash_supports_window_size:
        raise ValueError("Please update flash-attention to support window size.")
    #from flash_attn.ops.activations import swiglu
    #from flash_attn.ops.rms_norm import RMSNorm #as RMSNorm
except ImportError:
    logger.warning("Flash submodules not found, consider installing for better performance.")
    #swiglu = None
    #RMSNorm = None
from flash_attn.ops.activations import swiglu

import causal_conv1d_cuda

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    logger.warning("Causal Conv1d submodules not found, consider installing for better performance.")
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    logger.warning("Selective state updating submodules not found, consider installing for better performance.")
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    logger.warning("Triton LayerNorm submodules not found, consider installing for better performance.")
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from torch.cuda.amp import custom_bwd, custom_fwd

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

PHI_NORM_CLASS = RMSNorm #_RMSNorm #if RMSNorm is None else RMSNorm

# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with gemma->phi3, Gemma->Phi3
class _RotaryEmbedding(nn.Module):
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

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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
        #self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        #self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        y = self.fc1(hidden_states)
        # Special case for SwiGLU
        if self.config.hidden_act == "silu" and swiglu is not None:
            gate, y = y.chunk(2, dim=-1)
            y = swiglu(gate, y)
        else:
            gate, y = y.chunk(2, dim=-1)
            y = y * self.activation_fn(gate)
        return self.fc2(y)
        
        
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

class _FlashAttention2(nn.Module):
    """
    flash attention module. This module inherits from `Phi3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=False)
        self.rotary_emb = _RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta,)
        self.use_sliding_windows = (self.max_position_embeddings > self.config.sliding_window)
        #self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.Wqkv(hidden_states)
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
        # Because the input can be padded, the absolute sequence length depends on the max position id.
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

            if self.use_sliding_windows:
                key_cache = key_cache[:, :, -self.config.sliding_window:, :].contiguous()
                value_cache = value_cache[:, :, -self.config.sliding_window:, :].contiguous()
        else:
            key_cache = key_states
            value_cache = value_states
    
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_cache, self.num_key_value_groups)
        value_states = repeat_kv(value_cache, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, dropout=attn_dropout, windows = self.config.sliding_window)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.out_proj(attn_output)
        
        return attn_output, key_cache, value_cache

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward
    def _flash_attention_forward(self, query_states, key_states, value_states, dropout=0.0, softmax_scale=None, windows=-1):
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
        attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True, window_size=(windows, windows))
        return attn_output

PHI_ATTENTION_CLASSES = { "flash_attention_2": _FlashAttention2 }

# Make this function support inference (prefilling / generation)
class MambaInnerFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, out_proj_bias, 
                A, D = None, delta_bias = None, delta_softplus = True, checkpoint_lvl = 1, conv_state = None, ssm_state = None, activation = 'silu', recurrent_mode = False):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
                    
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        x, z = xz.chunk(2, dim=1) # b d l 
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None

        if recurrent_mode and L == 1:
            # generation mode:
            #print('x.shape:', x.shape, x.squeeze().shape)
            conv1d_out = causal_conv1d_update(x.squeeze(dim=-1), conv_state, conv1d_weight, conv1d_bias, activation)
            conv1d_out = conv1d_out.unsqueeze(dim=-1)
        else: 
            # prefill mode & training mode;
            if recurrent_mode:
                conv_state.copy_(F.pad(x, (conv_state.shape[-1] - x.shape[-1], 0)))         
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, None, None, True)

        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = True
        ctx.is_variable_C = True
        ctx.B_proj_bias_is_None = True
        ctx.C_proj_bias_is_None = True
                    
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
        B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
        C = x_dbl[:, -d_state:]  # (bl dstate)
        C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        
        D = D.contiguous()

        if recurrent_mode and L == 1:
            out_z = selective_state_update(ssm_state, conv1d_out.squeeze(dim=-1), delta.squeeze(dim=-1), A, B.squeeze(dim=-1), C.squeeze(dim=-1), D.float(), 
                                            z.squeeze(dim=-1), delta_bias, delta_softplus)
            out_z = out_z.unsqueeze(dim = -1)
        else:
            out, scan_intermediates, out_z = selective_scan_cuda.fwd(conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus)
            if recurrent_mode:
                ssm_state.copy_(scan_intermediates[:, :, -1, 1::2])
            
            ctx.delta_softplus = delta_softplus
            ctx.out_proj_bias_is_None = out_proj_bias is None
            ctx.checkpoint_lvl = checkpoint_lvl
            if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
                conv1d_out, delta = None, None
                
            ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                                  delta_proj_weight, out_proj_weight, conv1d_out, delta,
                                  A, B, C, D, delta_bias, scan_intermediates, out)
            
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)

        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
        
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dx_dbl[:, -d_state:] = dC  # (bl d)
        
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dD, ddelta_bias, None, None, None, None, None, None) 
        #ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, out_proj_bias, 
        #        A, D = None, delta_bias = None, delta_softplus = True, checkpoint_lvl = 1, conv_state = None, ssm_state = None, activation = 'silu', recurrent_mode = False):
                         
# reimplement the mamba layer. 
class _Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
        dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True, layer_idx=None, device=None, dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, conv_state = None, ssm_state = None, inference_mode = False):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states.to(dtype = self.in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            
        # if not self.training:
        #     xz = xz.to(torch.float32)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if inference_mode and (conv_state is None or ssm_state is None):
            conv_state = torch.zeros(batch, self.d_model * self.expand, self.d_conv, device=hidden_states.device, dtype = self.conv1d.weight.dtype)
            ssm_state = torch.zeros(batch, self.d_model * self.expand, self.d_state, device=hidden_states.device, dtype= self.dt_proj.weight.dtype)

        #xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, out_proj_bias, 
        #        A, D = None, delta_bias = None, delta_softplus = True, checkpoint_lvl = 1, conv_state = None, ssm_state = None, activation = 'silu', recurrent_mode = False):
        
        out = MambaInnerFn.apply(xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight,
                                self.out_proj.weight, self.out_proj.bias, A, self.D.float(), self.dt_proj.bias.float(), True, 1,
                                conv_state, ssm_state, 'silu', inference_mode)
        return out, conv_state, ssm_state

class _SambaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.mlp = _MLP(config)
        self.input_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0        
        if self.use_mamba:
            factory_kwargs = {"dtype": torch.float32}
            self.attn = _Mamba(config.hidden_size, layer_idx=layer_idx, **factory_kwargs)
        else:
            self.attn = PHI_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_cache = None,
        inference_mode = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_mamba:
            #  hidden_states, conv_state = None, ssm_state = None, inference_mode = False):
            # conv_state, ssm_state = None, None if past_cache is None else past_cache
            conv_state, ssm_state = past_cache if past_cache is not None else (None, None)
            #forward(self, hidden_states, conv_state = None, ssm_state = None, inference_mode = False):
            attn_outputs, conv_state, ssm_state = self.attn(hidden_states, conv_state = conv_state, ssm_state = ssm_state, inference_mode = inference_mode)
            #residual = residual.to(torch.float32) 
            #present_key_value = past_key_value
            next_cache = (conv_state, ssm_state)
        else:
            # Self Attention
            attn_outputs, key_cache, value_cache = self.attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                past_key_value=past_cache
            )
            next_cache = (key_cache, value_cache)

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        return hidden_states, next_cache
        
class _SambaPreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model_prefix = "model"
        self.supports_gradient_checkpointing = True
        self._no_split_modules = ["_SambaDecoderLayer"]
        self._skip_keys_device_placement = "past_caches"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = False
        self._supports_cache_class = True
        self._version = "0.0.5"
        self.config = config

class _SambaModel(_SambaPreTrainedModel):        
    def __init__(self, config):
        super().__init__(config)
        #self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) #, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [_SambaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_caches = None, 
        inference_mode = False,
    ):  
        batch_size, seq_length = input_ids.shape[:2]
        past_length = 0 if past_caches is None else past_caches[0]
        device = input_ids.device        
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device).unsqueeze(0).view(-1, seq_length)
            
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        next_caches = (past_length + seq_length, []) #List[Tuple[torch.FloatTensor, torch.FloatTensor]]
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            past_cache = None if past_caches is None else past_caches[1][layer_idx]
            if self.gradient_checkpointing and self.training:
                layer_outputs, next_cache = checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    position_ids,
                    past_cache,
                    inference_mode,
                )
            else:
                layer_outputs, next_cache = decoder_layer(
                    hidden_states,
                    position_ids=position_ids,
                    past_cache=past_cache,
                    inference_mode = inference_mode
                )
            hidden_states = layer_outputs
            next_caches[1].append(next_cache)
        hidden_states = self.final_layernorm(hidden_states) #.to(dtype=self.final_layernorm.weight.dtype))
        return hidden_states, next_caches

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


class _SambaForCausalLM(_SambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = _SambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        self.critic_head = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.xavier_normal_(self.critic_head.weight)
        nn.init.constant_(self.critic_head.bias, -8.0)

        self._tied_weights_keys = ["lm_head.weight"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_caches = None, # Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        inference_mode = False
    ):
        hidden_states, next_caches = self.model(
            input_ids = input_ids,
            past_caches = past_caches,
            inference_mode = inference_mode
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
        return loss, logits, critics, next_caches 
    
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

        pad_id = self.config.eos_token_id # self.config.pad_token_id
        eos_id = self.config.eos_token_id # eos_token_id
        bos_id = self.config.bos_token_id
        
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        # position starts with 0.
        # pos_ids = torch.full((bsz, total_len), 0, dtype=torch.long, device="cuda")
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
            
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        token_critics = torch.zeros_like(tokens, dtype=torch.float)
        #token_critics = 
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        
        _caches = None
        for cur_pos in range(min_prompt_len, total_len):
            _, logits, critics, _caches  = self.forward(tokens[:, prev_pos:cur_pos], past_caches = _caches, inference_mode = True)

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
            )
            #print('critics.shape', critics.shape)
            #print('token_critics[:, prev_pos: cur_pos].shape', token_critics[:, prev_pos: cur_pos].shape)
            token_critics[:, prev_pos: cur_pos] = critics.squeeze(-1) #[: -1]
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == eos_id
            )
            prev_pos = cur_pos
            # pos = torch.tensor([prev_pos+1] * bsz, dtype=torch.long, device='cuda')
            if all(eos_reached):
                break

        token_logprobs = token_logprobs.tolist()
        token_critics = token_critics.tolist()
        
        out_tokens, out_logprobs, out_critics = [], [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = len(prompt_tokens[i])
            toks = toks[start :]
            
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

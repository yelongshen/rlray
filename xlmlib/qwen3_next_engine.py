"""
Qwen3-Next Model Adapter for In-house LLM Engine

This file adapts the HuggingFace Qwen3-Next model to work with the in-house
llm_engine.py for fast inference with paged attention.

The in-house LLMEngine expects:
- model(input_ids=..., position_ids=..., cache_params=..., logits_to_keep=...) -> (logits, cache)
- cache_params=None for training, cache_params=Qwen3NextCacheParams for inference

Supports tensor parallelism for multi-GPU inference.

Usage:
    from qwen3_next_engine import Qwen3NextForLLMEngine, load_qwen3_next_for_engine
    
    # Single GPU
    model, tokenizer, config = load_qwen3_next_for_engine("Qwen/Qwen3-Coder-Next")
    engine = LLMEngine(model, config, "cuda")
    
    # Tensor Parallel (run with torchrun --nproc_per_node=2)
    model, tokenizer, config = load_qwen3_next_for_engine(
        "Qwen/Qwen3-Coder-Next", 
        tensor_parallel_size=2
    )
"""

import gc
import math
import sys
import time
import os as _os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, List, Union
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, paged attention will not work")

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as fla_recurrent_gated_delta_rule,
    )
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("Warning: fla not available, using pure PyTorch GatedDeltaNet (slow)")

try:
    from fused_moe_triton import fused_moe as triton_fused_moe
    TRITON_MOE_AVAILABLE = True
    print("Triton fused MoE kernel loaded successfully")
except ImportError:
    try:
        from xlmlib.fused_moe_triton import fused_moe as triton_fused_moe
        TRITON_MOE_AVAILABLE = True
        print("Triton fused MoE kernel loaded successfully (via xlmlib)")
    except ImportError:
        TRITON_MOE_AVAILABLE = False
        print("Warning: Triton fused MoE kernel not available, using fallback")

try:
    from context import set_context, get_context, reset_context
except ImportError:
    print("Warning: context module not found, using dummy implementation")
    class DummyContext:
        is_prefill = False
        slot_mapping = None
        context_lens = None
        block_tables = None
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = 0
        max_seqlen_k = 0
    _context = DummyContext()
    def get_context(): return _context
    def set_context(*args, **kwargs): pass
    def reset_context(): pass


# ============================================================================
# Tensor Parallel Utilities
# ============================================================================

_TENSOR_PARALLEL_GROUP = None
_TENSOR_PARALLEL_WORLD_SIZE = 1
_TENSOR_PARALLEL_RANK = 0


def init_tensor_parallel(tensor_parallel_size: int = 1):
    """Initialize tensor parallel group."""
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_WORLD_SIZE, _TENSOR_PARALLEL_RANK
    
    if tensor_parallel_size <= 1:
        _TENSOR_PARALLEL_WORLD_SIZE = 1
        _TENSOR_PARALLEL_RANK = 0
        return
    
    if not dist.is_initialized():
        import datetime
        # Set NCCL timeout BEFORE init_process_group (env var has no effect after init)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2),  # 2 hours for large MoE prefill
        )
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size % tensor_parallel_size == 0, \
        f"World size {world_size} must be divisible by tensor_parallel_size {tensor_parallel_size}"
    
    num_tp_groups = world_size // tensor_parallel_size
    
    for i in range(num_tp_groups):
        ranks = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_PARALLEL_GROUP = group
            _TENSOR_PARALLEL_WORLD_SIZE = tensor_parallel_size
            _TENSOR_PARALLEL_RANK = rank - i * tensor_parallel_size
    
    if _TENSOR_PARALLEL_RANK == 0:
        print(f"Initialized TP: rank={_TENSOR_PARALLEL_RANK}/{_TENSOR_PARALLEL_WORLD_SIZE}")


def get_tp_world_size() -> int:
    return _TENSOR_PARALLEL_WORLD_SIZE


def get_tp_rank() -> int:
    return _TENSOR_PARALLEL_RANK


def get_tp_group():
    return _TENSOR_PARALLEL_GROUP


# TP debug: counter for tracing collective operations
_tp_collective_counter = 0
_tp_debug_enabled = False  # Set True to trace every collective (very verbose)

def enable_tp_debug(enabled=True):
    global _tp_debug_enabled
    _tp_debug_enabled = enabled

class _ReduceFromTP(torch.autograd.Function):
    """All-reduce in tensor parallel group."""
    @staticmethod
    def forward(ctx, input_):
        if get_tp_world_size() == 1:
            return input_
        global _tp_collective_counter
        _tp_collective_counter += 1
        cnt = _tp_collective_counter
        if _tp_debug_enabled and cnt <= 20:
            print(f'    [TP rank={get_tp_rank()}] all_reduce #{cnt} shape={input_.shape}', flush=True)
        dist.all_reduce(input_, group=get_tp_group())
        if _tp_debug_enabled and cnt <= 20:
            print(f'    [TP rank={get_tp_rank()}] all_reduce #{cnt} DONE', flush=True)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_tp(input_):
    return _ReduceFromTP.apply(input_)


class _AllGatherFromTP(torch.autograd.Function):
    """All-gather in tensor parallel group."""
    @staticmethod
    def forward(ctx, input_):
        if get_tp_world_size() == 1:
            return input_
        global _tp_collective_counter
        _tp_collective_counter += 1
        cnt = _tp_collective_counter
        if _tp_debug_enabled and cnt <= 20:
            print(f'    [TP rank={get_tp_rank()}] all_gather #{cnt} shape={input_.shape}', flush=True)
        world_size = get_tp_world_size()
        output_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(output_list, input_, group=get_tp_group())
        if _tp_debug_enabled and cnt <= 20:
            print(f'    [TP rank={get_tp_rank()}] all_gather #{cnt} DONE', flush=True)
        return torch.cat(output_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = get_tp_world_size()
        rank = get_tp_rank()
        dim_size = grad_output.shape[-1] // world_size
        return grad_output[..., rank * dim_size:(rank + 1) * dim_size]


def all_gather_from_tp(input_):
    return _AllGatherFromTP.apply(input_)


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer for tensor parallelism."""
    def __init__(self, input_size: int, output_size: int, bias: bool = False, gather_output: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        world_size = get_tp_world_size()
        self.output_size_per_partition = output_size // world_size
        
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            output = all_gather_from_tp(output)
        return output


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer for tensor parallelism."""
    def __init__(self, input_size: int, output_size: int, bias: bool = False, input_is_parallel: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        
        world_size = get_tp_world_size()
        self.input_size_per_partition = input_size // world_size
        
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight)
        output = reduce_from_tp(output)
        if self.bias is not None:
            output = output + self.bias
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors.
    
    Handles partial rotary: if cos/sin have smaller dim than q/k,
    only apply rotary to the first rotary_dim dimensions.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Handle partial rotary (if cos/sin dim < q/k dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    
    # Apply rotary embeddings
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate with pass-through portion
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


@torch.jit.script
def store_kvcache(key_states: torch.Tensor, value_states: torch.Tensor, 
                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                  slot_mapping: torch.Tensor):
    """Store key/value states into paged KV cache."""
    slot_mapping = slot_mapping.long()
    k_cache.index_copy_(0, slot_mapping, key_states)
    v_cache.index_copy_(0, slot_mapping, value_states)


def get_rms_norm_eps(config):
    """Get RMS norm epsilon from config with fallback."""
    return getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm for Qwen3-Next.
    
    Note: Qwen3-Next uses (1 + weight) formula where weight is initialized to 0.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # Initialize to zeros to match HF (effective weight = 1 + 0 = 1)
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Qwen3-Next uses (1 + weight) formula
        return ((1.0 + self.weight.float()) * hidden_states).to(input_dtype)


class Qwen3NextRMSNormGated(nn.Module):
    """RMSNorm with gating for Qwen3-Next GatedDeltaNet."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32)).to(input_dtype)
        return hidden_states


# ============================================================================
# Linear Attention (Gated DeltaNet) - Helper Functions
# ============================================================================

def apply_mask_to_padding_states(hidden_states, attention_mask):
    """Tunes out the hidden states for padding tokens."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """L2 normalization matching FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Pure PyTorch implementation of chunked gated delta rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) 
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, 
    initial_state, 
    output_final_state,
    use_qk_l2norm_in_kernel=False
):
    """Pure PyTorch implementation of recurrent gated delta rule (for single token generation)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

class Qwen3NextCacheParams:
    """Unified cache for hybrid Qwen3-Next model.
    
    Memory budget allocation strategy:
    1. First compute fixed-cost linear attention caches (conv + recurrent states).
       These are O(1) per layer regardless of sequence length.
    2. Subtract from free GPU memory budget.
    3. Use remaining budget for paged KV cache blocks (full attention layers only).
    
    Memory costs per linear attention layer (per batch element):
      conv_state:      conv_dim * (kernel_size - 1) * 2 bytes  (bf16)
      recurrent_state: num_v_heads * key_dim * value_dim * 4 bytes  (fp32 for numerical stability)
    
    Memory cost per full attention KV block:
      2 * block_size * num_kv_heads * head_dim * 2 bytes  (bf16, k+v)
    """
    def __init__(
        self,
        config,
        batch_size: int, 
        free_memory_budget: int,
        device: torch.device,
        block_size: int = 256,
    ):
        self.batch_size = batch_size
        self.device = device
        self.block_size = block_size
        self.has_previous_state = False
        
        num_layers = config.num_hidden_layers
        layer_types = getattr(config, 'layer_types', ['full_attention'] * num_layers)
        
        # === Dimensions ===
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        num_kv_heads_global = config.num_key_value_heads
        
        # For TP: each rank only caches its partition of KV heads
        tp_world_size = get_tp_world_size()
        kv_is_replicated = num_kv_heads_global < tp_world_size
        if kv_is_replicated:
            num_kv_heads = num_kv_heads_global  # Replicated: each rank has all KV heads
        else:
            num_kv_heads = num_kv_heads_global // tp_world_size  # Sharded
        
        # GatedDeltaNet dimensions
        num_v_heads = getattr(config, 'linear_num_value_heads', 32)
        num_k_heads = getattr(config, 'linear_num_key_heads', 16)
        head_k_dim = getattr(config, 'linear_key_head_dim', 128)
        head_v_dim = getattr(config, 'linear_value_head_dim', 128)
        conv_kernel_size = getattr(config, 'linear_conv_kernel_dim', 4)
        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        
        # === Count layer types ===
        num_full_attn_layers = 0
        num_linear_attn_layers = 0
        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
            if lt == "full_attention":
                num_full_attn_layers += 1
            else:
                num_linear_attn_layers += 1
        
        # === Step 1: Compute fixed-cost linear attention cache memory ===
        # conv_state per layer: [batch, conv_dim, kernel_size - 1] in bf16
        conv_state_bytes_per_layer = batch_size * conv_dim * (conv_kernel_size - 1) * 2  # bf16 = 2 bytes
        # recurrent_state per layer: [batch, num_v_heads, key_dim, value_dim] in fp32
        recurrent_state_bytes_per_layer = batch_size * num_v_heads * head_k_dim * head_v_dim * 4  # fp32 = 4 bytes
        
        linear_attn_bytes_per_layer = conv_state_bytes_per_layer + recurrent_state_bytes_per_layer
        total_linear_attn_bytes = num_linear_attn_layers * linear_attn_bytes_per_layer
        
        # === Step 2: Remaining budget for paged KV cache ===
        remaining_budget = free_memory_budget - total_linear_attn_bytes
        assert remaining_budget > 0, (
            f"Not enough memory for linear attention caches alone: "
            f"need {total_linear_attn_bytes / 1e9:.2f} GB, "
            f"budget {free_memory_budget / 1e9:.2f} GB"
        )
        
        # KV block memory: each block stores block_size tokens for ALL full-attn layers
        # Per block: num_full_attn_layers * block_size * num_kv_heads * head_dim * 2bytes * 2(k+v)
        kv_block_bytes = num_full_attn_layers * block_size * num_kv_heads * head_dim * 2 * 2
        
        if kv_block_bytes > 0:
            num_kvcache_blocks = int(remaining_budget) // kv_block_bytes
        else:
            num_kvcache_blocks = 0
        
        self.num_kvcache_blocks = num_kvcache_blocks
        self.num_full_attn_layers = num_full_attn_layers
        self.num_linear_attn_layers = num_linear_attn_layers
        
        # === Print memory breakdown ===
        print(f"Qwen3NextCacheParams memory breakdown:")
        print(f"  Linear attention layers: {num_linear_attn_layers}")
        print(f"    conv_state per layer:      {conv_state_bytes_per_layer / 1e6:.2f} MB  "
              f"[{batch_size} x {conv_dim} x {conv_kernel_size - 1}] bf16")
        print(f"    recurrent_state per layer:  {recurrent_state_bytes_per_layer / 1e6:.2f} MB  "
              f"[{batch_size} x {num_v_heads} x {head_k_dim} x {head_v_dim}] fp32")
        print(f"    total linear attn cache:   {total_linear_attn_bytes / 1e6:.2f} MB")
        print(f"  Full attention layers: {num_full_attn_layers}")
        print(f"    KV block size:             {kv_block_bytes / 1e6:.2f} MB per block "
              f"({block_size} tokens x {num_full_attn_layers} layers)")
        print(f"    num KV blocks:             {num_kvcache_blocks}")
        print(f"    max KV cache tokens:       {num_kvcache_blocks * block_size}")
        print(f"    total KV cache:            {num_kvcache_blocks * kv_block_bytes / 1e9:.2f} GB")
        print(f"  Free memory budget:          {free_memory_budget / 1e9:.2f} GB")
        print(f"  Remaining after allocation:  {(remaining_budget - num_kvcache_blocks * kv_block_bytes) / 1e6:.2f} MB")
        
        # === Step 3: Allocate caches ===
        self.conv_states = []
        self.recurrent_states = []
        
        # Allocate paged KV cache for full attention layers only
        # Shape: [2, num_full_attn_layers, num_blocks, block_size, num_kv_heads, head_dim]
        if num_full_attn_layers > 0 and num_kvcache_blocks > 0:
            self.kv_cache = torch.zeros(
                2, num_full_attn_layers, num_kvcache_blocks, block_size,
                num_kv_heads, head_dim,
                dtype=torch.bfloat16, device=device
            )
        else:
            self.kv_cache = None
        
        # Allocate per-layer caches
        full_attn_idx = 0
        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
            
            if lt == "full_attention":
                # No conv/recurrent state for full attention
                self.conv_states.append(None)
                self.recurrent_states.append(None)
                full_attn_idx += 1
            else:
                # Conv state: [batch, conv_dim, kernel_size - 1] bf16
                conv_state = torch.zeros(
                    batch_size, conv_dim, conv_kernel_size - 1,
                    device=device, dtype=torch.bfloat16
                )
                self.conv_states.append(conv_state)
                
                # Recurrent state: [batch, num_v_heads, key_dim, value_dim] fp32
                recurrent_state = torch.zeros(
                    batch_size, num_v_heads, head_k_dim, head_v_dim,
                    device=device, dtype=torch.float32
                )
                self.recurrent_states.append(recurrent_state)
    
    def get_kv_cache(self, full_attn_idx: int):
        """Get (k_cache, v_cache) for a full attention layer by its index.
        
        Returns the paged KV cache in original [num_blocks, block_size, num_kv_heads, head_dim] shape.
        This is the shape expected by flash_attn_varlen_func and flash_attn_with_kvcache.
        For store_kvcache (index_copy_), the attention layer will .view() to flatten blocks.
        """
        if self.kv_cache is None:
            return None, None
        return self.kv_cache[0, full_attn_idx], self.kv_cache[1, full_attn_idx]
    
    def reset(self):
        """Reset cache for new generation."""
        self.has_previous_state = False
        with torch.inference_mode():
            for i, state in enumerate(self.recurrent_states):
                if state is not None:
                    self.recurrent_states[i].zero_()
            for i, state in enumerate(self.conv_states):
                if state is not None:
                    self.conv_states[i].zero_()
    
    def set_seq_position(self, position: int):
        """Mark that we have previous state after prefill."""
        if position > 0:
            self.has_previous_state = True



# ============================================================================
# Linear Attention (Gated DeltaNet) - Main Module
# ============================================================================

class Qwen3NextGatedDeltaNetForEngine(nn.Module):
    """
    Gated DeltaNet linear attention for Qwen3-Next.
    
    This implements the exact same logic as HuggingFace's Qwen3NextGatedDeltaNet.
    """
    def __init__(self, config, layer_idx: int, use_tp: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.use_tp = use_tp
        
        # Get dimensions from config
        self.num_v_heads = getattr(config, 'linear_num_value_heads', 32)
        self.num_k_heads = getattr(config, 'linear_num_key_heads', 16)
        self.head_k_dim = getattr(config, 'linear_key_head_dim', 128)
        self.head_v_dim = getattr(config, 'linear_value_head_dim', 128)
        self.conv_kernel_size = getattr(config, 'linear_conv_kernel_dim', 4)
        
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.layer_norm_epsilon = getattr(config, 'rms_norm_eps', 1e-6)
        self.activation = getattr(config, 'hidden_act', 'silu')
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        
        # Projections (exactly as HF)
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
        
        # Time step projection parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Output norm with gating
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        
        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        
        # Delta rule kernel: prefer FLA's Triton kernel, fallback to pure PyTorch
        if FLA_AVAILABLE:
            self.chunk_gated_delta_rule = fla_chunk_gated_delta_rule
        else:
            self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    
    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives query, key, value, z, b, a tensors from mixed projections.
        This matches HF's implementation exactly.
        """
        # Reshape to [B, L, num_k_heads, interleaved_dim]
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads, 
            2 * self.num_v_heads // self.num_k_heads
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads, 
            self.num_v_heads // self.num_k_heads
        ]

        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)

        # [B, L, num_k_heads, (num_v_heads/num_k_heads) * head_v_dim] -> [B, L, num_v_heads, head_v_dim]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)

        return query, key, value, z, b, a
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params = None,
    ) -> torch.Tensor:
        # Apply padding mask
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        use_precomputed_states = (
            cache_params is not None
            and hasattr(cache_params, 'has_previous_state')
            and cache_params.has_previous_state
            and seq_len == 1
        )
        
        # Detect varlen prefill mode: packed sequences with cu_seqlens.
        # FLA's chunk_gated_delta_rule supports cu_seqlens natively.
        cu_seqlens = None
        num_seqs = batch_size  # default: one state per batch element
        is_varlen = False
        
        if (not use_precomputed_states and cache_params is not None 
                and batch_size == 1 and FLA_AVAILABLE):
            context = get_context()
            if hasattr(context, 'is_prefill') and context.is_prefill:
                cu_seqlens_q = getattr(context, 'cu_seqlens_q', None)
                if cu_seqlens_q is not None and len(cu_seqlens_q) > 2:
                    cu_seqlens = cu_seqlens_q.long()
                    num_seqs = len(cu_seqlens) - 1
                    is_varlen = True
        
        # Get conv/recurrent states from cache.
        # Use num_seqs (not batch_size) to handle varlen where batch=1 but N sequences.
        conv_state = None
        recurrent_state = None
        if cache_params is not None and hasattr(cache_params, 'conv_states'):
            conv_state_buf = cache_params.conv_states[self.layer_idx]
            recurrent_state_buf = cache_params.recurrent_states[self.layer_idx]
            if conv_state_buf is not None:
                conv_state = conv_state_buf[:num_seqs]
            if recurrent_state_buf is not None:
                recurrent_state = recurrent_state_buf[:num_seqs]
        
        # Project hidden states
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        
        # Unpack projections
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        
        # Flatten Q, K, V for conv
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
        
        # Concatenate and apply causal conv
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, D, L]
        
        if use_precomputed_states and conv_state is not None:
            # Decode: single token generation, update conv state
            state_len = conv_state.shape[-1]
            hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx][:num_seqs].copy_(
                    hidden_states_new[:, :, -state_len:]
                )
            out = F.conv1d(hidden_states_new, self.conv1d.weight, 
                          self.conv1d.bias, padding=0, groups=self.conv_dim)
            mixed_qkv = F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype)
        elif is_varlen:
            # Varlen prefill: process each segment through conv1d independently
            # to prevent cross-contamination across sequence boundaries.
            conv_state_layer = cache_params.conv_states[self.layer_idx] if cache_params is not None else None
            cu_list = cu_seqlens.tolist()
            output_segments = []
            for i in range(num_seqs):
                start, end = cu_list[i], cu_list[i + 1]
                seg = mixed_qkv[:, :, start:end]  # [1, D, seg_len]
                seg_len = end - start
                # Save conv state: last (K-1) tokens of pre-conv input for this seq
                if conv_state_layer is not None:
                    if seg_len >= self.conv_kernel_size - 1:
                        conv_state_layer[i].copy_(seg[0, :, -(self.conv_kernel_size - 1):])
                    else:
                        conv_state_layer[i].zero_()
                        conv_state_layer[i, :, -seg_len:].copy_(seg[0])
                # Apply causal conv per-segment (conv1d has left-padding built in)
                seg_out = F.silu(self.conv1d(seg)[:, :, :seg_len])
                output_segments.append(seg_out)
            mixed_qkv = torch.cat(output_segments, dim=-1)  # [1, D, total_L]
        else:
            # Standard single-seq prefill conv path
            if cache_params is not None and hasattr(cache_params, 'conv_states'):
                if mixed_qkv.shape[-1] >= self.conv_kernel_size - 1:
                    cache_params.conv_states[self.layer_idx][:num_seqs].copy_(
                        mixed_qkv[:, :, -(self.conv_kernel_size - 1):]
                    )
                else:
                    cache_params.conv_states[self.layer_idx][:num_seqs].copy_(
                        F.pad(mixed_qkv, (self.conv_kernel_size - 1 - mixed_qkv.shape[-1], 0))
                    )
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, L, D]
        
        # Split back to Q, K, V
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        
        # Reshape
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)
        
        # Compute beta and g (gate)
        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        
        # Expand K heads to V heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        
        # Apply gated delta rule
        if use_precomputed_states:
            # Decode: recurrent kernel for single-token generation
            recurrent_fn = fla_recurrent_gated_delta_rule if FLA_AVAILABLE else torch_recurrent_gated_delta_rule
            core_attn_out, last_recurrent_state = recurrent_fn(
                query, key, value,
                g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        elif is_varlen:
            # Varlen prefill: pass cu_seqlens to FLA for native packed-sequence support.
            # FLA handles per-sequence initial/final states with shape [N, H, K, V].
            core_attn_out, last_recurrent_state = fla_chunk_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            # Standard single-seq prefill (or PyTorch fallback)
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        
        # Update cache: write per-seq states back
        if cache_params is not None and hasattr(cache_params, 'recurrent_states'):
            cache_params.recurrent_states[self.layer_idx][:num_seqs].copy_(last_recurrent_state)
        
        # Apply gated norm
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        
        # Output projection
        output = self.out_proj(core_attn_out)
        
        return output


# ============================================================================
# Mixture of Experts (MoE)
# ============================================================================

class Qwen3NextTopKRouter(nn.Module):
    """Router for MoE layers."""
    def __init__(self, config, use_tp: bool = False):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 64)
        self.top_k = getattr(config, 'num_experts_per_tok', 8)
        self.hidden_size = config.hidden_size
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))
    
    def forward(self, hidden_states):
        # hidden_states: [batch * seq_len, hidden_size]
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        router_top_values, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            router_top_values = router_top_values / router_top_values.sum(dim=-1, keepdim=True)
        
        return router_logits, router_top_values.to(hidden_states.dtype), router_indices


class Qwen3NextExpertsForEngine(nn.Module):
    """MoE Experts layer with tensor parallelism support."""
    def __init__(self, config, use_tp: bool = False):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 64)
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = getattr(config, 'moe_intermediate_size', 1408)
        self.use_tp = use_tp
        
        tp_world_size = get_tp_world_size() if use_tp else 1
        
        if use_tp and tp_world_size > 1:
            # Shard experts across TP ranks
            self.experts_per_rank = self.num_experts // tp_world_size
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.experts_per_rank, 2 * self.moe_intermediate_size, self.hidden_size)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.experts_per_rank, self.hidden_size, self.moe_intermediate_size)
            )
        else:
            self.experts_per_rank = self.num_experts
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_size, self.moe_intermediate_size)
            )
        
        self.act_fn = nn.SiLU()
        
        # Flag: whether weights are stored in transposed layout for Triton
        self._weights_transposed = False
        # Pre-allocated intermediate buffers for Triton kernel (set by prepare_buffers)
        self._intermediate_cache = None  # [max_N*K, 2*I]
        self._output_cache = None        # [max_N*K, H]
    
    def prepare_pretransposed_weights(self, max_batch_size: int = 64):
        """Transpose expert weights IN-PLACE and pre-allocate intermediate buffers.
        
        Changes weight layout from [E, 2I, H] / [E, H, I] (original HF)
        to [E, H, 2I] / [E, I, H] (what Triton kernel expects).
        
        Also pre-allocates intermediate buffers to avoid torch.zeros per call.
        """
        new_gate_up = self.gate_up_proj.data.transpose(1, 2).contiguous()
        new_down = self.down_proj.data.transpose(1, 2).contiguous()
        self.gate_up_proj = nn.Parameter(new_gate_up, requires_grad=False)
        self.down_proj = nn.Parameter(new_down, requires_grad=False)
        self._weights_transposed = True
        
        # Pre-allocate intermediate buffers for Triton MoE
        # max_flat = max_batch_size * top_k
        top_k = getattr(self, '_top_k', 8)  # fallback
        max_flat = max_batch_size * top_k
        two_intermediate = new_gate_up.shape[2]  # 2*I (transposed layout)
        self._intermediate_cache = torch.zeros(
            max_flat, two_intermediate, dtype=torch.bfloat16, device=new_gate_up.device)
        self._output_cache = torch.zeros(
            max_flat, self.hidden_size, dtype=torch.bfloat16, device=new_gate_up.device)
        
        buf_mb = (self._intermediate_cache.numel() + self._output_cache.numel()) * 2 / 1e6
        print(f'    [MoE] Weights transposed + buffers allocated: '
              f'gate_up={new_gate_up.shape}, down={new_down.shape}, '
              f'buf={buf_mb:.1f} MB (max_batch={max_batch_size})', flush=True)
    
    def forward(self, hidden_states, top_k_indices, top_k_weights):
        """
        MoE forward with three strategies:
        1. Triton fused kernel (preferred) — zero temp memory, 2 kernel launches
        2. bmm (small decode batches) — fast but gathers weights into temp memory
        3. Expert loop (fallback) — memory-safe for large prefill batches
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            top_k_indices: [num_tokens, top_k]
            top_k_weights: [num_tokens, top_k]
        """
        tp_rank = get_tp_rank() if self.use_tp else 0
        tp_world_size = get_tp_world_size() if self.use_tp else 1
        
        num_tokens = hidden_states.shape[0]
        top_k = top_k_indices.shape[1]
        
        # Strategy selection:
        # - Triton (pre-transposed): fastest at ALL batch sizes (1.2ms@bs=1 to 2.1ms@bs=256)
        #   With pre-transposed weights + pre-allocated buffers, Triton beats BMM even at bs=1.
        # - Loop: memory-safe fallback when Triton is not available
        if TRITON_MOE_AVAILABLE:
            return self._forward_triton(hidden_states, top_k_indices, top_k_weights,
                                         tp_rank, tp_world_size, num_tokens, top_k)
        elif num_tokens * top_k <= 1024:
            # BMM fallback for small batches when Triton unavailable
            return self._forward_fused(hidden_states, top_k_indices, top_k_weights,
                                       tp_rank, tp_world_size, num_tokens, top_k)
        else:
            # Loop fallback for large batches when Triton unavailable
            return self._forward_loop(hidden_states, top_k_indices, top_k_weights,
                                      tp_rank, tp_world_size)
    
    def _forward_loop(self, hidden_states, top_k_indices, top_k_weights, tp_rank, tp_world_size):
        """Expert-by-expert loop (memory-efficient for large batches like prefill)."""
        final_output = torch.zeros_like(hidden_states)
        
        for expert_local_idx in range(self.experts_per_rank):
            expert_global_idx = expert_local_idx + tp_rank * self.experts_per_rank
            
            expert_mask = (top_k_indices == expert_global_idx)
            if not expert_mask.any():
                continue
            
            token_indices, top_k_positions = torch.where(expert_mask)
            expert_input = hidden_states[token_indices]
            expert_weights = top_k_weights[token_indices, top_k_positions].unsqueeze(-1)
            
            if self._weights_transposed:
                # gate_up_proj[i] is [H, 2I], need input @ weight = [N, 2I]
                gate_up = expert_input @ self.gate_up_proj[expert_local_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                expert_output = self.act_fn(gate) * up
                # down_proj[i] is [I, H], need intermediate @ weight = [N, H]
                expert_output = expert_output @ self.down_proj[expert_local_idx]
            else:
                gate, up = F.linear(expert_input, self.gate_up_proj[expert_local_idx]).chunk(2, dim=-1)
                expert_output = self.act_fn(gate) * up
                expert_output = F.linear(expert_output, self.down_proj[expert_local_idx])
            
            expert_output = expert_output * expert_weights
            final_output.index_add_(0, token_indices, expert_output.to(final_output.dtype))
        
        if self.use_tp and tp_world_size > 1:
            final_output = reduce_from_tp(final_output)
        return final_output
    
    def _forward_triton(self, hidden_states, top_k_indices, top_k_weights,
                        tp_rank, tp_world_size, num_tokens, top_k):
        """Triton fused MoE kernel: zero temp memory, 2 kernel launches.
        
        Works for all batch sizes (decode and prefill).
        Supports TP via expert index remapping + all-reduce.
        """
        if self.use_tp and tp_world_size > 1:
            # Remap global expert indices to local: subtract rank offset
            local_indices = top_k_indices - tp_rank * self.experts_per_rank
            # Zero out weights for experts not on this rank
            valid_mask = (local_indices >= 0) & (local_indices < self.experts_per_rank)
            local_weights = top_k_weights * valid_mask.to(top_k_weights.dtype)
            # Clamp to valid range (masked experts have weight=0 so output is zeroed)
            local_indices = local_indices.clamp(0, self.experts_per_rank - 1)
        else:
            local_indices = top_k_indices
            local_weights = top_k_weights
        
        if self._weights_transposed:
            # Weights already in Triton layout: gate_up=[E, H, 2I], down=[E, I, H]
            output = triton_fused_moe(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                local_weights,
                local_indices,
                top_k=top_k,
                num_experts=self.experts_per_rank,
                w1_pre_transposed=self.gate_up_proj.data,
                w2_pre_transposed=self.down_proj.data,
                intermediate_cache=self._intermediate_cache,
                output_cache=self._output_cache,
            )
        else:
            # Original layout — fused_moe will transpose internally
            output = triton_fused_moe(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                local_weights,
                local_indices,
                top_k=top_k,
                num_experts=self.experts_per_rank,
            )
        
        # All-reduce across TP ranks (each rank computed its subset of experts)
        if self.use_tp and tp_world_size > 1:
            output = reduce_from_tp(output)
        
        return output
    
    def _forward_fused(self, hidden_states, top_k_indices, top_k_weights,
                       tp_rank, tp_world_size, num_tokens, top_k):
        """Fused bmm approach (fast for small batches like decode)."""
        
        # Map global expert indices to local indices for this TP rank
        if self.use_tp and tp_world_size > 1:
            local_indices = top_k_indices - tp_rank * self.experts_per_rank
            # Mask out experts not on this rank
            valid_mask = (local_indices >= 0) & (local_indices < self.experts_per_rank)
        else:
            local_indices = top_k_indices
            valid_mask = None
        
        # Flatten: each (token, expert_slot) pair becomes one computation
        flat_local_indices = local_indices.view(-1)              # [N * top_k]
        flat_weights = top_k_weights.view(-1, 1)                 # [N * top_k, 1]
        
        # Clamp indices to valid range for gathering (masked ones will be zeroed)
        flat_local_indices_clamped = flat_local_indices.clamp(0, self.experts_per_rank - 1)
        
        # Expand hidden states: each token repeated top_k times
        expanded_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_states.shape[-1])
        # expanded_hidden: [N * top_k, hidden_size]
        
        # Gather expert weights for each pair
        selected_gate_up = self.gate_up_proj[flat_local_indices_clamped]
        selected_down = self.down_proj[flat_local_indices_clamped]
        
        if self._weights_transposed:
            # gate_up: [N*K, H, 2*I], input [N*K, 1, H] @ [N*K, H, 2*I] → [N*K, 2*I]
            gate_up_out = torch.bmm(
                expanded_hidden.unsqueeze(1),
                selected_gate_up,
            ).squeeze(1)
        else:
            # gate_up: [N*K, 2*I, H], input [N*K, 1, H] @ [N*K, 2*I, H]^T → [N*K, 2*I]
            gate_up_out = torch.bmm(
                expanded_hidden.unsqueeze(1),
                selected_gate_up.transpose(1, 2),
            ).squeeze(1)
        
        gate, up = gate_up_out.chunk(2, dim=-1)
        expert_out = self.act_fn(gate) * up         # [N*K, I]
        
        if self._weights_transposed:
            # down: [N*K, I, H], input [N*K, 1, I] @ [N*K, I, H] → [N*K, H]
            expert_out = torch.bmm(
                expert_out.unsqueeze(1),
                selected_down,
            ).squeeze(1)
        else:
            # down: [N*K, H, I], input [N*K, 1, I] @ [N*K, H, I]^T → [N*K, H]
            expert_out = torch.bmm(
                expert_out.unsqueeze(1),
                selected_down.transpose(1, 2),
            ).squeeze(1)
        
        # Apply routing weights
        expert_out = expert_out * flat_weights
        
        # Zero out invalid experts (for TP)
        if valid_mask is not None:
            flat_valid = valid_mask.view(-1).unsqueeze(-1)  # [N*K, 1]
            expert_out = expert_out * flat_valid.to(expert_out.dtype)
        
        # Scatter-add back to token positions
        token_indices = torch.arange(num_tokens, device=hidden_states.device
                                     ).unsqueeze(1).expand(-1, top_k).reshape(-1)
        
        final_output = torch.zeros_like(hidden_states)
        final_output.index_add_(0, token_indices, expert_out.to(final_output.dtype))
        
        # All-reduce if using TP for experts
        if self.use_tp and tp_world_size > 1:
            final_output = reduce_from_tp(final_output)
        
        return final_output


class Qwen3NextSparseMoeBlockForEngine(nn.Module):
    """Sparse MoE block combining router, experts, and shared expert."""
    def __init__(self, config, use_tp: bool = False):
        super().__init__()
        self.gate = Qwen3NextTopKRouter(config, use_tp=use_tp)
        self.experts = Qwen3NextExpertsForEngine(config, use_tp=use_tp)
        
        # Shared expert uses shared_expert_intermediate_size
        shared_intermediate = getattr(config, 'shared_expert_intermediate_size', config.intermediate_size)
        self.shared_expert = Qwen3NextMLPForEngine(config, use_tp=use_tp, intermediate_size=shared_intermediate)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
    
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # Shared expert
        shared_output = self.shared_expert(hidden_flat)
        shared_gate = F.sigmoid(self.shared_expert_gate(hidden_flat))
        shared_output = shared_gate * shared_output
        
        # Routed experts
        _, routing_weights, selected_experts = self.gate(hidden_flat)
        expert_output = self.experts(hidden_flat, selected_experts, routing_weights)
        
        # Combine
        output = expert_output + shared_output
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output


class Qwen3NextRotaryEmbedding(nn.Module):
    """Rotary position embedding for Qwen3-Next."""
    def __init__(self, dim, max_position_embeddings=131072, base=1000000.0, device=None):
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
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextAttentionForEngine(nn.Module):
    """
    Qwen3-Next Attention adapted for in-house LLM Engine.
    
    Key changes from HuggingFace version:
    - Uses paged attention via cache_params (no stateful k_cache/v_cache on module)
    - cache_params=None means training, cache_params!=None means inference
    - Uses flash_attn_varlen_func and flash_attn_with_kvcache
    - Supports tensor parallelism
    """
    def __init__(self, config, layer_idx: int, full_attn_idx: int = 0, use_tp: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.full_attn_idx = full_attn_idx  # Index among full-attention layers only
        self.use_tp = use_tp
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Use config.head_dim if explicitly set (Qwen3-Next may have different head_dim)
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        
        # For tensor parallel, split heads across GPUs
        tp_world_size = get_tp_world_size() if use_tp else 1
        self.num_heads_per_partition = self.num_heads // tp_world_size
        
        # Handle GQA: if KV heads < TP world size, replicate KV instead of sharding
        self.kv_is_replicated = self.num_key_value_heads < tp_world_size
        if self.kv_is_replicated:
            self.num_kv_heads_per_partition = self.num_key_value_heads  # Keep all KV heads
        else:
            self.num_kv_heads_per_partition = self.num_key_value_heads // tp_world_size
        
        # GQA groups: ratio of Q heads to KV heads in this partition
        self.num_key_value_groups = self.num_heads_per_partition // self.num_kv_heads_per_partition
        
        if use_tp and tp_world_size > 1:
            # Projections with tensor parallelism
            # q_proj outputs q and gate concatenated, so output is num_heads * head_dim * 2
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, 
                self.num_heads * self.head_dim * 2, 
                bias=False, 
                gather_output=False
            )
            
            # KV projections: shard if enough KV heads, otherwise replicate
            if self.kv_is_replicated:
                # Replicate KV projections (standard linear, no sharding)
                self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
                self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            else:
                self.k_proj = ColumnParallelLinear(
                    self.hidden_size, 
                    self.num_key_value_heads * self.head_dim, 
                    bias=False, 
                    gather_output=False
                )
                self.v_proj = ColumnParallelLinear(
                    self.hidden_size, 
                    self.num_key_value_heads * self.head_dim, 
                    bias=False, 
                    gather_output=False
                )
            
            self.o_proj = RowParallelLinear(
                self.num_heads * self.head_dim, 
                self.hidden_size, 
                bias=False, 
                input_is_parallel=True
            )
        else:
            self.kv_is_replicated = False  # No TP, no replication needed
            # Standard projections
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # QK norm (Qwen3-Next specific)
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=get_rms_norm_eps(config))
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=get_rms_norm_eps(config))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        cache_params = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        
        # Use partition sizes for TP
        num_heads = self.num_heads_per_partition if self.use_tp else self.num_heads
        num_kv_heads = self.num_kv_heads_per_partition if self.use_tp else self.num_key_value_heads
        
        # Compute Q, K, V (Qwen3-Next has q_proj output q and gate interleaved per head)
        # HF: view to (bsz, q_len, num_heads, head_dim * 2) then chunk on last dim
        # This ensures Q and Gate are correctly paired per head
        qg = self.q_proj(hidden_states)
        qg = qg.view(bsz, q_len, num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)  # Each: (bsz, q_len, num_heads, head_dim)
        gate = gate.reshape(bsz, q_len, -1)  # Flatten gate: (bsz, q_len, num_heads * head_dim)
        
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, self.head_dim)
        
        # Apply QK norm (apply on the head_dim dimension)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Transpose for attention: [bsz, num_heads, q_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        # Apply rotary embeddings
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Transpose back for flash attention: [bsz, q_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2).to(hidden_states.dtype)
        key_states = key_states.transpose(1, 2).to(hidden_states.dtype)
        
        # Look up KV cache from cache_params (paged attention)
        k_cache_paged, v_cache_paged = None, None
        if cache_params is not None and hasattr(cache_params, 'get_kv_cache'):
            k_cache_paged, v_cache_paged = cache_params.get_kv_cache(self.full_attn_idx)
        
        key_cache = None
        value_cache = None
        page_attention = False
        
        if k_cache_paged is not None and v_cache_paged is not None:
            # Inference mode with paged attention
            # k_cache_paged shape: [num_blocks, block_size, num_kv_heads, head_dim]
            key_cache = None
            value_cache = None
            context = get_context()
            
            query_states = query_states.view(-1, num_heads, self.head_dim).contiguous()
            key_states = key_states.view(-1, num_kv_heads, self.head_dim).contiguous()
            value_states = value_states.view(-1, num_kv_heads, self.head_dim).contiguous()
            
            # Flatten cache for index_copy_: [num_blocks * block_size, num_kv_heads, head_dim]
            k_cache_flat = k_cache_paged.view(-1, k_cache_paged.shape[-2], k_cache_paged.shape[-1])
            v_cache_flat = v_cache_paged.view(-1, v_cache_paged.shape[-2], v_cache_paged.shape[-1])
            store_kvcache(key_states, value_states, k_cache_flat, v_cache_flat, context.slot_mapping)
            page_attention = True
        else:
            # Fallback: simple KV cache
            key_cache = key_states
            value_cache = value_states
        
        if page_attention and FLASH_ATTN_AVAILABLE:
            context = get_context()
            if context.is_prefill:
                attn_output = flash_attn_varlen_func(
                    query_states, k_cache_paged, v_cache_paged,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    causal=True,
                    block_table=context.block_tables
                )
            else:
                attn_output = flash_attn_with_kvcache(
                    query_states.unsqueeze(1),
                    k_cache_paged, v_cache_paged,
                    cache_seqlens=context.context_lens,
                    causal=True,
                    block_table=context.block_tables
                )
        else:
            # Fallback: standard attention
            key_states_expanded = key_cache.repeat_interleave(self.num_key_value_groups, dim=2)
            value_states_expanded = value_cache.repeat_interleave(self.num_key_value_groups, dim=2)
            
            # [bsz, num_heads, q_len, head_dim] @ [bsz, num_heads, head_dim, kv_len]
            query_for_attn = query_states.transpose(1, 2) if not page_attention else query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_for_attn = key_states_expanded.transpose(1, 2)
            value_for_attn = value_states_expanded.transpose(1, 2)
            
            attn_weights = torch.matmul(query_for_attn, key_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_for_attn.dtype)
            attn_output = torch.matmul(attn_weights, value_for_attn)
            attn_output = attn_output.transpose(1, 2)
        
        # Reshape and apply gate
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, key_cache, value_cache


class Qwen3NextMLPForEngine(nn.Module):
    """MLP for Qwen3-Next with tensor parallelism support."""
    def __init__(self, config, use_tp: bool = False, intermediate_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.use_tp = use_tp
        
        tp_world_size = get_tp_world_size() if use_tp else 1
        
        if use_tp and tp_world_size > 1:
            # gate_proj and up_proj are column parallel
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size, self.intermediate_size, bias=False, gather_output=False
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size, self.intermediate_size, bias=False, gather_output=False
            )
            # down_proj is row parallel
            self.down_proj = RowParallelLinear(
                self.intermediate_size, self.hidden_size, bias=False, input_is_parallel=True
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextDecoderLayerForEngine(nn.Module):
    """
    Hybrid decoder layer for Qwen3-Next.
    
    - Full attention layers: use our implementation (for paged attention support)
    - Linear attention layers: use our implementation (matches HF's GatedDeltaNet)
    """
    def __init__(self, config, layer_idx: int, full_attn_idx: int = -1, use_tp: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.full_attn_idx = full_attn_idx  # -1 for linear attention layers
        self.use_tp = use_tp
        
        # Determine layer type from config
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is not None and layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "full_attention"
        
        if self.layer_type == "linear_attention":
            # Linear attention - use our GatedDeltaNet implementation
            self.linear_attn = Qwen3NextGatedDeltaNetForEngine(config, layer_idx, use_tp=use_tp)
            self.self_attn = None
            self._use_hf_layer = False
            
            # LayerNorms
            self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
            self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
            
            # MLP - determine if MoE or dense (same logic as full attention)
            num_experts = getattr(config, 'num_experts', 0)
            decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
            mlp_only_layers = getattr(config, 'mlp_only_layers', [])
            
            use_moe = (num_experts > 0 and 
                      (layer_idx + 1) % decoder_sparse_step == 0 and 
                      layer_idx not in mlp_only_layers)
            
            if use_moe:
                self.mlp = Qwen3NextSparseMoeBlockForEngine(config, use_tp=use_tp)
                self.is_moe = True
            else:
                self.mlp = Qwen3NextMLPForEngine(config, use_tp=use_tp)
                self.is_moe = False
        else:
            # Full attention - use our implementation
            self.self_attn = Qwen3NextAttentionForEngine(config, layer_idx, full_attn_idx=full_attn_idx, use_tp=use_tp)
            self.linear_attn = None
            self._use_hf_layer = False
            
            # LayerNorms
            self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
            self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
            
            # MLP - determine if MoE or dense
            num_experts = getattr(config, 'num_experts', 0)
            decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
            mlp_only_layers = getattr(config, 'mlp_only_layers', [])
            
            use_moe = (num_experts > 0 and 
                      (layer_idx + 1) % decoder_sparse_step == 0 and 
                      layer_idx not in mlp_only_layers)
            
            if use_moe:
                self.mlp = Qwen3NextSparseMoeBlockForEngine(config, use_tp=use_tp)
                self.is_moe = True
            else:
                self.mlp = Qwen3NextMLPForEngine(config, use_tp=use_tp)
                self.is_moe = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        cache_params = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Sub-component debug: only during prefill (when _tp_debug_enabled), first 2 layers
        _dbg = _tp_debug_enabled and self.layer_idx < 2
        if _dbg:
            _r = get_tp_rank()
        
        # Token mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.layer_type == "linear_attention" and self.linear_attn is not None:
            if _dbg:
                torch.cuda.synchronize()
                print(f'      [layer {self.layer_idx} rank={_r}] entering GatedDeltaNet', flush=True)
            hidden_states = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                cache_params=cache_params,
            )
            if _dbg:
                torch.cuda.synchronize()
                print(f'      [layer {self.layer_idx} rank={_r}] GatedDeltaNet done', flush=True)
            k_cache, v_cache = None, None
        else:
            if _dbg:
                torch.cuda.synchronize()
                print(f'      [layer {self.layer_idx} rank={_r}] entering FullAttn', flush=True)
            hidden_states, k_cache, v_cache = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                cache_params=cache_params,
            )
            if _dbg:
                torch.cuda.synchronize()
                print(f'      [layer {self.layer_idx} rank={_r}] FullAttn done', flush=True)
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if _dbg:
            torch.cuda.synchronize()
            print(f'      [layer {self.layer_idx} rank={_r}] entering MLP/MoE (is_moe={self.is_moe})', flush=True)
        hidden_states = self.mlp(hidden_states)
        if _dbg:
            torch.cuda.synchronize()
            print(f'      [layer {self.layer_idx} rank={_r}] MLP/MoE done', flush=True)
        hidden_states = residual + hidden_states
        
        return hidden_states, k_cache, v_cache


class Qwen3NextModelForEngine(nn.Module):
    """Qwen3-Next Model backbone adapted for LLM Engine with TP support."""
    def __init__(self, config, use_tp: bool = False):
        super().__init__()
        self.config = config
        self.use_tp = use_tp
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Build layers with full_attn_idx tracking
        layer_types = getattr(config, 'layer_types', ['full_attention'] * config.num_hidden_layers)
        layers = []
        full_attn_counter = 0
        for layer_idx in range(config.num_hidden_layers):
            lt = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
            if lt == "full_attention":
                layers.append(Qwen3NextDecoderLayerForEngine(config, layer_idx, full_attn_idx=full_attn_counter, use_tp=use_tp))
                full_attn_counter += 1
            else:
                layers.append(Qwen3NextDecoderLayerForEngine(config, layer_idx, full_attn_idx=-1, use_tp=use_tp))
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
        
        # Get rope_theta with fallback (Qwen3-Next may use different attribute names)
        rope_theta = getattr(config, 'rope_theta', None)
        if rope_theta is None:
            rope_theta = getattr(config, 'rope_base', None)
        if rope_theta is None:
            rope_theta = getattr(config, 'rotary_pct_base', 1000000.0)  # Default
        
        # Use explicit head_dim if set, otherwise compute from hidden_size
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        
        self.rotary_emb = Qwen3NextRotaryEmbedding(
            head_dim,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 131072),
            base=rope_theta,
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params = None,
        _profile_layers: bool = False,
    ) -> Tuple[torch.Tensor, List]:
        batch_size, seq_length = input_ids.shape[:2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Compute rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        next_cache = []
        num_layers = len(self.layers)
        tp_rank = get_tp_rank()
        
        # Per-layer profiling: uses CUDA events for accurate GPU timing
        if _profile_layers:
            events_start = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
            events_attn = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
            events_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
        
        for layer_idx, layer in enumerate(self.layers):
            if _tp_debug_enabled and (layer_idx % 4 == 0 or layer_idx == num_layers - 1):
                torch.cuda.synchronize()
                print(f'    [backbone rank={tp_rank}] layer {layer_idx}/{num_layers} '
                      f'type={layer.layer_type} hidden={hidden_states.shape}', flush=True)
            
            if _profile_layers:
                events_start[layer_idx].record()
                # --- Attention ---
                residual = hidden_states
                hidden_states_ln = layer.input_layernorm(hidden_states)
                if layer.layer_type == "linear_attention" and layer.linear_attn is not None:
                    hidden_states_attn = layer.linear_attn(
                        hidden_states_ln, attention_mask=attention_mask, cache_params=cache_params,
                    )
                    k_cache, v_cache = None, None
                else:
                    hidden_states_attn, k_cache, v_cache = layer.self_attn(
                        hidden_states_ln, attention_mask=attention_mask,
                        position_ids=position_ids, cos=cos, sin=sin, cache_params=cache_params,
                    )
                hidden_states = residual + hidden_states_attn
                events_attn[layer_idx].record()
                # --- MLP/MoE ---
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
                events_end[layer_idx].record()
                next_cache.append((k_cache, v_cache))
            else:
                hidden_states, k_cache, v_cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cos=cos,
                    sin=sin,
                    cache_params=cache_params,
                )
                next_cache.append((k_cache, v_cache))
        
        if _tp_debug_enabled:
            torch.cuda.synchronize()
            print(f'    [backbone rank={tp_rank}] all {num_layers} layers done', flush=True)
        
        if _profile_layers:
            torch.cuda.synchronize()
            layer_types = getattr(self.config, 'layer_types', ['full_attention'] * num_layers)
            attn_times = []
            mlp_times = []
            total_times = []
            for i in range(num_layers):
                attn_ms = events_start[i].elapsed_time(events_attn[i])
                mlp_ms = events_attn[i].elapsed_time(events_end[i])
                total_ms = events_start[i].elapsed_time(events_end[i])
                lt = layer_types[i] if i < len(layer_types) else "full_attention"
                attn_times.append((i, lt, attn_ms))
                mlp_times.append((i, lt, mlp_ms))
                total_times.append((i, lt, total_ms))
            
            # Summary
            linear_attn_total = sum(t for _, lt, t in attn_times if lt == "linear_attention")
            full_attn_total = sum(t for _, lt, t in attn_times if lt == "full_attention")
            moe_total = sum(t for i, lt, t in mlp_times if hasattr(self.layers[i], 'is_moe') and self.layers[i].is_moe)
            dense_mlp_total = sum(t for i, lt, t in mlp_times if not (hasattr(self.layers[i], 'is_moe') and self.layers[i].is_moe))
            all_total = sum(t for _, _, t in total_times)
            
            print(f'\n  === LAYER PROFILING (rank={tp_rank}, batch={batch_size}, seq_len={seq_length}) ===', flush=True)
            print(f'  {"Layer":>6} {"Type":>17} {"Attn(ms)":>10} {"MLP(ms)":>10} {"Total(ms)":>10}', flush=True)
            print(f'  {"-"*57}', flush=True)
            for i in range(num_layers):
                _, lt, attn_ms = attn_times[i]
                _, _, mlp_ms = mlp_times[i]
                _, _, tot_ms = total_times[i]
                is_moe = hasattr(self.layers[i], 'is_moe') and self.layers[i].is_moe
                mlp_label = "MoE" if is_moe else "MLP"
                print(f'  {i:>6} {lt:>17} {attn_ms:>10.2f} {mlp_ms:>10.2f} {tot_ms:>10.2f}  {mlp_label}', flush=True)
            print(f'  {"-"*57}', flush=True)
            print(f'  GatedDeltaNet (36 layers): {linear_attn_total:>8.1f} ms', flush=True)
            print(f'  Full Attention (12 layers): {full_attn_total:>7.1f} ms', flush=True)
            print(f'  MoE MLP:                   {moe_total:>8.1f} ms', flush=True)
            print(f'  Dense MLP:                 {dense_mlp_total:>8.1f} ms', flush=True)
            print(f'  All layers total:          {all_total:>8.1f} ms', flush=True)
            print(f'  ===================================================\n', flush=True)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache


class Qwen3NextForLLMEngine(nn.Module):
    """
    Qwen3-Next For Causal LM adapted for in-house LLM Engine.
    Key interface requirement for LLMEngine:
    - forward(input_ids, position_ids, cache_params=..., logits_to_keep=...) -> (logits, cache)
    - cache_params=None for training, cache_params=Qwen3NextCacheParams for inference
    Supports tensor parallelism for multi-GPU inference.
    """
    def __init__(self, config, use_tp: bool = False):
        super().__init__()
        self.config = config
        self.use_tp = use_tp
        self.model = Qwen3NextModelForEngine(config, use_tp=use_tp)
        # lm_head can be column parallel or regular
        tp_world_size = get_tp_world_size() if use_tp else 1
        if use_tp and tp_world_size > 1:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False, gather_output=True
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def allocate_cache(
        self,
        batch_size: int, 
        free_memory_budget: int,
        device: torch.device,
        block_size: int = 256,
    ) -> Qwen3NextCacheParams:
        """
        Allocate cache for Qwen3-Next hybrid model.
        
        Strategy:
        1. Compute fixed-cost memory for linear attention (conv + recurrent states)
        2. Use remaining GPU memory for paged KV cache blocks (full attention only)
        3. Return cache_params to be passed as argument to forward()
        
        Args:
            batch_size: Number of sequences (for linear attention states)
            free_memory_budget: Available GPU memory in bytes
            device: Target device
            block_size: Paged KV cache block size (tokens per block)
        
        Returns:
            Qwen3NextCacheParams to pass to forward(cache_params=...)
        """
        cache = Qwen3NextCacheParams(
            config=self.config,
            batch_size=batch_size,
            free_memory_budget=free_memory_budget,
            device=device,
            block_size=block_size,
        )
        return cache

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[torch.Tensor] = None,
        cache_params = None,
        _profile_layers: bool = False,
    ) -> Tuple[torch.Tensor, List]:
        """
        Forward pass compatible with LLMEngine.
        
        Args:
            cache_params: None for training. Qwen3NextCacheParams for inference.
                          Contains paged KV cache (full attn) + conv/recurrent states (linear attn).
        
        Returns: (logits, past_key_values)
        """
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache_params=cache_params,
            _profile_layers=_profile_layers,
        )
        
        # Handle logits_to_keep for efficient prefill
        if logits_to_keep is not None:
            hidden_states = hidden_states.squeeze(0)[logits_to_keep]
            hidden_states = hidden_states.unsqueeze(0)
        
        if _tp_debug_enabled:
            print(f'    [rank={get_tp_rank()}] lm_head start, hidden={hidden_states.shape}', flush=True)
        logits = self.lm_head(hidden_states)
        if _tp_debug_enabled:
            print(f'    [rank={get_tp_rank()}] lm_head done, logits={logits.shape}', flush=True)
        
        return logits, next_cache


def load_qwen3_next_for_engine(
    model_path: str, 
    device: str = "cuda", 
    torch_dtype=torch.bfloat16,
    tensor_parallel_size: int = 1
):
    """
    Load Qwen3-Next model and convert to LLMEngine-compatible format.
    
    This loads the HuggingFace model and copies weights to our custom model.
    Supports tensor parallelism for multi-GPU inference.
    
    Args:
        model_path: HuggingFace model path or local path
        device: Device to load model to (e.g., "cuda", "cuda:0")
        torch_dtype: Data type for model weights
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        model: Qwen3NextForLLMEngine instance
        tokenizer: AutoTokenizer instance
        llm_config: Config object for LLMEngine
    """
    # Initialize tensor parallelism if needed
    use_tp = tensor_parallel_size > 1
    if use_tp:
        init_tensor_parallel(tensor_parallel_size)
        rank = get_tp_rank()
        device = f"cuda:{rank}"
        if rank == 0:
            print(f"Tensor parallel enabled: {tensor_parallel_size} GPUs")
    
    is_main = get_tp_rank() == 0
    if is_main:
        print(f"Loading Qwen3-Next from {model_path}...")
    
    # Load HuggingFace model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Print layer configuration (only rank 0)
    if is_main:
        layer_types = getattr(hf_config, 'layer_types', None)
        if layer_types:
            print(f"Layer types ({len(layer_types)} layers):")
            full_attn_count = sum(1 for t in layer_types if t == "full_attention")
            linear_attn_count = sum(1 for t in layer_types if t == "linear_attention")
            print(f"  - full_attention: {full_attn_count}")
            print(f"  - linear_attention: {linear_attn_count}")
            print(f"  - pattern: {layer_types[:10]}..." if len(layer_types) > 10 else f"  - pattern: {layer_types}")
        
        # Print MoE configuration
        num_experts = getattr(hf_config, 'num_experts', 0)
        decoder_sparse_step = getattr(hf_config, 'decoder_sparse_step', 1)
        if num_experts > 0:
            print(f"MoE config: num_experts={num_experts}, decoder_sparse_step={decoder_sparse_step}")
        
        # Print attention dimensions
        head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)
        num_q_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        gqa_ratio = num_q_heads // num_kv_heads
        print(f"Attention config:")
        print(f"  - Query heads: {num_q_heads}")
        print(f"  - KV heads: {num_kv_heads}")
        print(f"  - GQA ratio: {gqa_ratio} (Q heads per KV head)")
        print(f"  - head_dim: {head_dim}")
        if tensor_parallel_size > 1:
            kv_is_replicated = num_kv_heads < tensor_parallel_size
            if kv_is_replicated:
                print(f"  - KV heads replicated (num_kv_heads={num_kv_heads} < tp_size={tensor_parallel_size})")
            else:
                print(f"  - KV heads sharded: {num_kv_heads // tensor_parallel_size} per GPU")
        
        # Print GatedDeltaNet config
        print(f"GatedDeltaNet config:")
        print(f"  - linear_num_value_heads: {getattr(hf_config, 'linear_num_value_heads', 'NOT SET')}")
        print(f"  - linear_num_key_heads: {getattr(hf_config, 'linear_num_key_heads', 'NOT SET')}")
        print(f"  - linear_key_head_dim: {getattr(hf_config, 'linear_key_head_dim', 'NOT SET')}")
        print(f"  - linear_value_head_dim: {getattr(hf_config, 'linear_value_head_dim', 'NOT SET')}")
        print(f"  - linear_conv_kernel_dim: {getattr(hf_config, 'linear_conv_kernel_dim', 'NOT SET')}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",  # Load to CPU first for weight transfer
        trust_remote_code=True
    )
    
    # Create our engine-compatible model with TP support
    # Initialize model on target device to avoid CPU memory pressure
    if is_main:
        print(f"Creating engine model on {device}...", flush=True)
    model = Qwen3NextForLLMEngine(hf_config, use_tp=use_tp)
    model = model.to(device=device, dtype=torch_dtype)
    
    # Copy weights from HuggingFace model directly to GPU (handles TP sharding)
    is_main = get_tp_rank() == 0
    if is_main:
        print("Copying weights to engine-compatible model...")
    _copy_weights(hf_model, model, hf_config, use_tp=use_tp, target_device=device, target_dtype=torch_dtype)
    
    model.eval()
    
    # Pre-transpose MoE expert weights for Triton kernel (eliminates runtime copies)
    # Also pre-allocate intermediate buffers to avoid malloc per forward call
    if TRITON_MOE_AVAILABLE:
        # Use a reasonable default max_batch_size for buffer pre-allocation
        # (will be overridden if HybridLLMEngine is created with different size)
        default_max_batch = 64
        if is_main:
            print(f"Pre-transposing MoE expert weights + allocating buffers (max_batch={default_max_batch})...")
        for layer in model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                layer.mlp.experts.prepare_pretransposed_weights(max_batch_size=default_max_batch)
        if is_main:
            print("Pre-transposed MoE weights ready.")
    
    # Create LLMEngine-compatible config
    # Use hf_config directly — it has all fields needed by both LLMEngine and Qwen3NextCacheParams
    llm_config = hf_config
    llm_config.eos_token_id = tokenizer.eos_token_id
    llm_config.tensor_parallel_size = tensor_parallel_size
    
    # Clean up HF model
    del hf_model
    torch.cuda.empty_cache()
    
    if is_main:
        print("Model loaded and converted successfully!")
    return model, tokenizer, llm_config


def _copy_weights(hf_model, engine_model, config, use_tp: bool = False, 
                  target_device: str = "cuda", target_dtype=None):
    """Copy weights from HuggingFace model to our engine-compatible model.
    
    For tensor parallelism, this shards the weights across GPUs.
    Supports hybrid architecture: full attention + linear attention (GatedDeltaNet) + MoE.
    
    Memory optimization: Copies weights directly to target device and frees HF layers progressively.
    """
    tp_rank = get_tp_rank() if use_tp else 0
    tp_world_size = get_tp_world_size() if use_tp else 1
    is_main = tp_rank == 0
    
    if is_main:
        print(f"  TP rank: {tp_rank}, world_size: {tp_world_size}", flush=True)
        print(f"  Target device: {target_device}, dtype: {target_dtype}", flush=True)
    
    def _copy_to_device(src_tensor, dst_tensor):
        """Copy tensor from CPU to target device with dtype conversion."""
        dst_tensor.data.copy_(src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))
    
    # Copy embeddings (replicated across all ranks)
    if is_main:
        print("  Copying embeddings...", flush=True)
        print(f"    HF embed shape: {hf_model.model.embed_tokens.weight.shape}", flush=True)
        print(f"    Engine embed shape: {engine_model.model.embed_tokens.weight.shape}", flush=True)
    _copy_to_device(hf_model.model.embed_tokens.weight.data, engine_model.model.embed_tokens.weight)
    
    # Copy lm_head (sharded if TP)
    if is_main:
        print("  Copying lm_head...", flush=True)
        print(f"    HF lm_head shape: {hf_model.lm_head.weight.shape}", flush=True)
        print(f"    Engine lm_head shape: {engine_model.lm_head.weight.shape}", flush=True)
    if use_tp and tp_world_size > 1:
        vocab_size = config.vocab_size
        shard_size = vocab_size // tp_world_size
        start = tp_rank * shard_size
        end = start + shard_size
        _copy_to_device(hf_model.lm_head.weight.data[start:end], engine_model.lm_head.weight)
    else:
        _copy_to_device(hf_model.lm_head.weight.data, engine_model.lm_head.weight)
    
    # Free HF embeddings and lm_head to save CPU memory
    hf_model.model.embed_tokens = None
    hf_model.lm_head = None
    
    # Copy final norm (replicated)
    if is_main:
        print("  Copying final norm...", flush=True)
    _copy_to_device(hf_model.model.norm.weight.data, engine_model.model.norm.weight)
    hf_model.model.norm = None
    
    # Copy layers
    num_layers = config.num_hidden_layers
    if is_main:
        print(f"  Copying {num_layers} layers...", flush=True)
    for layer_idx in range(num_layers):
        if is_main and (layer_idx % 10 == 0 or layer_idx == num_layers - 1):
            print(f"    Layer {layer_idx}/{num_layers}...", flush=True)
        hf_layer = hf_model.model.layers[layer_idx]
        engine_layer = engine_model.model.layers[layer_idx]
        
        # Debug: print HF layer structure for first layer
        if is_main and layer_idx == 0:
            print(f"  DEBUG: HF layer 0 attributes: {[n for n in dir(hf_layer) if not n.startswith('_')]}", flush=True)
            print(f"  DEBUG: HF layer 0 modules: {list(hf_layer._modules.keys())}", flush=True)
        
        # Get layer type from config
        layer_types = getattr(config, 'layer_types', None)
        layer_type = layer_types[layer_idx] if layer_types and layer_idx < len(layer_types) else "full_attention"
        
        # HF model uses different attribute names: self_attn for full_attention, linear_attn for linear_attention
        hf_attn = None
        if hasattr(hf_layer, 'self_attn') and hf_layer.self_attn is not None:
            hf_attn = hf_layer.self_attn
            detected_type = "full_attention"
        elif hasattr(hf_layer, 'linear_attn') and hf_layer.linear_attn is not None:
            hf_attn = hf_layer.linear_attn
            detected_type = "linear_attention"
        else:
            if is_main:
                print(f"  Layer {layer_idx}: No attention module found (config says: {layer_type})")
                print(f"    Available: {list(hf_layer._modules.keys())}")
            detected_type = None
        
        # Detect layer type from structure - check actual weight sizes
        has_k_proj = (hf_attn is not None and 
                     hasattr(hf_attn, 'k_proj') and 
                     hf_attn.k_proj is not None and
                     hasattr(hf_attn.k_proj, 'weight') and
                     hf_attn.k_proj.weight.numel() > 0)
        
        has_gated_deltanet = (hf_attn is not None and 
                             hasattr(hf_attn, 'in_proj_qkvz') and 
                             hf_attn.in_proj_qkvz is not None and
                             hasattr(hf_attn.in_proj_qkvz, 'weight') and
                             hf_attn.in_proj_qkvz.weight.numel() > 0)
        
        # Debug: print attention structure for first few layers
        if is_main and layer_idx < 3:
            if hf_attn is not None:
                print(f"  Layer {layer_idx}: hf_attn modules: {list(hf_attn._modules.keys())}", flush=True)
                print(f"    has_k_proj={has_k_proj}, has_gated_deltanet={has_gated_deltanet}, detected_type={detected_type}", flush=True)
        
        # ========== ATTENTION WEIGHTS ==========
        if has_gated_deltanet or detected_type == "linear_attention":
            # Linear attention - copy GatedDeltaNet weights to our implementation
            if is_main and layer_idx < 3:
                print(f"    Layer {layer_idx}: Copying GatedDeltaNet weights to our implementation", flush=True)
            
            _copy_gated_deltanet_weights(hf_attn, engine_layer.linear_attn,
                                        config, use_tp, tp_rank, tp_world_size, layer_idx)
            
        elif has_k_proj or detected_type == "full_attention":
            # Full attention - copy weights to our implementation
            if is_main and layer_idx < 3:
                print(f"    Layer {layer_idx}: Copying full attention weights to our implementation", flush=True)
            _copy_full_attention_weights(hf_attn, engine_layer.self_attn, 
                                        config, use_tp, tp_rank, tp_world_size, layer_idx)
        else:
            if is_main:
                print(f"  Layer {layer_idx}: SKIPPING attention (unknown type)")
        
        # ========== MLP/MoE WEIGHTS ==========
        # Detect if this is MoE or dense MLP
        has_moe = (hasattr(hf_layer, 'mlp') and 
                  hasattr(hf_layer.mlp, 'experts') and 
                  hf_layer.mlp.experts is not None)
        
        has_dense_mlp = (hasattr(hf_layer, 'mlp') and 
                        hasattr(hf_layer.mlp, 'gate_proj') and 
                        hf_layer.mlp.gate_proj is not None and
                        not has_moe)
        
        if has_moe:
            if is_main and layer_idx < 3:
                print(f"    Layer {layer_idx}: Copying MoE weights", flush=True)
            _copy_moe_weights(hf_layer.mlp, engine_layer.mlp, config, 
                             use_tp, tp_rank, tp_world_size, layer_idx)
        elif has_dense_mlp:
            if is_main and layer_idx < 3:
                print(f"    Layer {layer_idx}: Copying dense MLP weights", flush=True)
            _copy_dense_mlp_weights(hf_layer.mlp, engine_layer.mlp, config,
                                   use_tp, tp_rank, tp_world_size, layer_idx)
        else:
            if is_main:
                print(f"  Layer {layer_idx}: SKIPPING MLP (unknown type)")
        
        # ========== LAYER NORMS ==========
        if hasattr(hf_layer, 'input_layernorm') and hf_layer.input_layernorm is not None:
            _copy_to_device(hf_layer.input_layernorm.weight.data, engine_layer.input_layernorm.weight)
        if hasattr(hf_layer, 'post_attention_layernorm') and hf_layer.post_attention_layernorm is not None:
            _copy_to_device(hf_layer.post_attention_layernorm.weight.data, engine_layer.post_attention_layernorm.weight)
        
        # Free HF layer to save CPU memory
        hf_model.model.layers[layer_idx] = None
        
        # Periodic garbage collection to free memory more aggressively
        if layer_idx % 10 == 0:
            gc.collect()
    
    # Verify a sample of weights after copying
    if is_main:
        print("  Verifying weight copy...", flush=True)
        embed_sum = engine_model.model.embed_tokens.weight.data.abs().sum().item()
        lmhead_sum = engine_model.lm_head.weight.data.abs().sum().item()
        print(f"    Embedding weight abs sum: {embed_sum:.4f}", flush=True)
        print(f"    LM head weight abs sum: {lmhead_sum:.4f}", flush=True)
        # Check first attention layer weights
        for layer in engine_model.model.layers:
            if layer.linear_attn is not None:
                # Linear attention uses our GatedDeltaNet implementation
                qkvz_sum = layer.linear_attn.in_proj_qkvz.weight.data.abs().sum().item()
                print(f"    First linear attention in_proj_qkvz abs sum: {qkvz_sum:.4f}", flush=True)
                break
            elif layer.self_attn is not None:
                # Full attention uses our implementation
                q_sum = layer.self_attn.q_proj.weight.data.abs().sum().item()
                print(f"    First full attention q_proj abs sum: {q_sum:.4f}", flush=True)
                break
    
    if is_main:
        print("  Weight copy complete!", flush=True)


def _copy_full_attention_weights(hf_attn, engine_attn, config, use_tp, tp_rank, tp_world_size, layer_idx):
    """Copy weights for full attention layers with optional TP sharding.
    
    Note: ColumnParallelLinear shards by output dimension, not by heads.
    So we need to match that sharding pattern.
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    # Use config.head_dim if explicitly set (Qwen3-Next may have different head_dim)
    head_dim = getattr(config, 'head_dim', config.hidden_size // num_heads)
    
    # Check if KV is replicated (when num_kv_heads < tp_world_size)
    kv_is_replicated = num_kv_heads < tp_world_size
    
    # Debug print for first layer
    is_main = tp_rank == 0
    if is_main and layer_idx == 0:
        print(f"  _copy_full_attention_weights layer {layer_idx}:", flush=True)
        print(f"    HF q_proj shape: {hf_attn.q_proj.weight.shape}", flush=True)
        print(f"    Engine q_proj shape: {engine_attn.q_proj.weight.shape}", flush=True)
        print(f"    HF k_proj shape: {hf_attn.k_proj.weight.shape}", flush=True)
        print(f"    Engine k_proj shape: {engine_attn.k_proj.weight.shape}", flush=True)
        print(f"    kv_is_replicated={kv_is_replicated}", flush=True)
    
    if use_tp and tp_world_size > 1:
        # ColumnParallelLinear shards by output_size // world_size
        # q_proj output: num_heads * head_dim * 2 (QK with gate)
        q_output_size = num_heads * head_dim * 2
        q_shard = q_output_size // tp_world_size
        q_start = tp_rank * q_shard
        q_end = q_start + q_shard
        engine_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data[q_start:q_end])
        
        if kv_is_replicated:
            # KV heads are replicated - copy full weights
            engine_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
            engine_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
        else:
            # k_proj output: num_kv_heads * head_dim
            kv_output_size = num_kv_heads * head_dim
            kv_shard = kv_output_size // tp_world_size
            k_start = tp_rank * kv_shard
            k_end = k_start + kv_shard
            engine_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data[k_start:k_end])
            # v_proj: same sharding as k_proj
            engine_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data[k_start:k_end])
        
        # o_proj: RowParallelLinear shards input_size // world_size
        o_input_size = num_heads * head_dim
        o_shard = o_input_size // tp_world_size
        o_start = tp_rank * o_shard
        o_end = o_start + o_shard
        engine_attn.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data[:, o_start:o_end])
    else:
        engine_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data)
        engine_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
        engine_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
        engine_attn.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data)
    
    # QK norms (replicated)
    if hasattr(hf_attn, 'q_norm') and hf_attn.q_norm is not None:
        engine_attn.q_norm.weight.data.copy_(hf_attn.q_norm.weight.data)
    if hasattr(hf_attn, 'k_norm') and hf_attn.k_norm is not None:
        engine_attn.k_norm.weight.data.copy_(hf_attn.k_norm.weight.data)


def _copy_gated_deltanet_weights(hf_attn, engine_attn, config, use_tp, tp_rank, tp_world_size, layer_idx):
    """Copy weights for GatedDeltaNet (linear attention) layers.
    
    Note: GatedDeltaNet weights are NOT sharded with TP - they're replicated.
    This is because GatedDeltaNet has complex interleaved dimensions that don't
    shard cleanly, and the weights are relatively small compared to full attention.
    """
    is_main = tp_rank == 0
    
    # Debug: print all parameters and buffers for first layer
    if is_main and layer_idx == 0:
        print(f"    GatedDeltaNet layer 0 parameters: {[n for n, p in hf_attn.named_parameters()]}", flush=True)
        print(f"    GatedDeltaNet layer 0 buffers: {[n for n, b in hf_attn.named_buffers()]}", flush=True)
    
    # Just copy all weights without sharding - replicate across all ranks
    
    # in_proj_qkvz: projects to [q, k, v, z]
    if hasattr(hf_attn, 'in_proj_qkvz') and hf_attn.in_proj_qkvz is not None:
        engine_attn.in_proj_qkvz.weight.data.copy_(hf_attn.in_proj_qkvz.weight.data)
    
    # in_proj_ba: projects to [beta, alpha]
    if hasattr(hf_attn, 'in_proj_ba') and hf_attn.in_proj_ba is not None:
        engine_attn.in_proj_ba.weight.data.copy_(hf_attn.in_proj_ba.weight.data)
    
    # conv1d: causal convolution
    if hasattr(hf_attn, 'conv1d') and hf_attn.conv1d is not None:
        engine_attn.conv1d.weight.data.copy_(hf_attn.conv1d.weight.data)
        if hf_attn.conv1d.bias is not None:
            engine_attn.conv1d.bias.data.copy_(hf_attn.conv1d.bias.data)
    
    # dt_bias: per-head bias
    if hasattr(hf_attn, 'dt_bias') and hf_attn.dt_bias is not None:
        engine_attn.dt_bias.data.copy_(hf_attn.dt_bias.data)
        if is_main and layer_idx == 0:
            print(f"    Copied dt_bias: {hf_attn.dt_bias.shape}", flush=True)
    elif is_main and layer_idx == 0:
        print(f"    WARNING: dt_bias not found in HF model", flush=True)
    
    # A_log: per-head parameter
    if hasattr(hf_attn, 'A_log') and hf_attn.A_log is not None:
        engine_attn.A_log.data.copy_(hf_attn.A_log.data)
        if is_main and layer_idx == 0:
            print(f"    Copied A_log: {hf_attn.A_log.shape}", flush=True)
    elif is_main and layer_idx == 0:
        print(f"    WARNING: A_log not found in HF model", flush=True)
    
    # norm (gated RMSNorm)
    if hasattr(hf_attn, 'norm') and hf_attn.norm is not None:
        engine_attn.norm.weight.data.copy_(hf_attn.norm.weight.data)
    
    # out_proj
    if hasattr(hf_attn, 'out_proj') and hf_attn.out_proj is not None:
        engine_attn.out_proj.weight.data.copy_(hf_attn.out_proj.weight.data)


def _copy_dense_mlp_weights(hf_mlp, engine_mlp, config, use_tp, tp_rank, tp_world_size, layer_idx):
    """Copy weights for dense MLP layers with optional TP sharding."""
    intermediate_size = config.intermediate_size
    
    if use_tp and tp_world_size > 1:
        mlp_shard = intermediate_size // tp_world_size
        mlp_start = tp_rank * mlp_shard
        mlp_end = mlp_start + mlp_shard
        
        engine_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data[mlp_start:mlp_end])
        engine_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data[mlp_start:mlp_end])
        engine_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data[:, mlp_start:mlp_end])
    else:
        engine_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
        engine_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
        engine_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)


def _copy_moe_weights(hf_mlp, engine_mlp, config, use_tp, tp_rank, tp_world_size, layer_idx):
    """Copy weights for MoE layers with optional TP sharding.
    
    For MoE, we can either:
    1. Expert parallelism: Different ranks handle different experts
    2. Tensor parallelism within experts: Each expert's weights are sharded
    
    We use expert parallelism when num_experts >= tp_world_size, 
    otherwise fall back to sharding within experts.
    """
    num_experts = getattr(config, 'num_experts', 0)
    intermediate_size = getattr(config, 'moe_intermediate_size', config.intermediate_size)
    
    # Router weights (replicated - small)
    if hasattr(hf_mlp, 'gate') and hf_mlp.gate is not None:
        engine_mlp.gate.weight.data.copy_(hf_mlp.gate.weight.data)
    
    # Expert weights
    if hasattr(hf_mlp, 'experts') and hf_mlp.experts is not None:
        if use_tp and tp_world_size > 1 and num_experts >= tp_world_size:
            # Expert parallelism: each rank handles a subset of experts
            experts_per_rank = num_experts // tp_world_size
            start_expert = tp_rank * experts_per_rank
            end_expert = start_expert + experts_per_rank
            
            # gate_up_proj: [num_experts, 2 * intermediate, hidden]
            engine_mlp.experts.gate_up_proj.data.copy_(
                hf_mlp.experts.gate_up_proj.data[start_expert:end_expert]
            )
            # down_proj: [num_experts, hidden, intermediate]
            engine_mlp.experts.down_proj.data.copy_(
                hf_mlp.experts.down_proj.data[start_expert:end_expert]
            )
        elif use_tp and tp_world_size > 1:
            # Tensor parallelism within experts (when num_experts < tp_world_size)
            # Replicate experts, shard intermediate dimension
            mlp_shard = intermediate_size // tp_world_size
            mlp_start = tp_rank * mlp_shard
            mlp_end = mlp_start + mlp_shard
            
            # gate_up_proj: [num_experts, 2 * intermediate, hidden]
            # Shard the middle dimension
            engine_mlp.experts.gate_up_proj.data.copy_(
                hf_mlp.experts.gate_up_proj.data[:, mlp_start*2:mlp_end*2, :]
            )
            # down_proj: [num_experts, hidden, intermediate]
            engine_mlp.experts.down_proj.data.copy_(
                hf_mlp.experts.down_proj.data[:, :, mlp_start:mlp_end]
            )
        else:
            # No TP - copy full weights
            engine_mlp.experts.gate_up_proj.data.copy_(hf_mlp.experts.gate_up_proj.data)
            engine_mlp.experts.down_proj.data.copy_(hf_mlp.experts.down_proj.data)
    
    # Shared expert (replicated or sharded like dense MLP)
    if hasattr(hf_mlp, 'shared_expert') and hf_mlp.shared_expert is not None:
        shared_intermediate = getattr(config, 'shared_expert_intermediate_size', intermediate_size)
        
        if use_tp and tp_world_size > 1:
            mlp_shard = shared_intermediate // tp_world_size
            mlp_start = tp_rank * mlp_shard
            mlp_end = mlp_start + mlp_shard
            
            engine_mlp.shared_expert.gate_proj.weight.data.copy_(
                hf_mlp.shared_expert.gate_proj.weight.data[mlp_start:mlp_end]
            )
            engine_mlp.shared_expert.up_proj.weight.data.copy_(
                hf_mlp.shared_expert.up_proj.weight.data[mlp_start:mlp_end]
            )
            engine_mlp.shared_expert.down_proj.weight.data.copy_(
                hf_mlp.shared_expert.down_proj.weight.data[:, mlp_start:mlp_end]
            )
        else:
            engine_mlp.shared_expert.gate_proj.weight.data.copy_(hf_mlp.shared_expert.gate_proj.weight.data)
            engine_mlp.shared_expert.up_proj.weight.data.copy_(hf_mlp.shared_expert.up_proj.weight.data)
            engine_mlp.shared_expert.down_proj.weight.data.copy_(hf_mlp.shared_expert.down_proj.weight.data)
    
    # Shared expert gate (replicated)
    if hasattr(hf_mlp, 'shared_expert_gate') and hf_mlp.shared_expert_gate is not None:
        engine_mlp.shared_expert_gate.weight.data.copy_(hf_mlp.shared_expert_gate.weight.data)


# ============================================================================
# Qwen3-Next LLM Engine (Hybrid: Full Attention + Linear Attention)
# ============================================================================

import os
from collections import deque
from itertools import count
from copy import copy

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def get_gpu_memory():
    """Get GPU memory info: (total, used, free) in bytes."""
    torch.cuda.synchronize()
    if PYNVML_AVAILABLE:
        nvmlInit()
        visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
        cuda_device_idx = torch.cuda.current_device()
        cuda_device_idx = visible_device[cuda_device_idx]
        handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()
        return mem_info.total, mem_info.used, mem_info.free
    else:
        # Fallback using torch
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_mem
        used = torch.cuda.memory_allocated()
        return total, used, total - used


class HybridModelRunner:
    """
    Generic model runner for models with cache_params-based inference.
    
    Works with any model that implements:
    - model.allocate_cache(batch_size, free_memory_budget, device, block_size) -> cache_params
      where cache_params has .num_kvcache_blocks attribute
    - model.forward(input_ids, position_ids, cache_params=..., logits_to_keep=...) -> (logits, cache)
    
    This is a drop-in replacement for llm_engine.ModelRunner that uses stateless
    cache_params instead of wiring k_cache/v_cache to model layers.
    """

    def __init__(self, model, llm_config, device, temperature=0.6, top_k=0, max_batch_size=64):
        self.block_size = 256
        self.default_dtype = torch.bfloat16
        self.model = model
        self.llm_config = llm_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.temperature = temperature
        self.top_k = top_k  # 0 means no top-k filtering
        self.max_batch_size = max_batch_size
        
        # Allocate hybrid cache (paged KV + linear attention states)
        self.cache_params = self.allocate_cache(self.model, self.llm_config, 0.70, max_batch_size)
        self.num_kvcache_blocks = self.cache_params.num_kvcache_blocks
        
        # Synchronize all TP ranks after cache allocation, before any forward pass.
        # Without this barrier, rank 0 may start the first model forward (which contains
        # NCCL all_reduce ops) while rank 1 is still doing large cudaMalloc for cache.
        # This can cause NCCL deadlock because the large allocation blocks the GPU.
        if get_tp_world_size() > 1:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=get_tp_group())
            print(f'  [rank={get_tp_rank()}] TP barrier after cache allocation - all ranks ready', flush=True)
        
        # CUDA Graph support for decode
        self.use_cuda_graph = False  # Will be enabled after first decode warmup
        self.cuda_graphs = {}        # batch_size -> captured graph
        self.graph_static_inputs = {}  # batch_size -> static input buffers
        self.graph_static_outputs = {} # batch_size -> static output buffer
        self._decode_warmup_done = False
        self._decode_step_counter = 0
        self._profile_done = False

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_cache(self, model, llm_config, gpu_memory_utilization=0.85, max_batch_size=1):
        """Allocate cache by calling model.allocate_cache().
        
        Uses torch.cuda.mem_get_info for accurate free memory on the target device.
        The batch_size controls linear attention state allocation (conv + recurrent).
        """
        # Query free memory on the correct device (not the default device)
        device_idx = self.device.index if self.device.index is not None else 0
        free, total = torch.cuda.mem_get_info(device_idx)
        # Reserve some memory for activations and overhead
        usable_free = int(free * gpu_memory_utilization)
        
        if usable_free <= 0:
            print(f"WARNING: No free GPU memory for cache allocation on device {device_idx}! free={free/1e9:.2f} GB")
            usable_free = int(free * 0.5)  # Try with 50% of whatever is left
        
        print(f'[allocate_cache] device={device_idx}, free={free/1e9:.2f} GB, '
              f'total={total/1e9:.2f} GB, usable={usable_free/1e9:.2f} GB', flush=True)
        
        cache = model.allocate_cache(
            batch_size=max_batch_size,  # Allocate linear attention states for full batch
            free_memory_budget=usable_free,
            device=self.device,
            block_size=self.block_size,
        )
        
        print(f'max kv cache length: {cache.num_kvcache_blocks * self.block_size}, batch_size={max_batch_size}')
        return cache

    def prepare_block_tables(self, seqs):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)

    def prepare_prefill(self, seqs):
        from context import set_context
        
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        assert len(input_ids) == len(slot_mapping)
        assert len(input_ids) == cu_seqlens_q[-1]

        context_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs):
        from context import set_context
        
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        num_seqs = len(seqs)
        
        # For CUDA graph: use static context buffers if available
        if self.use_cuda_graph and num_seqs in self.cuda_graphs:
            key = num_seqs
            if not hasattr(self, '_static_context'):
                self._static_context = {}
            if key not in self._static_context:
                # Create static buffers for context
                max_blocks = max(len(seq.block_table) for seq in seqs)
                self._static_context[key] = {
                    'slot_mapping': torch.zeros(num_seqs, dtype=torch.int32, device=self.device),
                    'context_lens': torch.zeros(num_seqs, dtype=torch.int32, device=self.device),
                    'block_tables': torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=self.device),
                }
            
            ctx = self._static_context[key]
            ctx['slot_mapping'].copy_(torch.tensor(slot_mapping, dtype=torch.int32))
            ctx['context_lens'].copy_(torch.tensor(context_lens, dtype=torch.int32))
            
            # Update block tables (may need to resize if max_blocks changed)
            max_blocks = max(len(seq.block_table) for seq in seqs)
            if ctx['block_tables'].shape[1] < max_blocks:
                ctx['block_tables'] = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=self.device)
            bt = self.prepare_block_tables(seqs)
            ctx['block_tables'][:, :bt.shape[1]].copy_(bt)
            
            set_context(False, slot_mapping=ctx['slot_mapping'], 
                       context_lens=ctx['context_lens'], 
                       block_tables=ctx['block_tables'][:, :bt.shape[1]])
            
            input_ids_t = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
            positions_t = torch.tensor(positions, dtype=torch.int64, device=self.device)
            return input_ids_t, positions_t
        
        # Standard path (non-graph)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        context = get_context()
        if is_prefill:
            print(f'  [run_model rank={get_tp_rank()}] prefill: input_ids={input_ids.shape}, positions={positions.shape}', flush=True)
            # Enable TP collective tracing for first prefill to debug hangs
            global _tp_collective_counter
            _tp_collective_counter = 0
            enable_tp_debug(get_tp_world_size() > 1)
            t0 = time.time()
            logits, _ = self.model(
                input_ids=input_ids.unsqueeze(0),
                position_ids=positions.unsqueeze(0),
                cache_params=self.cache_params,
                logits_to_keep=context.cu_seqlens_q[1:] - 1,
            )
            t1 = time.time()
            self.cache_params.has_previous_state = True
            enable_tp_debug(False)  # Disable verbose TP tracing after prefill
            print(f'  [run_model rank={get_tp_rank()}] prefill done: logits={logits.shape}, '
                  f'time={t1-t0:.3f}s, tp_collectives={_tp_collective_counter}', flush=True)
        else:
            num_seqs = input_ids.shape[0]
            self._decode_step_counter += 1
            
            # Profile on decode step 10 (after warmup, before too long)
            do_profile = (self._decode_step_counter == 10 and not self._profile_done
                          and get_tp_rank() == 0)
            
            # Try CUDA graph replay for decode
            if self.use_cuda_graph and num_seqs in self.cuda_graphs:
                logits = self._run_decode_graph(input_ids, positions, num_seqs)
            else:
                # Eager decode (also used for warmup before graph capture)
                logits, _ = self.model(
                    input_ids=input_ids.unsqueeze(1),
                    position_ids=positions.unsqueeze(1),
                    cache_params=self.cache_params,
                    _profile_layers=do_profile,
                )
                logits = logits.squeeze(1)
                
                if do_profile:
                    self._profile_done = True
                
                # After first eager decode, try to capture graph for this batch size
                if not self._decode_warmup_done and num_seqs > 1:
                    self._try_capture_decode_graph(num_seqs)
                    self._decode_warmup_done = True
        
        return logits.squeeze(0)
    
    def _try_capture_decode_graph(self, batch_size):
        """Attempt to capture a CUDA graph for decode with the given batch size.
        
        Known limitations:
        - TP > 1: DISABLED. If capture fails on one rank but succeeds on another,
          NCCL collectives (all_reduce/all_gather) inside the graph won't match
          the eager path on the failed rank → deadlock.
        - Triton MoE kernel: Incompatible with CUDA graphs because:
          (a) moe_align_block_size() allocates dynamic tensors (torch.zeros, argsort, scatter)
          (b) .item() call causes CPU-GPU sync during capture
          (c) Grid sizes depend on router decisions which change every step
        - FLA recurrent kernel: May use internal Triton ops with dynamic allocations.
        The try/except fallback handles (b) and (c) gracefully for TP=1.
        """
        # Skip CUDA graph capture for tensor parallel — partial capture failure
        # across ranks would cause NCCL deadlock (mismatched collectives).
        if get_tp_world_size() > 1:
            print(f'  [CUDA Graph] Skipped: not supported with tensor parallelism (TP={get_tp_world_size()})', flush=True)
            return
        
        try:
            print(f'  [CUDA Graph] Attempting to capture graph for batch_size={batch_size}...', flush=True)
            
            # Create static input buffers (fixed addresses for graph)
            static_input_ids = torch.zeros(batch_size, 1, dtype=torch.int64, device=self.device)
            static_positions = torch.zeros(batch_size, 1, dtype=torch.int64, device=self.device)
            
            # CUDA graphs must be captured on a non-default stream
            capture_stream = torch.cuda.Stream(device=self.device)
            
            # Warmup runs on the capture stream (required before capture)
            torch.cuda.synchronize(self.device)
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    static_logits, _ = self.model(
                        input_ids=static_input_ids,
                        position_ids=static_positions,
                        cache_params=self.cache_params,
                    )
            torch.cuda.synchronize(self.device)
            
            # Capture the graph on the non-default stream
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(capture_stream):
                with torch.cuda.graph(graph, stream=capture_stream):
                    static_logits, _ = self.model(
                        input_ids=static_input_ids,
                        position_ids=static_positions,
                        cache_params=self.cache_params,
                    )
            torch.cuda.synchronize(self.device)
            
            self.cuda_graphs[batch_size] = graph
            self.graph_static_inputs[batch_size] = (static_input_ids, static_positions)
            self.graph_static_outputs[batch_size] = static_logits
            self.use_cuda_graph = True
            
            print(f'  [CUDA Graph] Successfully captured for batch_size={batch_size}', flush=True)
            
        except Exception as e:
            print(f'  [CUDA Graph] Failed to capture for batch_size={batch_size}: {e}', flush=True)
            print(f'  [CUDA Graph] Falling back to eager decode', flush=True)
            self.use_cuda_graph = False
    
    def _run_decode_graph(self, input_ids, positions, batch_size):
        """Replay captured CUDA graph for decode."""
        static_input_ids, static_positions = self.graph_static_inputs[batch_size]
        
        # Copy new inputs into static buffers (graph captures these addresses)
        static_input_ids.copy_(input_ids.unsqueeze(1))
        static_positions.copy_(positions.unsqueeze(1))
        
        # Replay the graph
        self.cuda_graphs[batch_size].replay()
        
        # Return output from static buffer
        return self.graph_static_outputs[batch_size].squeeze(1).clone()

    @torch.inference_mode()
    def _swap_linear_state(self, slot_a, slot_b):
        """Swap linear attention (conv + recurrent) states between two batch slots."""
        cache = self.cache_params
        for i in range(len(cache.conv_states)):
            if cache.conv_states[i] is not None:
                tmp = cache.conv_states[i][slot_a].clone()
                cache.conv_states[i][slot_a].copy_(cache.conv_states[i][slot_b])
                cache.conv_states[i][slot_b].copy_(tmp)
            if cache.recurrent_states[i] is not None:
                tmp = cache.recurrent_states[i][slot_a].clone()
                cache.recurrent_states[i][slot_a].copy_(cache.recurrent_states[i][slot_b])
                cache.recurrent_states[i][slot_b].copy_(tmp)

    def run(self, seqs, is_prefill: bool):
        import time as _time
        t_start = _time.time()
        
        if is_prefill:
            if FLA_AVAILABLE and len(seqs) > 1:
                t_prep = _time.time()
                input_ids, positions = self.prepare_prefill(seqs)
                t_prep_done = _time.time()
                logits = self.run_model(input_ids, positions, True)
                t_model = _time.time()
                print(f'  [run] varlen prefill: {len(seqs)} seqs, '
                      f'prep={t_prep_done-t_prep:.3f}s, model={t_model-t_prep_done:.3f}s', flush=True)
            else:
                # Fallback: per-seq prefill loop (when FLA not available or single seq)
                all_logits = []
                for seq_idx, seq in enumerate(seqs):
                    if seq_idx > 0:
                        self._swap_linear_state(0, seq_idx)
                    input_ids, positions = self.prepare_prefill([seq])
                    logits = self.run_model(input_ids, positions, True)
                    all_logits.append(logits.view(-1))  # [vocab]
                    if seq_idx > 0:
                        self._swap_linear_state(0, seq_idx)
                logits = torch.stack(all_logits, dim=0)  # [num_seqs, vocab]
        else:
            t_prep = _time.time()
            input_ids, positions = self.prepare_decode(seqs)
            t_prep_done = _time.time()
            logits = self.run_model(input_ids, positions, is_prefill)
            t_model = _time.time()
        
        # Ensure logits is always 2D [num_seqs, vocab] for uniform sampling
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        if self.temperature <= 0:
            # Greedy decoding
            token_ids = logits.argmax(dim=-1)  # [num_seqs]
        else:
            # Sampling with temperature + optional top-k
            scaled_logits = (logits / self.temperature).to(torch.float32)
            if self.top_k > 0:
                # Zero out everything outside top-k
                topk_vals, _ = torch.topk(scaled_logits, min(self.top_k, scaled_logits.size(-1)), dim=-1)
                scaled_logits[scaled_logits < topk_vals[..., -1:]] = float('-inf')
            probs = torch.softmax(scaled_logits, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [num_seqs]
        
        # For TP: broadcast sampled tokens from rank 0 so all ranks stay in sync
        if get_tp_world_size() > 1:
            dist.broadcast(token_ids, src=0, group=get_tp_group())
        
        reset_context()
        token_list = token_ids.tolist()
        # Ensure token_list is always a flat list
        if isinstance(token_list, int):
            token_list = [token_list]
        
        t_end = _time.time()
        if is_prefill:
            print(f'  [run] prefill -> sampled tokens: {token_list}', flush=True)
        # Log timing every 50 decode steps for the first seq
        if not is_prefill and len(seqs) > 0 and seqs[0].num_completion_tokens % 50 == 1:
            print(f'  [run] decode timing: batch={len(seqs)}, '
                  f'prep={t_prep_done-t_prep:.4f}s, model={t_model-t_prep_done:.4f}s, '
                  f'sample={t_end-t_model:.4f}s, total={t_end-t_start:.4f}s, '
                  f'throughput={len(seqs)/(t_end-t_start):.1f} tok/s', flush=True)
        return token_list


class HybridLLMEngine:
    """
    Generic LLM Engine for models with cache_params-based inference.
    
    Works with any model that implements the allocate_cache() + forward(cache_params=) protocol.
    Uses HybridModelRunner instead of the stateful ModelRunner from llm_engine.py.
    """
    def __init__(self, model, llm_config, device, temperature=0.6, top_k=0, max_batch_size=64, prefill_one_by_one=False):
        # Increase NCCL timeout for batch prefill (many sequential forward passes)
        if 'NCCL_TIMEOUT' not in _os.environ:
            _os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour
        self.prefill_one_by_one = prefill_one_by_one
        # Ensure imports work for both `from phi4 import ...` (needs xlmlib/ on path)
        # and `from xlmlib.fused_linear_cross_entropy import ...` (needs parent on path).
        # We register xlmlib as a namespace to avoid triggering __init__.py -> samba.
        _script_dir = _os.path.dirname(_os.path.abspath(__file__))
        _parent_dir = _os.path.dirname(_script_dir)
        if _script_dir not in sys.path:
            sys.path.insert(0, _script_dir)
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        # Prevent xlmlib/__init__.py from running (it imports samba which needs selective_scan_cuda)
        import types
        if 'xlmlib' not in sys.modules:
            sys.modules['xlmlib'] = types.ModuleType('xlmlib')
            sys.modules['xlmlib'].__path__ = [_script_dir]
        
        from llm_engine import Scheduler, Sequence
        
        self.model_runner = HybridModelRunner(model, llm_config, device, temperature=temperature, top_k=top_k, max_batch_size=max_batch_size)
        self.scheduler = Scheduler(llm_config, self.model_runner.block_size, self.model_runner.num_kvcache_blocks)
        self.scheduler.max_num_seqs = max_batch_size  # Limit concurrent sequences

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        import time as _time
        t0 = _time.time()
        seqs, is_prefill = self.scheduler.schedule()
        t_sched = _time.time()
        
        if is_prefill and self.prefill_one_by_one and len(seqs) > 1:
            # Process only 1 prefill per step to avoid NCCL timeout.
            # Preempt the rest back to waiting for next step().
            extra_seqs = seqs[1:]
            seqs = seqs[:1]
            for seq in extra_seqs:
                self.scheduler.preempt(seq)
        
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        t_run = _time.time()
        self.scheduler.postprocess(seqs, token_ids)
        t_post = _time.time()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # Print decode progress every 50 tokens
        if not is_prefill and len(seqs) > 0:
            seq = seqs[0]
            if seq.num_completion_tokens % 50 == 1:
                print(f'  [step] decode: seq_id={seq.seq_id}, completion_tokens={seq.num_completion_tokens}, '
                      f'last_token={seq.last_token}, running={len(self.scheduler.running)}, '
                      f'finished={len(outputs)}, '
                      f'sched={t_sched-t0:.4f}s, run={t_run-t_sched:.4f}s, post={t_post-t_run:.4f}s, '
                      f'step_total={t_post-t0:.4f}s', flush=True)
        return outputs

    def generate(self, prompts: List[List[int]], max_tokens: int = 32768) -> List[List[int]]:
        from llm_engine import Sequence
        
        for prompt in prompts:
            seq = Sequence(prompt)
            seq.max_tokens = max_tokens
            self.scheduler.add(seq)

        outputs = {}
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        return [outputs[seq_id] for seq_id in sorted(outputs)]


# Simple test
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--test_engine", action="store_true", help="Test with LLMEngine")
    parser.add_argument("--prompt", type=str, default="What is 2+2?")
    parser.add_argument("--gpu_ids", type=str, default=None, help="GPU IDs to use (e.g., '0' or '2,3')")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Tensor parallel size (use with torchrun)")
    parser.add_argument("--compare_hf", action="store_true", help="Compare with HF model output")
    args = parser.parse_args()
    
    # Set GPU for single-GPU mode
    if args.gpu_ids is not None and args.tensor_parallel == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    # For TP mode, device is set by init_tensor_parallel
    device = "cuda:0" if args.tensor_parallel == 1 else "cuda"
    
    # These early prints are before TP is initialized, but only rank 0 should print
    # For TP, use RANK env var; for single GPU, always print
    import os as os_module
    rank_env = int(os_module.environ.get("RANK", "0"))
    if rank_env == 0:
        print(f"Testing load_qwen3_next_for_engine with {args.model_path}")
        print(f"Tensor parallel size: {args.tensor_parallel}")
    
    # Test 0: Compare with HF model (before weight transfer cleanup)
    if args.compare_hf:
        print("\n=== Test 0: Compare HF vs Engine Model ===")
        
        # Load HF model for comparison
        hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        hf_model.eval()
        
        test_input_text = args.prompt
        test_input = tokenizer.encode(test_input_text, return_tensors="pt").to("cuda:0")
        
        # HF model forward
        with torch.no_grad():
            hf_output = hf_model(test_input)
            hf_logits = hf_output.logits
            hf_next_token_id = hf_logits[0, -1, :].argmax().item()
            hf_next_token = tokenizer.decode([hf_next_token_id])
        
        print(f"HF model predicted next token: '{hf_next_token}' (id={hf_next_token_id})")
        print(f"HF logits shape: {hf_logits.shape}")
        print(f"HF logits last position stats: min={hf_logits[0,-1].min().item():.4f}, max={hf_logits[0,-1].max().item():.4f}, mean={hf_logits[0,-1].mean().item():.4f}")
        
        # Also capture HF intermediate states for comparison
        print("\nHF intermediate states:")
        hf_embed = hf_model.model.embed_tokens(test_input)
        print(f"  HF embedding output: shape={hf_embed.shape}, mean={hf_embed.mean().item():.6f}, std={hf_embed.std().item():.6f}")
        
        # HF model greedy generation
        hf_generated = hf_model.generate(
            test_input, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        hf_generated_text = tokenizer.decode(hf_generated[0], skip_special_tokens=True)
        print(f"HF model generated: {hf_generated_text}")
        
        # Move HF model to CPU to free GPU memory before loading engine model
        print("\nMoving HF model to CPU to free GPU memory...")
        hf_model = hf_model.to("cpu")
        torch.cuda.empty_cache()
        
        # Keep HF model for later comparison
        _hf_model_for_compare = hf_model
    else:
        _hf_model_for_compare = None
    
    model, tokenizer, config = load_qwen3_next_for_engine(
        args.model_path, 
        device=device,
        tensor_parallel_size=args.tensor_parallel
    )
    
    is_main = get_tp_rank() == 0
    if is_main:
        print(f"Model loaded: {type(model)}")
        print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # Get the actual device model is on
    model_device = next(model.parameters()).device
    
    # Test 1: Direct forward pass (all ranks must run, only rank 0 prints)
    if is_main:
        print("\n=== Test 1: Direct Forward Pass ===")
    test_input_text = args.prompt
    test_input = tokenizer.encode(test_input_text, return_tensors="pt").to(model_device)
    
    # Compare embedding outputs if HF model available
    if args.compare_hf and _hf_model_for_compare is not None:
        print("\nComparing embedding outputs:")
        with torch.no_grad():
            engine_embed = model.model.embed_tokens(test_input)
            # HF model is on CPU, compute embedding on CPU then compare
            test_input_cpu = test_input.to("cpu")
            hf_embed = _hf_model_for_compare.model.embed_tokens(test_input_cpu).to(model_device)
            embed_diff = (engine_embed - hf_embed).abs().max().item()
            print(f"  Engine embedding: mean={engine_embed.mean().item():.6f}, std={engine_embed.std().item():.6f}")
            print(f"  HF embedding:     mean={hf_embed.mean().item():.6f}, std={hf_embed.std().item():.6f}")
            print(f"  Max absolute diff: {embed_diff:.8f}")
            
            # Compare layer 0 output
            print("\nComparing layer 0 output:")
            # Engine layer 0
            engine_layer0 = model.model.layers[0]
            engine_hidden = engine_embed.clone()
            engine_residual = engine_hidden
            engine_hidden_ln = engine_layer0.input_layernorm(engine_hidden)
            
            # HF layer 0 (move to GPU temporarily)
            hf_layer0 = _hf_model_for_compare.model.layers[0].to(model_device)
            hf_hidden = hf_embed.clone()
            hf_residual = hf_hidden
            hf_hidden_ln = hf_layer0.input_layernorm(hf_hidden)
            
            ln_diff = (engine_hidden_ln - hf_hidden_ln).abs().max().item()
            print(f"  After input_layernorm: max diff = {ln_diff:.8f}")
            
            # Linear attention output
            if engine_layer0.linear_attn is not None:
                # Our implementation returns just the tensor, HF returns tuple
                engine_attn_out = engine_layer0.linear_attn(engine_hidden_ln)
                hf_attn_out = hf_layer0.linear_attn(hf_hidden_ln)
                # HF returns output directly now (not wrapped)
                if isinstance(hf_attn_out, tuple):
                    hf_attn_out = hf_attn_out[0]
                attn_diff = (engine_attn_out - hf_attn_out).abs().max().item()
                print(f"  After linear_attn: max diff = {attn_diff:.8f}")
                print(f"    Engine: mean={engine_attn_out.mean().item():.6f}, std={engine_attn_out.std().item():.6f}")
                print(f"    HF:     mean={hf_attn_out.mean().item():.6f}, std={hf_attn_out.std().item():.6f}")
                
                engine_hidden = engine_residual + engine_attn_out
                hf_hidden = hf_residual + hf_attn_out
            
            # Post attention layernorm
            engine_residual = engine_hidden
            hf_residual = hf_hidden
            engine_hidden_post = engine_layer0.post_attention_layernorm(engine_hidden)
            hf_hidden_post = hf_layer0.post_attention_layernorm(hf_hidden)
            
            post_ln_diff = (engine_hidden_post - hf_hidden_post).abs().max().item()
            print(f"  After post_attention_layernorm: max diff = {post_ln_diff:.8f}")
            
            # MLP/MoE output
            engine_mlp_out = engine_layer0.mlp(engine_hidden_post)
            hf_mlp_out = hf_layer0.mlp(hf_hidden_post)
            mlp_diff = (engine_mlp_out - hf_mlp_out).abs().max().item()
            print(f"  After MLP/MoE: max diff = {mlp_diff:.8f}")
            print(f"    Engine: mean={engine_mlp_out.mean().item():.6f}, std={engine_mlp_out.std().item():.6f}")
            print(f"    HF:     mean={hf_mlp_out.mean().item():.6f}, std={hf_mlp_out.std().item():.6f}")
            
            # Move HF layer back to CPU
            hf_layer0 = hf_layer0.to("cpu")
            torch.cuda.empty_cache()
    
    with torch.no_grad():
        logits, _ = model(test_input)
    if is_main:
        print(f"Input string: {test_input_text}")
        print(f"Input shape: {test_input.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Engine logits last position stats: min={logits[0,-1].min().item():.4f}, max={logits[0,-1].max().item():.4f}, mean={logits[0,-1].mean().item():.4f}")
        # Get predicted next token
        next_token_id = logits[0, -1, :].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print(f"Predicted next token: '{next_token}' (id={next_token_id})")
        print("Direct forward pass: OK")
    
    # Cleanup HF model if it was kept
    if args.compare_hf and _hf_model_for_compare is not None:
        del _hf_model_for_compare
        torch.cuda.empty_cache()
    
    # Test 1.5: Simple greedy generation (without LLMEngine)
    if is_main:
        print("\n=== Test 1.5: Simple Greedy Generation ===")
    max_new_tokens = 50
    generated_ids = test_input.clone()
    eos_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass
            logits, _ = model(generated_ids)
            # Get next token (greedy)
            next_token_id = logits[0, -1, :].argmax().item()
            # Check for EOS
            if next_token_id == eos_token_id:
                break
            # Append to sequence
            next_token_tensor = torch.tensor([[next_token_id]], device=model_device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
    
    if is_main:
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Input: {test_input_text}")
        print(f"Generated ({generated_ids.shape[1] - test_input.shape[1]} new tokens): {generated_text}")
    
    # Test 2 & 3: With HybridLLMEngine (single engine, reused across all tests)
    if args.test_engine:
        if is_main:
            print("\n=== Test 2: HybridLLMEngine Generation ===")
        
        config.eos_token_id = tokenizer.eos_token_id
        
        # Batch prompts for Test 3
        batch_prompts_text = [
            "What is 2+2?",
            "What is the capital of France?",
            "Solve: 3 * 7 =",
            "Write a haiku about rain.",
        ]
        n_rollout = 2
        
        # Create ONE engine with max_batch_size covering all tests (2, 3a, 3b)
        max_bs = max(len(batch_prompts_text), n_rollout, 1)
        engine = HybridLLMEngine(model, config, str(model_device), 
                                  temperature=0, max_batch_size=max_bs)
        
        # --- Test 2: Single prompt generation ---
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(args.prompt)
        
        if is_main:
            print(f"Prompt: {args.prompt}")
            print(f"Input string: {args.prompt}")
            print(f"Input tokens: {len(input_ids)}")
        
        import time
        start = time.time()
        output_ids = engine.generate([input_ids], max_tokens=64)[0]
        elapsed = time.time() - start
        
        if is_main:
            new_tokens = len(output_ids)
            response = tokenizer.decode(output_ids, skip_special_tokens=False)
            full_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            print(f"Generated {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
            print(f"Output string: {full_output}")
            print(f"Response: {response}")
    
    # Test 3: Batch-wise generation (reusing same engine from Test 2)
    if args.test_engine:
        if is_main:
            print("\n=== Test 3: Batch-wise Generation (reusing engine) ===")
        
        batch_size = len(batch_prompts_text)
        rollout_prompt_text = "What is 5+3?"
        
        # --- Test 3a: multiple different prompts (greedy) ---
        engine.model_runner.temperature = 0
        engine.model_runner.top_k = 0
        
        if is_main:
            print(f"\n--- Test 3a: {batch_size} different prompts (greedy, reusing engine with max_batch_size={max_bs}) ---")
        
        batch_input_ids = [tokenizer.encode(p) for p in batch_prompts_text]
        
        if is_main:
            for i, (text, ids) in enumerate(zip(batch_prompts_text, batch_input_ids)):
                print(f"  Prompt {i}: '{text}' ({len(ids)} tokens)")
        
        # Reset cache before new batch
        engine.model_runner.cache_params.reset()
        
        import time
        start = time.time()
        batch_outputs = engine.generate(batch_input_ids, max_tokens=64)
        elapsed = time.time() - start
        
        total_tokens = sum(len(o) for o in batch_outputs)
        if is_main:
            print(f"\n  Batch generation: {batch_size} prompts, {total_tokens} total tokens in {elapsed:.2f}s "
                  f"({total_tokens/elapsed:.1f} tok/s)")
            for i, output_ids in enumerate(batch_outputs):
                response = tokenizer.decode(output_ids, skip_special_tokens=True)
                print(f"  Output {i} ({len(output_ids)} tokens): {response[:150]}...")
        
        # --- Test 3b: Rollout-style (same prompt, sampling) ---
        # Switch sampling parameters on the same engine (no reallocation)
        engine.model_runner.temperature = 0.7
        engine.model_runner.top_k = 50
        
        if is_main:
            print(f"\n--- Test 3b: Same prompt x{n_rollout} rollouts (temperature=0.7, top_k=50, reusing engine) ---")
        
        rollout_ids = tokenizer.encode(rollout_prompt_text)
        rollout_batch = [rollout_ids] * n_rollout
        
        if is_main:
            print(f"  Prompt: '{rollout_prompt_text}' ({len(rollout_ids)} tokens) x{n_rollout}")
        
        # Reset cache before new batch
        engine.model_runner.cache_params.reset()
        
        start = time.time()
        rollout_outputs = engine.generate(rollout_batch, max_tokens=64)
        elapsed = time.time() - start
        
        total_tokens = sum(len(o) for o in rollout_outputs)
        if is_main:
            print(f"  Rollout generation: {n_rollout} copies, {total_tokens} total tokens in {elapsed:.2f}s "
                  f"({total_tokens/elapsed:.1f} tok/s)")
            for i, output_ids in enumerate(rollout_outputs):
                response = tokenizer.decode(output_ids, skip_special_tokens=True)
                print(f"  Rollout {i} ({len(output_ids)} tokens): {response[:150]}...")
            
            # Check diversity: with temperature>0, outputs should differ
            if n_rollout >= 2:
                same = rollout_outputs[0] == rollout_outputs[1]
                print(f"  Rollout 0 == Rollout 1: {same} (expect False with temperature>0)")
        
        print("\nBatch-wise generation test: OK")


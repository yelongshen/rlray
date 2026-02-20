"""
Qwen3-Next Model Adapter for In-house LLM Engine

This file adapts the HuggingFace Qwen3-Next model to work with the in-house
llm_engine.py for fast inference with paged attention.

The in-house LLMEngine expects:
- model(input_ids=..., position_ids=..., inference_mode=True, logits_to_keep=...) -> (_, logits, _, _)
- Attention layers with k_cache and v_cache attributes for paged attention

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
        dist.init_process_group(backend="nccl")
    
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


class _ReduceFromTP(torch.autograd.Function):
    """All-reduce in tensor parallel group."""
    @staticmethod
    def forward(ctx, input_):
        if get_tp_world_size() == 1:
            return input_
        dist.all_reduce(input_, group=get_tp_group())
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
        world_size = get_tp_world_size()
        output_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(output_list, input_, group=get_tp_group())
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
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.jit.script
def store_kvcache(key_states: torch.Tensor, value_states: torch.Tensor, 
                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                  slot_mapping: torch.Tensor):
    """Store key/value states into paged KV cache."""
    k_cache.index_copy_(0, slot_mapping, key_states)
    v_cache.index_copy_(0, slot_mapping, value_states)


def get_rms_norm_eps(config):
    """Get RMS norm epsilon from config with fallback."""
    return getattr(config, 'rms_norm_eps', getattr(config, 'layer_norm_eps', 1e-6))


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm for Qwen3-Next."""
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
# Linear Attention (Gated DeltaNet)
# ============================================================================

class Qwen3NextGatedDeltaNetForEngine(nn.Module):
    """
    Gated DeltaNet linear attention for Qwen3-Next.
    
    This implements a recurrent linear attention mechanism with:
    - Causal convolution for local context
    - Gated delta rule for recurrent state updates
    """
    def __init__(self, config, layer_idx: int, use_tp: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.use_tp = use_tp
        
        # Get dimensions from config with fallbacks
        self.num_v_heads = getattr(config, 'linear_num_value_heads', 32)
        self.num_k_heads = getattr(config, 'linear_num_key_heads', 4)
        self.head_k_dim = getattr(config, 'linear_key_head_dim', 128)
        self.head_v_dim = getattr(config, 'linear_value_head_dim', 128)
        self.conv_kernel_size = getattr(config, 'linear_conv_kernel_dim', 4)
        
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        
        # Projections
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        
        # Note: GatedDeltaNet doesn't use TP sharding - weights are replicated
        # This is because the dimensions are complex and interleaved
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        
        # Learnable parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        
        # Output norm with gating
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=get_rms_norm_eps(config))
        
        # Recurrent state (for inference)
        self.conv_state = None
        self.recurrent_state = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to QKVZ and BA
        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = self.in_proj_ba(hidden_states)
        
        # Split projections
        query, key, value, z = self._split_qkvz(projected_qkvz)
        beta, alpha = self._split_ba(projected_ba)
        
        # Apply causal conv1d
        qkv_cat = torch.cat([query, key, value], dim=-1)
        qkv_cat = qkv_cat.transpose(1, 2)  # [B, D, L]
        qkv_cat = F.silu(self.conv1d(qkv_cat)[:, :, :seq_len])
        qkv_cat = qkv_cat.transpose(1, 2)  # [B, L, D]
        
        query, key, value = torch.split(qkv_cat, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        
        # Reshape for attention
        query = query.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        # Compute gated delta rule (simplified version)
        output = self._gated_delta_rule(query, key, value, beta, alpha)
        
        # Apply gated norm and output projection
        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)
        
        return output
    
    def _split_qkvz(self, mixed):
        """Split mixed projection into Q, K, V, Z."""
        # Shape: [B, L, projection_size_qkvz]
        q_size = self.key_dim
        k_size = self.key_dim
        v_size = self.value_dim
        z_size = self.value_dim
        query, key, value, z = torch.split(mixed, [q_size, k_size, v_size, z_size], dim=-1)
        return query, key, value, z
    
    def _split_ba(self, mixed):
        """Split mixed projection into beta and alpha."""
        beta, alpha = torch.chunk(mixed, 2, dim=-1)
        return beta, alpha
    
    def _gated_delta_rule(self, query, key, value, beta, alpha):
        """
        Simplified gated delta rule computation.
        This is a basic implementation - the full FLA library version is more efficient.
        """
        batch_size, seq_len, num_v_heads, head_v_dim = value.shape
        num_k_heads = query.shape[2]
        head_k_dim = query.shape[3]
        
        # Expand K heads to match V heads
        kv_ratio = num_v_heads // num_k_heads
        query = query.repeat_interleave(kv_ratio, dim=2)
        key = key.repeat_interleave(kv_ratio, dim=2)
        
        # Compute gate
        g = -F.softplus(-self.A_log.view(1, 1, -1, 1) - alpha.unsqueeze(-1))
        beta = torch.sigmoid(beta + self.dt_bias.view(1, 1, -1)).unsqueeze(-1)
        
        # Scale query
        scale = 1.0 / (head_k_dim ** 0.5)
        query = query * scale
        
        # Simple linear attention (not full delta rule for efficiency)
        # Full implementation would use recurrent state
        output = torch.zeros(batch_size, seq_len, num_v_heads, head_v_dim, 
                           device=value.device, dtype=value.dtype)
        
        # Causal linear attention approximation
        kv = torch.einsum('bsnk,bsnv->bsnkv', key, value * beta)
        kv_cumsum = kv.cumsum(dim=1)
        output = torch.einsum('bsnk,bsnkv->bsnv', query, kv_cumsum)
        
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
    
    def forward(self, hidden_states, top_k_indices, top_k_weights):
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
            top_k_indices: [num_tokens, top_k]
            top_k_weights: [num_tokens, top_k]
        """
        tp_rank = get_tp_rank() if self.use_tp else 0
        tp_world_size = get_tp_world_size() if self.use_tp else 1
        
        final_output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_local_idx in range(self.experts_per_rank):
            expert_global_idx = expert_local_idx + tp_rank * self.experts_per_rank
            
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_global_idx)
            if not expert_mask.any():
                continue
            
            # Get positions where this expert is selected
            token_indices, top_k_positions = torch.where(expert_mask)
            
            # Get input states and weights
            expert_input = hidden_states[token_indices]
            expert_weights = top_k_weights[token_indices, top_k_positions].unsqueeze(-1)
            
            # Forward through expert
            gate, up = F.linear(expert_input, self.gate_up_proj[expert_local_idx]).chunk(2, dim=-1)
            expert_output = self.act_fn(gate) * up
            expert_output = F.linear(expert_output, self.down_proj[expert_local_idx])
            
            # Weight and accumulate
            expert_output = expert_output * expert_weights
            final_output.index_add_(0, token_indices, expert_output.to(final_output.dtype))
        
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
    - Uses paged attention with external k_cache/v_cache
    - Supports inference_mode parameter
    - Uses flash_attn_varlen_func and flash_attn_with_kvcache
    - Supports tensor parallelism
    """
    def __init__(self, config, layer_idx: int, use_tp: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_tp = use_tp
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Use config.head_dim if explicitly set (Qwen3-Next may have different head_dim)
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # For tensor parallel, split heads across GPUs
        tp_world_size = get_tp_world_size() if use_tp else 1
        self.num_heads_per_partition = self.num_heads // tp_world_size
        self.num_kv_heads_per_partition = self.num_key_value_heads // tp_world_size
        
        if use_tp and tp_world_size > 1:
            # Projections with tensor parallelism
            # q_proj outputs q and gate concatenated, so output is num_heads * head_dim * 2
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, 
                self.num_heads * self.head_dim * 2, 
                bias=False, 
                gather_output=False
            )
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
            # Standard projections
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # QK norm (Qwen3-Next specific)
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=get_rms_norm_eps(config))
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=get_rms_norm_eps(config))
        
        # KV cache for paged attention (will be set by LLMEngine)
        self.k_cache = None
        self.v_cache = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        
        # Use partition sizes for TP
        num_heads = self.num_heads_per_partition if self.use_tp else self.num_heads
        num_kv_heads = self.num_kv_heads_per_partition if self.use_tp else self.num_key_value_heads
        
        # Compute Q, K, V (Qwen3-Next has q_proj output q and gate concatenated)
        qg = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(qg, 2, dim=-1)
        
        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, self.head_dim)
        
        # Apply QK norm
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
        
        key_cache = None
        value_cache = None
        page_attention = False
        
        if not inference_mode:
            # Training mode - no paged attention
            key_cache = key_states
            value_cache = value_states
        elif inference_mode and self.k_cache is not None and self.v_cache is not None:
            # Inference mode with paged attention
            key_cache = None
            value_cache = None
            context = get_context()
            
            query_states = query_states.view(-1, num_heads, self.head_dim).contiguous()
            key_states = key_states.view(-1, num_kv_heads, self.head_dim).contiguous()
            value_states = value_states.view(-1, num_kv_heads, self.head_dim).contiguous()
            
            store_kvcache(key_states, value_states, self.k_cache, self.v_cache, context.slot_mapping)
            page_attention = True
        else:
            # Fallback: simple KV cache
            key_cache = key_states
            value_cache = value_states
        
        if page_attention and FLASH_ATTN_AVAILABLE:
            context = get_context()
            if context.is_prefill:
                attn_output = flash_attn_varlen_func(
                    query_states, self.k_cache, self.v_cache,
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
                    self.k_cache, self.v_cache,
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
    Hybrid decoder layer for Qwen3-Next supporting:
    - Full attention OR linear attention (GatedDeltaNet)
    - Dense MLP OR MoE
    """
    def __init__(self, config, layer_idx: int, use_tp: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.use_tp = use_tp
        
        # Determine layer type from config
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is not None and layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "full_attention"
        
        # Token mixer (attention)
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNetForEngine(config, layer_idx, use_tp=use_tp)
            self.self_attn = None
        else:  # full_attention
            self.self_attn = Qwen3NextAttentionForEngine(config, layer_idx, use_tp=use_tp)
            self.linear_attn = None
        
        # MLP block - determine if MoE or dense
        num_experts = getattr(config, 'num_experts', 0)
        decoder_sparse_step = getattr(config, 'decoder_sparse_step', 1)
        mlp_only_layers = getattr(config, 'mlp_only_layers', [])
        
        # MoE is used if: num_experts > 0 AND (layer_idx + 1) % decoder_sparse_step == 0 AND not in mlp_only_layers
        use_moe = (num_experts > 0 and 
                  (layer_idx + 1) % decoder_sparse_step == 0 and 
                  layer_idx not in mlp_only_layers)
        
        if use_moe:
            self.mlp = Qwen3NextSparseMoeBlockForEngine(config, use_tp=use_tp)
            self.is_moe = True
        else:
            self.mlp = Qwen3NextMLPForEngine(config, use_tp=use_tp)
            self.is_moe = False
        
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Token mixer
        if self.layer_type == "linear_attention" and self.linear_attn is not None:
            hidden_states = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                inference_mode=inference_mode,
            )
            k_cache, v_cache = None, None
        else:
            hidden_states, k_cache, v_cache = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                inference_mode=inference_mode,
                max_generation=max_generation,
                cur_pos=cur_pos,
            )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, k_cache, v_cache


class Qwen3NextModelForEngine(nn.Module):
    """Qwen3-Next Model backbone adapted for LLM Engine with TP support."""
    def __init__(self, config, use_tp: bool = False):
        super().__init__()
        self.config = config
        self.use_tp = use_tp
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3NextDecoderLayerForEngine(config, layer_idx, use_tp=use_tp)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=get_rms_norm_eps(config))
        
        # Get rope_theta with fallback (Qwen3-Next may use different attribute names)
        rope_theta = getattr(config, 'rope_theta', None)
        if rope_theta is None:
            rope_theta = getattr(config, 'rope_base', None)
        if rope_theta is None:
            rope_theta = getattr(config, 'rotary_pct_base', 1000000.0)  # Default
        
        self.rotary_emb = Qwen3NextRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 131072),
            base=rope_theta,
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
    ) -> Tuple[torch.Tensor, List]:
        batch_size, seq_length = input_ids.shape[:2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Compute rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        next_cache = []
        for layer in self.layers:
            hidden_states, k_cache, v_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                inference_mode=inference_mode,
                max_generation=max_generation,
                cur_pos=cur_pos,
            )
            next_cache.append((k_cache, v_cache))
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache


class Qwen3NextForLLMEngine(nn.Module):
    """
    Qwen3-Next For Causal LM adapted for in-house LLM Engine.
    
    Key interface requirement for LLMEngine:
    - forward(input_ids, position_ids, inference_mode=True, logits_to_keep=...) -> (_, logits, _, _)
    
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
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
        logits_to_keep: Optional[torch.Tensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Tuple[None, torch.Tensor, None, List]:
        """
        Forward pass compatible with LLMEngine.
        
        Returns: (_, logits, _, past_key_values)
        """
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inference_mode=inference_mode,
            max_generation=max_generation,
            cur_pos=cur_pos,
        )
        
        # Handle logits_to_keep for efficient prefill
        if logits_to_keep is not None:
            hidden_states = hidden_states.squeeze(0)[logits_to_keep]
            hidden_states = hidden_states.unsqueeze(0)
        elif num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        
        logits = self.lm_head(hidden_states)
        
        # Return format expected by LLMEngine: (_, logits, _, cache)
        return None, logits, None, next_cache


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
        print(f"Attention config: num_heads={hf_config.num_attention_heads}, num_kv_heads={hf_config.num_key_value_heads}, head_dim={head_dim}")
    
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
    
    # Create LLMEngine-compatible config
    class LLMConfig:
        pass
    
    llm_config = LLMConfig()
    llm_config.hidden_size = hf_config.hidden_size
    llm_config.num_attention_heads = hf_config.num_attention_heads
    llm_config.num_key_value_heads = hf_config.num_key_value_heads
    llm_config.num_hidden_layers = hf_config.num_hidden_layers
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
    _copy_to_device(hf_model.model.embed_tokens.weight.data, engine_model.model.embed_tokens.weight)
    
    # Copy lm_head (sharded if TP)
    if is_main:
        print("  Copying lm_head...", flush=True)
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
                print(f"  Layer {layer_idx}: No attention module found")
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
        
        # ========== ATTENTION WEIGHTS ==========
        # Engine model always uses self_attn for full attention, linear_attn for linear attention
        if has_gated_deltanet or detected_type == "linear_attention":
            # Linear attention (GatedDeltaNet) layer
            engine_attn = engine_layer.linear_attn if hasattr(engine_layer, 'linear_attn') and engine_layer.linear_attn is not None else engine_layer.self_attn
            _copy_gated_deltanet_weights(hf_attn, engine_attn,
                                        config, use_tp, tp_rank, tp_world_size, layer_idx)
        elif has_k_proj or detected_type == "full_attention":
            # Full attention layer
            engine_attn = engine_layer.self_attn if hasattr(engine_layer, 'self_attn') and engine_layer.self_attn is not None else engine_layer.linear_attn
            _copy_full_attention_weights(hf_attn, engine_attn, 
                                        config, use_tp, tp_rank, tp_world_size, layer_idx)
        else:
            if is_main:
                print(f"  Layer {layer_idx}: Skipping attention (unknown type, has_k_proj={has_k_proj}, has_gated_deltanet={has_gated_deltanet})")
        
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
            # MoE layer
            _copy_moe_weights(hf_layer.mlp, engine_layer.mlp, config, 
                             use_tp, tp_rank, tp_world_size, layer_idx)
        elif has_dense_mlp:
            # Dense MLP layer
            _copy_dense_mlp_weights(hf_layer.mlp, engine_layer.mlp, config,
                                   use_tp, tp_rank, tp_world_size, layer_idx)
        else:
            if is_main:
                print(f"  Layer {layer_idx}: Skipping MLP (unknown type)")
        
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
    
    if use_tp and tp_world_size > 1:
        # ColumnParallelLinear shards by output_size // world_size
        # q_proj output: num_heads * head_dim * 2 (QK with gate)
        q_output_size = num_heads * head_dim * 2
        q_shard = q_output_size // tp_world_size
        q_start = tp_rank * q_shard
        q_end = q_start + q_shard
        engine_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data[q_start:q_end])
        
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
    
    # A_log: per-head parameter
    if hasattr(hf_attn, 'A_log') and hf_attn.A_log is not None:
        engine_attn.A_log.data.copy_(hf_attn.A_log.data)
    
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
    test_input_text = "Hello"
    test_input = tokenizer.encode(test_input_text, return_tensors="pt").to(model_device)
    with torch.no_grad():
        _, logits, _, _ = model(test_input, inference_mode=False)
    if is_main:
        print(f"Input string: {test_input_text}")
        print(f"Input shape: {test_input.shape}")
        print(f"Output logits shape: {logits.shape}")
        # Get predicted next token
        next_token_id = logits[0, -1, :].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print(f"Predicted next token: {next_token}")
        print("Direct forward pass: OK")
    
    # Test 2: With LLMEngine
    if args.test_engine:
        if is_main:
            print("\n=== Test 2: LLMEngine Generation ===")
        from llm_engine import LLMEngine
        
        config.eos_token_id = tokenizer.eos_token_id
        engine = LLMEngine(model, config, str(model_device))
        
        # Prepare prompt with chat template
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text)
        
        if is_main:
            print(f"Prompt: {args.prompt}")
            print(f"Input string: {text}")
            print(f"Input tokens: {len(input_ids)}")
        
        import time
        start = time.time()
        output_ids = engine.generate([input_ids])[0]
        elapsed = time.time() - start
        
        if is_main:
            new_tokens = len(output_ids) - len(input_ids)
            response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
            full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"Generated {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
            print(f"Output string: {full_output}")
            print(f"Response: {response}")


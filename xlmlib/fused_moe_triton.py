"""
Fused MoE Triton Kernel for Qwen3-Next Engine.

Adapted from vLLM's fused_moe implementation (Apache 2.0 license).

Key fixes vs initial version:
- Added A_DIVIDE_BY_TOPK flag: 1st kernel divides by top_k (input=hidden_states[N,H]),
  2nd kernel indexes directly (input=intermediate[N*K,I])
- Fixed moe_align_block_size to use scatter_add for counting
- Use int64 for pointer arithmetic to prevent overflow
- Use zeros (not empty) for output caches to handle padded positions

Usage:
    from fused_moe_triton import fused_moe
    output = fused_moe(hidden_states, gate_up_proj, down_proj, 
                        topk_weights, topk_ids, top_k)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    A_DIVIDE_BY_TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Input row index
    if A_DIVIDE_BY_TOPK:
        a_row = offs_token // top_k
    else:
        a_row = offs_token

    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (a_row[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be + 
              offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(tl.bfloat16)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(topk_ids, block_size, num_experts):
    """Sort tokens by expert and pad to block_size alignment.
    
    Fully vectorized on GPU — no Python loops or .item() calls
    (except the final total_padded which is needed for grid sizing).
    """
    flat_ids = topk_ids.view(-1).long()
    num_flat = flat_ids.shape[0]
    device = topk_ids.device
    
    # Count tokens per expert (vectorized)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    ones = torch.ones(num_flat, dtype=torch.int32, device=device)
    expert_counts.scatter_add_(0, flat_ids, ones)
    
    # Pad to block_size
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size
    
    # Cumulative offsets per expert
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = padded_counts.cumsum(0)
    total_padded = expert_offsets[-1].item()
    
    # --- Vectorized sorted_token_ids construction ---
    sorted_order = flat_ids.argsort(stable=True)
    
    expert_start_in_sorted = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_start_in_sorted[1:] = expert_counts.long().cumsum(0)
    
    pos_in_sorted = torch.arange(num_flat, dtype=torch.int64, device=device)
    expert_of_sorted = flat_ids[sorted_order]
    rank_in_expert = pos_in_sorted - expert_start_in_sorted[expert_of_sorted]
    
    write_positions = expert_offsets[expert_of_sorted].long() + rank_in_expert
    
    sorted_token_ids = torch.full((total_padded,), num_flat, dtype=torch.int32, device=device)
    sorted_token_ids[write_positions] = sorted_order.int()
    
    # --- Vectorized expert_ids construction (no Python loop) ---
    num_blocks = total_padded // block_size
    # Create expert_ids by expanding each expert's block range
    block_offsets = (expert_offsets // block_size).long()  # [num_experts+1]
    # For each block, find which expert it belongs to via searchsorted
    block_indices = torch.arange(num_blocks, dtype=torch.int64, device=device)
    # searchsorted finds the rightmost expert_offset <= block_index
    expert_ids = torch.searchsorted(block_offsets[1:], block_indices, right=True).int()
    
    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def fused_moe(hidden_states, gate_up_proj, down_proj, topk_weights, topk_ids,
              top_k, num_experts=-1, activation="silu",
              w1_pre_transposed=None, w2_pre_transposed=None,
              intermediate_cache=None, output_cache=None):
    """Fused MoE forward: 2 Triton kernel launches + SiLU activation.
    
    Args:
        gate_up_proj: [E, 2*I, H] (original) or unused if w1_pre_transposed given
        down_proj: [E, H, I] (original) or unused if w2_pre_transposed given 
        w1_pre_transposed: [E, H, 2*I] — if provided, use directly (skip transpose)
        w2_pre_transposed: [E, I, H] — if provided, use directly (skip transpose)
        intermediate_cache: Pre-allocated [N*K, 2*I] buffer (optional, avoids alloc)
        output_cache: Pre-allocated [N*K, H] buffer (optional, avoids alloc)
    """
    # Ensure CUDA device is set correctly for Triton (required for multi-GPU)
    if hidden_states.is_cuda:
        torch.cuda.set_device(hidden_states.device)
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    
    # Derive dimensions from pre-transposed weights if available
    if w1_pre_transposed is not None:
        E = w1_pre_transposed.shape[0]
        two_intermediate = w1_pre_transposed.shape[2]
    else:
        E = gate_up_proj.shape[0]
        two_intermediate = gate_up_proj.shape[1]
    intermediate_size = two_intermediate // 2
    if num_experts < 0:
        num_experts = E
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 64
    num_flat = num_tokens * top_k
    
    # Sort tokens by expert
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, num_experts)
    total_padded = num_tokens_post_padded.item()
    
    # --- Kernel 1: gate_up projection ---
    # Use pre-allocated buffer or allocate new one
    if intermediate_cache is not None and intermediate_cache.shape[0] >= num_flat:
        int_cache = intermediate_cache[:num_flat]
        int_cache.zero_()
    else:
        int_cache = torch.zeros(num_flat, two_intermediate,
                                dtype=hidden_states.dtype, device=hidden_states.device)
    
    hidden_c = hidden_states.contiguous()
    
    if w1_pre_transposed is not None:
        w1 = w1_pre_transposed
    else:
        w1 = gate_up_proj.transpose(1, 2).contiguous()
    
    grid1 = (triton.cdiv(total_padded, BLOCK_SIZE_M) * triton.cdiv(two_intermediate, BLOCK_SIZE_N),)
    fused_moe_kernel[grid1](
        hidden_c, w1, int_cache,
        topk_weights.view(-1), sorted_token_ids, expert_ids, num_tokens_post_padded,
        two_intermediate, hidden_size, num_flat,
        hidden_c.stride(0), hidden_c.stride(1),
        w1.stride(0), w1.stride(1), w1.stride(2),
        int_cache.stride(0), int_cache.stride(1),
        MUL_ROUTED_WEIGHT=False, top_k=top_k, A_DIVIDE_BY_TOPK=True,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # --- Activation ---
    gate = int_cache[:, :intermediate_size]
    up = int_cache[:, intermediate_size:]
    activated = torch.nn.functional.silu(gate) * up  # [N*K, I]
    
    # --- Kernel 2: down projection ---
    if output_cache is not None and output_cache.shape[0] >= num_flat:
        out_cache = output_cache[:num_flat]
        out_cache.zero_()
    else:
        out_cache = torch.zeros(num_flat, hidden_size,
                                dtype=hidden_states.dtype, device=hidden_states.device)
    
    activated_c = activated.contiguous()
    
    if w2_pre_transposed is not None:
        w2 = w2_pre_transposed
    else:
        w2 = down_proj.transpose(1, 2).contiguous()
    
    grid2 = (triton.cdiv(total_padded, BLOCK_SIZE_M) * triton.cdiv(hidden_size, BLOCK_SIZE_N),)
    fused_moe_kernel[grid2](
        activated_c, w2, out_cache,
        topk_weights.view(-1), sorted_token_ids, expert_ids, num_tokens_post_padded,
        hidden_size, intermediate_size, num_flat,
        activated_c.stride(0), activated_c.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out_cache.stride(0), out_cache.stride(1),
        MUL_ROUTED_WEIGHT=True, top_k=top_k, A_DIVIDE_BY_TOPK=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reduce over top_k
    return out_cache.view(num_tokens, top_k, hidden_size).sum(dim=1)

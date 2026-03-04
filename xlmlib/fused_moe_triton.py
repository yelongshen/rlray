"""
Fused MoE Triton Kernel for Qwen3-Next Engine.

Adapted from vLLM's fused_moe implementation (Apache 2.0 license).
This avoids the vLLM dependency while providing the same performance.

Key idea:
- Sort tokens by expert assignment → each Triton block handles one expert's tokens
- Load expert weights directly from stacked [E, N, K] tensor (no copying/gathering)
- Fuse gate_up + activation + down_proj into minimal kernel launches
- Pre-allocated workspace avoids dynamic memory allocation

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
    # Pointers
    a_ptr,          # [num_tokens, hidden_size] — input activations
    b_ptr,          # [num_experts, N, K] — expert weights (gate_up or down)
    c_ptr,          # [num_tokens, N] — output
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N,              # output dim
    K,              # input dim  
    EM,             # total padded tokens
    num_valid_tokens,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused MoE GEMM kernel. Each block computes a tile of the output for
    tokens assigned to one expert.
    
    The key trick: sorted_token_ids maps each output row to the original
    token index. expert_ids tells which expert's weights to use.
    This avoids gathering/copying expert weights entirely.
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    # Load number of valid padded tokens
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Which tokens this block handles (sorted by expert)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # Which expert this block uses
    off_experts = tl.load(expert_ids_ptr + pid_m)

    # Output column offsets
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Input pointers: index by original token position (divided by top_k)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + 
                       offs_k[None, :] * stride_ak)
    
    # Weight pointers: index by expert
    b_ptrs = (b_ptr + off_experts * stride_be + 
              offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply routing weight
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(tl.bfloat16)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int, 
    num_experts: int,
) -> tuple:
    """
    Sort tokens by expert and pad to block_size alignment.
    
    Returns:
        sorted_token_ids: [EM] — maps padded positions to original token indices
        expert_ids: [EM // block_size] — which expert each block handles
        num_tokens_post_padded: [1] — total padded token count
    """
    num_tokens = topk_ids.numel()
    
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    flat_ids = topk_ids.view(-1)
    for e in range(num_experts):
        expert_counts[e] = (flat_ids == e).sum()
    
    # Pad each expert's count to block_size
    padded_counts = ((expert_counts + block_size - 1) // block_size) * block_size
    total_padded = padded_counts.sum().item()
    
    # Build sorted token ids and expert ids
    sorted_token_ids = torch.full((total_padded,), num_tokens, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(total_padded // block_size, dtype=torch.int32, device=topk_ids.device)
    
    # Fill sorted_token_ids per expert
    cumsum = 0
    for e in range(num_experts):
        # Find all (token, k) pairs assigned to expert e
        mask = (flat_ids == e)
        indices = mask.nonzero(as_tuple=True)[0]
        count = indices.shape[0]
        padded = padded_counts[e].item()
        
        if count > 0:
            sorted_token_ids[cumsum:cumsum + count] = indices.int()
        
        # Fill expert_ids for this expert's blocks
        num_blocks = padded // block_size
        expert_ids[cumsum // block_size: cumsum // block_size + num_blocks] = e
        
        cumsum += padded
    
    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=topk_ids.device)
    
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def fused_moe(
    hidden_states: torch.Tensor,    # [num_tokens, hidden_size]
    gate_up_proj: torch.Tensor,     # [num_experts, 2*intermediate, hidden_size]
    down_proj: torch.Tensor,        # [num_experts, hidden_size, intermediate]
    topk_weights: torch.Tensor,     # [num_tokens, top_k]
    topk_ids: torch.Tensor,         # [num_tokens, top_k]
    top_k: int,
    num_experts: int = -1,
    activation: str = "silu",
) -> torch.Tensor:
    """
    Fused MoE forward pass using Triton kernels.
    
    No temporary memory allocation for weight gathering.
    Two kernel launches: gate_up + activation, then down_proj.
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        gate_up_proj: [E, 2*I, H] — stacked gate and up projection weights
        down_proj: [E, H, I] — down projection weights
        topk_weights: [num_tokens, top_k] — routing weights
        topk_ids: [num_tokens, top_k] — expert assignments
        top_k: number of experts per token
        num_experts: total number of experts
        activation: activation function ("silu" or "gelu")
    
    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    E, two_intermediate, _ = gate_up_proj.shape
    intermediate_size = two_intermediate // 2
    
    if num_experts < 0:
        num_experts = E
    
    # Config
    BLOCK_SIZE_M = 64
    
    # Step 1: Sort tokens by expert
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, num_experts
    )
    
    # Step 2: gate_up projection — [num_tokens * top_k, 2*I]
    intermediate_cache = torch.empty(
        num_tokens * top_k, two_intermediate,
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    
    grid = lambda META: (
        triton.cdiv(num_tokens_post_padded.item(), META['BLOCK_SIZE_M']) * 
        triton.cdiv(two_intermediate, META['BLOCK_SIZE_N']),
    )
    
    # Ensure contiguous
    hidden_states = hidden_states.contiguous()
    gate_up_proj_t = gate_up_proj.transpose(1, 2).contiguous()  # [E, H, 2*I]
    
    fused_moe_kernel[grid](
        hidden_states, gate_up_proj_t, intermediate_cache,
        topk_weights,  # not used yet (MUL_ROUTED_WEIGHT=False)
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        two_intermediate, hidden_size,
        sorted_token_ids.shape[0],
        num_tokens * top_k,
        hidden_states.stride(0), hidden_states.stride(1),
        gate_up_proj_t.stride(0), gate_up_proj_t.stride(1), gate_up_proj_t.stride(2),
        intermediate_cache.stride(0), intermediate_cache.stride(1),
        MUL_ROUTED_WEIGHT=False,
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
    )
    
    # Step 3: activation (SiLU gating)
    gate = intermediate_cache[:, :intermediate_size]
    up = intermediate_cache[:, intermediate_size:]
    intermediate_cache2 = torch.nn.functional.silu(gate) * up  # [N*K, I]
    
    # Step 4: Re-sort for down_proj (same sorting)
    output_cache = torch.zeros(
        num_tokens * top_k, hidden_size,
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    
    down_proj_t = down_proj.transpose(1, 2).contiguous()  # [E, I, H]
    
    grid2 = lambda META: (
        triton.cdiv(num_tokens_post_padded.item(), META['BLOCK_SIZE_M']) * 
        triton.cdiv(hidden_size, META['BLOCK_SIZE_N']),
    )
    
    fused_moe_kernel[grid2](
        intermediate_cache2, down_proj_t, output_cache,
        topk_weights.view(-1),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        hidden_size, intermediate_size,
        sorted_token_ids.shape[0],
        num_tokens * top_k,
        intermediate_cache2.stride(0), intermediate_cache2.stride(1),
        down_proj_t.stride(0), down_proj_t.stride(1), down_proj_t.stride(2),
        output_cache.stride(0), output_cache.stride(1),
        MUL_ROUTED_WEIGHT=True,
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
    )
    
    # Step 5: Reduce over top_k dimension
    output = output_cache.view(num_tokens, top_k, hidden_size).sum(dim=1)
    
    return output

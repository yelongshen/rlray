import io
import os

import torch
import torch.nn as nn
import pytest

from flash_attn import flash_attn_func, flash_attn_varlen_func

import math
import triton
import triton.language as tl

# two usage:
#    1. query_state.length == key_states.length == value_states.length (attention with sliding_window_size 4) 
#    2. query_state.length == 1 (or n), while key_states.length == value_states.length = m ( m > n), 

device = 'cuda:0'

#from vllm.attention.backends.flash_attn import flash_attn_varlen_func
#from vllm.attention.backends.mla.common import _get_graph_runner_block_tables
block_size = 4
seq_lens = [7, 10, 13, 16, 20]  # => ceil(7/4)=2 blocks, ceil(10/4)=3 blocks
# Batch size = 2, block_size = 4
# Sequence lengths
num_heads, head_dim = 2, 2

max_blocks = max((l + block_size - 1) // block_size for l in seq_lens)
# Create dummy tensors: [total_tokens, num_heads, head_dim]
total_tokens = sum(seq_lens)

q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)

cu_seqlens = [0]
for l in seq_lens:
    cu_seqlens.append(cu_seqlens[-1] + l)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def test_kv_store():
    # 2,3,5, block-size:4
    key = torch.randn(10, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)

    k_cache = torch.zeros(6, 4, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(6, 4, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    print(key)
    print(value)

    slot_mapping = []
    block_idx = 0
    _seq_len = [2,3,5]
    for sl in _seq_len:
        num_blocks = (sl + 4 - 1) // 4
        for i in range(0, num_blocks):
            start = (block_idx + i) * 4 # block start.
            if i != num_blocks - 1:
                end = start + 4
            else:
                end = start + sl - 4 * i 
            slot_mapping.extend(list(range(start, end)))
        block_idx += num_blocks
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=device, pin_memory=True).cuda(non_blocking=True)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    print(k_cache)
    print(v_cache)

# block-wise kv cache store.
test_kv_store()

def test_paged_attention():
    # obtain the max_block number.
    # Cumulated lengths for varlen API
    
    cu_seqlens_q = cu_seqlens_k = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)

    # Build logical block_tables per sequence (allocate sequential physical blocks)
    block_tables = []
    phys_idx = 0
    for sl in seq_lens:
        logical_blocks = (sl + block_size - 1) // block_size # 2, 3, 4, 4, 5
        block_tables.append(list(range(phys_idx, phys_idx + logical_blocks))) # 0, 1, 2, 
        phys_idx += logical_blocks

    # Map to full-sized GPU block_table tensor with padding = -1
    #block_table = _get_graph_runner_block_tables(len(seq_lens), block_tables)
    #assert block_table.shape == (len(seq_lens), max_blocks)
    # Check padding on row 0, positions beyond 2 blocks
    #assert (block_table[0, 2:] == -1).all()

    # Call paged flash attention
    out = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=max(seq_lens),
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max(seq_lens),
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=1.0,
        causal=True,
        #block_table=block_table
    )
    # Output shape: same as input q
    assert out.shape == q.shape

    return out

    # A basic sanity check: output should be finite and non-NaN
    #assert torch.isfinite(out).all()

def test_vanilla_attention():
    outputs_naive = []
    for i in range(len(seq_lens)):
        q_i = q[cu_seqlens[i]:cu_seqlens[i+1]]  # (L_i, nheads, headdim)
        k_i = k[cu_seqlens[i]:cu_seqlens[i+1]]
        v_i = v[cu_seqlens[i]:cu_seqlens[i+1]]
        o_i = flash_attn_func(q_i.unsqueeze(0), k_i.unsqueeze(0), v_i.unsqueeze(0), softmax_scale=1.0, causal=True)

        outputs_naive.append(o_i.squeeze(0)[-1])

    # 将朴素实现的输出拼接起来
    out_naive = torch.cat(outputs_naive, dim=0)  # (total_q, nheads, headdim)
    return out_naive


o1 = test_paged_attention()
o2 = test_vanilla_attention()

print(o1)
print(o2)
assert torch.allclose(o1, o2, atol=1e-1)

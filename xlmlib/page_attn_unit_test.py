import io
import os

import torch
import torch.nn as nn
import pytest

from flash_attn import flash_attn_func, flash_attn_varlen_func

import math

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
num_heads, head_dim = 2, 8

max_blocks = max((l + block_size - 1) // block_size for l in seq_lens)
# Create dummy tensors: [total_tokens, num_heads, head_dim]
total_tokens = sum(seq_lens)

q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)

cu_seqlens = [0]
for l in seq_lens:
    cu_seqlens.append(cu_seqlens[-1] + l)

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

        outputs_naive.append(o_i.squeeze(0))

    # 将朴素实现的输出拼接起来
    out_naive = torch.cat(outputs_naive, dim=0)  # (total_q, nheads, headdim)
    return out_naive


o1 = test_paged_attention()
o2 = test_vanilla_attention()

print(o1)
print(o2)
assert torch.allclose(o1, o2, atol=1e-1)

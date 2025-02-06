import torch
import io
import os

import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

# two usage:
#    1. query_state.length == key_states.length == value_states.length (attention with sliding_window_size 4) 
#    2. query_state.length == 1 (or n), while key_states.length == value_states.length = m ( m > n), 

# query size
bs = 2
seq_len = 16
n_head = 6
head_dim = 8
window = 4

device = 'cuda:0'

query = torch.randn(bs, n_head, seq_len, head_dim, device = device)
key = torch.randn(bs, n_head, seq_len, head_dim, device = device)
value = torch.randn(bs, n_head, seq_len, head_dim, device = device)

def vanilla_attention():
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    attn_weights = attn_weights.masked_fill(attention_mask < 0.1, float('-inf'))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
    attn_output = torch.matmul(attn_weights, value)
    
    print(attn_output.shape)
    return attn_output

def flash_attention():
    f_query = query.transpose(1, 2)
    f_key = key.transpose(1, 2)
    f_value = value.transpose(1, 2)

    attn_output = flash_attn_func(
            f_query,
            f_key,
            f_value,
            0,
            softmax_scale=None,
            causal=True)
    return attn_output

o1 = vanilla_attention()
o2 = flash_attention()
assert torch.allclose(o1, o2, atol=1e-4)

#def flash_attn

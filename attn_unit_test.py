import io
import os

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

import math
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

query = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)
key = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)
value = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)

a_query = torch.randn(bs, n_head, 1, head_dim, dtype = torch.bfloat16, device = device)
a_key = torch.randn(bs, n_head, window, head_dim, dtype = torch.bfloat16, device = device)
a_value = torch.randn(bs, n_head, window, head_dim, dtype = torch.bfloat16, device = device)

def vanilla_attention():
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    attn_weights = attn_weights.masked_fill(attention_mask < 0.1, float('-inf'))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
    attn_output = torch.matmul(attn_weights, value)
    
    print(attn_output.shape)
    return attn_output

def vanilla_sliding_attention():
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    for i in range(seq_len):
        start = max(i - window + 1, 0)
        attention_mask[i][:start] = 0
        
    attn_weights = attn_weights.masked_fill(attention_mask < 0.1, float('-inf'))
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
    attn_output = torch.matmul(attn_weights, value)
    
    print(attn_output.shape)
    return attn_output

def vanilla_step_attention():
    attn_weights = torch.matmul(a_query, a_key.transpose(2, 3)) / math.sqrt(head_dim)
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
    attn_output = torch.matmul(attn_weights, a_value)
    
    print(attn_output.shape)
    return attn_output


def flash_attention():
    f_query = query.transpose(1, 2) #.contiguous()
    f_key = key.transpose(1, 2) #.contiguous()
    f_value = value.transpose(1, 2) #.contiguous()

    attn_output = flash_attn_func(
            f_query,
            f_key,
            f_value,
            0,
            softmax_scale=None,
            causal=True)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    print(attn_output.shape)
    return attn_output
    
def flash_sliding_attention():
    f_query = query.transpose(1, 2) #.contiguous()
    f_key = key.transpose(1, 2) #.contiguous()
    f_value = value.transpose(1, 2) #.contiguous()

    attn_output = flash_attn_func(
            f_query,
            f_key,
            f_value,
            0,
            softmax_scale=None,
            causal=True, 
            window_size=(
                window-1,
                window-1,
            )
    )
    
    attn_output = attn_output.transpose(1, 2) #.contiguous()
    print(attn_output.shape)
    return attn_output
    
def flash_step_attention():
    f_query = a_query.transpose(1, 2) #.contiguous()
    f_key = a_key.transpose(1, 2) #.contiguous()
    f_value = a_value.transpose(1, 2) #.contiguous()

    attn_output = flash_attn_func(
            f_query,
            f_key,
            f_value,
            0,
            softmax_scale=None,
            causal=True, 
            window_size=(
                window-1,
                window-1,
            )
    )
    
    attn_output = attn_output.transpose(1, 2) #.contiguous()
    print(attn_output.shape)
    return attn_output
    
    


#torch.Size([2, 6, 16, 8])
#torch.Size([2, 16, 6, 8])

#o1 = vanilla_attention()
#o2 = flash_attention()

#o1 = vanilla_sliding_attention()
#o2 = flash_sliding_attention()

o1 = vanilla_step_attention()
o2 = flash_step_attention()

print(o1)
print(o2)
assert torch.allclose(o1, o2, atol=1e-2)

#def flash_attn

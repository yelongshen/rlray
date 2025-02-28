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
window = 8
n_kv_head = 2

kv_group = n_head // n_kv_head

#self.num_heads = 8  # More query heads
#self.num_kv_heads = 2  # Fewer key-value heads (GQA)

device = 'cuda:0'

query = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)
key = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)
value = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)

a_query = torch.randn(bs, n_head, 1, head_dim, dtype = torch.bfloat16, device = device)
a_key = torch.randn(bs, n_head, window, head_dim, dtype = torch.bfloat16, device = device)
a_value = torch.randn(bs, n_head, window, head_dim, dtype = torch.bfloat16, device = device)

# consider full attention first. 
query = torch.randn(bs, n_head, seq_len,  head_dim, dtype = torch.bfloat16, device = device)
key = torch.randn(bs, n_kv_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)
value = torch.randn(bs, n_kv_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    _bs, _n_head, _len, _head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(_bs, _h_head, n_rep, _len, _head_dim)
    return hidden_states.reshape(_bs, _h_head * n_rep, _len, _head_dim)

def vanilla_gqa():
    _key_states = repeat_kv(key, kv_group)
    _value_states = repeat_kv(value, kv_group)
    
    attn_weights = torch.matmul(query, _key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    attn_weights = attn_weights.masked_fill(attention_mask < 0.1, float('-inf'))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
    attn_output = torch.matmul(attn_weights, _value_states)
    print(attn_output.shape)
    return attn_output

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

    #m_key = torch.cat([a_key, torch.randn(bs, n_head, 1, head_dim, dtype = torch.bfloat16, device = device)], dim = 2)
    #m_value = torch.cat([a_value, torch.randn(bs, n_head, 1, head_dim, dtype = torch.bfloat16, device = device)], dim = 2)
                       
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

#o1 = vanilla_step_attention()
#o2 = flash_step_attention()

o1 = vanilla_gqa()
o2 = flash_attention()

print(o1)
print(o2)
assert torch.allclose(o1, o2, atol=1e-2)

#def flash_attn

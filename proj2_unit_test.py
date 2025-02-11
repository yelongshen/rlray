import io
import os

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

import math

from einops import rearrange, repeat
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

import selective_scan_cuda
import causal_conv1d_cuda

bs = 2
seqlen = 16

# meta-hyper
d_model = 6
d_state = 16
d_conv = 4
expand = 2

##### calculate
d_inner = int(expand * d_model)
dt_rank = math.ceil(d_model / 2) 
device = 'cuda:0'
hidden_states = torch.randn(bs, seqlen, d_model, device = device)
in_proj = nn.Linear(d_model, d_inner * 2, bias=True, device = device)

conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, bias=True, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, device=device)
activation = "silu"
act = nn.SiLU()

x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False, device=device)
dt_proj = nn.Linear(dt_rank, d_inner, bias=True, device=device)

# bs, d_model * expand, d_conv = 4 
conv_state = torch.zeros(bs, d_model * expand, d_conv, device = device)
ssm_state = torch.zeros(bs, d_model * expand, d_state, device = device)

A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device = device), "n -> d n", d = d_inner,).contiguous()
A_log = torch.log(A)  # Keep A_log in fp32
A_log = nn.Parameter(A_log)

# D "skip" parameter
D = nn.Parameter(torch.ones(d_inner, device=device))  # Keep in fp32
out_proj = nn.Linear(d_inner, d_model, bias=True, device = device) #, **factory_kwargs)

def conv_case1():
    global conv1d
    global hidden_states
  
    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen)
  
    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)

    # xz : torch.Size([2, 24, 16]  
    # x : torch.Size([2, 12, 16])
    # z : torch.Size([2, 12, 16])
  
    conv1d_weight = conv1d.weight
    conv1d_bias = conv1d.bias
    
    conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
    conv1d_bias = conv1d_bias.contiguous() 
        
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, None, None, True)

    # standard mode: 
    return conv1d_out

def conv_case2():
    global conv1d
    global hidden_states

    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen)

    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)

    # x : bs, dim, seqlength
    #x = hidden_states
    x = causal_conv1d_fn(
                    x = x,
                    weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                    bias=conv1d.bias,
                    activation=activation
                )
    return x

def conv_case3():
    # split x into two part, first 15, second 1.
    
    global conv1d
    global hidden_states
    global conv_state
  
    h1 = hidden_states[:, :seqlen-1, :].contiguous()
    h2 = hidden_states[:, -1:, :].contiguous()
  
    xz1 = rearrange(
            in_proj.weight @ rearrange(h1.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen-1)

    if in_proj.bias is not None:
        xz1 = xz1 + rearrange(in_proj.bias.to(dtype=xz1.dtype), "d -> d 1")

    x1, z1 = xz1.chunk(2, dim=1)

    # x : bs, dim, seqlength
    #x = hidden_states
    x1 = causal_conv1d_fn(
                    x = x1,
                    weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                    bias=conv1d.bias,
                    activation=activation
                )

    # save the x into conv_states. 
    conv_state.copy_(F.pad(x1, (d_conv - x1.shape[-1], 0))) 


    xz2 = rearrange(
            in_proj.weight @ rearrange(h2.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=1)

    if in_proj.bias is not None:
        xz2 = xz2 + rearrange(in_proj.bias.to(dtype=xz2.dtype), "d -> d 1")

    x2, z2 = xz2.chunk(2, dim=1)

    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
    conv_state[:, :, -1] = x2
    x2 = torch.sum(conv_state * rearrange(conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
    x2 = x2 + conv1d.bias
    x2 = act(x2).to(dtype=dtype)

    #x2 = causal_conv1d_update(
    #            x2.squeeze(),
    #            conv_state,
    #            rearrange(conv1d.weight, "d 1 w -> d w"),
    #            conv1d.bias,
    #            activation,
    #        )

    return torch.cat([x1, x2.unsqueeze(dim=-1)], dim=-1) #x2
  
x1 = conv_case1()
x2 = conv_case2()
x3 = conv_case3()
print(x1.shape, x1)
print(x2.shape, x2)
print(x3.shape, x3)

assert torch.allclose(x1, x2, atol=1e-2)
assert torch.allclose(x1, x3, atol=1e-2)


#x1 = 
#case1()
#x2 = case2()

#print(x1.shape)
#print(x1)

#print(x2.shape)
#print(x2)


#assert torch.allclose(x1, x2, atol=1e-2)

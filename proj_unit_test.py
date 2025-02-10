import io
import os

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

import math

from einops import rearrange, repeat


bs = 2
seqlen = 16


d_model = 6
d_inner = 12

device = 'cuda:0'


hidden_states = torch.randn(bs, seqlen, d_model, device = device)
in_proj = nn.Linear(d_model, d_inner * 2, bias=True, device = device)

#x = 

def case1():
    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)
    # if not self.training:
    #     xz = xz.to(torch.float32)
    #A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    return x, z

def case2():
    xz = in_proj(hidden_states.to(dtype = in_proj.weight.dtype).squeeze(1))  # (B 2D)

    #xz = self.in_proj(hidden_states.to(dtype = self.in_proj.weight.dtype).squeeze(1))  # (B 2D)
    x, z = xz.chunk(2, dim=-1)  # (B D)

    return x, z 
    #x, z = xz.chunk(2, dim=-1)  # (B D)

    #if have conv_state
    #x = causal_conv1d_update(
    #            x,
    #            conv_state,
    #            rearrange(self.conv1d.weight, "d 1 w -> d w"),
    #            self.conv1d.bias,
    #            self.activation,
    #        )
    

    #return xz


x1, z1 = case1()
x2, z2 = case2()

print(x1.shape)
print(x1)

print(x2.shape)
print(x2)

print(z1.shape)
print(z1)

print(z2.shape)
print(z2)

assert torch.allclose(x1, x2, atol=1e-2)
assert torch.allclose(z1, z2, atol=1e-2)


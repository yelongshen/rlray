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
            in_proj.weight @ rearrange(hidden_states.to(dtype = self.in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
    if in_proj.bias is not None:
        xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    # if not self.training:
    #     xz = xz.to(torch.float32)
    #A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    return xz

def case2():
    xz = in_proj(hidden_states.to(dtype = self.in_proj.weight.dtype).squeeze(1))  # (B 2D)
    return xz


o1 = case1()
o2 = case2()

print(o1)
print(o2)
assert torch.allclose(o1, o2, atol=1e-2)


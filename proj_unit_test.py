import io
import os

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

import math

from einops import rearrange, repeat
import torch.nn.functional as F


bs = 2
seqlen = 16


d_model = 6
d_inner = 12

device = 'cuda:0'
d_conv = 4


hidden_states = torch.randn(bs, seqlen, d_model, device = device)
in_proj = nn.Linear(d_model, d_inner * 2, bias=True, device = device)

#x = 
conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, device=device)
activation = "silu"
act = nn.SiLU()


x_proj = nn.Linear(d_inner, self.dt_rank + self.d_state * 2, bias=False, device=device)


def case1():
    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

    # xz : torch.Size([2, 24, 16]  
    # x : torch.Size([2, 12, 16])
    # z : torch.Size([2, 12, 16])
    
    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)

    # copy the last d_conv x.
    conv_state.copy_(F.pad(x, (d_conv - x.shape[-1], 0)))  # Update state (B D W)

    x = causal_conv1d_fn(
                    x = x, # b, dim, l 
                    weight = rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias = self.conv1d.bias,
                    activation = self.activation,
                )
    # x = self.act(self.conv1d(x)[..., :seqlen])
    # x : b, dim, l
    x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    
    # if not self.training:
    #     xz = xz.to(torch.float32)
    #A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    return x, z

def case2():
    xz = in_proj(hidden_states.to(dtype = in_proj.weight.dtype).squeeze(1))  # (B 2D)

    #xz = self.in_proj(hidden_states.to(dtype = self.in_proj.weight.dtype).squeeze(1))  # (B 2D)
    x, z = xz.chunk(2, dim=-1)  # (B D)

    # x : torch.Size([2, 16, 12])
    # z : torch.Size([2, 16, 12])
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


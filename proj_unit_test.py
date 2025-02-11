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

#x = 
conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, bias=True, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1, device=device)
activation = "silu"
act = nn.SiLU()

x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False, device=device)
dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

# bs, d_model * expand, d_conv = 4 
conv_state = torch.zeros(bs, d_model * expand, d_conv, device = device)

ssm_state = torch.zeros(bs, d_model * expand, d_state, device = device)

A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d = d_inner,).contiguous()
A_log = torch.log(A)  # Keep A_log in fp32
A_log = nn.Parameter(A_log)

# D "skip" parameter
D = nn.Parameter(torch.ones(d_inner))  # Keep in fp32
out_proj = nn.Linear(d_inner, d_model, bias=True) #, **factory_kwargs)

def case1():
    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen)

    # xz : torch.Size([2, 24, 16]  
    # x : torch.Size([2, 12, 16])
    # z : torch.Size([2, 12, 16])
    
    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)

    A = -torch.exp(A_log.float()) 
    
    # x : bs, d_inner (12), seqlen, 
    # copy the last d_conv x.
    conv_state.copy_(F.pad(x, (d_conv - x.shape[-1], 0)))  # Update state (B D W)

    #print('o1 before x.shape',x.shape)
    x = causal_conv1d_fn(
                    x = x, # b, dim, l 
                    weight = rearrange(conv1d.weight, "d 1 w -> d w"),
                    bias = conv1d.bias,
                    activation = activation,
                )
    #print('o1 after x.shape', x.shape)

    x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            
    dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
    dt = dt_proj.weight @ dt.t()
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    #forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
    #            return_last_state=False):
    
    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
        
    D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    
    if B.dim() == 3:
        B = rearrange(B, "b dstate l -> b 1 dstate l")
        #ctx.squeeze_B = True
    
    if C.dim() == 3:
        C = rearrange(C, "b dstate l -> b 1 dstate l")
        #ctx.squeeze_C = True

    out, scan_intermediates, out_z = selective_scan_cuda.fwd(x, dt, A, B, C, D, z, dt_proj.bias, True)
    

    print(out.shape)
    print(scan_intermediates.shape)
    print(out_z.shape)

    return out
    
    # x = self.act(self.conv1d(x)[..., :seqlen])
    # x : b, dim, l
    #print('x.shape:', x.shape)
    
    #x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

    #print('x_dbl.shape:', x_dbl.shape)

    #return x #x_dbl
    
    #        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #        dt = self.dt_proj.weight @ dt.t()
    #        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    #        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    #        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    # if not self.training:
    #     xz = xz.to(torch.float32)
    #A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    #return x, z

def case2():
    xz = in_proj(hidden_states.to(dtype = in_proj.weight.dtype).squeeze(1))  # (B 2D)
    #xz = self.in_proj(hidden_states.to(dtype = self.in_proj.weight.dtype).squeeze(1))  # (B 2D)
    x, z = xz.chunk(2, dim=-1)  # (B D)
    # x : torch.Size([2, 16, 12])
    # z : torch.Size([2, 16, 12])
    # x: bs, seqlen, dim 
    print('o2 before x.shape',x.shape)
    x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(conv1d.weight, "d 1 w -> d w"),
                conv1d.bias,
                activation)

    print('o2 after x.shape', x.shape)
    #x_db = self.x_proj(x)  # (B dt_rank+2*d_state)

    return x
    #return x, z 
    #x, z = xz.chunk(2, dim=-1)  # (B D)
    #if have conv_state
    #x = causal_conv1d_update(
    #            x,
    #            conv_state,
    #            rearrange(self.conv1d.weight, "d 1 w -> d w"),
    #            self.conv1d.bias,
    #            self.activation,
    #        )
    

#x1 = 
case1()
#x2 = case2()

#print(x1.shape)
#print(x1)

#print(x2.shape)
#print(x2)


#assert torch.allclose(x1, x2, atol=1e-2)


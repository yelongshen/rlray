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

import triton
print(triton.__version__)

from mamba_ssm.ops.triton.selective_state_update import selective_state_update

bs = 2
seqlen = 3

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
    global x_proj
    global dt_proj
    global A_log
    global D
    global out_proj
    
    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen)
  
    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)

    _conv_state = torch.zeros(bs, d_model * expand, d_conv, device = device)

    _conv_state.copy_(F.pad(x, (d_conv - x.shape[-1], 0))) 

    # xz : torch.Size([2, 24, 16]  
    # x : torch.Size([2, 12, 16])
    # z : torch.Size([2, 12, 16])
  
    conv1d_weight = conv1d.weight
    conv1d_bias = conv1d.bias
    
    conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
    conv1d_bias = conv1d_bias.contiguous() 
    
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, None, None, True)

    x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj.weight)  # (bl d)
    delta = rearrange(dt_proj.weight @ x_dbl[:, :dt_rank].t(), "d (b l) -> b d l", l = seqlen)

    A = -torch.exp(A_log.float())
    
    B = x_dbl[:, dt_rank:dt_rank + d_state]  # (bl dstate)
    B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=seqlen).contiguous()

    C = x_dbl[:, -d_state:]  # (bl dstate)
    C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=seqlen).contiguous()

    D = D.contiguous()

    #delta_softplus=True,
    out, scan_intermediates, out_z = selective_scan_cuda.fwd(conv1d_out, delta, A, B, C, D.float(), z, dt_proj.bias.float(), True)

    fout = F.linear(rearrange(out_z, "b d l -> b l d"), out_proj.weight, out_proj.bias)

    _ssm_state = scan_intermediates[:, :, -1, 1::2]
    
    # standard mode: 
    return fout, _conv_state, _ssm_state # _ssm_state not there yet. 

def conv_case2():
    global conv1d
    global hidden_states
    global x_proj
    global dt_proj
    global A_log
    global D
    global out_proj

    xz = rearrange(
            in_proj.weight @ rearrange(hidden_states.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen)

    if in_proj.bias is not None:
        xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)

    _conv_state = torch.zeros(bs, d_model * expand, d_conv, device = device)
    _conv_state.copy_(F.pad(x, (d_conv - x.shape[-1], 0))) 
    
    # x : bs, dim, seqlength
    #x = hidden_states
    x = causal_conv1d_fn(
                    x = x,
                    weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                    bias=conv1d.bias,
                    activation=activation
                )

    x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            
    dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
    dt = dt_proj.weight @ dt.t()

    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
        
    D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()

    A = -torch.exp(A_log.float())
    
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
        
    if B.dim() == 3:
        B = rearrange(B, "b dstate l -> b 1 dstate l")
        #ctx.squeeze_B = True
    if C.dim() == 3:
        C = rearrange(C, "b dstate l -> b 1 dstate l")
        #ctx.squeeze_C = True
            
    out, x, rest = selective_scan_cuda.fwd(x, dt, A, B, C, D.float(), z, dt_proj.bias.float(), True)

    y = rearrange(rest, "b d l -> b l d")
    
    fout = out_proj(y)
    
    _ssm_state = x[:, :, -1, 1::2] # (batch, dim, dstate)

    return fout, _conv_state, _ssm_state # _ssm_state not there yet. 
    

def conv_case3():
    # split x into two part, first 15, second 1.
    
    global conv1d
    global hidden_states
    #global conv_state
    global x_proj
    global dt_proj
    global A_log
    global D
    global out_proj

    A = -torch.exp(A_log.float())
    D = D.contiguous()
    
    h1 = hidden_states[:, :seqlen-1, :].contiguous()
    h2 = hidden_states[:, -1:, :].contiguous()

    
    xz1 = rearrange(
            in_proj.weight @ rearrange(h1.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen-1)

    if in_proj.bias is not None:
        xz1 = xz1 + rearrange(in_proj.bias.to(dtype=xz1.dtype), "d -> d 1")

    x1, z1 = xz1.chunk(2, dim=1)

    _conv_state = torch.zeros(bs, d_model * expand, d_conv, device = device)
    
    # save the x into conv_states. 
    _conv_state.copy_(F.pad(x1, (d_conv - x1.shape[-1], 0))) 

    # x : bs, dim, seqlength
    #x = hidden_states
    x1 = causal_conv1d_fn(
                    x = x1,
                    weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                    bias=conv1d.bias,
                    activation=activation
                )
    
    x1_dbl = x_proj(rearrange(x1, "b d l -> (b l) d"))  # (bl d)
    
    dt1, B1, C1 = torch.split(x1_dbl, [dt_rank, d_state, d_state], dim=-1)
    dt1 = dt_proj.weight @ dt1.t()

    dt1 = rearrange(dt1, "d (b l) -> b d l", l=seqlen-1)
    B1 = rearrange(B1, "(b l) dstate -> b dstate l", l=seqlen-1).contiguous()
    C1 = rearrange(C1, "(b l) dstate -> b dstate l", l=seqlen-1).contiguous()

    if x1.stride(-1) != 1:
        x1 = x1.contiguous()
    if dt1.stride(-1) != 1:
        dt1 = dt1.contiguous()
        
    if B1.stride(-1) != 1:
        B1 = B1.contiguous()
    if C1.stride(-1) != 1:
        C1 = C1.contiguous()

    if z1 is not None and z1.stride(-1) != 1:
        z1 = z1.contiguous()
        
    if B1.dim() == 3:
        B1 = rearrange(B1, "b dstate l -> b 1 dstate l")
        #ctx.squeeze_B = True
    if C1.dim() == 3:
        C1 = rearrange(C1, "b dstate l -> b 1 dstate l")
        #ctx.squeeze_C = True
            
    out1, x1, rest1 = selective_scan_cuda.fwd(x1, dt1, A, B1, C1, D.float(), z1, dt_proj.bias.float(), True)
    y1 = rearrange(rest1, "b d l -> b l d")
    fout = out_proj(y1)
    _ssm_state = x1[:, :, -1, 1::2] # (batch, dim, dstate)
    
    ################### segment 2
    xz2 = rearrange(
            in_proj.weight @ rearrange(h2.to(dtype = in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=1)
    if in_proj.bias is not None:
        xz2 = xz2 + rearrange(in_proj.bias.to(dtype=xz2.dtype), "d -> d 1")
    x2, z2 = xz2.chunk(2, dim=1)
    x2 = x2.squeeze()
    z2 = z2.squeeze()
    
    #conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
    #conv_state[:, :, -1] = x2
    #x2 = torch.sum(conv_state * rearrange(conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
    #x2 = x2 + conv1d.bias
    #x2 = act(x2)#.to(dtype=dtype)
    x2 = causal_conv1d_update(
                x2,
                _conv_state,
                rearrange(conv1d.weight, "d 1 w -> d w"),
                conv1d.bias,
                activation,
            )


    x2_db = x_proj(x2)  # (B dt_rank+2*d_state)
    dt2, B2, C2 = torch.split(x2_db, [dt_rank, d_state, d_state], dim=-1)
    # Don't add dt_bias here
    dt2 = F.linear(dt2, dt_proj.weight)  # (B d_inner)

    y2 = selective_state_update(_ssm_state, x2, dt2, A, B2, C2, D.float(), z2, dt_proj.bias.float(), True)

    fout2 = out_proj(y2)
    
    return torch.cat([fout, fout2.unsqueeze(dim=1)], dim=1), _conv_state, _ssm_state

  
x1,c1,s1 = conv_case1()
x2,c2,s2 = conv_case2()
x3,c3,s3 = conv_case3()

print('--------------------------------')
print(x1.shape, x1)
print(x2.shape, x2)    
print(x3.shape, x3)

#print(x3.shape, x3)
#x3,c3 = conv_case3()
print('--------------------------------')
print(c1.shape, c1)
print(c2.shape, c2)
print(c3.shape, c3)

#print(c3.shape, c3)

print('--------------------------------')
print(s1.shape, s1)
print(s2.shape, s2)
print(s3.shape, s3)

assert torch.allclose(c1, c2, atol=1e-2)
#assert torch.allclose(c1, c3, atol=1e-2)
assert torch.allclose(x1, x2, atol=1e-2)

assert torch.allclose(s1, s2, atol=1e-2)

assert torch.allclose(c1, c3, atol=1e-2)

assert torch.allclose(x1, x3, atol=1e-2)

assert torch.allclose(s1, s3, atol=1e-2)

#assert torch.allclose(x1, x3, atol=1e-2)

#x1 = 
#case1()
#x2 = case2()

#print(x1.shape)
#print(x1)

#print(x2.shape)
#print(x2)


#assert torch.allclose(x1, x2, atol=1e-2)

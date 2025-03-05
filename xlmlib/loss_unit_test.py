import io
import os

import torch
import torch.nn as nn

import math
import torch.nn.functional as F

from fused_linear_cross_entropy import FusedLinearCrossEntropyFunction
hidden_size = 16
vocab_size = 24
bsz = 3
seqlen = 8

device = 'cuda:0'

#query = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)

lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype = torch.bfloat16, device = device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

states = torch.randn(bsz, seqlen, hidden_size, dtype = torch.bfloat16, device = device)
label = torch.zeros(bsz, seqlen, dtype = torch.long, device = device)

def vanilla_linear_softmax():
    global lm_head
    global states
    global label
    
    logits = lm_head(states)

    loss = criterion(logits.reshape(-1, vocab_size), label.reshape(-1)) 

    loss = -F.cross_entropy(
                input=logits.reshape(-1, vocab_size), #.transpose(1, 2),
                target=label.reshape(-1),
                reduction="none"
            ).view(bsz, seqlen)
    
    print('loss1:', loss)
    return loss

def fused_linear_softmax():
    global lm_head
    global states
    global label
    
    loss = FusedLinearCrossEntropyFunction.apply(states.view(-1, states.size(-1)), lm_head.weight, label.reshape(-1), reduction='none')

    print('loss2:', loss)
    return loss
loss1 = vanilla_linear_softmax()
loss2 = fused_linear_softmax()

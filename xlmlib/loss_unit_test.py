import io
import os

import torch
import torch.nn as nn

import math

hidden_size = 16
vocab_size = 32
bsz = 3
seqlen = 8

device = 'cuda:0'

#query = torch.randn(bs, n_head, seq_len, head_dim, dtype = torch.bfloat16, device = device)

lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype = torch.bfloat16, device = device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

states = torch.randn(bsz, seqlen, hidden_size, dtype = torch.bfloat16, device = device)
label = torch.zero(bsz, seqlen, dtype = torch.long, device = device)

def vanilla_linear_softmax():
    global lm_head
    global data

    logits = lm_head(states)

    loss = criterion(logits.reshape(-1, vocab_size), label.reshape(-1)) 

    print('loss:', loss)
    return loss
  
loss = vanilla_linear_softmax()


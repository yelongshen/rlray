import os
import sys
import io
import logging

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from replaybuffer import ReplayBuffer, Sample

def sft_gradient(llm, llm_config, buffer, buffer_size, device, weight = 0.1):
    llm.train() 
    step = 0
    batch_size = 1
    micro_training_steps = buffer_size / batch_size
    vocab_size = llm_config.vocab_size
    mini_sft_loss = 0
    
    # rl training steps;
    while step < micro_training_steps:
        mini_data = buffer.pop(batch_size)

        input_tokens = [d.tokens for d in mini_data]
        _response_idx = mini_data[0].masks.index(1)
      
        #advantages = [d.advantages for d in mini_data]
        #advantages = [d.normalized_advantages for d in mini_data]
        #returns = [d.returns for d in mini_data]
        
        # do tokens padding & alignment with batchsize  > 1   
        input_tokens = torch.tensor(input_tokens).to(torch.long).to(device)    
        
        _batch_size, _seq_len = input_tokens.shape
        # generation token index. 
        # re-evaluate the policy.     
        # return: next_token_loss, logits, critics, next_decoder_cache 
        _, logits, critics, _ = llm(input_tokens)
    
        logprobs = -F.cross_entropy(
            input = logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
            target = input_tokens.reshape(-1)[1:], 
            reduction = "none",
            #ignore_index = pad_id,
        ).reshape(1, -1)

        logprobs = logprobs[:, _response_idx-1:]
        # we shall do advantage normalization. 
        
        _total_loss = - (logprobs.mean() *  weight + critics.mean() * 0.0) / micro_training_steps # (_policy_loss + critic_alpha * _critic_loss) / micro_training_steps 
        
        # final loss of clipped objective PPO objective. 
        # take gradient step
        mini_sft_loss = mini_sft_loss + _total_loss.detach()  

        _total_loss.backward()
        #print(' _policy_loss:', _policy_loss, ' , _critic_loss:', _critic_loss, ' , device:', device)
        step = step + 1
    return mini_sft_loss #, mini_critic_loss

    
        

        

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

def ppo_gradient(llm, llm_config, buffer, buffer_size, device, critic_alpha=0.01):
    llm.train() 
    step = 0
    batch_size = 1
    max_seq_len = 4096
    micro_training_steps = buffer_size / batch_size

    vocab_size = llm_config.vocab_size
    
    mseLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
    mini_policy_loss = 0
    mini_critic_loss = 0
    # rl training steps;
    while step < micro_training_steps:
        mini_data = buffer.pop(batch_size)

        input_tokens = [d.tokens for d in mini_data]
        old_logprobs = [d.probs for d in mini_data]
        #advantages = [d.advantages for d in mini_data]
        advantages = [d.normalized_advantages for d in mini_data]
        returns = [d.returns for d in mini_data]
      
        # do tokens padding & alignment with batchsize  > 1   
        input_tokens = torch.tensor(input_tokens).to(torch.long).to(device)    
        old_logprobs = torch.tensor(old_logprobs).to(torch.bfloat16).to(device).detach()
        advantages = torch.tensor(advantages).to(torch.bfloat16).to(device).detach()
        returns = torch.tensor(returns).to(torch.bfloat16).to(device).detach()
        
        _batch_size, _seq_len = input_tokens.shape
        # generation token index. 
        _response_idx = mini_data[0].masks.index(1)

        # re-evaluate the policy.     
        # return: next_token_loss, logits, critics, next_decoder_cache 
        _, logits, critics, _ = llm(input_tokens)
    
        logprobs = -F.cross_entropy(
            input = logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
            target = input_tokens.reshape(-1)[1:], 
            reduction = "none",
            #ignore_index = pad_id,
        ).reshape(1, -1)

        # critics align with the ground truth. 
        critics = critics.reshape(_batch_size, _seq_len)
        critics = critics[:, _response_idx-1:-1] 
        logprobs = logprobs[:, _response_idx-1:]

        # we shall do advantage normalization. 
        # let's try to stablizae the training. 
        ratios = torch.exp(logprobs - old_logprobs.detach() + 1e-10)
        #if debug:
        #    print('ratio:', ratios)
            
        eps_clip = 0.5
        surr1 = ratios * advantages       
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        _policy_loss = -torch.min(surr1, surr2).mean() 
        _critic_loss = mseLoss(critics, returns).mean() 

        _total_loss = (_policy_loss + critic_alpha * _critic_loss) / micro_training_steps 

        
        _total_loss.backward()
            
        # final loss of clipped objective PPO objective. 
        # take gradient step
        mini_critic_loss = mini_critic_loss + _critic_loss.detach() / micro_training_steps 
        mini_policy_loss = mini_policy_loss + _policy_loss.detach() / micro_training_steps

        step = step + 1

        #print(' _policy_loss:', _policy_loss, ' , _critic_loss:', _critic_loss, ' , device:', device)
        
    return mini_policy_loss, mini_critic_loss


def gradient_v2_pass(llm, input_tokens, advantages, _response_idx, vocab_size, returns, critic_alpha, mseLoss, loss_scalar):
    
    _batch_size, _seq_len = input_tokens.shape
    
    logprobs, _, critics, _ = llm(input_tokens[:, :-1], labels=input_tokens[:, 1:], fuse_loss = True)
    
    #logprobs = F.cross_entropy(
    #    input = logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
    #    target = input_tokens.reshape(-1)[1:], 
    #    reduction = "none",
    #        #ignore_index = pad_id,
    #).reshape(1, -1)
        
    # critics align with the ground truth. 
    critics = critics.reshape(_batch_size, _seq_len-1)
    critics = critics[:, _response_idx-1:] 
    logprobs = logprobs[:, _response_idx-1:]

    _policy_loss = (advantages * logprobs).mean() #  -torch.min(surr1, surr2).sum() 
    _critic_loss = mseLoss(critics, returns).mean() 

    _total_loss = (_policy_loss + critic_alpha * _critic_loss) * loss_scalar 
    _total_loss.backward()

    return _policy_loss.detach(), _critic_loss.detach()

# 1. dpo style objective function. 
def ppo_gradient_v2(llm, llm_config, buffer, buffer_size, device, critic_alpha=0.01):
    llm.train() 
    step = 0
    batch_size = 1
    max_seq_len = 4096
    micro_training_steps = buffer_size / batch_size

    vocab_size = llm_config.vocab_size
    
    mseLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
    mini_policy_loss = 0
    mini_critic_loss = 0

    loss_scalar = 1.0 / micro_training_steps
    # rl training steps;
    while step < micro_training_steps:
        mini_data = buffer.pop(batch_size)

        input_tokens = [d.tokens for d in mini_data]
        #old_logprobs = [d.probs for d in mini_data]
        #advantages = [d.advantages for d in mini_data]
        advantages = [d.normalized_advantages for d in mini_data]
        returns = [d.returns for d in mini_data]
      
        # do tokens padding & alignment with batchsize  > 1   
        input_tokens = torch.tensor(input_tokens).to(torch.long).to(device)    
        #old_logprobs = torch.tensor(old_logprobs).to(torch.bfloat16).to(device).detach()
        advantages = torch.tensor(advantages).to(torch.bfloat16).to(device).detach()
        returns = torch.tensor(returns).to(torch.bfloat16).to(device).detach()
        
        _response_idx = mini_data[0].masks.index(1)

        step = step + 1
        
        if step >= micro_training_steps:
            _policy_loss, _critic_loss = gradient_v2_pass(llm, input_tokens, advantages, _response_idx, vocab_size, returns, critic_alpha, mseLoss, loss_scalar)
        else:
            with llm.no_sync():
                _policy_loss, _critic_loss = gradient_v2_pass(llm, input_tokens, advantages, _response_idx, vocab_size, returns, critic_alpha, mseLoss, loss_scalar)
            
        # final loss of clipped objective PPO objective. 
        # take gradient step
        mini_critic_loss = mini_critic_loss + _critic_loss * loss_scalar # / micro_training_steps 
        mini_policy_loss = mini_policy_loss + _policy_loss * loss_scalar # / micro_training_steps
            
        #print(' _policy_loss:', _policy_loss, ' , _critic_loss:', _critic_loss, ' , device:', device)
    return mini_policy_loss, mini_critic_loss
    

def ppo_train(llm, llm_config, optimizer, scheduler, buffer, buffer_size, device, critic_alpha=0.01, mode=1):
    optimizer.zero_grad()
    if mode == 1:
        mini_policy_loss, mini_critic_loss = ppo_gradient(llm, llm_config, buffer, buffer_size, device, critic_alpha)
    elif mode == 2:
        mini_policy_loss, mini_critic_loss = ppo_gradient_v2(llm, llm_config, buffer, buffer_size, device, critic_alpha)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    return mini_policy_loss, mini_critic_loss
    # 
    #pad_id = llm_config.pad_token_id
    
        

        

#ppo algorithm

import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
from queue import Queue
import threading
import time
import random
import argparse

from typing import List, Optional, Tuple, Union

import torch.nn as nn

#from vllm import LLM, SamplingParams

from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError
from functools import partial
#from contextlib import redirect_stdout
import sys


#from datasets import load_dataset

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import logging

#from peft import LoraConfig
#from trl import SFTTrainer
#from transformers import TrainingArguments, BitsAndBytesConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
#from transformers import AdamW
#import numpy as np 
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW


import os
import io
import pickle
import traceback
import copy
import datetime
from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError
from functools import partial
from contextlib import redirect_stdout
import sys

from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3MLP, Phi3PreTrainedModel, Phi3Model, Phi3DecoderLayer
from transformers.models.phi3.configuration_phi3 import Phi3Config

from transformers import AutoConfig

import torch.nn as nn
import multiprocessing

import signal
from transformers.activations import ACT2FN

from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

from transformers.cache_utils import Cache, DynamicCache

from phimodel import _Phi3ForCausalLM

import torch.nn.functional as F

import datetime


import signal
import psutil  # To check process status before killing

import concurrent.futures
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk

import re

import random

from collections import deque

from math_util import compare_math_answers, process_math_prompt, process_math_answer

import numpy as np

# ReplayBuffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # every sample is very different. 
        self.epsilon = 1e-8
        self.alpha = 0.01       
        self.lock = threading.Lock()

    # experience : <prompt, response, reward, tokens, masks, seq_rewards> 
    def push(self, experience):
        """ Add new experience to the buffer """
        with self.lock:
            self.buffer.append(experience) 
            
    def clear(self):
        self.buffer.clear()

    def pop(self, batch_size):
        """
        Pop the oldest batch of experiences from the buffer (FIFO order).
        Args:
            batch_size (int): Number of experiences to pop.
        Returns:
            List of popped experiences.
        """
        with self.lock:
            batch_size = min(batch_size, len(self.buffer))  # Ensure we don't pop more than available
            data = [self.buffer.popleft() for _ in range(batch_size)]  # Pop oldest elements
            return data
        
    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def get_rewards(self):
        with self.lock:
            rewards = []
            for d in self.buffer:
                prompt, response, reward, tokens, masks, seq_rewards = d
                rewards.append(reward)
            return rewards
            
    def mean_reward(self):
        rewards = self.get_rewards()
        return np.mean(rewards)

    def avg_responselen(self):
        with self.lock:
            response_len = []
            for d in self.buffer:
                prompt, response, reward, tokens, masks, seq_rewards = d
                response_len.append(len(response))
            return np.mean(response_len)
        
    def z_score_normalization(self):
        """Standardize rewards using mean and standard deviation."""
        rewards = self.get_rewards()
        
        mean = np.mean(rewards)
        std = np.std(rewards) + self.epsilon
        return (rewards - mean) / std


def train(args, llm, llm_config, optimizer, scheduler, buffer, buffer_size, device):
    llm.train()
    
    #critic_loss = 0.0
    #policy_loss = 0.0

    #mini_c_loss = 0.0
    #mini_p_loss = 0.0
    # update_step = 0

    # accumulate gradient for the whole batchsize.     
    step = 0
    batch_size = 1
    max_seq_len = 4096
    micro_training_steps = buffer_size / batch_size

    # 
    pad_id = llm_config.pad_token_id
    vocab_size = llm_config.vocab_size
    
    optimizer.zero_grad()
    mseLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')

    # rl training steps;
    while step < micro_training_steps:
        mini_data = buffer.pop(batch_size)

        input_tokens = []
        # data clean up. 
        for d in mini_data:
            # d : <prompt, response, reward, tokens, masks, seq_rewards> 
            prompt, response, reward, tokens, masks, seq_rewards = d
            input_tokens.append(tokens)
        
        #_tokens = [d[0] for d in data]
        #_masks = [d[1] for d in data]
        #_probs = [d[2] for d in data]
        #_rewards = [d[3] for d in data]
        #_crits = [d[4] for d in data] 
        # do tokens padding & alignment with batchsize  > 1
         
        input_tokens = torch.tensor(input_tokens).to(torch.long).to(device)    
        _batch_size, _seq_len = input_tokens.shape
        
        # re-evaluate the policy.     
        # return: next_token_loss, logits, critics, next_decoder_cache 
        _, logits, critics, _ = model(input_tokens)
    
        logprobs = -F.cross_entropy(
            input = logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
            target = input_tokens.reshape(-1)[1:], 
            reduction = "none",
            ignore_index = pad_id,
        ).reshape(1, -1)

        # critics align with the ground truth. 
        critics = critics.reshape(_batch_size, _seq_len)
        critics = critics[:, _idx-1:-1] 
        
        old_logprobs = torch.tensor(_probs).to(model.device)
        _idx = _masks[0].index(1)
        ratios = torch.exp(logprobs[:, _idx-1:] - old_logprobs.detach() + 1e-10)
            
            
            gamma = 0.95
            rewards = []
            discounted_reward = 0
            
            #baselines = []
            discounted_baseline = ema_reward # avg_reward
            
            for reward in reversed(_rewards[0]): 
                discounted_baseline = gamma * discounted_baseline
                #baselines.insert(0, discounted_baseline) 

                discounted_reward = reward + (gamma * discounted_reward)
                rewards.insert(0, discounted_reward - discounted_baseline)
                
            # Normalizing the rewards
            rewards = torch.tensor([rewards], dtype=torch.bfloat16).to(model.device)
        
            # calculate advantages
            advantages = rewards.detach() # - old_state_values.detach()

            eps_clip = 0.2
            # Finding Surrogate Loss  
            surr1 = ratios * advantages # (optimize logprobs)
            
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages

            _p_loss = -torch.min(surr1, surr2).mean()
            _c_loss = mseLoss(critics, rewards).mean()
            
            # final loss of clipped objective PPO objective. 
            loss = (_p_loss + 0.02 * _c_loss) / gradient_accumulation_steps  #- 0.01 * dist_entropy
            
            # take gradient step
            mini_c_loss = mini_c_loss + _c_loss.detach()
            mini_p_loss = mini_p_loss + _p_loss.detach()
            
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                update_step = update_step + 1

                mini_c_loss = mini_c_loss / gradient_accumulation_steps
                mini_p_loss = mini_p_loss / gradient_accumulation_steps

                critic_loss = critic_loss * (update_step - 1) / update_step + mini_c_loss / update_step
                policy_loss = policy_loss * (update_step - 1) / update_step + mini_p_loss / update_step

                if rank == 0:
                    print('mini_c_loss: ', mini_c_loss, 'critic_loss: ', critic_loss)
                    print('mini_p_loss: ', mini_p_loss, 'policy_loss: ', policy_loss)
                    print('avg reward: ', avg_reward, 'ema reward: ', ema_reward)
                    
                mini_c_loss = 0.0
                mini_p_loss = 0.0
    
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Update the learning rate

                if update_step % 8 == 0:
                    #print('enter update phase', rank)
                    #dist.barrier(learndp)
                    #print('enter update phase, barrier 1', rank)
                    
                    # notify the producer to boardcast the model weight to 
                    #if rank == 0:
                    #print('enter model update message phase', rank)
                    if rank == 0:
                        print('learner samples: ', update_step * 8 * gradient_accumulation_steps)
                        notify_model_update()
                    #print('waiting for model update phase 1', rank)                    
                    dist.barrier() #mdg)
                    #print('waiting for model update phase 2', rank)                    
                    allmodel_sync(model) #, device_ids=[local_rank], mdg=mdg)
                    #print('waiting for model update phase 3', rank)                    
                    dist.barrier()
                    print('*************** learner model update ******************************', rank)
                    #rpc.rpc_sync(f"worker-{buffer_rank}", notify_model_update, args=_info, timeout=0)
                    #print('wait on the learndp barrier 2', rank)
                    #dist.barrier(learndp)
                    #print('leave update phase, barrier 1', rank)
            step = step + 1


def main(args):
    # on-policy ppo experiments with phi3.5 model on math dataset. 
    local_rank = int(os.environ['LOCAL_RANK']) 
    print('local rank', local_rank) 
    rank = int(os.environ['RANK']) 
    print('rank', rank) 
    world_size = int(os.environ['WORLD_SIZE']) 
    print('WORLD_SIZE', world_size)  
    
    gpus_per_node = 8 
    node_idx = rank // gpus_per_node 
    
    torch.cuda.set_device(local_rank) 
    device = torch.device(f"cuda:{local_rank}") 
    # init distributed process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=5))    
    #rpc.init_rpc(f"worker-{rank}", rank=rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions()) # consider 2 nodes, 16 gpus in this example.
    local_model_path = args.pretrained_model # "/mnt/blob-aimsllmeus2-data/phimodels2/" 
    print('model_path', local_model_path) 
    #model_name = "microsoft/Phi-3.5-mini-instruct" 
    # define: local model path 
    # Load model configuration from local path 
    llm_config = AutoConfig.from_pretrained(local_model_path, local_files_only=True) 
    #llm_config = AutoConfig.from_pretrained(local_model_path) 
    #llm_config = AutoConfig.from_pretrained(model_name) 
    # load model. 
    #llm = AutoModelForCausalLM.from_pretrained( 
    #    model_name,  
    #    device_map='cpu', 
    #    torch_dtype=torch.bfloat16,  
    #    trust_remote_code=True,  
    #) 
    # Load model from local path using the configuration. 
    llm = AutoModelForCausalLM.from_pretrained( 
        local_model_path, 
        device_map="cpu", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        local_files_only=True 
    ) 
    
    #.to(device) 
    llm_model = _Phi3ForCausalLM(llm_config) 
    
    missing_keys, unexpected_keys = llm_model.load_state_dict(llm.state_dict(), strict=False) 
    llm_model = llm_model.to(torch.bfloat16).to(device) 
    llm_model.model.gradient_checkpointing = True 
    
    llm = llm_model 

    # setup model distribution.
    vocab_size = llm_config.vocab_size 
    eos_token_id = llm_config.eos_token_id #": 32000,
    
    llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[rank]) 
    print('distributed language model creation.') 

    optimizer = torch.optim.AdamW(llm.parameters(), lr=1.0e-6) 
    #num_epochs = 3 
    num_training_steps = 100000 # num_epochs * len(train_dataloader)
    
    scheduler = get_linear_schedule_with_warmup( 
        optimizer, num_warmup_steps=1500, num_training_steps=num_training_steps 
    ) 
    print('model optimization initialization...') 

    # load tokenizer. 
    #tokenizer = AutoTokenizer.from_pretrained(model_name) 
    # Load tokenizer from local path 
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True) 
    tokenizer.model_max_length = 4096 
    tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) 
    tokenizer.padding_side = 'right' 
    
    print('initial llm model ....') 
    ############################################################

    # load dataset....
    datafile = 'math_level3to5_data_processed_with_qwen_prompt.json' 
    dataset = load_dataset('json', data_files=datafile) 
    print(f"loaded {dataset} with data_files={datafile}") 

    # 
    sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank, shuffle=True) 
    dataloader = DataLoader(dataset['train'], batch_size=1, sampler=sampler) 

    ### initialize replaybuffer.
    llm.eval()
    
    buffer_size = 64
    buffer = ReplayBuffer(buffer_size)
    ### 
    
    for epoch in range(0, 100):
        sampler.set_epoch(epoch)  # Set epoch for shuffling
        acc_reward = 0
        acc_num = 0
        
        for batch_idx, d in enumerate(dataloader):
            qwen_prompt = d['input']
            vanilla_prompts = d['question']    
            answers_1 = d['answer']
            answers_2 = d['gt_answer']
            answers_3 = d['ground_truth_answer']
            answers_4 = d['target']
            answers = answers_1 + answers_2 + answers_3 + answers_4
            
            # features: ['input', 'answer', 'gt_answer', 'subject', 'level', 'question', 'ground_truth_answer', 'target'],

            prompt = process_math_prompt(vanilla_prompts[0])
      
            x1 = tokenizer([prompt], add_special_tokens=False, max_length=1024, truncation=True)

            input_ids = x1['input_ids']
            
            outputs, probs, crits = llm.generate(input_ids, max_gen_len = 3000)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            #processed_response, extract_answer, reward
            response, extracted_answer, reward = process_math_answer(response, answers)

            y1 = tokenizer([response], add_special_tokens=False, max_length=3000, truncation=True)
            
            outputs = y1['input_ids]
                
            acc_reward = acc_reward + reward
            acc_num = acc_num + 1
            
            if local_rank == 0:
                print('batch idx', batch_idx)
                print('\n\n\nquestion: ************\n')
                print(prompt)
                print('\n\n\nresponse: *************\n')
                print(response)
                print('\n\n\nextracted answer: ************\n')
                print(extracted_answer)
                print('\n\n\nground truth: *************\n')
                print(answers)
                print('\n\n\nreward: **********\n')
                print(reward)
                print('\n\n')
            
            # prompt_tokens: List[List[int]],
            all_tokens = []
            all_masks = []
            all_rewards = []
            for input_id, output_id in zip(input_ids, outputs):
                _ids = input_id + output_id 
                _masks = [0] * len(input_id) + [1] * len(output_id) 
                _rewards = [0] * (len(_ids)-1) + [reward] 
                
                all_tokens.append(_ids)
                all_masks.append(_masks)
                all_rewards.append(_rewards)

            #<prompt, response, reward, tokens, masks, seq_rewards>
            experience = (prompt, response, reward, probs[0], crits[0], all_tokens[0], all_masks[0], all_rewards[0])
            buffer.push(experience)
            
            
            if len(buffer) >= buffer_size:
                avg_reward = buffer.mean_reward()
                avg_len = buffer.avg_responselen()
                
                print('progress: ', batch_idx, ', average_reward: ', avg_reward, ', avg_responselen: ', avg_len , ', rank: ', rank)

                ## start the model training; 
                
                
                
                buffer.clear()
                
            
        print('final average reward: ', acc_reward / acc_num, '\nacc_num: ',acc_num)
    # one node inference; one node training; as an example; 
    # suppose we use 4 gpus for vllm and 4 gpus 
    #if rank in [0,1,2,3,4,5,6,7]:
    #    learn(learndp) #, mdg)
    #else:
    #    play(learndp) #, mdg)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="none", help="path to pretrained ckpt.")
    args = parser.parse_args()
    
    main(args)

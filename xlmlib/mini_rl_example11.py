import os
import io
import pickle
import traceback
import copy
import datetime
import sys
import threading
import time
import random
import argparse
import signal
import psutil  # To check process status before killing
import re
import multiprocessing
import logging
import json
from types import SimpleNamespace

import numpy as np
from queue import Queue
from typing import List, Optional, Tuple, Union, Any, Dict, Optional
import concurrent.futures
from concurrent.futures import TimeoutError
from functools import partial
from contextlib import redirect_stdout
from dataclasses import dataclass
from collections import deque


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import AdamW

from accelerate import Accelerator

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from safetensors.torch import load_file

from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from transformers import get_linear_schedule_with_warmup
from transformers.activations import ACT2FN


from samba import _SambaForCausalLM
from replaybuffer import ReplayBuffer, Sample
from ppo import ppo_gradient
from sft import sft_gradient
from math_util import compare_math_answers, process_math_prompt, process_math_answer


def initmodel_sync(model:_SambaForCausalLM):
    with torch.no_grad():
        torch.distributed.broadcast(model.critic_head.weight, 0, async_op=False)
        torch.distributed.broadcast(model.critic_head.bias, 0, async_op=False)

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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)    

    # Step 3: Load and merge multiple safetensor state_dicts
    dist.barrier()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.weight_path is None:
        llm_model, llm_config, tokenizer = _SambaForCausalLM.load_hfckpt(args.pretrained_model)
    else:
        llm_model, llm_config, tokenizer = _SambaForCausalLM.load_customckpt(args.pretrained_model, args.weight_path)
        
    # Load tokenizer from local path 
    #tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True) 
    #tokenizer.model_max_length = 4096 
    tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) 
    tokenizer.padding_side = 'right'     
    new_special_tokens = ['<think>', '</think>', '<answer>', '</answer>']
    tokenizer.add_tokens(new_special_tokens)

    vocab_size = llm_config.vocab_size 
    eos_token_id = llm_config.eos_token_id #": 32000,
  
    llm_model = llm_model.to(torch.bfloat16).to(device) 
    llm_model.model.gradient_checkpointing = True 
    dist.barrier()
    initmodel_sync(llm_model)    
    llm = llm_model 
    print('initial llm model ....') 
    # setup model distribution.
    llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[local_rank]) 
    print('distributed language model creation.') 

    # load dataset....
    datafile = 'math_level3to5_data_processed_with_qwen_prompt.json' 
    dataset = load_dataset('json', data_files=datafile) 
    print(f"loaded {dataset} with data_files={datafile}") 
    sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank, shuffle=True) 
    dataloader = DataLoader(dataset['train'], batch_size=1, sampler=sampler) 

    # setup optimization.
    optimizer = torch.optim.AdamW(llm.parameters(), lr=args.lr) # 1.0e-6) 
    num_training_steps = dataset['train'].num_rows * args.epoch * args.n_rollout * 1.0 / (args.replay_size * world_size) # num_epochs * len(train_dataloader)    
    warmup_steps = args.warmup_step * num_training_steps
    scheduler = get_linear_schedule_with_warmup( 
        optimizer, num_warmup_steps = int(warmup_steps), num_training_steps = int(num_training_steps) 
    )     
    
    if local_rank == 0:
        print('num_training_steps', int(num_training_steps), ' warmup_steps', int(warmup_steps), ' learning rate', args.lr)
    print('model optimization initialization...') 

    ### initialize replaybuffer.
    llm.eval()
    buffer = ReplayBuffer(args.replay_size)

    ## load sft dataset.
    if args.sft_data is not None:
        sft_dataset = load_dataset('json', data_files=args.sft_data) 
        print(f"loaded {sft_dataset} with data_files={args.sft_data}")
        print(sft_dataset['train'])
        sft_sampler = torch.utils.data.distributed.DistributedSampler(sft_dataset['train'], num_replicas=world_size, rank=rank, shuffle=True) 
        sft_dataloader = DataLoader(sft_dataset['train'], batch_size=1, sampler=sft_sampler) 
        sft_iter = iter(sft_dataloader)

        sft_buffer = ReplayBuffer(args.sft_replay_size)
        #dataset_b = DatasetB()
        #dataloader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=True)
        #iter_b = iter(dataloader_b)
    
    if args.profile:
        elapsed_time_generation = 0
        elapsed_time_train = 0
        elapsed_time_reward = 0
    
    #rl_update = 0    
    for epoch in range(0, args.epoch):
        sampler.set_epoch(epoch)  # Set epoch for shuffling
        acc_reward = 0
        acc_num = 0
        topk_reward = 0
        topk_num = 0
        for batch_idx, d in enumerate(dataloader):
            qwen_prompt = d['input']
            vanilla_prompts = d['question']    
            answers_1 = d['answer']
            answers_2 = d['gt_answer']
            answers_3 = d['ground_truth_answer']
            answers_4 = d['target']
            answers = answers_1 + answers_2 + answers_3 + answers_4
            # features: ['input', 'answer', 'gt_answer', 'subject', 'level', 'question', 'ground_truth_answer', 'target']
            prompt = process_math_prompt(vanilla_prompts[0])
            
            x1 = tokenizer([prompt] * args.n_rollout, add_special_tokens=False, max_length=1024, truncation=True)
            input_ids = x1['input_ids']

            topk_hit = 0    
            if args.profile:
                start_time = time.perf_counter()    
            outputs, probs, crits = llm.module.generate(input_ids, max_gen_len = 4096)
            if args.profile:
                end_time = time.perf_counter()
                elapsed_time_generation = elapsed_time_generation + end_time - start_time
                
            if batch_idx == 0 and local_rank == 0: # and rollout == 0:
                print('probs.shape', len(probs[0]))
                print('crits.shape', len(crits[0]))
                print('outputs.shape', len(outputs[0]))

            
            response = tokenizer.decode(outputs[0])
            response_mapping = tokenizer(response, return_offsets_mapping=True)

                if args.profile:
                    start_time = time.perf_counter()
                #processed_response, extract_answer, reward
                mid_response, extracted_answer, reward = process_math_answer(response, answers, tokenizer)
                
                if args.profile:
                    end_time = time.perf_counter()
                    elapsed_time_reward = elapsed_time_reward + end_time - start_time
                    
                def getindex(char_pos, offset_mapping):
                    for token_idx, (start, end) in enumerate(offset_mapping):
                        if start <= char_pos < end:
                             return token_idx
                    return None
                response_idx = getindex(len(mid_response), response_mapping.offset_mapping)
                # 5 token space. 
                if response_idx is not None and len(outputs[0]) > response_idx + 5:
                    outputs[0] = outputs[0][ : response_idx]
                    probs[0] = probs[0][ : response_idx]
                    crits[0] = crits[0][ : response_idx]
                response = tokenizer.decode(outputs[0])

                if reward >= 0.5:
                    topk_hit = 1
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
                    print('\n\ncrits: *******\n')
                    print(np.mean(crits[0]), crits[0][-1])
                    print('\n\nprobs: *******\n')
                    print(np.mean(probs[0]))
                    print('\n\n')
                    
                # prompt_tokens: List[List[int]],
                all_tokens = []
                all_masks = []
                output_rewards = []
                for input_id, output_id in zip(input_ids, outputs):
                    _ids = input_id + output_id 
                    _masks = [0] * len(input_id) + [1] * len(output_id) 
                    _rewards = [0] * (len(output_id)-1) + [reward] 
                    
                    all_tokens.append(_ids)
                    all_masks.append(_masks)
                    output_rewards.append(_rewards)
    
                #<prompt, response, reward, probs, crits, tokens, masks, seq_rewards>    
                experience = Sample(prompt = prompt, response = response, reward = reward, probs = probs[0], crits = crits[0], seq_rewards = output_rewards[0],
                                    tokens = all_tokens[0], masks = all_masks[0])
                buffer.push(experience)
            
            topk_reward = topk_reward + topk_hit
            topk_num = topk_num + 1
            if len(buffer) >= args.replay_size:    
                avg_reward = buffer.mean_reward()
                avg_response_len = buffer.avg_responselen()
                print('progress: ', batch_idx, ', avg_reward: ', avg_reward, ', avg_response_len: ', avg_response_len , ', rank: ', rank)
                print('topk_reward: ', topk_reward * 1.0 / topk_num, ', topk_num: ', topk_num,   ', rank: ', rank)
                print('acc_reward: ',  acc_reward / acc_num, ', acc_num: ', acc_num, ', rank: ', rank)
                
                dist.barrier()
                if args.advantage == 'distgae':
                    buffer.calculate_advantage()
                    buffer.distributed_advantage_norm(device, dist)
                elif args.advantage == 'group':
                    buffer.calculate_group_advantage(group = args.n_rollout)

                # insert new SFT data into replay buffer;
                for _idx in range(0, args.sft_replay_size):
                    try: 
                        sft_data = next(sft_iter)
                    except:
                        sft_iter = iter(sft_dataloader)
                        sft_data = next(sft_iter)

                    prompt = sft_data['messages'][0]['content'][0]
                    response = sft_data['messages'][1]['content'][0]

                    prompt_tokens =  tokenizer([prompt], add_special_tokens=False, max_length=1024, truncation=True)
                    response_tokens = tokenizer(['\n\n' + response], add_special_tokens=False, max_length=7168, truncation=True)

                    prompt_tokens = prompt_tokens['input_ids'][0] 
                    response_tokens = response_tokens['input_ids'][0] + [tokenizer.eos_token_id]

                    all_tokens = prompt_tokens + response_tokens 
                    masks = [0] * len(prompt_tokens) + [1] * len(response_tokens)
                    experience = Sample(prompt = prompt, response = response, reward = 1.0, tokens = all_tokens, masks = masks)
                    sft_buffer.push(experience)

                
                if args.profile:
                    start_time = time.perf_counter()
                    
                #optimizer, scheduler,
                optimizer.zero_grad()
                policy_loss_log, critic_loss_log = ppo_gradient(llm, llm_config, buffer, args.replay_size, device, critic_alpha = args.critic_alpha)

                sft_loss_log = 0.0
                if args.sft_replay_size > 0:
                    sft_loss_log = sft_gradient(llm, llm_config, sft_buffer, args.sft_replay_size, device, weight = args.sft_weight)                    
                    
                optimizer.step()
                scheduler.step()

                if args.profile:
                    end_time = time.perf_counter()
                    elapsed_time_train = elapsed_time_train + end_time - start_time
                    
                if local_rank == 0:
                    print('policy_loss_log: ', policy_loss_log, ', critic_loss_log: ', critic_loss_log, ', sft_loss_log: ', sft_loss_log, ', lr:', scheduler.get_last_lr() )
                    if args.profile:
                        print('elapsed_time_generation:', elapsed_time_generation)
                        print('elapsed_time_train:', elapsed_time_train)
                        print('elapsed_time_reward:', elapsed_time_reward)
                        
                ## start the model training; 
                
                buffer.clear()    
                if args.sft_replay_size > 0:
                    sft_buffer.clear()
                llm.eval()

                # Save only on rank 0
                if rank == 0 and scheduler._step_count % args.save_per_steps == 0:
                    checkpoint = {
                        "step": scheduler._step_count,
                        "model_state_dict": llm.module.state_dict(),  # Remove DDP wrapper
                    }
                    save_path = f"{args.save_ckpt}/ckpt_{scheduler._step_count}.pth"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(checkpoint, save_path)
                    print(f"Checkpoint saved at: {save_path}")
        
        print('final average reward: ', acc_reward / acc_num, ', acc_num: ', acc_num)
        print('final topk reward: ', topk_reward * 1.0 / topk_num, ', topk_num: ', topk_num)
        #if rank == 0 and args.save_ckpt is not None:
        # Synchronize all processes to ensure rank 0 saves first
        dist.barrier()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="none", help="path to pretrained ckpt.")
    parser.add_argument("--weight_path", default=None, type=str, help="customized model weight path.")
    parser.add_argument("--save_per_steps", type=int, default=40, help="save ckpt per steps.")
    
    parser.add_argument("--save_ckpt", type=str, default=None, help="path to save ckpt.")
    parser.add_argument("--replay_size", type=int, default=64, help="size of replay buffer.")
    parser.add_argument("--warmup_step", type=float, default=0.1, help="warmup steps.")
    parser.add_argument("--lr", type=float, default=1e-6, help="peak learning rate.")
    parser.add_argument("--epoch", type=int, default=30, help="number of epoches.")
    parser.add_argument("--n_rollout", type=int, default=1, help="number of rollout per sample.")
    parser.add_argument("--advantage", type=str, default="distgae", choices=["distgae", "group"], help="Choose the advantage function.")
    parser.add_argument("--critic_alpha", type=float, default=0.01, help="alpha for critic loss.")
    parser.add_argument("--sft_data", type=str, default=None, help="path to sft data.")
    parser.add_argument("--sft_replay_size", type=int, default=0, help="SFT update batch size.")
    parser.add_argument("--sft_weight", type=float, default=0.1, help="token weight of sft dataset.")
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    
    assert args.replay_size % args.n_rollout == 0, 'pls make sure replay_size mod n_rollout == 0'
    
    main(args)

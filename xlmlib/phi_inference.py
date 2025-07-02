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

import numpy as np
from queue import Queue
from typing import List, Optional, Tuple, Union, Any, Dict, Optional
import concurrent.futures
from concurrent.futures import TimeoutError
from functools import partial

from contextlib import redirect_stdout
from contextlib import contextmanager

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
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3MLP, Phi3PreTrainedModel, Phi3Model, Phi3DecoderLayer
from transformers.models.phi3.configuration_phi3 import Phi3Config

from phimodel import _Phi3ForCausalLM
from phi4 import _Phi4ForCausalLM

from replaybuffer import ReplayBuffer, Sample
from ppo import ppo_train 
from math_util import compare_math_answers, process_math_prompt, process_math_answer

from llm_engine import LLMEngine

def initmodel_sync(model:_Phi3ForCausalLM):
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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(days=365 * 10))    

    # Step 3: Load and merge multiple safetensor state_dicts
    dist.barrier()
    local_model_path = args.pretrained_model 
    print('model_path', local_model_path) 
    
    if args.model_type == 'phi3':
        llm_config = AutoConfig.from_pretrained(local_model_path, local_files_only=True) 
        vocab_size = llm_config.vocab_size 
        eos_token_id = llm_config.eos_token_id #": 32000,
        safetensor_files = [
            f"{local_model_path}/model-00001-of-00002.safetensors",
            f"{local_model_path}/model-00002-of-00002.safetensors"
        ]
        model_state_dict = {}
        for file in safetensor_files:
            part_state_dict = load_file(file, device="cpu")  # Load each part
            model_state_dict.update(part_state_dict)  # Merge into one dictionary
        print('load model weight ... ')
        llm_model = _Phi3ForCausalLM(llm_config) 
        # Step 4: Apply the merged state_dict to the model
        missing_keys, unexpected_keys = llm_model.load_state_dict(model_state_dict, strict=False) 
        print('missing_keys: ', missing_keys)
        print('unexpected_keys: ', unexpected_keys)    
        
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True) 
        tokenizer.model_max_length = 4096 
    
    elif args.model_type == 'phi4':

        llm_model, llm_config, tokenizer = _Phi4ForCausalLM.load_hfckpt(args.pretrained_model)
        tokenizer.model_max_length = 32768 
    
    #tokenizer.pad_token = tokenizer.unk_token # use unk rather than eos token to prevent endless generation
    #tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) 
    #tokenizer.padding_side = 'right' 

    llm_model = llm_model.to(torch.bfloat16).to(device) 
    llm_model.model.gradient_checkpointing = True 
    dist.barrier()
    initmodel_sync(llm_model)    

    llm = llm_model 
    print('initial llm model ....') 
    # setup model distribution.
    #llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[local_rank]) 
    print('distributed language model creation.') 
    
    #total_memory, used_memory, free_memory = get_gpu_memory()
    #print('total_memory', total_memory, 'used_memory', used_memory, 'free_memory', free_memory)

    engine = LLMEngine(llm_model, llm_config, device)
    #def __init__(self, model, llm_config, device):
    
    #allocate_kv_cache(llm, llm_config, device, gpu_memory_utilization = 0.90)

    # Load tokenizer from local path 
    # load dataset....
    datafile = 'math_level3to5_data_processed_with_qwen_prompt.json' 
    dataset = load_dataset('json', data_files=datafile) 
    print(f"loaded {dataset} with data_files={datafile}") 
    sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank, shuffle=True) 
    dataloader = DataLoader(dataset['train'], batch_size=64, sampler=sampler) 

    # setup optimization.
    #optimizer = torch.optim.AdamW(llm.parameters(), lr=args.lr) # 1.0e-6) 
    #num_training_steps = dataset['train'].num_rows * args.epoch * args.n_rollout * 1.0 / (args.replay_size * world_size) # num_epochs * len(train_dataloader)    
    #warmup_steps = args.warmup_step * num_training_steps
    #scheduler = get_linear_schedule_with_warmup( 
    #    optimizer, num_warmup_steps = int(warmup_steps), num_training_steps = int(num_training_steps) 
    #)     
    
    #if local_rank == 0:
    #    print('num_training_steps', int(num_training_steps), ' warmup_steps', int(warmup_steps), ' learning rate', args.lr)
    #print('model optimization initialization...') 

    ### initialize replaybuffer.
    llm.eval()
    buffer_size = args.replay_size
    buffer = ReplayBuffer(buffer_size)
    ### 

    time_start = time.perf_counter()  

    
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

            batch_prompts = []
            for inner_prompt in vanilla_prompts:
                prompt = process_math_prompt(inner_prompt, prompt_type = 'v17')
                batch_prompts.append(prompt)

            x1 = tokenizer(batch_prompts, add_special_tokens=False, max_length=1024, truncation=True)
            input_ids = x1['input_ids']

            topk_hit = 0
            for rollout in range(0, args.n_rollout):
                outputs = engine.generate(input_ids)

                #outputs, probs, crits = llm.generate(input_ids, max_gen_len = 32768, temperature = 0.7, top_p = 0.95)
                #if batch_idx == 0 and local_rank == 0 and rollout == 0:
                #    print('probs.shape', len(probs[0]))
                #    print('crits.shape', len(crits[0]))
                #    print('outputs.shape', len(outputs[0]))
                
                batch_responses = []
                for _i in range(0, len(outputs)):
                    response = tokenizer.decode(outputs[_i])
                    batch_responses.append(response)

                    print('batch idx', _i, 'device', rank)
                    print('\n\n\nquestion: ************\n')
                    print(vanilla_prompts[_i])
                    print('\n\n\nresponse: *************\n')
                    print(batch_responses[_i])
                    print('\n\n\nground truth: *************\n')
                    print(answers_1[_i])

                #response_mapping = tokenizer(response, return_offsets_mapping=True)
                #processed_response, extract_answer, reward
                #mid_response, extracted_answer, reward = process_math_answer(response, answers, tokenizer)
                #def getindex(char_pos, offset_mapping):
                #    for token_idx, (start, end) in enumerate(offset_mapping):
                #        if start <= char_pos < end:
                #             return token_idx
                #    return None
                #response_idx = getindex(len(mid_response), response_mapping.offset_mapping)
                # 5 token space. 
                #if response_idx is not None and len(outputs[0]) > response_idx + 5:
                #    outputs[0] = outputs[0][ : response_idx]
                #    probs[0] = probs[0][ : response_idx]
                #    crits[0] = crits[0][ : response_idx]
                #response = tokenizer.decode(outputs[0])

                reward = 0.0
                if reward > 0.5:
                    topk_hit = 1
                acc_reward = acc_reward + reward
                acc_num = acc_num + 1
                
                # prompt_tokens: List[List[int]],
                all_tokens = []
                all_masks = []
                output_rewards = []
                for prompt, response, input_id, output_id in zip(batch_prompts, batch_responses, input_ids, outputs):
                    _ids = input_id + output_id 
                    _masks = [0] * len(input_id) + [1] * len(output_id) 
                    _rewards = [0] * (len(output_id)-1) + [reward] 
                    
                    all_tokens.append(_ids)
                    all_masks.append(_masks)
                    output_rewards.append(_rewards)
    
                    #<prompt, response, reward, probs, crits, tokens, masks, seq_rewards>    
                    experience = Sample(prompt = prompt, response = response, reward = reward, tokens = all_tokens[0], masks = all_masks[0], seq_rewards = output_rewards[0])
                    buffer.push(experience)

            topk_reward = topk_reward + topk_hit
            topk_num = topk_num + 1
            if len(buffer) >= buffer_size:    
                avg_reward = buffer.mean_reward()
                avg_response_len = buffer.avg_responselen()
                
                #buffer.calculate_advantage()
                
                print('progress: ', batch_idx, ', avg_reward: ', avg_reward, ', avg_response_len: ', avg_response_len , ', rank: ', rank)
                print('topk_reward: ', topk_reward * 1.0 / topk_num, ', topk_num: ', topk_num, ', rank: ', rank)

                dist.barrier()
                ############# profiling
                time_end = time.perf_counter()  

                elapsed_time_generation = time_end - time_start


                time_start = time.perf_counter()  

                _n_tokens = torch.tensor([avg_response_len * len(buffer)], dtype=torch.float, device = device)
                dist.all_reduce(_n_tokens, op=dist.ReduceOp.SUM)

                #if local_rank == 0:
                print('elapsed_time_generation:', elapsed_time_generation, 'total_tokens:', _n_tokens, 'token per sec:', _n_tokens/elapsed_time_generation)

                #buffer.distributed_advantage_norm(device, dist)
                #policy_loss_log, critic_loss_log = ppo_train(llm, llm_config, optimizer, scheduler, buffer, buffer_size, device)
                #print('policy_loss_log: ', policy_loss_log)
                #print('critic_loss_log: ', critic_loss_log)
                ## start the model training; 
                buffer.clear()    
                llm.eval()
                
        print('final average reward: ', acc_reward / acc_num, ', acc_num: ', acc_num)
        print('final topk reward: ', topk_reward * 1.0 / topk_num, ', topk_num: ', topk_num)
        
        dist.barrier()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="none", help="path to pretrained ckpt.")
    parser.add_argument("--n_rollout", type=int, default=1, help="number of rollout per sample.")
    parser.add_argument("--epoch", type=int, default=1, help="number of epoches.")
    parser.add_argument("--replay_size", type=int, default=128, help="size of replay buffer.")
    parser.add_argument("--model_type", type=str, default="phi3", choices=["phi3", "phi4"], help="choose model type.")
    
    args = parser.parse_args()
    
    main(args)

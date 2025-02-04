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

from replaybuffer import ReplayBuffer

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

from safetensors.torch import load_file



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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=5))    
    
    dist.barrier()
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

    safetensor_files = [
        f"{local_model_path}/model-00001-of-00002.safetensors",
        f"{local_model_path}/model-00002-of-00002.safetensors"
    ]

    # Step 3: Load and merge multiple safetensor state_dicts
    model_state_dict = {}
    for file in safetensor_files:
        part_state_dict = load_file(file, device="cpu")  # Load each part
        model_state_dict.update(part_state_dict)  # Merge into one dictionary

    print('load model weight ... ')
    #llm = AutoModelForCausalLM.from_pretrained( 
    #    local_model_path, 
    #    device_map="cpu", 
    #    torch_dtype=torch.bfloat16, 
    #    trust_remote_code=True, 
    #    local_files_only=True 
    #) 

    #.to(device) 
    llm_model = _Phi3ForCausalLM(llm_config) 
    
    # Step 4: Apply the merged state_dict to the model
    #missing_keys, unexpected_keys = llm_model.load_state_dict(llm.state_dict(), strict=False) 
    missing_keys, unexpected_keys = llm_model.load_state_dict(model_state_dict, strict=False) 

    print('missing_keys: ', missing_keys)
    print('unexpected_keys: ', unexpected_keys)
    
    llm_model = llm_model.to(torch.bfloat16).to(device) 
    llm_model.model.gradient_checkpointing = True 

    dist.barrier()
    initmodel_sync(llm_model)
    
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
    
    for epoch in range(0, 1):
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
            
            outputs, probs, crits = llm.module.generate(input_ids, max_gen_len = 3000)

            if batch_idx == 0 and local_rank == 0:
                print('probs.shape', len(probs[0]))
                print('crits.shape', len(crits[0]))
                print('outputs.shape', len(outputs[0]))
                #probs.shape 538
                #crits.shape 538
                #outputs.shape 538
                #batch idx 0
            response = tokenizer.decode(outputs[0])
            response_mapping = tokenizer(response, return_offsets_mapping=True)
            
            #processed_response, extract_answer, reward
            mid_response, extracted_answer, reward = process_math_answer(response, answers, tokenizer)
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

            #<prompt, response, reward, probs, crits, tokens, masks, seq_rewards>
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

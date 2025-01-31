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

from math_util import compare_math_answers, process_math_prompt, process_math_answer

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

    # Define local model path
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
    # Load model from local path using the configuration
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
    
    llm = llm_model
    #dist.barrier()
    #print('before model sync, model parameters', 'rank', rank, llm_model.critic_head.weight)
    #initmodel_sync(llm_model)
    #dist.barrier()
    #print('after model sync, model parameters', 'rank', rank, llm_model.critic_head.weight)

    # load tokenizer.
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    
    tokenizer.model_max_length = 4096
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    print('initial llm model ....')
    ############################################################

    # load dataset....
    datafile = 'math_level3to5_data_processed_with_qwen_prompt.json'
    dataset = load_dataset('json', data_files=datafile)
    print(f"loaded {dataset} with data_files={datafile}")

    # 
    sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset['train'], batch_size=1, sampler=sampler)

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
            
            outputs, probs, crits = llm.generate(input_ids, max_gen_len = 3000)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            #processed_response, extract_answer, reward
            response, extracted_answer, reward = process_math_answer(response, answers)

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
                
            if batch_idx % 10 == 0:
                print('generating: ', batch_idx, ', average_reward: ', acc_reward / acc_num, ', rank:', rank)

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

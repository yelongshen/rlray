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

def main():
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

    # load model. 
    model_name = "microsoft/Phi-3.5-mini-instruct"
    llm = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map='cpu',
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    )#.to(device)
    llm_config = AutoConfig.from_pretrained(model_name)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        for batch_idx, d in enumerate(dataloader):
            qwen_prompt = d['input']
            vanilla_prompt = d['question']         
            # features: ['input', 'answer', 'gt_answer', 'subject', 'level', 'question', 'ground_truth_answer', 'target'],
            print('qwen_prompt:', qwen_prompt)
            print('vanilla_prompt:', vanilla_prompt)

            x1 = tokenizer(qwen_prompt, add_special_tokens=False, max_length=8, truncation=True)
            print('qwen_ids1:', x1['input_ids'])

            x2 = tokenizer(qwen_prompt, add_special_tokens=False, max_length=16, truncation=True)
            print('qwen_ids2:', x2['input_ids'])

            y1 = tokenizer(vanilla_prompt, add_special_tokens=False, max_length=1024, truncation=True)
            print('vanilla_ids:', y1['input_ids'])

            input_ids = y1['input_ids']
            outputs, probs, crits = llm.generate(input_ids, max_gen_len = 3000)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print('response:', response)
            
            break
            #data, target = data.to(device), target.to(device)
    # one node inference; one node training; as an example; 
    # suppose we use 4 gpus for vllm and 4 gpus 
    #if rank in [0,1,2,3,4,5,6,7]:
    #    learn(learndp) #, mdg)
    #else:
    #    play(learndp) #, mdg)
if __name__ == "__main__":
    main()

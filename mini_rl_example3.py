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

from vllm import LLM, SamplingParams

from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError
from functools import partial
#from contextlib import redirect_stdout
import sys

from datasets import load_dataset

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



def play():
    # Load a model
    
    print('start llm data ...')
    
    rank = int(os.environ['RANK'])

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # give up huggingface model.
    
    model_name = "microsoft/Phi-3.5-mini-instruct"
    llm = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    # llm = LLM(model="microsoft/Phi-3-mini-4k-instruct", disable_custom_all_reduce=True, enforce_eager=True ) #, device_map=f"cuda:{rank}") # "facebook/opt-6.7b")  # You can specify any Hugging Face model here
    # llm.llm_engine.model_executor.driver_workerinit_process_group(
    #            master_address, master_port, rank_offset, world_size, group_name)
    # Set sampling parameters
    print('initial llm model ....')
    
    #sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024)
    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    print('start sampling data ...')
    # Generate response
    #outputs = []
    for epoch in range(0, 100):
        for i in range(0, len(train)):
            if i % 8 != local_rank:
                continue
            example = train[i]
            soluts = example['solutions']
            problem = example['description']

            inputs = tokenizer(problem, return_tensors="pt").to("cuda")
            #print('input_ids.shape', inputs["input_ids"].shape)

            outputs = llm.generate(inputs["input_ids"], max_length=4096)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            #o = llm.generate([problem], sampling_params)
            #completion = o[0].outputs[0].text

            completion = response
            data = problem + completion

            target_rank = 4
            #rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))
            time.sleep(1)

            print('push to buffer ... ') #, data)
            #if check_model_update():
            #    llm.model.load_state_dict()
        #print(ans)
        #outputs.append(ans)

def learn():   
    print('start to learn ....') 
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    
    #rank = int(os.environ['RANK'])
    torch.random.manual_seed(0) 
    
    device = torch.device(f"cuda:{local_rank}")
    # give up huggingface model.
    
    model_name = "microsoft/Phi-3.5-mini-instruct"

    model = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ).to(device)
    print('done with model creation.')

    dist.init_process_group(backend="nccl", rank=local_rank, world_size=8)
    #dist.init_process_group(backend="nccl", rank)

    print('dist initialization ...', local_rank)

    dist.barrier()

    print('dist barrier success')

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    print('distributed model creation.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #num_epochs = 3
    num_training_steps = 10000 # num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
    )

    print('model optimization initialization...')
    
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    tokenizer.model_max_length = 4096
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    i_num = 0
    batch_size = 1
    max_seq_len = 4096

    print('done...')


def main():
    # system parameters:
    # args.ngpu_per_node
    # args.nnode_actor
    # args.nnode_learner
    #world_size = 8

    local_rank = int(os.environ['LOCAL_RANK'])
    print('local rank', local_rank)

    rank = int(os.environ['RANK'])
    print('rank', rank)
    #rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    gpus_per_node = 8
    node_idx = rank // gpus_per_node

    world_size = int(os.environ['WORLD_SIZE'])
    print('WORLD_SIZE', world_size)

    # suppose we use 4 gpus for vllm and 4 gpus 
    if rank in [0,1,2,3,4,5,6,7]:
        #print('rank', rank, 'play')
        play()
    else:
        learn()
        #for i in range(0, 1000000):
        #    print('rank', rank, 'sleep.....')
        #    time.sleep(1)
    #else:
    #    learn()
    #if rank in [1,2,3,4,5,6,7]:
    #    
    #       
    #          time.sleep(1)
    #    learn()
    
if __name__ == "__main__":
    main()

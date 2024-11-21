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

#from accelerate import Accelerator
from torch.utils.data import DataLoader
#from transformers import AdamW
#import numpy as np 

#from transformers import get_linear_schedule_with_warmup
#from torch.optim import AdamW



def play():
    # Load a model
    print('start llm data ...')
    
    rank = int(os.environ['RANK'])

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    llm = LLM(model="microsoft/Phi-3-mini-4k-instruct") #, device_map=f"cuda:{rank}") # "facebook/opt-6.7b")  # You can specify any Hugging Face model here
    # llm.llm_engine.model_executor.driver_workerinit_process_group(
    #            master_address, master_port, rank_offset, world_size, group_name)
    # Set sampling parameters
    print('initial llm model ....')
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024)

    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    print('start sampling data ...')

    #outputs = []
    for epoch in range(0, 100):
        for i in range(0, len(train)):
            example = train[i]
            soluts = example['solutions']
            problem = example['description']

            o = llm.generate([problem], sampling_params)

            completion = o[0].outputs[0].text

            data = problem + completion

            target_rank = 4
            #rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))
            time.sleep(1)

            print('push to buffer ... ', data)
            #if check_model_update():
            #    llm.model.load_state_dict()

        #print(ans)
        #outputs.append(ans)


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

    # suppose we use 4 gpus for vllm and 4 gpus 
    
    if rank in [0]:
        #print('rank', rank, 'play')
        play()
    #else:

    #if rank in [1,2,3,4,5,6,7]:
    #    for i in range(0, 1000000):
    #        print('rank', rank, 'sleep.....')
    #          time.sleep(1)
    #    learn()

if __name__ == "__main__":
    main()
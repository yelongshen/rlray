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
from contextlib import redirect_stdout
import sys

from datasets import load_dataset

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

import logging

from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np 

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.lock = threading.Lock()

    def add(self, experience):
        """ Add new experience to the buffer """
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Sample a batch of experiences from the buffer """
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        with self.lock:
            return len(self.buffer)

replaybuffer = ReplayBuffer(500000)

# Consumer side: Function to add experiences to its local buffer
def add_to_buffer(experience):
    #print('debug consumer side add.....',  int(os.environ['RANK']) )
    global replaybuffer
    replaybuffer.add(experience)

def len_buffer():
    global replaybuffer
    return len(replaybuffer)

def pop_from_buffer(batchsize):
    global replaybuffer
    return replaybuffer.sample(batchsize)
############################################### 

def rev_experience_len(server_worker='worker1'):
    return rpc.rpc_sync(server_worker, len_buffer)

def rev_experience_data(server_worker='worker1', batchsize=2):
    return rpc.rpc_sync(server_worker, pop_from_buffer, args=(batchsize, ))


################################ Model Buffer


class ModelBuffer:
    def __init__(self):
        self.buffer = None
        self.is_new = False
        self.lock = threading.Lock()

    def push(self, model):
        """ Add new experience to the buffer """
        with self.lock:
            self.buffer = model
            self.is_new = True

    def check_new(self):
        with self.lock:
            return self.is_new

    def pull(self, host_model):
        with self.lock:
            host_model.data.copy_(self.buffer)
            self.is_new = False


#def check_model_update()
# rpc communication. 


def play():
    # Load a model
    llm = LLM(model="microsoft/Phi-3-mini-4k-instruct") # "facebook/opt-6.7b")  # You can specify any Hugging Face model here
    # llm.llm_engine.model_executor.driver_workerinit_process_group(
    #            master_address, master_port, rank_offset, world_size, group_name)
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024)

    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

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
            rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))

            time.sleep(1)
            print('push to buffer')
            #if check_model_update():
            #    llm.model.load_state_dict()

        #print(ans)
        #outputs.append(ans)

def learn():    
    dist.init_process_group(backend="nccl")

    rank = int(os.environ['RANK'])
    torch.random.manual_seed(0) 
    
    device = torch.device(f"cuda:{rank}")
    # give up huggingface model.
    model = AutoModelForCausalLM.from_pretrained( 
        "microsoft/Phi-3-mini-4k-instruct",  
        device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ).to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #num_epochs = 3
    num_training_steps = 10000 # num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=num_training_steps
    )

    model.train()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    i_num = 0
    batch_size = 2
    max_seq_len = 128

    while i_num < 1000:
        l = len(replaybuffer) if rank == 4 else rpc.rpc_sync('worker4', len_buffer)
        # data sample threshold there. 
        if l > 20:
            # Sample batch of experiences from the replay buffer
            #(x, y) = buffer.sample(2)
            #print("[Consumer] Sampled batch:", x, y)
            data = replaybuffer.sample(batch_size) if rank == 4 else rev_experience_data('worker4', batch_size)
            print("[Consumer] Sampled batch:", data)
            #time.sleep(1)
            batch = tokenizer(data, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors="pt")

            input_ids = batch["input_ids"]
    
            # Shift input_ids to create labels for next-token prediction
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Mask the last token
            
            # Return the dictionary with input_ids, attention_mask, and labels
            batch["labels"] = labels

            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            print('loss', loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            i_num = i_num + 1

            time.sleep(1)
            #if i_num % 100 == 0:
            #    # model weight sync.
            #    send_model_weight_to_producer(model.weight.cpu())
            #    print('push model weight...........')

def main():
    # system parameters:
    # args.ngpu_per_node
    # args.nnode_actor
    # args.nnode_learner
    world_size = 8
    rank = int(os.environ['RANK'])
    
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # suppose we use 4 gpus for vllm and 4 gpus 
    if rank in [0,1,2,3]:
        play()

    if rank in [4,5,6,7]:
        learn()

if __name__ == "__main__":
    main()
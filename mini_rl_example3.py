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


#buff = []
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
        return len(self.buffer)
buffer = ReplayBuffer(100000)

def add_to_buffer(experience):
    print('[debug] consumer side add.....',  int(os.environ['RANK']) )
    global buffer
    buffer.add(experience)

def len_buffer():
    global buffer
    return len(buffer)

def pop_from_buffer(batchsize):
    global buffer
    return buffer.sample(batchsize)
################################################################################################

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

            if inputs["input_ids"].shape[1] > 4000:
                continue
            outputs = llm.generate(inputs["input_ids"], max_length=4096)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            #o = llm.generate([problem], sampling_params)
            #completion = o[0].outputs[0].text

            completion = response
            data = problem + completion

            buffer_rank = 8
            #rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))
            #time.sleep(1)
            #print('push to buffer ... ') #, data)
            rpc.rpc_sync(f"worker-{buffer_rank}", add_to_buffer, args=(data,))
            
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
    buffer_rank = 8
    batch_size = 1
    sample_idx = 0
    step = 0
    gradient_accumulation_steps = 32
    optimizer.zero_grad()

    while step < 40000:
        l = len(buffer) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", len_buffer) #rev_experience_len('worker2')
        if l > 20:
            data = buffer.sample(batch_size) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", pop_from_buffer, args=(batch_size, )) #rev_experience_data('worker2', 2)
            inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt").to(device)
            
            #labels = batch["labels"].to(device)

            input_ids = inputs["input_ids"]
    
            # Shift input_ids to create labels for next-token prediction
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Mask the last token
            
            # Return the dictionary with input_ids, attention_mask, and labels
            inputs["labels"] = labels

            batch = {k: v.to(device) for k,v in inputs.items()}
            outputs = model(**batch)

            loss = outputs.loss
            print('loss:', loss, 'rank', rank)
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Update the learning rate

            step = step + 1
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
    
    rpc.init_rpc(f"worker-{rank}", rank=rank, world_size=16) # consider 2 nodes, 16 gpus in this example.
    
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

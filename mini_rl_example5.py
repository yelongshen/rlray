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



######################################################################## REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] # every sample is very different. 
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

buffer = ReplayBuffer(4096)
player_node = 1
learner_node = 1
buffer_rank = 0

def add_to_buffer(_tokens, _masks, _probs, _reward, _crits):
    global buffer
    experience = (_tokens, _masks, _probs, _reward, _crits)
        
    buffer.add(experience)

def len_buffer():
    global buffer
    return len(buffer)

def pop_from_buffer(batchsize):
    global buffer
    return buffer.sample(batchsize)

################################################################################################
class ModelUpdateMessage:
    def __init__(self):
        self.is_new = False
        self.lock = threading.Lock()

    def push(self):
        """ Add new experience to the buffer """
        with self.lock:
            self.is_new = True

    def check(self):
        with self.lock:
            return self.is_new

    def pull(self):
        with self.lock:
            self.is_new = False

msg = ModelUpdateMessage()

def msg_push():
    global msg
    msg.push()

def notify_model_update():
    global msg
    for worker in range(8, 16):
        rpc.rpc_sync(f"worker-{worker}", msg_push, timeout=0)

def allmodel_sync(model:_Phi3ForCausalLM): #, device_ids, mdg):
    with torch.no_grad():
        for i, (name, param) in enumerate(model.state_dict().items()):
            torch.distributed.broadcast(param, 0, async_op=False)
    global msg
    msg.pull()
######################################################################## MODEL BUFFER


def code_extraction(input_text):
    lines = input_text.splitlines()
    code_lines = []
    in_code_block = False

    for line in lines:
        if line.strip() == "```python":  # Start of Python code block
            in_code_block = True
        elif line.strip() == "```":  # End of code block
            if in_code_block:
                break  # End the extraction when the code block ends
        elif in_code_block:
            code_lines.append(line)

    return "\n".join(code_lines)

def evaluate_program(program, test_input, test_output):    
    def run_program(conn, program, test_input):        
        try:            
            local_stdout = io.StringIO()            
            local_stdin = io.StringIO(test_input)            
            sys.stdout = local_stdout            
            sys.stdin = local_stdin            
            local_globals = {}            
            exec(program, local_globals)            
            output = local_stdout.getvalue().strip()            
            conn.send(output)  
            # Send output through Pipe        
        except Exception as e:            
            conn.send(f"Error: {str(e)}")        
        finally:            
            conn.close()  
    parent_conn, child_conn = multiprocessing.Pipe()    
    process = multiprocessing.Process(target=run_program, args=(child_conn, program, test_input))    
    process.start()    
    process.join(5)    
    if process.is_alive():        
        logging.error("Process timed out. Forcibly killing...")        
        os.kill(process.pid, signal.SIGKILL)        
        process.join()    
    if parent_conn.poll():  
        # Check if there's data to read        
        output = parent_conn.recv()  
        # Non-blocking receive        
        return test_output.strip() == output.strip()    
    else:        
        return "Error: No output from program"

######################################################################################### #create distributed process group for model syncize.

def initmodel_sync(model:_Phi3ForCausalLM):
    mgroup = [x for x in range(8 * (player_node + learner_node))]
    gp = torch.distributed.new_group(mgroup)
    with torch.no_grad():
        torch.distributed.broadcast(model.critic_head.weight, 0, group=gp, async_op=False)
        torch.distributed.broadcast(model.critic_head.bias, 0, group=gp, async_op=False)
            
##########################################################################################
def play(learndp): #, mdg):
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
        device_map='cpu',
        #device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    )#.to(device)
    llm_config = AutoConfig.from_pretrained(model_name)
    llm_model = _Phi3ForCausalLM(llm_config)
    
    missing_keys, unexpected_keys = llm_model.load_state_dict(llm.state_dict(), strict=False)
    llm_model = llm_model.to(torch.bfloat16).to(device)

    print('before model sync, model parameters', 'rank', rank, llm_model.critic_head.weight)
    initmodel_sync(llm_model)
    print('after model sync, model parameters', 'rank', rank, llm_model.critic_head.weight)


    llm = llm_model

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    print('initial llm model ....')
    
    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    instruction_prefix = ''
    instruction_postfix = '\n\nplease only reply with the source code in python. \n'
    print('start sampling data ...')


    # Generate response
    for epoch in range(0, 100):
        total_reward = 0
        total_count = 0
        
        print('start to trigger play ...........................\n\n')
        for i in range(0, len(train)):
            if i % 8 != local_rank :
                continue
            example = train[i]
            soluts = example['solutions']
            problem = example['description']

            problem = instruction_prefix + problem + instruction_postfix

            x = tokenizer([problem])
            input_ids = x['input_ids']
            
            if len(input_ids[0]) > 2000: # inputs["input_ids"].shape[1] > 2000:
                continue
            
            outputs, probs, crits = llm.generate(input_ids, max_gen_len = 2048)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            program = code_extraction(response)
            
            tests = example['public_tests']
            correct = 0
            total = 0
            
            for test_input, test_output in zip(tests['input'], tests['output']):
                o = evaluate_program(program, test_input, test_output)
                if o == True:
                    correct = correct + 1
                total = total + 1
                
                
            reward_score = correct * 1.0 / (total+0.0001)
            print('success rate...................', reward_score,'\n\n')

            total_reward = total_reward + reward_score
            total_count = total_count + 1
            
            completion = response
            data = problem + completion

            _tokens = input_ids[0] + outputs[0] 
            _masks = [0] * len(input_ids[0]) + [1] * len(outputs[0])
            _probs = probs[0]    
            _crits = crits[0]
            _reward = [0.0] * (len(_crits)-1) + [reward_score]
            # send data into replaybuffer.
            _info = (_tokens, _masks, _probs, _reward, _crits)
            rpc.rpc_sync(f"worker-{buffer_rank}", add_to_buffer, args=_info, timeout=0)

            if msg.check():
                #print('waiting on player barrier 1', rank)
                dist.barrier() #mdg)
                #print('waiting on player barrier 1', rank)
                allmodel_sync(llm) #, device_ids=[local_rank], mdg=mdg)
                #print('waiting on player barrier 2', rank)
                dist.barrier()
                #dist.barrier(mdg)
                print('player model update....', rank)
                
        print('end to trigger play ...........................\n\n')
        print('average reward: ', total_reward / (total_count + 0.00001), '\n\n') 
        
def learn(learndp): #, mdg):   
    print('start to learn ....') 
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.random.manual_seed(0) 
    
    device = torch.device(f"cuda:{local_rank}")
    
    model_name = "microsoft/Phi-3.5-mini-instruct"
    llm = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map='cpu',
        #device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    )#.to(device)
    llm_config = AutoConfig.from_pretrained(model_name)

    model = _Phi3ForCausalLM(llm_config)
    missing_keys, unexpected_keys = model.load_state_dict(llm.state_dict(), strict=False)
    model = model.to(torch.bfloat16).to(device)

    model.model.gradient_checkpointing = True
    print('before model sync, model parameters', 'rank', rank, model.critic_head.weight)
    initmodel_sync(model)
    print('after model sync, model parameters', 'rank', rank, model.critic_head.weight)    
    print('done with model creation.')
    
    vocab_size = llm_config.vocab_size


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],  process_group=learndp)
    print('distributed model creation.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-6)
    #num_epochs = 3
    num_training_steps = 10000 # num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
    )

    print('model optimization initialization...')

    model.train()
    i_num = 0
    batch_size = 1
    max_seq_len = 4096

    print('done...')
    #buffer_rank = 8
    batch_size = 1
    sample_idx = 0
    step = 0
    gradient_accumulation_steps = 32
    optimizer.zero_grad()

    pad_id = llm_config.pad_token_id

    print('model.device', model.device)
    # 
    mseLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')

    critic_loss = 0.0
    policy_loss = 0.0

    mini_c_loss = 0.0
    mini_p_loss = 0.0

    update_step = 0
    
    # rl training steps;
    while step < num_training_steps:
        # receive data from buffer_rank
        #l = len(buffer) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", len_buffer, timeout=0) #rev_experience_len('worker2')
        try:
            l = len(buffer) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", len_buffer, timeout=10)  # Add timeout
        except:
            print(f"RPC Error while getting buffer length on rank {rank}: {e}")
            l = 0  # Default value or take alternative action
        
        if l < 32:
            time.sleep(1)    
        else:
            torch.cuda.empty_cache()
            try:
                data = buffer.sample(batch_size) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", pop_from_buffer, args=(batch_size, ), timeout=10) #rev_experience_data('worker2', 2)
            print('done with fetching the data', rank)
            dist.barrier(learndp)
            print('getting all data done...', rank)
            _tokens = [d[0] for d in data]
            _masks = [d[1] for d in data]
            _probs = [d[2] for d in data]
            _rewards = [d[3] for d in data]
            _crits = [d[4] for d in data] 

            if step == 0:
                print('example:', _tokens, _masks, _probs, _rewards, _crits)

            _tokens = torch.tensor(_tokens).to(torch.long).to(model.device)
            
            # re-evaluate the policy.     
            _, logits, critics, _ = model(_tokens)

            _b, _seq = _tokens.shape
            
            logprobs = -F.cross_entropy(
                input=logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
                target=_tokens.reshape(-1)[1:], 
                reduction="none",
                ignore_index=pad_id,
            ).reshape(1, -1)
            
            critics = critics.reshape(_b, _seq)
            
            old_logprobs = torch.tensor(_probs).to(model.device)
            _idx = _masks[0].index(1)
            ratios = torch.exp(logprobs[:, _idx-1:] - old_logprobs.detach() + 1e-10)
            
            critics = critics[:, _idx-1:-1] 
            
            gamma = 0.95
            rewards = []
            discounted_reward = 0
            for reward in reversed(_rewards[0]): 
                discounted_reward = reward + (gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            # Normalizing the rewards
            rewards = torch.tensor([rewards], dtype=torch.bfloat16).to(model.device)

            # global reward sync. 
            #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
            # convert list to tensor
            #old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            #old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            #old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
            #old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

            old_state_values = torch.tensor(critics).to(torch.bfloat16).to(model.device)
            # calculate advantages
            advantages = rewards.detach() - old_state_values.detach()

            eps_clip = 0.2
            # Finding Surrogate Loss  
            surr1 = ratios * advantages # (optimize logprobs)
            
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages

            _p_loss = -torch.min(surr1, surr2).mean()
            _c_loss = mseLoss(critics, rewards).mean()
            
            # final loss of clipped objective PPO objective. 
            loss = _p_loss + 0.5 * _c_loss  #- 0.01 * dist_entropy
            
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
                    print('mini_c_loss: ', mini_c_loss, 'critic_loss', critic_loss)
                    print('mini_p_loss: ', mini_p_loss, 'policy_loss', policy_loss)
 
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
def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    print('local rank', local_rank)

    rank = int(os.environ['RANK'])
    print('rank', rank)
    
    # init distributed process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=16, timeout=datetime.timedelta(minutes=5))
    
    dist.barrier()
    learndp = torch.distributed.new_group([0,1,2,3,4,5,6,7])    
    dist.barrier() #learndp)
    print('dist learndp barrier success', rank)

    #dist.barrier()
    #mdg = torch.distributed.new_group([0, 8, 9, 10, 11, 12, 13, 14, 15])
    #dist.barrier()
    #print('dist mdg barrier success', rank)
    
    # rpc.init_rpc(f"worker-{rank}", backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500"))
    rpc.init_rpc(f"worker-{rank}", rank=rank, world_size=16, rpc_backend_options=rpc.TensorPipeRpcBackendOptions()) # consider 2 nodes, 16 gpus in this example.

    gpus_per_node = 8
    node_idx = rank // gpus_per_node

    world_size = int(os.environ['WORLD_SIZE'])
    print('WORLD_SIZE', world_size)

    # one node inference; one node training; as an example; 
    # suppose we use 4 gpus for vllm and 4 gpus 
    if rank in [0,1,2,3,4,5,6,7]:
        learn(learndp) #, mdg)
    else:
        play(learndp) #, mdg)
    
if __name__ == "__main__":
    main()

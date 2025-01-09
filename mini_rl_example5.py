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

def add_to_buffer(_tokens, _masks, _probs, _reward, _crits): # experience):
    #print('[debug] consumer side add.....',  int(os.environ['RANK']) )
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
            #host_model.data.copy_(self.buffer)
            self.is_new = False

msg = ModelUpdateMessage()

def msg_push():
    global msg
    msg.push()

def notify_model_update():
    global msg
    for worker in range(8, 16):
        rpc.rpc_sync(f"worker-{worker}", msg_push, timeout=0)

def allmodel_sync(model:_Phi3ForCausalLM, device_ids, mdg):
    #mgroup = [x for x in range(8 * (player_node + learner_node))]
    #gp = torch.distributed.new_group(mgroup)
    global msg
    for param in model.state_dict().values():
        torch.distributed.broadcast(param, 0, group=mdg, device_ids = device_ids, async_op=False)
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

    torch.distributed.broadcast(model.critic_head.weight, 0, group=gp, async_op=False)
    torch.distributed.broadcast(model.critic_head.bias, 0, group=gp, async_op=False)
    #return gp
    #mp_groups = [[0,1,2,3], [4,5,6,7]]
    #groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size, pipeline_length, model_parallel_size)
    #ranks = groups[i, :, k].tolist()
    # initial parallel group.
    #for g in mp_groups:
    #    group = torch.distributed.new_group(g)
    #    if args.rank in g:
    #        mp_group = group #, backend='gloo') # model parallel group.
    #        mp_master_rank = g[0]
    #        MP_Group = mp_group
            
##########################################################################################
def play():
    # Load a model
    print('start llm data ...')
    
    rank = int(os.environ['RANK'])

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank) # local_rank)
    #device = torch.device(f"cuda:{local_rank}")
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
    #print(llm_config)
    #print('attn_implemention', llm_config._attn_implementation)
    #print('config.rope_scaling', llm_config.rope_scaling)
    
    #llm_state_dict = llm.state_dict()
    #print('state key begin.......')
    #for key in llm_state_dict:
    #    print(key)
    #print('state key end  .......')
    # Load configuration from a pre-trained model
    
    #Phi3rCausalLM(Phi3ForCausalLM):
    #def __init__(self, config, base_model, is_critic=False):
    #llm_model = Phi3rCausalLM(llm_config, llm, is_critic=True)
    llm_model = _Phi3ForCausalLM(llm_config)
    
    missing_keys, unexpected_keys = llm_model.load_state_dict(llm.state_dict(), strict=False)
    llm_model = llm_model.to(torch.bfloat16).to(device)

    print('before model sync, model parameters', 'rank', rank, llm_model.critic_head.weight)
    initmodel_sync(llm_model)
    print('after model sync, model parameters', 'rank', rank, llm_model.critic_head.weight)

    # critic_model = Phi3rCausalLM(llm_config, llm, is_critic=True) # Phi4LM(llm, r=8, lora_alpha=1.0)
    #phi4rllm = Phi4rLM(llm_config)
    # to avoid copy two times of model-weight.
    #missing_keys, unexpected_keys = phi4rllm.load_state_dict(llm_state_dict, strict=False)
    #print("Missing keys:", missing_keys)
    #print("Unexpected keys:", unexpected_keys)
   
    #critic_model = critic_model.to(device)
    #print(phi4rllm)
    
    llm = llm_model
    #base_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

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

    instruction_prefix = ''
    instruction_postfix = '\n\nplease only reply with the source code in python. \n'
    print('start sampling data ...')

    learndp = torch.distributed.new_group([0,1,2,3,4,5,6,7])
    dist.barrier(learndp)

    ### model distributed group.
    print('player: rank', rank, 'create mdg...')
    mdg = torch.distributed.new_group([0, 8, 9, 10, 11, 12, 13, 14, 15])
    dist.barrier(mdg)
    print('dist mdg barrier success..', rank)


    # Generate response
    #outputs = []
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
            #inputs = tokenizer(problem, return_tensors="pt").to("cuda")
            #print('input_ids.shape', inputs["input_ids"].shape)

            if len(input_ids[0]) > 2000: # inputs["input_ids"].shape[1] > 2000:
                continue
            #print('input_ids', input_ids)
            #prompt_tokens: List[List[int]],
            #max_gen_len: int,
            #llm.begin_generation()
            #outputs = llm.generate(inputs["input_ids"], max_length=4096)
            outputs, probs, crits = llm.generate(input_ids, max_gen_len = 2048)
            #llm.end_generation()
            #for _i in range(0, len(llm.critic_list)):
            #    print('critic', _i, llm.critic_list[_i], llm.critic_list[_i].shape)
            #print('outputs', outputs)

            #critic_model(inputs["input_ids"])
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            #print('*************', problem + response) 
            #o = llm.generate([problem], sampling_params)
            #completion = o[0].outputs[0].text

            #print('code response start .........................................\n\n')
            #print(response)
            #print('code response end .........................................\n\n')
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

            # discrete tokens, word probabilities, mask, critics. 
            # send data into replaybuffer.

            _info = (_tokens, _masks, _probs, _reward, _crits)
            rpc.rpc_sync(f"worker-{buffer_rank}", add_to_buffer, args=_info, timeout=0)

            if msg.check():
                allmodel_sync(llm, device_ids=[local_rank], mdg=mdg)
            #buffer_rank = 8
            #rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))
            #time.sleep(1)
            #print('push to buffer ... ') #, data)
            
            #if check_model_update():
            #    llm.model.load_state_dict()
            
        #print(ans)
        #outputs.append(ans)
        print('end to trigger play ...........................\n\n')
        print('average reward: ', total_reward / (total_count + 0.00001), '\n\n') 
        #break
        
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
    llm = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map='cpu',
        #device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    )#.to(device)
    llm_config = AutoConfig.from_pretrained(model_name)

    #model_name = "microsoft/Phi-3.5-mini-instruct"

    #model = AutoModelForCausalLM.from_pretrained( 
    #    model_name,  
    #    device_map="cpu",  
    #    torch_dtype=torch.bfloat16,  
    #    trust_remote_code=True,  
    #) #.to(device)
    #tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    #model.gradient_checkpointing_enable()

    model = _Phi3ForCausalLM(llm_config)
    missing_keys, unexpected_keys = model.load_state_dict(llm.state_dict(), strict=False)
    model = model.to(torch.bfloat16).to(device)

    model.model.gradient_checkpointing = True
        
    print('before model sync, model parameters', 'rank', rank, model.critic_head.weight)
    initmodel_sync(model)
    print('after model sync, model parameters', 'rank', rank, model.critic_head.weight)
     
        
    print('done with model creation.')
    ##  精诚所至，金石为开。
    ##  天地万物皆同力。
    
    #mgroup = [x for x in range(8 * (player_node + learner_node))]
    #gp = torch.distributed.new_group(mgroup)

    vocab_size = llm_config.vocab_size
    
    print('dist learndp group initialization ...', rank)
    learndp = torch.distributed.new_group([0,1,2,3,4,5,6,7])
    dist.barrier(learndp)
    #dist.init_process_group(backend="nccl", rank=local_rank, world_size=8)
    #dist.init_process_group(backend="nccl", rank)
    
    #dist.barrier(learndp)
    print('dist learndp barrier success')

    

    print('learner: rank', rank, 'create mdg...')
    mdg = torch.distributed.new_group([0, 8, 9, 10, 11, 12, 13, 14, 15])
    dist.barrier(mdg)
    print('dist mdg barrier success')

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

    #tokenizer.model_max_length = 4096
    #tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    #tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    #tokenizer.padding_side = 'right'

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
        l = len(buffer) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", len_buffer, timeout=0) #rev_experience_len('worker2')
        if l < 8:
            time.sleep(1)    
        else:
            torch.cuda.empty_cache()
            
            data = buffer.sample(batch_size) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", pop_from_buffer, args=(batch_size, ), timeout=0) #rev_experience_data('worker2', 2)

            #(_tokens, _masks, _probs, _reward, _crits)
                
            _tokens = [d[0] for d in data]
            _masks = [d[1] for d in data]
            _probs = [d[2] for d in data]
            _rewards = [d[3] for d in data]
            _crits = [d[4] for d in data] 
                
            #inputs = tokenizer(text, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").to(device)
            #if inputs["input_ids"].shape[1] > 4096:
            #    continue
            #if inputs['input_ids'].shape[1] < 16:
            #    continue
                
            #labels = batch["labels"].to(device)
            #input_ids = inputs["input_ids"]
            if step == 0:
                print('example:', _tokens, _masks, _probs, _rewards, _crits)

            _tokens = torch.tensor(_tokens).to(torch.long).to(model.device)
            
            # re-evaluate the policy.     
            _, logits, critics, _ = model(_tokens)

            # _tokens : batch_size, sequence_length
            # logits : batch_size, sequence_length, vocab_size;
            # critics : batch_size, sequence_length, 1 
            _b, _seq = _tokens.shape
            
            logprobs = -F.cross_entropy(
                input=logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
                target=_tokens.reshape(-1)[1:], 
                reduction="none",
                ignore_index=pad_id,
            ).reshape(1, -1)
            
            critics = critics.reshape(_b, _seq)
            
            #print('logprobs.shape', logprobs.shape)
            old_logprobs = torch.tensor(_probs).to(model.device)
            #print('old_logprobs.shape', old_logprobs.shape)   
            #print('len(_masks)', len(_masks[0]))
            #print('len(_rewards)', len(_rewards[0]))
            #print('len(_crits)', len(_crits[0]))
            
            #print(_masks)
            _idx = _masks[0].index(1)
            #print('_masks.index', _idx)
            
            #print('_probs.shape', 
            #_idx 389
            #logprobs.shape torch.Size([785])
            #old_logprobs.shape torch.Size([1, 785])
            #critics.shape torch.Size([1, 786])

            ###### PPO algorithm here.     
            #print('_idx', _idx)
            #print('logprobs.shape', logprobs.shape)
            #print('old_logprobs.shape', old_logprobs.shape)
            #print('critics.shape', critics.shape) 
            ratios = torch.exp(logprobs[:, _idx-1:] - old_logprobs.detach())
            
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
            #print('ratios.shape', ratios.shape)
            #print('advantages.shape', advantages.shape)
            #print('state_values.shape', critics.shape)
            #print('rewards.shape', rewards.shape)
            
            surr1 = ratios * advantages # (optimize logprobs)
            
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages

            _p_loss = -torch.min(surr1, surr2).mean()
            _c_loss = mseLoss(critics, rewards).mean()
            
            # final loss of clipped objective PPO objective. 
            loss = _p_loss + 0.5 * _c_loss  #- 0.01 * dist_entropy
            
            # take gradient step
            #self.optimizer.zero_grad()
            #loss.mean().backward()
            #self.optimizer.step()
            #loss = loss.mean()
            
            #print('loss', loss)
            mini_c_loss = mini_c_loss + _c_loss.detach()
            mini_p_loss = mini_p_loss + _p_loss.detach()
            # Shift input_ids to create labels for next-token prediction
            #labels = input_ids.clone()
            #labels[:, :-1] = input_ids[:, 1:]
            #labels[:, -1] = -100  # Mask the last token
            # Return the dictionary with input_ids, attention_mask, and labels
            #inputs["labels"] = labels

            #batch = {k: v.to(device) for k,v in inputs.items()}
            #print('1. forward', rank, inputs['input_ids'].shape)

            #time.sleep(10)
            #outputs = model(**batch)

            #loss = outputs.loss
            #print('loss:', loss, 'rank', rank,'step', step, 'shape', inputs['input_ids'].shape)

            #print('2. backward', rank, inputs['input_ids'].shape)
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                #print('3. optimization', rank)
                #critic_loss = 0.0
                #policy_loss = 0.0
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

                if update_step % 10 == 0:
                    # notify the producer to boardcast the model weight to 
                    if rank == 0:
                        notify_model_update()
                        allmodel_sync(model, device_ids=[local_rank], mdg=mdg)

                    #rpc.rpc_sync(f"worker-{buffer_rank}", notify_model_update, args=_info, timeout=0)
                dist.barrier(learndp)

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
    
    # init distributed process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=16)

    # rpc.init_rpc(f"worker-{rank}", backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500"))
    rpc.init_rpc(f"worker-{rank}", rank=rank, world_size=16, rpc_backend_options=rpc.TensorPipeRpcBackendOptions()) # consider 2 nodes, 16 gpus in this example.

    #rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    gpus_per_node = 8
    node_idx = rank // gpus_per_node

    world_size = int(os.environ['WORLD_SIZE'])
    print('WORLD_SIZE', world_size)

    # one node inference; one node training; as an example; 
    # suppose we use 4 gpus for vllm and 4 gpus 
    if rank in [0,1,2,3,4,5,6,7]: #, 8, 9, 10, 11, 12, 13, 14, 15]:
        #print('rank', rank, 'play')
        #play()
        learn()
    else:
        play()
        #learn()
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

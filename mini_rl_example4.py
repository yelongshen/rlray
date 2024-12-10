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

from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3MLP, Phi3PreTrainedModel, Phi3Model
from transformers.models.phi3.configuration_phi3 import Phi3Config

from transformers import AutoConfig

import torch.nn as nn
import multiprocessing

import signal
from transformers.activations import ACT2FN

class LoRALayer(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=1.0):
        super(LoRALayer, self).__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_down = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, original_linear.out_features, bias=False)
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / (r ** 0.5)
        
        nn.init.zeros_(self.lora_down.weight)
        nn.init.normal_(self.lora_up.weight)
    
    def forward(self, x):
        original_output = self.original_linear(x)
        lora_update = self.lora_up(self.lora_down(x)) * self.scaling
        return original_output + lora_update

class LoRAMLP(nn.Module):
    def __init__(self, original_mlp, r=8, lora_alpha=1.0):
        super(LoRAMLP, self).__init__()
        self.original_mlp = original_mlp
        self.r = r 
        self.lora_up = LoRALayer(original_mlp.gate_up_proj, r, lora_alpha)
        self.lora_down = LoRALayer(original_mlp.down_proj, r, lora_alpha)
        
        self.activation_fn = ACT2FN[original_mlp.config.hidden_act]
    def forward(self, x):
        up_states = self.lora_up(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.lora_down(up_states)


class Phi3rDecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Phi3Config, base_model):
        #super().__init__()
        self.config = config
        self.self_attn = base_model.self_attn # PHI3_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        self.mlp = LoRAMLP(base_model.mlp) # Phi3MLP(config)
        
        self.input_layernorm = base_model.input_layernorm # Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = base_model.resid_attn_dropout # nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = base_model.resid_mlp_dropout # nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = base_model.post_attention_layernorm # Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
class Phi3rModel(Phi3Model):
    def __init__(self, config: Phi3Config, base_model):
        Phi3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = base_model.embed_tokens # nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = base_model.embed_dropout # nn.Dropout(config.embd_pdrop)
        
        self._attn_implementation = config._attn_implementation
        self.norm = base_model.norm # Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(Phi3rDecoderLayer(config, base_model.layers[layer_idx]))
        #[Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.post_init()


class Phi3rCausalLM(Phi3ForCausalLM):
    def __init__(self, config, base_model):
        Phi3PreTrainedModel.__init__(self, config)
        
        self.model = Phi3rModel(config, base_model.model)
        self.vocab_size = config.vocab_size
        self.lm_head = base_model.lm_head # nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

#def wrap_up_lora(base_model, cx = 0):
#    new_model = base_model
#    for layer_idx in range(0, len(base_model.model.layers)):
#        if cx == 0:
#            new_model.model.layers[layer_idx].mlp = LoRAMLP(base_model.model.layers[layer_idx].mlp)
#        else:
#            new_model.model.layers[layer_idx].mlp = LoRAMLP(base_model.model.layers[layer_idx].mlp.original_mlp)
#    return new_model

#def wrap_up_critic(base_model):
#    new_model = base_model
#    new_model.lm_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
#    nn.init.zeros_(new_model.lm_head.weight)
#    return new_model
    
#class Phi4Critic(nn.Module):
#    def __init__(self, base_model, r=8, lora_alpha=1.0):
#        super().__init__()
        
        
#class HydraMLP(nn.Module):
#    def __init__(self, base_model, r=8, lora_alpha=1.0):
#        super(HydraNLP, self).__init__()
#        self.base_model = base_model
#        self.lm_mlp = LoRAMLP(base_model, r, lora_alpha)
#        self.critic_mlp = LoRAMLP(base_model, r, lora_alpha)
#    def forward(self, x1, x2):
#        y1 = self.lm_mlp(x1)
#        y2 = self.critic_mlp(x2)
#        return (y1, y2)
        
        
# parallel KV cache. 

#class Phi4hyMLP(nn.Module):
#    def __init__(self, original_mlp, lora_r=16):
#        super().__init__()
#        self.original_mlp = mlp
#        self.lora_down = nn.Linear(original_mlp.
#        self.config = config
#        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
#        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
#        self.activation_fn = ACT2FN[config.hidden_act]
#    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
#        up_states = self.gate_up_proj(hidden_states)
#        gate, up_states = up_states.chunk(2, dim=-1)
#        up_states = up_states * self.activation_fn(gate)
#        return self.down_proj(up_states)
        
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.lock = threading.Lock()
        
    def add(self, experience, reward):
        """ Add new experience to the buffer """
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (experience, reward)
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """ Sample a batch of experiences from the buffer """
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
            return batch
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)
buffer = ReplayBuffer(1024)

def add_to_buffer(experience, reward):
    #print('[debug] consumer side add.....',  int(os.environ['RANK']) )
    global buffer
    buffer.add(experience, reward)

def len_buffer():
    global buffer
    return len(buffer)

def pop_from_buffer(batchsize):
    global buffer
    return buffer.sample(batchsize)
################################################################################################

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
        device_map='cpu',
        #device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ).to(device)
    #llm_state_dict = llm.state_dict()

    # Load configuration from a pre-trained model
    llm_config = AutoConfig.from_pretrained(model_name)

    actor_model = wrap_up_lora(llm)
    critic_model = wrap_up_critic(wrap_up_lora(llm)) # Phi4LM(llm, r=8, lora_alpha=1.0)
    
    #phi4rllm = Phi4rLM(llm_config)
    
    # to avoid copy two times of model-weight.
    
    #missing_keys, unexpected_keys = phi4rllm.load_state_dict(llm_state_dict, strict=False)
    #print("Missing keys:", missing_keys)
    #print("Unexpected keys:", unexpected_keys)
   
    actor_model = actor_model.to(device)
    critic_model = critic_model.to(device)
    #print(phi4rllm)
    
    llm = actor_model
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
    # Generate response
    #outputs = []
    for epoch in range(0, 100):
        total_reward = 0
        total_count = 0
        
        print('start to trigger play ...........................\n\n')
        for i in range(0, len(train)):
            if i % 16 != local_rank:
                continue
            example = train[i]
            soluts = example['solutions']
            problem = example['description']

            problem = instruction_prefix + problem + instruction_postfix
            
            inputs = tokenizer(problem, return_tensors="pt").to("cuda")
            #print('input_ids.shape', inputs["input_ids"].shape)

            if inputs["input_ids"].shape[1] > 2000:
                continue
            outputs = llm.generate(inputs["input_ids"], max_length=4096)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

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

            buffer_rank = 8
            #rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))
            #time.sleep(1)
            #print('push to buffer ... ') #, data)
        
            #rpc.rpc_sync(f"worker-{buffer_rank}", add_to_buffer, args=(data, reward_score), timeout=0)
            
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

    model = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)

    model.gradient_checkpointing_enable()

    print('done with model creation.')

    dist.init_process_group(backend="nccl", rank=local_rank, world_size=8)
    #dist.init_process_group(backend="nccl", rank)

    print('dist initialization ...', local_rank)

    dist.barrier()

    print('dist barrier success')

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    print('distributed model creation.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0e-6)
    #num_epochs = 3
    num_training_steps = 10000 # num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
    )

    print('model optimization initialization...')

    model.train()

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
    time.sleep(10000)
    
    while step < 40000:
        l = 0 # len(buffer) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", len_buffer, timeout=0) #rev_experience_len('worker2')
        if l > 20:
            torch.cuda.empty_cache()
            
            data = None # buffer.sample(batch_size) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", pop_from_buffer, args=(batch_size, ), timeout=0) #rev_experience_data('worker2', 2)
            text = [d[0] for d in data]
            score = [d[1] for d in data]
            
            inputs = tokenizer(text, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").to(device)
            if inputs["input_ids"].shape[1] > 4096:
                continue

            if inputs['input_ids'].shape[1] < 16:
                continue
                
            #labels = batch["labels"].to(device)
            input_ids = inputs["input_ids"]

            if step == 0:
                print('example input_ids', input_ids)
                
            # Shift input_ids to create labels for next-token prediction
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Mask the last token
            # Return the dictionary with input_ids, attention_mask, and labels
            inputs["labels"] = labels

            batch = {k: v.to(device) for k,v in inputs.items()}
            #print('1. forward', rank, inputs['input_ids'].shape)

            #time.sleep(10)
            outputs = model(**batch)

            loss = outputs.loss
            #print('loss:', loss, 'rank', rank,'step', step, 'shape', inputs['input_ids'].shape)

            #print('2. backward', rank, inputs['input_ids'].shape)
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                #print('3. optimization', rank)
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

    # rpc.init_rpc(f"worker-{rank}", backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29500"))
    
    rpc.init_rpc(f"worker-{rank}", rank=rank, world_size=16) # consider 2 nodes, 16 gpus in this example.
    
    #rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    gpus_per_node = 8
    node_idx = rank // gpus_per_node

    world_size = int(os.environ['WORLD_SIZE'])
    print('WORLD_SIZE', world_size)

    # suppose we use 4 gpus for vllm and 4 gpus 
    if rank in [0,1,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15]:
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

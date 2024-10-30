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

replaybuffer = ReplayBuffer(500000)

# Consumer side: Function to add experiences to its local buffer
def add_to_buffer(experience):
    #print('debug consumer side add.....',  int(os.environ['RANK']) )
    global replaybuffer
    replaybuffer.add(experience)

# rpc communication. 


def play():
    # Load a model
    llm = LLM(model="microsoft/Phi-3-mini-4k-instruct") # "facebook/opt-6.7b")  # You can specify any Hugging Face model here

    #llm.llm_engine.model_executor.driver_workerinit_process_group(
    #            master_address, master_port, rank_offset, world_size, group_name
    #        )
    #  
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024)

    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    outputs = []
    for i in range(0, len(train)):
        example = train[i]
        soluts = example['solutions']
        problem = example['description']

        o = llm.generate([problem], sampling_params)

        completion = o[0].outputs[0].text

        data = problem + completion

        target_rank = 4
        rpc.rpc_sync(f"worker{rank}", add_to_buffer, args=(data,))

        print(ans)
        outputs.append(ans)

def learn():    
    training_config = {
        "bf16": True,
        "do_eval": False,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 1,
        "max_steps": -1,
        "output_dir": "./checkpoint_dir",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 4,
        "per_device_train_batch_size": 4,
        "remove_unused_columns": True,
        "save_steps": 100,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
    }

    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "modules_to_save": None,
    }
    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)

    torch.random.manual_seed(0) 
    model = AutoModelForCausalLM.from_pretrained( 
        "microsoft/Phi-3-mini-4k-instruct",  
        device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ) 

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

    ################
    # Model Loading
    ################
    checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    
    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )

    ###########
    # Training
    ###########
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_test_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )
    train_result = trainer.train()
    metrics = train_result.metrics


##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example




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
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

import re

def preprocess_orm800k_box_responsev1(sequence, answer):
    temp_query = ""
    temp_response = ""

    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]

    temp_response = sequence.split("ASSISTANT:\n")[-1]
    
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"

    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The answer is:")[0]

    #response_list = temp_response.split("<|reserved_special_token_0|>")

    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"

    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"

    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    
    if temp_answer == answer:
        box_match = 1.0 # 0.5
    else:
        box_match = 0.0 # -0.5
    return processed_solution, temp_answer, box_match
    
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
        acc_reward = 0
        acc_num = 0
        for batch_idx, d in enumerate(dataloader):
            qwen_prompt = d['input']
            vanilla_prompt = d['question']    
            vanilla_answer = d['answer']

            # features: ['input', 'answer', 'gt_answer', 'subject', 'level', 'question', 'ground_truth_answer', 'target'],
            #print('qwen_prompt:', qwen_prompt)
            #print('vanilla_prompt:', vanilla_prompt)

            candidate_prompt_1 = "Think step by step and provide the final answer at the end in this format: 'The final answer is: <your_answer>.'\n"

            candidate_prompt_2 = "Think step by step and conclude with the final answer in this format: 'The final answer is: <your_answer>.' Ensure <your_answer> is the simplest form. \n\n"

            candidate_prompt_3 = "Think step by step and provide the answer at the end in this format: 'The answer is: <your_answer>.'\n"

            pattern = r'The answer is:\s*(.*?)\s*\.'
            #pattern = r'The final answer is:\s*(.*?)\s*\.'

            prefix_instruct = candidate_prompt_3
            postfix_instruct = ''
            #instruction_prefix = ''
            #instruction_postfix = '\n\nplease only reply with the source code in python. \n'

            #['<|im_start|>system\n<|im_end|>\n<|im_start|>user\nLet $\\mathbf{A}$ and $\\mathbf{B}$ be $2 \\times 2$ matrices such that $\\det \\mathbf{A} = -1$ and $\\det \\mathbf{B} = 3.$  Find $\\det (3 \\mathbf{A} \\mathbf{B}).$<|im_end|>\n<|im_start|>assistant']

            prompt = prefix_instruct + vanilla_prompt[0] + postfix_instruct
            
            x1 = tokenizer([prompt], add_special_tokens=False, max_length=1024, truncation=True)
            #print('qwen_ids1:', x1['input_ids'])

            #x2 = tokenizer(qwen_prompt, add_special_tokens=False, max_length=16, truncation=True)
            #print('qwen_ids2:', x2['input_ids'])

            #y1 = tokenizer(vanilla_prompt, add_special_tokens=False, max_length=1024, truncation=True)
            #print('vanilla_ids:', y1['input_ids'])

            input_ids = x1['input_ids']
            
            outputs, probs, crits = llm.generate(input_ids, max_gen_len = 3000)
            
            response = [tokenizer.decode(outputs[0], skip_special_tokens=True)]

            #pattern = r'The final answer is:\s*(?:\[(.*?)\]|(\d+)|"(.*?)"|(\$?\\frac{\d+}{\d+}\$?))'
            #pattern = r'The final answer is:\s*(.*?)\s*\.'

            #pattern = r"The final answer is: \[(.*?)\]"
            
            #pattern = r"The answer is: \\boxed\{(.*?)\}"
            #missing_answer_indices = [
            #    i for i, query in enumerate(response) if not re.search(pattern, query, re.DOTALL)
            #]

            processed_queries = []
            box_match_list = []
            for query, answer in zip(response, vanilla_answer):
                match = re.search(pattern, query, re.DOTALL)
                p_answer = "none"
                box_match = 0.0
                p_query = query
                if match:
                    extracted_answer = match.group(1) #or match.group(2) or match.group(3) or match.group(4)
                    #print("Extracted Answer:", extracted_answer)
                    p_answer = extracted_answer
                    if p_answer == answer:
                        box_match = 1.0 # 0.5
                        
                    pos = match.end() 
                    p_query = query[:pos]
                    #print(r[:pos])
                    
                #else:
                #    print("No match found.")
                    
                #if match:
                #    extracted_answer = match.group(1)  # Extracts "3, 4, 5, 6, 7"   
                #    box_match = 1.0
                #    #print("Extracted Answer:", extracted_answer)
                #query, p_answer, box_match = preprocess_orm800k_box_responsev1(query, answer)
                
                processed_queries.append(p_query)
                box_match_list.append(box_match)

                acc_reward = acc_reward + box_match
                acc_num = acc_num + 1
            #queries = processed_queries
            #reward_sequences, reward_attention_mask = preprocess_prm_reward(queries, self.tokenizer, self.prompt_max_len, **generate_kwargs)

                if local_rank == 0:
                    print('batch idx', batch_idx)
                    print('\n\n\nraw question: ************\n')
                    print(prompt)
                    print('\n\n\nraw response: *************\n')
                    print(p_query)
                    print('\n\n\npredict answer: ************\n')
                    print(p_answer)
                    print('\n\n\nground truth: *************\n')
                    print(vanilla_answer)
                    print('\n\n\nmatch: **********\n')
                    print(box_match)
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
    main()

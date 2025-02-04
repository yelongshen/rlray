
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
import argparse

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

import random

from collections import deque

from math_util import compare_math_answers, process_math_prompt, process_math_answer

import numpy as np

from safetensors.torch import load_file

# first process all the wards. 

def train(args, llm, llm_config, optimizer, scheduler, buffer, buffer_size, device):
    llm.train()    
    #critic_loss = 0.0
    #policy_loss = 0.0
    #mini_c_loss = 0.0
    #mini_p_loss = 0.0
    # update_step = 0

    # processing reward.
    buffer.calculate_advantage()
    buffer.distributed_advantage_norm(device)
  
    # accumulate gradient for the whole batchsize.     
    step = 0
    batch_size = 1
    max_seq_len = 4096
    micro_training_steps = buffer_size / batch_size
    
    # 
    pad_id = llm_config.pad_token_id
    vocab_size = llm_config.vocab_size
    
    optimizer.zero_grad()
    mseLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
    mini_policy_loss = 0
    mini_critic_loss = 0
    # rl training steps;
    while step < micro_training_steps:
        mini_data = buffer.pop(batch_size)

        input_tokens = [d.tokens for d in mini_data]
        old_logprobs = [d.probs for d in mini_data]
        #advantages = [d.advantages for d in mini_data]
        advantages = [d.normalized_advantages for d in mini_data]
        returns = [d.returns for d in mini_data]
      
        # do tokens padding & alignment with batchsize  > 1   
        input_tokens = torch.tensor(input_tokens).to(torch.long).to(device)    
        old_logprobs = torch.tensor(old_logprobs).to(device)
        advantages = torch.tensor(advantages).to(device).detach()
        returns = torch.tensor(returns).to(device).detach()
      
        _batch_size, _seq_len = input_tokens.shape
        # generation token index. 
        _response_idx = mini_data[0].masks.index(1)

        # re-evaluate the policy.     
        # return: next_token_loss, logits, critics, next_decoder_cache 
        _, logits, critics, _ = llm(input_tokens)
    
        logprobs = -F.cross_entropy(
            input = logits.reshape(-1, vocab_size)[:-1,:], #.transpose(1, 2),
            target = input_tokens.reshape(-1)[1:], 
            reduction = "none",
            ignore_index = pad_id,
        ).reshape(1, -1)

        # critics align with the ground truth. 
        critics = critics.reshape(_batch_size, _seq_len)
        critics = critics[:, _response_idx-1:-1] 
        logprobs = logprobs[:, _response_idx-1:]

        # we shall do advantage normalization. 
        ratios = torch.exp(logprobs - old_logprobs.detach() + 1e-10)
        
        eps_clip = 0.2
        surr1 = ratios * advantages       
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        _policy_loss = -torch.min(surr1, surr2).mean() 
        _critic_loss = mseLoss(critics, returns).mean() 

        _total_loss = (_policy_loss + 0.01 * _critic_loss) / micro_training_steps 
      
        # final loss of clipped objective PPO objective. 
            
        # take gradient step
        mini_critic_loss = mini_critic_loss + _critic_loss.detach() / micro_training_steps 
        mini_policy_loss = mini_policy_loss + _policy_loss.detach() / micro_training_steps
            
        _total_loss.backward()
        step = step + 1
        if step % micro_training_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    return mini_policy_loss, mini_critic_loss
        

        

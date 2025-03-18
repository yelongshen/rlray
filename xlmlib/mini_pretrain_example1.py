import os
import io
import pickle
import traceback
import copy
import datetime
import sys
import threading
import time
import random
import argparse
import signal
import psutil  # To check process status before killing
import re
import multiprocessing
import logging
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import numpy as np
from queue import Queue
from typing import List, Optional, Tuple, Union, Any, Dict, Optional
import concurrent.futures
from concurrent.futures import TimeoutError
from functools import partial
from contextlib import redirect_stdout
from dataclasses import dataclass
from collections import deque


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import AdamW

from accelerate import Accelerator

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from safetensors.torch import load_file

from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from transformers import get_linear_schedule_with_warmup
from transformers.activations import ACT2FN


#from replaybuffer import ReplayBuffer, Sample, AsyncReplayBuffer
#from ppo import ppo_gradient, ppo_gradient_v2
#from sft import sft_gradient
#from math_util import compare_math_answers, process_math_prompt, process_math_answer

class fabric:
    world_size = 0
    local_rank = 0
    rank = 0
    device = None
    
    @classmethod
    def create(cls, world_size, local_rank, global_rank, device):
        cls.world_size = world_size
        cls.local_rank = local_rank
        cls.rank = global_rank
        cls.device = device

def sync_model_weights(model):
    """Broadcast model weights from rank 0 to all other processes."""
    for param in model.state_dict().values():
        dist.broadcast(param, src=0)  # Send parameters from rank 0 to all ranks
        
def main(args):
    ########################################### on-policy ppo experiments with phi3.5 model on math dataset. 
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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)    

    # Step 3: Load and merge multiple safetensor state_dicts
    dist.barrier()

    fabri.create(world_size, local_rank, rank, device)
        
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.model_type == 'tformer400m':
        from tformer import _TformerForCausalLM
        llm_model, llm_config = _TformerForCausalLM.init_400m()
            
    # Load tokenizer from local path 
    llm_model = llm_model.to(torch.bfloat16).to(device) 
    llm_model.model.gradient_checkpointing = False 
    dist.barrier()
    
    sync_model_weights(llm_model)    
    llm = llm_model 
    print('initial llm model ....') 
    # setup model distribution.
    llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[local_rank]) 
    print('distributed language model creation.') 

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="none", help="path to pretraining dataset.")
    parser.add_argument("--model_type", type=str, default="tformer400m", choices=["tformer400m", "xformer400m"], help="choose model type.")
    
    parser.add_argument("--warmup_step", type=float, default=0.1, help="warmup steps.")
    parser.add_argument("--lr", type=float, default=1e-6, help="peak learning rate.")
    parser.add_argument("--epoch", type=int, default=30, help="number of epoches.")
    
    parser.add_argument("--save_per_steps", type=int, default=40, help="save ckpt per steps.")
    parser.add_argument("--save_ckpt", type=str, default=None, help="path to save ckpt.")
    
    args = parser.parse_args()
    
    main(args)

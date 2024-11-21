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

#from vllm import LLM, SamplingParams
from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError
from functools import partial
#from contextlib import redirect_stdout
import sys

#from datasets import load_dataset

import torch 
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

import logging

#from peft import LoraConfig
#from trl import SFTTrainer
#from transformers import TrainingArguments, BitsAndBytesConfig

from accelerate import Accelerator
from torch.utils.data import DataLoader
#from transformers import AdamW
#import numpy as np 

#from transformers import get_linear_schedule_with_warmup
#from torch.optim import AdamW


def main():
    # system parameters:
    # args.ngpu_per_node
    # args.nnode_actor
    # args.nnode_learner
    #world_size = 8
    rank = int(os.environ['RANK'])
    
    print('rank', rank)

    #rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # suppose we use 4 gpus for vllm and 4 gpus 
    #if rank in [0]:
    #    print('rank', rank, 'play')
    #    play()

    #if rank in [1,2,3,4,5,6,7]:
    #    for i in range(0, 1000000):
    #        print('rank', rank, 'sleep.....')
    #        time.sleep(1)
    #    learn()

if __name__ == "__main__":
    main()
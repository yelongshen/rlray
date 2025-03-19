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

from transformers import get_linear_schedule_with_warmup
from transformers.activations import ACT2FN


import glob
from packed_dataset import PackedDataset
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


def create_dataloader(
    batch_size: int, block_size: int, data_dir, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    filenames = sorted(glob.glob(f'{data_dir}/{split}*'))
    random.seed(seed)
    random.shuffle(filenames)
    print('length of files', len(filenames), split)

    #rank = int(os.environ['RANK']) 
    #world_size = int(os.environ['WORLD_SIZE']) 

    dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size = block_size,
            shuffle=shuffle,
            seed=seed+fabric.rank,
            num_processes=fabric.world_size,
            process_rank=fabric.rank,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    #return dataset

#encoder : transformer (i.e., 4 layers)
#decoder : transformer (i.e., 2/0 layers)
#recurr : transformer (i.e., 2 layers)
#for seq_data (4k length) in training:
#    state = encoder(data)
#    decode_state = []
#    past_state = None
#    #split state into chunks (i.e., 128 token per chunk)
#    for c_i in state:
#        new_states, new_c_i = recurr([past_state, c_i]) # we can set max_recurr_step (T) here.
#        past_states = [new_states, new_c_i]
#        decode_state += [new_c_i]
#    logits = decoder(decode_state)
#    next_token_loss(logits, data)
    
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

    fabric.create(world_size, local_rank, rank, device)
        
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.model_type == 'tformer400m':
        from xlmlib.tformer import _TformerForCausalLM
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

    # setup optimization.
    #optimizer = torch.optim.AdamW(llm.parameters(), lr=args.lr) # 1.0e-6) 
    
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    optimizer = torch.optim.AdamW(
        llm.parameters(), lr=args.lr, weight_decay=weight_decay, betas=(beta1, beta2), fused=True)
    num_training_steps = args.num_training_step #dataset['train'].num_rows * args.epoch * args.n_rollout * 1.0 / (args.replay_size * world_size) # num_epochs * len(train_dataloader)    
    warmup_steps = args.warmup_step * num_training_steps
    scheduler = get_linear_schedule_with_warmup( 
        optimizer, num_warmup_steps = int(warmup_steps), num_training_steps = int(num_training_steps) 
    )     
    print('distributed model optimization created.')

    seq_len = 4096
    train_loader = create_dataloader(args.micro_batch_size, seq_len+1, args.data, split='train')
    
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    loss_scalar = 1.0 / gradient_accumulation_steps
    
    def gradient_pass(input, target, scalar):
        loss, _, _ = llm(input_ids=input, labels=target, fuse_loss=args.fuse_loss)
        avg_loss = loss.mean()
        (avg_loss * scalar).backward()
        return avg_loss.detach()
        

    micro_loss_log = 0
    micro_step = 0
    step_log = 20 # print loss every 100 steps.
    avg_loss_log = 0
    avg_step = 0
    for data_idx, train_data in enumerate(train_loader):
        if data_idx >= num_training_steps * gradient_accumulation_steps:
            break
        #print('data', data.shape, data) 
        input_ids = train_data[:, 0 : seq_len].contiguous().to(fabric.device)
        targets = train_data[:, 1 : seq_len + 1].contiguous().to(fabric.device)

        is_grad_sync = (data_idx + 1) % gradient_accumulation_steps == 0
        if is_grad_sync:
            _loss = gradient_pass(input_ids, targets, loss_scalar)
        else:
            with llm.no_sync():
                _loss = gradient_pass(input_ids, targets, loss_scalar)

        if args.debug and fabric.rank == 0:
            print('one pass loss...', _loss)
            
        micro_loss_log = micro_loss_log + _loss
        micro_step = micro_step + 1
        if is_grad_sync:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if (scheduler._step_count+1) % args.save_per_steps == 0 and rank == 0:
                checkpoint = {
                        "step": scheduler._step_count,
                        "model_state_dict": llm.module.state_dict(),  # Remove DDP wrapper
                        }
                save_path = f"{args.save_ckpt}/ckpt_{scheduler._step_count}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"Checkpoint saved at: {save_path}")
            
            if (scheduler._step_count+1) % step_log == 0:
                micro_loss_log = micro_loss_log / micro_step
                
                dist.all_reduce(micro_loss_log, op=dist.ReduceOp.SUM)  # Sum losses across GPUs
                micro_loss_log = micro_loss_log / fabric.world_size

                avg_loss_log = avg_loss_log + micro_loss_log
                avg_step = avg_step + 1
            
                if rank == 0:  # Print only on rank 0
                    print(f"data: {data_idx + 1}, update: {scheduler._step_count + 1}, lr: {scheduler.get_last_lr()}, loss: {micro_loss_log.item():.4f}, avg_loss: {avg_loss_log.item() / avg_step:.4f}")

                micro_loss_log = 0
                micro_step = 0

        
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="none", help="path to pretraining dataset.")
    parser.add_argument("--micro_batch_size", type=int, default=4, help='batch size.')
    parser.add_argument("--batch_size", type=int, default=256, help='overall batch size.')
    
    parser.add_argument("--model_type", type=str, default="tformer400m", choices=["tformer400m", "xformer400m"], help="choose model type.")
    
    parser.add_argument("--num_training_step", type=int, default=100000, help="number of training step.")
    parser.add_argument("--warmup_step", type=float, default=0.1, help="warmup steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="peak learning rate.")
    parser.add_argument("--epoch", type=int, default=30, help="number of epoches.")
    
    parser.add_argument("--save_per_steps", type=int, default=1000, help="save ckpt per steps.")
    parser.add_argument("--save_ckpt", type=str, default=None, help="path to save ckpt.")
    parser.add_argument('--fuse_loss', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    
    main(args)

import inspect
import math
import warnings
import json
from typing import List, Optional, Tuple, Union
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from einops import rearrange, repeat
from torch import nn

import threading
from dataclasses import dataclass


from dataclasses import dataclass
from .replaybuffer import AsyncReplayBuffer

@dataclass
class Request:
    id : int
    prompt : str
    prompt_tokens : List[int]

# distributed inference engine.
# usage:
# on multinode:
# _engine = _inference_engine(model, rank, world_size)
# _engine.start() # start on thread. 
# on rank_0: 
#    load prompt set
#    for prompt 
#    _engine.pool.add(request)
#    
# on rank_0:
# for 
# prompt: text --> speech , text LLM 
# processing _engine.results
class _inference_engine:
    def __init__(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size

    def create_request_pool(self, requests):
        self.request_buffer = AsyncReplayBuffer(len(requests))
        for i in range(0, len(requests)):
            self.request_buffer.push(requests[i])
        # add buffer for results.
        self.result_buffer = AsyncReplayBuffer(len(requests))
        
    def start(self):        
        def perform_request():
            # Simulate a time-consuming calculation
            print("inference started...")
            while len(self.request_buffer) > 0:
                 data, avg_reward, ema_reward = pop_from_buffer(batch_size) if rank == buffer_rank else rpc.rpc_sync(f"worker-{buffer_rank}", pop_from_buffer, args=(batch_size, ), timeout=10)
            
            return result
        
        # Create a Thread object targeting the perform_calculation function
        self.inference_thread = threading.Thread(target=perform_request)
        # Start the thread
        self.inference_thread.start()

    def join(self):
        # Wait for the thread to complete
        self.inference_thread.join()

        
        

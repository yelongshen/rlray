import os
import sys
import io

import threading
import signal
import time
import random
import logging

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
from queue import Queue

from concurrent.futures import TimeoutError
from functools import partial

from accelerate import Accelerator
from dataclasses import dataclass
import torch

import numpy as np
import math
import torch.distributed.rpc as rpc

@dataclass
class Sample:
    prompt : str
    response : str
    reward : float
    tokens : List[int]
    masks : List[int]
    norm_reward: float = None
    probs : Optional[List[float]] = None
    crits : Optional[List[float]] = None
    seq_rewards : Optional[List[float]] = None
    advantages : Optional[List[float]] = None
    returns : Optional[List[float]] = None
    normalized_advantages : Optional[List[float]] = None

# ReplayBuffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # every sample is very different. 
        self.epsilon = 1e-8
        self.alpha = 0.01       

    # experience : #<prompt, response, reward, probs, crits, tokens, masks, seq_rewards>
    def push(self, experience):
        """ Add new experience to the buffer """
        #with self.lock:
        self.buffer.append(experience) 
            
    def clear(self):
        self.buffer.clear()

    def pop(self, batch_size):
        """
        Pop the oldest batch of experiences from the buffer (FIFO order).
        Args:
            batch_size (int): Number of experiences to pop.
        Returns:
            List of popped experiences.
        """
        #with self.lock:
        batch_size = min(batch_size, len(self.buffer))  # Ensure we don't pop more than available
        data = [self.buffer.popleft() for _ in range(batch_size)]  # Pop oldest elements
        return data
        
    def __len__(self):
        return len(self.buffer)

    def calculate_group_passk(self, group=8):
        assert len(self.buffer) % group == 0 
        rewards = [d.reward for d in self.buffer]
        
        passk = 0
        for g in range(0, len(self.buffer) // group):
            _rewards = rewards[g * group: (g+1) * group]
            _all_reward = np.sum(_rewards)

            if _all_reward > 0:
                passk += 1
        return passk, len(self.buffer) // group
        
    def calculate_positive_advantage(self, gamma=0.995, group = 8):
        assert len(self.buffer) % group == 0 
        rewards = [d.reward for d in self.buffer]

        norm_rewards = []
        for g in range(0, len(self.buffer) // group):
            _rewards = rewards[g * group: (g+1) * group]
            _all_reward = np.sum(_rewards)
            
            _norm_rewards = [ (_r / (_all_reward + 1e-2)) for _r in _rewards]  # _norm(_rewards)
            norm_rewards = norm_rewards + _norm_rewards
            
        for idx, d in enumerate(self.buffer):
            d.norm_reward = norm_rewards[idx]

        for d in self.buffer:
            #prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d
            acc_reward = d.reward
            d.advantages = []
            d.returns = []
            for r, c in zip(reversed(d.seq_rewards), reversed(d.crits)):
                acc_reward = gamma * acc_reward #+ d.norm_reward
                #advantage = acc_reward # - c
                d.advantages.insert(0, d.norm_reward)
                d.returns.insert(0, acc_reward)
            
            d.normalized_advantages = d.advantages
       
        
    def calculate_group_advantage(self, gamma=0.995, group = 8):
        assert len(self.buffer) % group == 0 
        rewards = [d.reward for d in self.buffer]

        def _norm(x):
            _mean = np.mean(x)
            _l2 = [(_r - _mean) **2 for _r in x]
            return [(_r - _mean) / math.sqrt(np.sum(_l2)/len(x) + 1e-4) for _r in x]

        norm_rewards = []
        for g in range(0, len(self.buffer) // group):
            _rewards = rewards[g * group: (g+1) * group]
            _norm_rewards = _norm(_rewards)
            norm_rewards = norm_rewards + _norm_rewards
            
        for idx, d in enumerate(self.buffer):
            d.norm_reward = norm_rewards[idx]

        for d in self.buffer:
            #prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d
            acc_reward = d.reward
            d.advantages = []
            d.returns = []
            for r, c in zip(reversed(d.seq_rewards), reversed(d.crits)):
                acc_reward = gamma * acc_reward #+ d.norm_reward
                #advantage = acc_reward # - c
                d.advantages.insert(0, d.norm_reward)
                d.returns.insert(0, acc_reward)

            d.normalized_advantages = d.advantages
            
    def calculate_advantage(self, gamma=0.9995):
        for d in self.buffer:
            #prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d
            acc_reward = 0
            d.advantages = []
            d.returns = []
            for r, c in zip(reversed(d.seq_rewards), reversed(d.crits)):
                acc_reward = gamma * acc_reward + r 
                advantage = acc_reward - c
                d.advantages.insert(0, advantage)
                d.returns.insert(0, acc_reward)
            
    def distributed_advantage_norm(self, device, dist):
        world_size = dist.get_world_size()
        full_advantages = [adv for dat in self.buffer for adv in dat.advantages]
        _sum = torch.tensor([np.sum(full_advantages)], dtype=torch.float, device = device)
        _count = torch.tensor([len(full_advantages)], dtype=torch.float, device = device)
        dist.all_reduce(_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(_count, op=dist.ReduceOp.SUM)
        _global_mean = _sum / _count

        print('total count of tokens:', _count, _global_mean)
        _global_mean_value = _global_mean[0].item()
        
        l2_advantages = [(adv - _global_mean_value) ** 2 for adv in full_advantages]
        
        _sq = torch.tensor([np.sum(l2_advantages)], dtype=torch.float, device = device)        
        dist.all_reduce(_sq, op=dist.ReduceOp.SUM)
        _global_variance = _sq / _count
        _global_std = torch.sqrt(_global_variance)

        _global_std_value = _global_std[0].item()
        for d in self.buffer:
            d.normalized_advantages = []
            for adv in d.advantages:
                d.normalized_advantages.append( (adv - _global_mean_value) / (_global_std_value + 1e-2))
    
    def mean_reward(self):
        rewards = [d.reward for d in self.buffer]
        return np.mean(rewards)

    def avg_responselen(self):
        response_len = [len(d.probs) for d in self.buffer]
        return np.mean(response_len)
            
    def z_score_normalization(self):
        """Standardize rewards using mean and standard deviation."""
        rewards = [d.reward for d in self.buffer]
        mean = np.mean(rewards)
        std = np.std(rewards) + self.epsilon
        return (rewards - mean) / std

class AsyncReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.lock = threading.Lock()

    def push(self, data):
        """ Add new experience to the buffer """
        with self.lock:
            super().push(data) 

    def pop(self, batchsize):
        with self.lock:
            x = super().pop(batchsize)
            if len(x) == 0: #< batchsize:
                return [None]
            else:
                return x

    def __len__(self):
        with self.lock:
            return super().__len__()

# the class is served as Rpc factory.
class RpcReplayBuffer(AsyncReplayBuffer):
    RpcFactory = {}
    RpcMain = {}
    
    def __init__(self, capacity, main_worker):
        super().__init__(capacity)
        self.main_worker = main_worker
        
    @staticmethod
    def Register(buffer_name, main_worker, is_main, capacity=0):
        if is_main:
            RpcReplayBuffer.RpcFactory[buffer_name] = RpcReplayBuffer(capacity, main_worker)
        else:
            RpcReplayBuffer.RpcMain[buffer_name] = main_worker

    @staticmethod
    def Push(buffer_name, data):
        if buffer_name in RpcReplayBuffer.RpcFactory:
            RpcReplayBuffer.RpcFactory[buffer_name].push(data)
        else:
            main_worker = RpcReplayBuffer.RpcMain[buffer_name]
            rpc.rpc_async(main_worker, RpcReplayBuffer.Push, args=(buffer_name, data), timeout=0)

    @staticmethod
    def Pop(buffer_name):
        if buffer_name in RpcReplayBuffer.RpcFactory:
            return RpcReplayBuffer.RpcFactory[buffer_name].pop(1)[0]
        else:
            main_worker = RpcReplayBuffer.RpcMain[buffer_name]

            future = rpc.rpc_async(main_worker, RpcReplayBuffer.Pop, args=(buffer_name,))
            try:
                return future.wait(timeout=60)  # Wait at most 2 seconds
            except: # RuntimeError:  # Handle timeout
                return None
            
            #return rpc.rpc_sync(main_worker, RpcReplayBuffer.Pop, args=(buffer_name, ), timeout=0)
            #try:
            #    return rpc.rpc_sync(main_worker, RpcReplayBuffer.Pop, args=(buffer_name,), timeout=5)  # Set timeout to 2s
            #except:  # Catch timeout exception
            #    return None
            
    @staticmethod
    def Length(buffer_name):
        if buffer_name in RpcReplayBuffer.RpcFactory:
            return len(RpcReplayBuffer.RpcFactory[buffer_name])
        else:
            main_worker = RpcReplayBuffer.RpcMain[buffer_name]
            return rpc.rpc_sync(main_worker, RpcReplayBuffer.Length, args=(buffer_name, ), timeout=10)

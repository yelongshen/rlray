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

@dataclass
class Sample:
    prompt : str
    response : str
    reward : float
    probs : List[float]
    crits : List[float]
    tokens : List[int]
    masks : List[int]
    seq_rewards : List[float]
    advantages : Optional[List[float]] = None
    returns : Optional[List[float]] = None
    normalized_advantages : Optional[List[float]] = None
    norm_reward: float = None
# ReplayBuffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # every sample is very different. 
        self.epsilon = 1e-8
        self.alpha = 0.01       
        #self.lock = threading.Lock()

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

    def group_advantage(self, group = 8):
        assert len(self.buffer) % group == 0 
        rewards = [d.reward for d in self.buffer]

        def _norm(x):
            _mean = numpy.mean(x)
            _l2 = [(_r - _mean) **2 for _r in x]
            return [(_r - _mean) / math.sqrt(np.sum(_l2)/len(x) + 1e-4) for _r in x]

        norm_rewards = []
        for g in range(0, len(self.buffer) / group):
            _rewards = rewards[g * group: (g+1) * group]
            _norm_rewards = _norm(_rewards)
            norm_rewards = norm_rewards + _norm_rewards
            
        for idx, d in enumerate(self.buffer):
            d.norm_reward = norm_rewards[idx]
            
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


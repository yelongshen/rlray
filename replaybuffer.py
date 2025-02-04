
import os
import sys

import torch
import torch.distributed as dist
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
import logging

#from peft import LoraConfig
#from trl import SFTTrainer
#from transformers import TrainingArguments, BitsAndBytesConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader

import os
import io
import pickle
import traceback
import copy
import datetime
from typing import Any, Dict, Optional
from functools import partial
import sys


import signal
from dataclasses import dataclass


import re

from collections import deque

import numpy as np
from typing import Optional

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
# ReplayBuffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # every sample is very different. 
        self.epsilon = 1e-8
        self.alpha = 0.01       
        self.lock = threading.Lock()

    # experience : #<prompt, response, reward, probs, crits, tokens, masks, seq_rewards>
    def push(self, experience):
        """ Add new experience to the buffer """
        with self.lock:
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
        with self.lock:
            batch_size = min(batch_size, len(self.buffer))  # Ensure we don't pop more than available
            data = [self.buffer.popleft() for _ in range(batch_size)]  # Pop oldest elements
            return data
        
    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def get_rewards(self):
        with self.lock:
            rewards = []
            for d in self.buffer:
                rewards.append(d.reward)
            return rewards

    def calculate_advantage(self, gamma=0.9995):
        with self.lock:
            #d.advantages = []
            #d.returns = [] 
            #full_advantages = []
            #full_returns = []
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
            
    def distributed_advantage_norm(self, device):
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
        rewards = self.get_rewards()
        return np.mean(rewards)

    def avg_responselen(self):
        with self.lock:
            response_len = []
            for d in self.buffer:
                #prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d
                response_len.append(len(d.probs))
            return np.mean(response_len)
        
    def z_score_normalization(self):
        """Standardize rewards using mean and standard deviation."""
        rewards = self.get_rewards()
        
        mean = np.mean(rewards)
        std = np.std(rewards) + self.epsilon
        return (rewards - mean) / std


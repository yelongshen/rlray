
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
                prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d
                rewards.append(reward)
            return rewards

    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage =  (rewards[t]- values[t]) + 
                                              gamma * lambda_ * (1 - dones[t]) * last_advantage +  
                                              gamma * values[t + 1] * (1 - dones[t]) 
        returns = advantages + values[:-1]  # Compute targets for value function
        return advantages, returns
        
    def calculate_advantage(self, gamma=0.9995):
        with self.lock:
            full_advantages = []
            full_returns = []
            for d in self.buffer:
                prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d

                acc_reward = 0
                _advantages = []
                _returns = []
                for r, c in zip(reversed(seq_rewards), reversed(ctits)):
                    acc_reward = gamma * acc_reward + r 
                    advantage = acc_reward - c
                    _advantages.insert(0, advantage)
                    _returns.insert(0, acc_reward)
                    
                full_advantages.append(_advantages)
                full_returns.append(_returns)
                
            return full_advantages, full_returns

    
            
    def mean_reward(self):
        rewards = self.get_rewards()
        return np.mean(rewards)

    def avg_responselen(self):
        with self.lock:
            response_len = []
            for d in self.buffer:
                prompt, response, reward, probs, crits, tokens, masks, seq_rewards = d
                response_len.append(len(probs))
            return np.mean(response_len)
        
    def z_score_normalization(self):
        """Standardize rewards using mean and standard deviation."""
        rewards = self.get_rewards()
        
        mean = np.mean(rewards)
        std = np.std(rewards) + self.epsilon
        return (rewards - mean) / std


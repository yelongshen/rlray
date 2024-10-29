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

from vllm import LLM, SamplingParams
from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError
from functools import partial
from contextlib import redirect_stdout
import sys

from datasets import load_dataset

def play():
    # Load a model
    llm = LLM(model="microsoft/Phi-3-mini-4k-instruct") # "facebook/opt-6.7b")  # You can specify any Hugging Face model here
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=50)

    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    outputs = []
    for i in range(0, len(train)):
        example = train[i]
        soluts = example['solutions']
        problem = example['description']

        ans = llm.generate([problem], sampling_params)
        print(ans)
        outputs.append(ans)

def main():
    # system parameters:
    # args.ngpu_per_node
    # args.nnode_actor
    # args.nnode_learner
    play()

if __name__ == "__main__":
    main()
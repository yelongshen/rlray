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


def main():
    # system parameters:
    # args.ngpu_per_node
    # args.nnode_actor
    # args.nnode_learner
    
    # Load a model
    llm = LLM(model="facebook/opt-6.7b")  # You can specify any Hugging Face model here

    # Define your prompt
    prompt = "Once upon a time in a futuristic world,"

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)

    # Run inference
    output = llm.generate([prompt], sampling_params)

    # Print the output
    print(output)


if __name__ == "__main__":
    main()
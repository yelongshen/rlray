import os
import time
import random

import ray
import torch
import torch.distributed as dist

import socket

import subprocess
import re


def start_ray_head():
    """
    Launches "ray start --head" using subprocess,
    captures its output, and returns the process and any discovered cluster address.
    """

    # 1) Build the command
    cmd = ["ray", "start", "--head", "--port=6379"]

    # 2) Run the command
    #    - We use Popen if we want the process to stay alive,
    #      so we can later stop it or parse its output live.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True  # so we get string output
    )

    # 3) Read the output line by line to find the head address
    head_address = None
    while True:
        line = process.stdout.readline()
        if not line:  # means the process ended or no more output
            break

        print("[ray start output] ", line.strip())

        # "ray start" often prints a line like:
        # "Started a Ray head node at address 192.168.1.10:6379"
        match = re.search(r"address\s+([0-9.]+:\d+)", line)
        if match:
            head_address = match.group(1)

        # If ray start has effectively "finished" initialization,
        # it prints instructions or sits waiting for requests.

        # You can decide how to detect readiness. Some users:
        #   - look for a particular line
        #   - or just sleep a bit

        # For simplicity, let's break once we see "address"
        if head_address is not None:
            break
    return process, head_address

def start_ray_worker(address):
    
    """
    Launches "ray start --head" using subprocess,
    captures its output, and returns the process and any discovered cluster address.
    """
    # ray start --address='10.0.0.5:6380'
    # 1) Build the command
    cmd = ["ray", "start", f"--address={address}:6379"]

    # 2) Run the command
    #    - We use Popen if we want the process to stay alive,
    #      so we can later stop it or parse its output live.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True  # so we get string output
    )

    # 3) Read the output line by line to find the head address
    head_address = None
    while True:
        line = process.stdout.readline()
        if not line:  # means the process ended or no more output
            break

        print("[ray start output] ", line.strip())

        # "ray start" often prints a line like:
        # "Started a Ray head node at address 192.168.1.10:6379"
        match = re.search(r"address\s+([0-9.]+:\d+)", line)
        if match:
            head_address = match.group(1)

        # If ray start has effectively "finished" initialization,
        # it prints instructions or sits waiting for requests.

        # You can decide how to detect readiness. Some users:
        #   - look for a particular line
        #   - or just sleep a bit

        # For simplicity, let's break once we see "address"
        if head_address is not None:
            break
    return process, head_address

# this function is called after dist.init_process_group
def ray_init(global_rank, local_rank, world_size):
    # We'll create a Ray cluster only once (on rank 0).
    # Then every other rank will connect to it.
    # If you have multiple nodes, node_rank=0 with global_rank=0
    # typically means "primary process on the first node."

    # ------------------------------------------------------------------
    # (C) Start Ray on rank 0, node 0
    # ------------------------------------------------------------------
    #head_address = None  # We will fill this in if we are rank 0
    is_rank0 = (global_rank == 0)
    head_address = None
    if is_rank0:
        head_address = socket.gethostbyname(socket.gethostname())
        print('ip_address:', head_address)
    # ------------------------------------------------------------------
    # (D) Broadcast the Ray head address to all ranks
    # ------------------------------------------------------------------
    # We'll do a simple PyTorch broadcast from rank 0 → everyone else.
    # 1. Convert string to bytes
    # 2. Make a fixed-size tensor for broadcast
    # (If you want a more robust approach, you can store in C10d store or
    #  handle variable lengths, but here's a basic example.)
    max_len = 200  # assume address won't exceed 200 chars
    address_tensor = torch.ByteTensor(max_len).fill_(0).cuda()
    if is_rank0 and head_address is not None:
        encoded = head_address.encode("utf-8")
        address_tensor[: len(encoded)] = torch.ByteTensor(list(encoded)).cuda()
    dist.broadcast(address_tensor, src=0)
    head_address = address_tensor.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00")
    # Barrier to ensure everyone has the same info now
    dist.barrier()

    is_node0 = local_rank == 0

    if is_node0:
        if is_rank0:
            start_ray_head()
        else:
            start_ray_worker(head_address)
    dist.barrier()
    init_info = ray.init(address="auto")
    print(init_info)
    print(ray.cluster_resources()) 
    print(ray.runtime_context)
    dist.barrier()
  

import os
import time
import random

import ray
import torch
import torch.distributed as dist

import socket


# this function is called after dist.init_process_group
def ray_init(global_rank, world_size):
    # We'll create a Ray cluster only once (on rank 0).
    # Then every other rank will connect to it.
    # If you have multiple nodes, node_rank=0 with global_rank=0
    # typically means "primary process on the first node."

    # ------------------------------------------------------------------
    # (C) Start Ray on rank 0, node 0
    # ------------------------------------------------------------------
    head_address = None  # We will fill this in if we are rank 0
    is_rank0 = (global_rank == 0)
    if is_rank0:
        # Start a Ray HEAD on the first node
        # We'll pick a port or let Ray choose automatically
        print(f"[global_rank={global_rank}] Starting Ray ...")

        # (C1) Start Ray HEAD
        # You could specify `node_ip_address`, `dashboard_port`, `resources`, etc.
        init_info = ray.init(
            num_cpus=world_size,  # example, set to your liking
            num_gpus=world_size,  # or how many you want Ray to see
            _node_ip_address=socket.gethostbyname(socket.gethostname()),
        )
        # The “address” that others should connect to
        #address_info = ray.get_runtime_context().address_info
        print(init_info)
        head_address = init_info["redis_address"] 
        
        #head_address = address_info["address"]  # e.g. "ray://<ip>:<port>"
        print(f"[Rank 0] Ray head address: {head_address}")

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

    # ------------------------------------------------------------------
    # (E) Non-zero ranks connect to the Ray cluster
    # ------------------------------------------------------------------
    if not is_rank0:
        print(f"[global_rank={global_rank}] Connecting to Ray head at {head_address}")
        if not head_address:
            raise RuntimeError("Got empty Ray head address. Something went wrong.")
        # Initialize Ray in "client" mode
        ray.init(address=head_address)
    dist.barrier()
  

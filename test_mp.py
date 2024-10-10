import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    world_size = 2
    rank = int(os.environ['RANK'])
    print(rank)
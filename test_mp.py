import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # Initialize the process group (NCCL is preferred for GPU communication)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_init_model(rank):
    # rank in predictor mode.
    if rank == 0:
        model = torch.nn.Linear(4, 16).to(rank)
    # rank in trainer model.
    elif rank == 1:
        model = torch.nn.Linear(4, 16).to(rank)

if __name__ == "__main__":
    world_size = 2
    rank = int(os.environ['RANK'])

    num_pred_gpus = 1
    num_trainer_gpus = 1

    print('gpu rank', rank)
    model = get_init_model(rank) #

    print('done!')

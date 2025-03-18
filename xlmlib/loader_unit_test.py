import os
import glob
import argparse
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from packed_dataset import PackedDataset


def create_dataloader(
    batch_size: int, block_size: int, data_dir, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    #datasets = []
    #data_config = train_data_config if split == "train" else val_data_config
    #for prefix, _ in data_config:
    filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
    random.seed(seed)
    random.shuffle(filenames)
    print('length of files', len(filenames), prefix)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="none", help="path to pretrained data.")
    args = parser.parse_args()    

    ################################################### gpu setup.
    local_rank = int(os.environ['LOCAL_RANK']) 
    print('local rank', local_rank) 
    rank = int(os.environ['RANK']) 
    print('rank', rank) 
    world_size = int(os.environ['WORLD_SIZE']) 
    print('WORLD_SIZE', world_size)      
    gpus_per_node = 8 
    node_idx = rank // gpus_per_node 
    torch.cuda.set_device(local_rank) 
    device = torch.device(f"cuda:{local_rank}") 
    # init distributed process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)    
    ########################################################

    create_dataloader(8, 4096, args.data, split='train')
    create_dataloader(8, 4096, args.data, split='valid')
    

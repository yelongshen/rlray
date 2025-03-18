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
    filenames = sorted(glob.glob(f'{data_dir}/{split}*'))
    random.seed(seed)
    random.shuffle(filenames)
    print('length of files', len(filenames), split)

    rank = int(os.environ['RANK']) 
    world_size = int(os.environ['WORLD_SIZE']) 

    dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size = block_size,
            shuffle=shuffle,
            seed=seed+rank,
            num_processes=world_size,
            process_rank=rank,
        )

def test_dataloader(loader, test_iters):
    for data_idx, data in enumerate(loader):
        if data_idx >= test_iters:
            break
        print('data', data.shape, data) 
        #for i, length in enumerate([4096, 8192, 12288, 16384]):   #[2048, 4096, 8192, 16384]
        #    input_ids = val_data[:, 0 : length].contiguous()
        #    targets = val_data[:, 1 : length + 1].contiguous()
        #    logits = model(input_ids).logits
        #    loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        #    losses[k,i] = loss.item()
            
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

    train_loader = create_dataloader(8, 4096, args.data, split='train')
    valid_loader = create_dataloader(8, 4096, args.data, split='valid')

    print('start to test train loader', rank)
    test_dataloader(train_loader, 10)
    print('start to test valid loader', rank)
    test_dataloader(valid_loader, 8)

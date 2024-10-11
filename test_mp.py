import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
from queue import Queue
import threading
import time
import random

def get_init_model(rank):
    # rank in predictor mode.
    if rank == 0:
        model = torch.nn.Linear(4, 16).to(rank)
    # rank in trainer model.
    elif rank == 1:
        model = torch.nn.Linear(4, 16).to(rank)
    return model

#buff = []
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.lock = threading.Lock()

    def add(self, experience):
        """ Add new experience to the buffer """
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Sample a batch of experiences from the buffer """
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)
    
buffer = ReplayBuffer(100000)
# Consumer side: Function to add experiences to its local buffer
def add_to_buffer(experience):
    global buffer
    buffer.add(experience)

def send_experience_to_consumer(experience, consumer_worker="worker1"):
    # Use RPC to send experience data to the consumer node
    rpc.rpc_sync(consumer_worker, add_to_buffer, args=(experience,))


if __name__ == "__main__":
    world_size = 2
    rank = int(os.environ['RANK'])

    num_pred_gpus = 1
    num_trainer_gpus = 1

    print('gpu rank', rank)
    model = get_init_model(rank) #

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # predictor running.
    if rank == 0:
        for i in range(0, 100000):
            x = torch.randn(2, 4).to(rank)
            y = model(x)
            send_experience_to_consumer((x.cpu(),y.cpu()), consumer_worker="worker1")
            print("[Producer] Sent experience to consumer.")
            time.sleep(1)
    # trainer running loop.
    if rank == 1:
        i = 0
        while i < 1000:
            if len(buffer) > 2:
                # Sample batch of experiences from the replay buffer
                (x, y) = buffer.sample(2)
                print("[Consumer] Sampled batch:", x, y)
                time.sleep(1)
                i = i + 1
    print('done!')

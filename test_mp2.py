import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
from queue import Queue
import threading
import time
import random


## so this can be used for 

def get_init_model(rank):
    # rank in predictor mode.
    if rank in [0, 1]:
        model = torch.nn.Linear(4, 16).to(rank)
    # rank in trainer model.
    elif rank in [2, 3]:
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

class ModelBuffer:
    def __init__(self):
        self.buffer = None
        self.is_new = False
        self.lock = threading.Lock()

    def push(self, model):
        """ Add new experience to the buffer """
        with self.lock:
            self.buffer = model
            self.is_new = True

    def check_new(self):
        with self.lock:
            return self.is_new

    def pull(self, host_model):
        with self.lock:
            host_model.data.copy_(self.buffer)
            self.is_new = False
            
buffer = ReplayBuffer(100000)
# Consumer side: Function to add experiences to its local buffer
def add_to_buffer(experience):
    print('debug consumer side add.....',  int(os.environ['RANK']) )
    global buffer
    buffer.add(experience)

def len_buffer():
    global buffer
    return len(buffer)

model_buffer = ModelBuffer()
def sync_weight(model_weight):
    global model_buffer
    model_buffer.push(model_weight)

def send_experience_to_consumer(experience, consumer_worker="worker1", proceduer="worker0"):
    # Use RPC to send experience data to the consumer node
    rpc.rpc_sync(consumer_worker, add_to_buffer, args=("i am a big big girl, in a big big world by" + proceduer,))


def send_model_weight_to_producer(model_weight):
    rpc.rpc_sync("worker0", sync_weight, args=(model_weight,))

def rev_experience_len(server_worker='worker1'):
    return rpc.rpc_sync(server_worker, len_buffer)

if __name__ == "__main__":
    world_size = 4
    rank = int(os.environ['RANK'])

    num_pred_gpus = 2
    num_trainer_gpus = 2

    print('gpu rank', rank)
    model = get_init_model(rank) #

    #dist.init_process_group('nccl', rank=rank, world_size=world_size)
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # predictor running.
    if rank in [0, 1]:
        for i in range(0, 100000):
            x = torch.randn(2, 4).to(rank)
            y = model(x)
            send_experience_to_consumer((x.cpu(),y.cpu()), consumer_worker="worker2", proceduer = str(rank))
            print("[Producer] Sent experience to consumer. {rank}")
            time.sleep(1)

            #if model_buffer.check_new():
            #    model_buffer.pull(model.weight)
            #    print('pull model weight.............')
    # trainer running loop.
    if rank in [2, 3]:
        i = 0
        while i < 1000:
            l = len(buffer) if rank == 2 else rev_experience_len('worker2')

            print('work', rank, l)

            time.sleep(1)

            i = i + 1
                # Sample batch of experiences from the replay buffer
                #(x, y) = buffer.sample(2)
                #print("[Consumer] Sampled batch:", x, y)

            #    z = buffer.sample(2)
            #    print("[Consumer] Sampled batch:", z)
            #    time.sleep(1)
            #    i = i + 1
                
                #if i % 20 == 0:
                #    # model weight sync.
                #    send_model_weight_to_producer(model.weight.cpu())
                #    print('push model weight...........')
    print('done!')
import os
import time
import random

import ray
import torch
import torch.distributed as dist

# ------------------------- REPLAY BUFFER ACTOR -------------------------
@ray.remote
class ReplayBufferActor:
    """A simple Replay Buffer stored in a Ray actor."""
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def add(self, item):
        """Add 'item' to the buffer if capacity not exceeded."""
        if len(self.buffer) >= self.capacity:
            # For simplicity: drop oldest if full (or do something else)
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size=1):
        """Sample a random subset of items from the buffer."""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

# ------------------------- PRODUCER FUNCTION -------------------------
def producer_loop(replay_handle, num_iters=5):
    """
    Producer generates data and pushes it to 'replay_handle'.
    'replay_handle' is a remote actor for the replay buffer.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    for i in range(num_iters):
        # Simulate producing some data
        data_tensor = torch.randn((2, 2), device=device)
        data_list = data_tensor.tolist()  # Convert for replay buffer (must be serializable)
        
        print(f"[Producer rank {rank}] Iter {i}: adding data {data_list}")
        replay_handle.add.remote((rank, data_list, i))

        time.sleep(0.5)  # Simulate time spent producing

# ------------------------- CONSUMER FUNCTION -------------------------
def consumer_loop(replay_handle, num_iters=5):
    """
    Consumer reads data from 'replay_handle' repeatedly.
    'replay_handle' is a remote actor for the replay buffer.
    """
    rank = dist.get_rank()

    for i in range(num_iters):
        # Sample 2 items
        samples_future = replay_handle.sample.remote(batch_size=2)
        samples = ray.get(samples_future)

        print(f"[Consumer rank {rank}] Iter {i}: got samples => {samples}")

        time.sleep(0.5)  # Simulate time spent consuming/processing

# ------------------------- MAIN -------------------------
def main():
    # 1. Initialize Ray (only needs to be done once per script)
    #    If you have a Ray cluster, you'd do ray.init(address="auto") or similar.
    #    For local usage:
    ray.init()

    # 2. Initialize Torch Distributed. We expect 4 processes total.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank
    )

    # Set the GPU device for this rank
    torch.cuda.set_device(local_rank)

    # 3. Create / Retrieve the replay buffer actor
    #    - Rank 0 creates it and gives it a name ("global_replay")
    #    - Other ranks look it up by name
    if global_rank == 0:
        replay_actor = ReplayBufferActor.options(
            name="global_replay",
            lifetime="detached"  # so it remains after rank 0 finishes
        ).remote(capacity=100)
        print("[Rank 0] Created ReplayBufferActor.")
    dist.barrier()  # ensure actor is created before others search for it

    if global_rank != 0:
        replay_actor = ray.get_actor("global_replay")
        print(f"[Rank {global_rank}] Retrieved ReplayBufferActor handle.")

    # 4. Producer or Consumer?
    #    - Let's say ranks 0,1 => Producer; ranks 2,3 => Consumer
    if global_rank < 2:
        producer_loop(replay_actor, num_iters=500)
    else:
        consumer_loop(replay_actor, num_iters=500)

    # 5. Final barrier to ensure all tasks complete
    dist.barrier()

    # Rank 0 can do some final shutdown or reporting
    if global_rank == 0:
        print("[Rank 0] All done. Shutting down.")
        # Optionally kill the actor or keep it for next usage

    # Cleanup
    dist.destroy_process_group()
    ray.shutdown()

if __name__ == "__main__":
    main()

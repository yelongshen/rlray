from dataclasses import fields
from time import perf_counter
from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Optional, Tuple, Union, Any, Dict


import pickle
import torch
import torch.distributed as dist


from collections import deque

from pynvml import *

from collections import deque
import xxhash
import numpy as np

from context import set_context, get_context, reset_context
from phi4 import normalize_probs


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: List[int]):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        
        #self.num_tokens = 
        self.num_prompt_tokens = len(token_ids)

        # 
        self.num_cached_tokens = 0
        self.block_table = [] 
        self.temperature = 0.7 # sampling_params.temperature
        self.max_tokens = 32768 # sampling_params.max_tokens
        #self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return len(self.token_ids) #self.num_tokens

    def __lt__(self, other):
        return self.seq_id < other.seq_id

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        state = vars(self).copy()
        if self.num_completion_tokens:
            state.pop("token_ids")
        return state


# 增大batch size. 怎么优化batch sequence length. 怎么极大化memory usage. 
def get_gpu_memory():
    torch.cuda.synchronize()
    nvmlInit()
    visible_device = list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(',')))
    cuda_device_idx = torch.cuda.current_device()
    cuda_device_idx = visible_device[cuda_device_idx]
    handle = nvmlDeviceGetHandleByIndex(cuda_device_idx)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total
    used_memory = mem_info.used
    free_memory = mem_info.free
    nvmlShutdown()
    return total_memory, used_memory, free_memory



def compute_hash(token_ids: List[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: List[int]):
        assert hash != -1
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def __repr__(self):
        return f"{(self.block_id, self.ref_count, self.hash)}"


class BlockManager:

    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()


    def _allocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # the i's block.

            h = compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # reuse the cache.
                seq.num_cached_tokens += self.block_size

                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id) # the first block, second block, 

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence):
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

# 充当军师角色。
class Scheduler:
    def __init__(self, llm_config, block_size, num_kvcache_blocks):
        self.block_size = block_size
        self.num_kvcache_blocks = num_kvcache_blocks

        self.llm_config = llm_config

        self.max_num_seqs = 1024 # config.max_num_seqs
        self.max_num_batched_tokens = 32768 # config.max_num_batched_tokens
        #self.eos = config.eos
        
        #max_num_batched_tokens: int = 32768 
        #max_num_seqs: int = 512
        #max_model_len: int = 4096
        #kvcache_block_size: int = 256
        #num_kvcache_blocks: int = -1
        
        self.block_manager = BlockManager(block_size, num_kvcache_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # decoding or prefill stage. 
    def schedule(self) -> Tuple[List[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # if we can't schedule the sequence, break. 
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            num_seqs += 1
            self.block_manager.allocate(seq) # allocate seq. 
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

            # prefill     
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq) 
        running = deque(scheduled_seqs)
        running.extend(self.running)
        self.running = running
        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) :
        eos_id1 = self.llm_config.eos_token_id
        eos_id2 = 200020 # eos_token_id

        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (token_id == eos_id1 or token_id == eos_id2) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq) ## deallocate.
                self.running.remove(seq)


# 执行首领。
class ModelRunner:

    def __init__(self, model, llm_config, device):

        self.block_size = 256 #config.kvcache_block_size

        #self.enforce_eager = config.enforce_eager
        #self.world_size = config.tensor_parallel_size
        #self.rank = rank
        #self.event = event
        #dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        #torch.cuda.set_device(rank)

        self.default_dtype = torch.bfloat16 #get_default_dtype()

        #torch.set_default_dtype(hf_config.torch_dtype)
        #torch.set_default_device("cuda")

        self.model = model # Qwen3ForCausalLM(hf_config)
        self.llm_config = llm_config
        self.device = device
        #load_model(self.model, config.model)
        #self.sampler = Sampler()

        # gpu memory resource.
        self.num_kvcache_blocks = self.allocate_kv_cache(self.model, self.llm_config, 0.85)
        
        self.temperature = 0.6
        # 这个操作是为了加速decoding。 
        #if not self.enforce_eager:
        #    self.capture_cudagraph()


    def call(self, method_name, *args):
        #if self.world_size > 1 and self.rank == 0:
        #    self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_kv_cache(self, llm, llm_config, gpu_memory_utilization = 0.90):
        #config = self.config
        #hf_config = config.hf_config

        total, used, _ = get_gpu_memory()
        free = total * gpu_memory_utilization - used

        #
        #kvcache_block_size: int = 256
        block_size = 256
        head_dim = llm_config.hidden_size // llm_config.num_attention_heads

        #key_cache = torch.zeros(bsz, max_generation, self.num_key_value_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype) 
        #value_cache = torch.zeros(bsz, max_generation, self.num_key_value_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
                
        # assume kv_cache is stored on one GPU only. 
        num_kv_heads = llm_config.num_key_value_heads # // dist.get_world_size()
        block_bytes = 2 * llm_config.num_hidden_layers * block_size * num_kv_heads * head_dim * torch.finfo(torch.bfloat16).bits // 8 #torch.bfloat16.itemsize

        # how many kv blocks. 
        num_kvcache_blocks = int(free) // block_bytes
        
        kv_cache = torch.zeros(2, llm_config.num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=self.device)
        
        print('max kv cache length:', num_kvcache_blocks * block_size)

        # setup k_cache.
        layer_id = 0
        for module in llm.model.layers:
            module.self_attn.k_cache = kv_cache[0, layer_id]
            module.self_attn.v_cache = kv_cache[1, layer_id]
            layer_id += 1

        return num_kvcache_blocks

    def prepare_block_tables(self, seqs: List[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: List[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = None
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))

        assert len(input_ids) == len(slot_mapping)
        assert len(input_ids) == cu_seqlens_q[-1]

        #if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
        context_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: List[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    #def prepare_sample(self, seqs: List[Sequence]):
    #    temperatures = []
    #    for seq in seqs:
    #        temperatures.append(seq.temperature)
    #    temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    #    return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):

        context = get_context()
        if is_prefill:
            _, logits, _, _ = self.model(input_ids=input_ids.unsqueeze(0), position_ids=positions.unsqueeze(0), logits_to_keep=context.cu_seqlens_q[1:]-1)
        else:
            _, logits, _, _ = self.model(input_ids=input_ids.unsqueeze(0), position_ids=positions.unsqueeze(0))

        logits = logits.squeeze(0)
            #self.model
        return logits

        #if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        #    return self.model.compute_logits(self.model(input_ids, positions))
        #else:
        #    bs = input_ids.size(0)    
        #    self.reset_graph_vars()
        #    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        #    graph_vars = self.graph_vars
        #    graph_vars["input_ids"][:bs] = input_ids
        #    graph_vars["positions"][:bs] = positions
        #    graph_vars["slot_mapping"][:bs] = context.slot_mapping
        #    graph_vars["context_lens"][:bs] = context.context_lens
        #    graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        #    graph.replay()
        #    return self.model.compute_logits(graph_vars["outputs"][:bs])

    
    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        #temperatures = self.prepare_sample(seqs)

        #print('is_prefill', is_prefill)
        #print('input_ids', input_ids.shape)
        
        #print('positions', positions.shape)

        logits = self.run_model(input_ids, positions, is_prefill)
        
        #print('logits', logits.shape)
        probs = torch.softmax( (logits / self.temperature).to(torch.float32), dim=-1)
        
        #print('probs', probs.shape)
        #norm_probs = normalize_probs(probs, 0.9)
        token_ids = torch.multinomial(probs, num_samples=1)

        #print('token_ids', token_ids.shape)
        #token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids.squeeze(dim=1).tolist()


# 入口托管。
class LLMEngine:
    def __init__(self, model, llm_config, device):
        self.model_runner = ModelRunner(model, llm_config, device) 
        self.scheduler = Scheduler(llm_config, self.model_runner.block_size, self.model_runner.num_kvcache_blocks) 

        #config)
        #atexit.register(self.exit)
        #config, 0, self.events)


    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        # sequence & tasks. 筹划decoding. 
        seqs, is_prefill = self.scheduler.schedule()

        # do prefill first.
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        self.scheduler.postprocess(seqs, token_ids)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        #num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs #, num_tokens

    #def is_finished(self):
    #    return self.scheduler.is_finished()

    # do we expect 
    def generate(
        self,
        prompts: List[List[int]],
    ) -> List[List[int]]:
        # step 1. fill all the request into schedule.
        for prompt in prompts: #zip(prompts, sampling_params):
            self.scheduler.add(Sequence(prompt))
            #self.add_request(prompt) #, sp

        outputs = {}
        #prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        #outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        return outputs

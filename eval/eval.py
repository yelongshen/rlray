import argparse
import io
import sys
import os
import torch
import json
from typing import Iterable, Union, Any
import torch.distributed as dist
import torch.distributed.rpc as rpc
import time


import xlmlib
from xlmlib import _SambaForCausalLM
from xlmlib import RpcReplayBuffer

def load_jsonl(file) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()
# implement a easy version of distributed inference pipeline. 
# define a inference engine to automatically process request. 

def setup_dist_eval(args):
    # on-policy ppo experiments with phi3.5 model on math dataset. 
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

    #model, config, tokenizer = _SambaForCausalLM.load_hfckpt(args.model_path)
    #model.eval()
    
    rpc_worker_name = f"worker-{rank}"
    rpc.init_rpc(rpc_worker_name, rank=rank, world_size=world_size, rpc_backend_options=rpc.TensorPipeRpcBackendOptions()) 

    request_buffer_name = 'request_buffer'
    request_buffer_worker = f"worker-{0}"
    
    # load file.
    if rank == 0:
        prompt_list = []
        for data in load_jsonl(args.data_path):
            prompt = data['problem']
            prompt_list.append(prompt)
        RpcReplayBuffer.Register(request_buffer_name, request_buffer_worker, True, capacity = len(prompt_list))
        
        for prompt in prompt_list:
            RpcReplayBuffer.Push(request_buffer_name, prompt)
    else:
        RpcReplayBuffer.Register(request_buffer_name, request_buffer_worker, False)

    dist.barrier()
    while RpcReplayBuffer.Length(request_buffer_name) > 0:
        prompt = RpcReplayBuffer.Pop(request_buffer_name)
        print(prompt, ', rpc:', rpc_worker_name)
        time.sleep(1)
        
        #prompt = 'I am a big big girl, in a'
        #_tokens = tokenizer([prompt], add_special_tokens=False, max_length=1024, truncation=False)
        #_tokens = _tokens['input_ids']
        #model = model.to(torch.bfloat16).to(device) 
        #outputs, _, _ = model.generate(input_ids, max_gen_len = 1000)
        #response = tokenizer.decode(outputs[0])
        #print('response: ', response)
    
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="aime24", type=str)
    parser.add_argument("--model_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
  
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
  
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    #set_seed(args.seed)
    setup_dist_eval(args)

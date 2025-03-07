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
from dataclasses import dataclass
import numpy
import datetime

import xlmlib
from xlmlib import _SambaForCausalLM, _Phi4ForCausalLM
from xlmlib import RpcReplayBuffer
from xlmlib import process_math_prompt, process_math_answer


@dataclass
class Request:
    id : str
    prompt : str
    answer : str

@dataclass
class Result(Request):
    responselen : int
    reward : float
    
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

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TIMEOUT"] = "600000"  # 600 seconds (10 minutes)


    # Initialize distributed process group
    dist.init_process_group(backend="nccl",  rank = rank, world_size = world_size, timeout = datetime.timedelta(seconds=600000))

    # init distributed process group.
    #dist.init_process_group(backend="nccl")    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.model_type == 'samba':
        if args.weight_path is None:
            model, config, tokenizer = _SambaForCausalLM.load_hfckpt(args.model_path)
        else:
            model, config, tokenizer = _SambaForCausalLM.load_customckpt(args.model_path, args.weight_path)
    elif args.model_type == 'phi4':
        if args.weight_path is None:
            model, config, tokenizer = _Phi4ForCausalLM.load_hfckpt(args.model_path)
        else:
            model, config, tokenizer = _Phi4ForCausalLM.load_customckpt(args.model_path, args.weight_path)
            
    model = model.to(torch.bfloat16).to(device) 
    model.eval()
    
    rpc_worker_name = f"worker-{rank}"
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        _channels=["cuda_ipc", "cuda_basic"]
    )
    rpc.init_rpc(
        name=rpc_worker_name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )
    
    #rpc.init_rpc(rpc_worker_name, rank=rank, world_size=world_size) #, rpc_backend_options=rpc.TensorPipeRpcBackendOptions()) 

    request_buffer_name = 'request_buffer'
    request_buffer_worker = f"worker-{0}"

    result_buffer_name = 'result_buffer'
    result_buffer_worker = f"worker-{0}"

    assert args.n_rollout % args.batch_size == 0
    # load file.
    if rank == 0:
        request_list = []
        idx = 0
        for data in load_jsonl(args.data_path):
            prompt = data['problem']
            ans = data['answer']
            solution = data['solution']
            if 'id' in data:
                id = str(data['id'])
            elif 'unique_id' in data:
                id = str(data['unique_id'])
            else:
                id = str(idx)
            idx += 1

            for n in range(0, args.n_rollout // args.batch_size):
                request_list.append(Request(id = id, prompt = prompt, answer = ans))

        RpcReplayBuffer.Register(request_buffer_name, request_buffer_worker, True, capacity = len(request_list))
        RpcReplayBuffer.Register(result_buffer_name, result_buffer_worker, True, capacity = len(request_list) * args.batch_size)
        
        for req in request_list:
            RpcReplayBuffer.Push(request_buffer_name, req)
    else:
        RpcReplayBuffer.Register(request_buffer_name, request_buffer_worker, False)
        RpcReplayBuffer.Register(result_buffer_name, result_buffer_worker, False)

    force_reflection = '\nWait, please verify the answer again. \n'
    force_tokens = tokenizer([force_reflection], add_special_tokens=False, max_length=1024)
    force_tokens = force_tokens['input_ids'][0]
    
    dist.barrier()
    while True: 
        req = RpcReplayBuffer.Pop(request_buffer_name)
        if req is None:
            break    
        prompt = process_math_prompt(req.prompt, prompt_type = args.prompt_type)
        _tokens = tokenizer([prompt] * args.batch_size, add_special_tokens=False, max_length=1024, truncation=False)
        input_ids = _tokens['input_ids']
        #temperature: float = 0.7,
        #top_p: float = 0.95,
        #early_stop = True,
        print('start to generate.....')
        outputs, _, _ = model.generate(input_ids, max_gen_len = args.max_generation, temperature = args.temperature, top_p = args.top_p, early_stop=args.early_stop, force_wait_tokens = force_tokens)
        print('end to generate .....')
        assert len(outputs) == args.batch_size
        
        for output in outputs:
            response = tokenizer.decode(output)
            mid_response, extracted_answer, reward = process_math_answer(response, [req.answer], tokenizer, prompt_type = args.prompt_type)
            
            if args.debug:
                print('######################\n\n')
                print('prompt:\n', prompt)
                print('response:\n', response)
                print('filterd response:\n', mid_response)
                print('extracted_answer:\n', extracted_answer)
                print('gold answer:\n', req.answer)
                print('reward:', reward)

            
            RpcReplayBuffer.Push(result_buffer_name, Result(id = req.id, prompt = req.prompt, answer = req.answer, responselen = len(output), reward = reward))
        print('push to replaybuffer')
        
    dist.barrier()
    if rank == 0:
        print('eval length', RpcReplayBuffer.Length(result_buffer_name))

        avg_response = 0
        eval_results = {}
        for i in range(0, len(request_list) * args.batch_size):
            result = RpcReplayBuffer.Pop(result_buffer_name)
            if result is None:
                break
            if not result.id in eval_results:
                eval_results[result.id] = []
            eval_results[result.id].append(result.reward)

            avg_response = avg_response + result.responselen

        avg_response = avg_response / (len(request_list) * args.batch_size)
        #total_reward = 0
        pass_1 = 0
        pass_n = 0
        total_count = 0
        for mkey in eval_results:
            rlist = eval_results[mkey]
            pass_1 = pass_1 + numpy.mean(rlist)
            pass_n = pass_n + min(1, numpy.sum(rlist))
            total_count = total_count + 1
            
        print(f'average pass@1:', pass_1 * 1.0 / total_count, len(eval_results))
        print(f'average pass@{args.n_rollout}:', pass_n * 1.0 / total_count)
        print(f'average response len:{avg_response}')
    rpc.shutdown(graceful=True)

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="aime24", type=str)
    parser.add_argument("--model_path", default="gpt-4", type=str)
    parser.add_argument("--model_type", type=str, default="samba", choices=["samba", "phi4"], help="choose model type.")
    parser.add_argument("--prompt_type", type=str, default="v8", choices=["v8", "v9", "v10", "v11"], help="choose prompt type.")
    parser.add_argument("--batch_size", default=1, type=int)
    
    parser.add_argument("--weight_path", default=None, type=str)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--n_rollout", default=1, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_generation", default=4096, type=int)    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    #set_seed(args.seed)
    setup_dist_eval(args)

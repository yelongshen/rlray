import xlmlib
from xlmlib import _SambaForCausalLM

import argparse
import io
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="none", help="path to pretrained ckpt.")
    
    args = parser.parse_args()

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

    model, config, tokenizer = _SambaForCausalLM.load_hfckpt(args.pretrained_model)

    prompt = 'I am a big big girl, in a'
    
    x_tokens = tokenizer([prompt], add_special_tokens=False, max_length=100, truncation=True)
    
    input_ids = x_tokens['input_ids']

    model.eval()
    model = model.to(torch.bfloat16).to(device) 
  
    outputs, _, _ = model.generate(input_ids, max_gen_len = 1000)
    response = tokenizer.decode(outputs[0])
    print('response: ', response)



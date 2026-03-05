"""
Compare HF generate vs our qwen3_next_engine on the first AIME24 problem.

Checks:
1. Same prompt tokenization
2. Same first-token logits
3. Generation quality

Usage:
    # Single GPU
    python xlmlib/test_aime24_compare.py --model_path ./models/Qwen_Qwen3-Coder-Next/ --gpu_ids 0
    
    # TP=2
    torchrun --nproc_per_node=2 xlmlib/test_aime24_compare.py --model_path ./models/Qwen_Qwen3-Coder-Next/ --tensor_parallel 2
"""

import os, sys, json, time, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_util import process_math_prompt


def get_aime24_problem_0():
    """Load first AIME24 problem."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "aime24_test.jsonl")
    with open(data_path) as f:
        data = json.loads(f.readline())
    return data["problem"], str(data["answer"])


def build_prompt(problem_text, tokenizer, prompt_type="chat"):
    """Build prompt exactly as eval_qwen3_next does."""
    prompt_text = process_math_prompt(problem_text, prompt_type=prompt_type)
    if prompt_type in ('chat', 'v_chat'):
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt_text


def test_engine(args):
    """Test our engine on the first AIME24 problem."""
    from qwen3_next_engine import (
        load_qwen3_next_for_engine, HybridLLMEngine, get_tp_rank
    )
    
    problem, gold_answer = get_aime24_problem_0()
    is_main = int(os.environ.get("RANK", "0")) == 0
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"AIME24 Problem 0: {problem[:80]}...")
        print(f"Gold answer: {gold_answer}")
        print(f"{'='*60}")
    
    # Load model
    device = "cuda:0" if args.tensor_parallel == 1 else "cuda"
    if args.gpu_ids and args.tensor_parallel == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    model, tokenizer, config = load_qwen3_next_for_engine(
        args.model_path, device=device,
        tensor_parallel_size=args.tensor_parallel
    )
    
    is_main = get_tp_rank() == 0
    
    # Build prompt
    prompt_text = build_prompt(problem, tokenizer, prompt_type="chat")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    if is_main:
        print(f"\n--- Prompt ---")
        print(prompt_text)
        print(f"\nPrompt tokens: {len(input_ids)}")
        print(f"Last 10 tokens decoded: '{tokenizer.decode(input_ids[-10:], skip_special_tokens=False)}'")
    
    # ===== Test 1: Direct forward pass (no KV cache) =====
    if is_main:
        print(f"\n{'='*60}")
        print("TEST 1: Direct forward pass (no KV cache)")
        print(f"{'='*60}")
    
    input_tensor = torch.tensor([input_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits, _ = model(input_tensor)
    
    if is_main:
        last_logits = logits[0, -1, :]
        top5_vals, top5_ids = torch.topk(last_logits, 5)
        print(f"Logits shape: {logits.shape}")
        print(f"Top-5 next tokens (greedy):")
        for val, tid in zip(top5_vals, top5_ids):
            token_str = tokenizer.decode([tid.item()], skip_special_tokens=False)
            print(f"  {tid.item():>8} ({val.item():>8.3f}): '{token_str}'")
        
        greedy_token = top5_ids[0].item()
        print(f"\nGreedy next token: {greedy_token} = '{tokenizer.decode([greedy_token])}'")
    
    # ===== Test 2: Engine generation (with KV cache) =====
    if is_main:
        print(f"\n{'='*60}")
        print("TEST 2: Engine generation (HybridLLMEngine)")
        print(f"{'='*60}")
    
    config.eos_token_id = tokenizer.eos_token_id
    engine = HybridLLMEngine(
        model, config, str(next(model.parameters()).device),
        temperature=0.0, max_batch_size=1, tokenizer=tokenizer
    )
    
    t0 = time.time()
    output_ids = engine.generate([input_ids], max_tokens=2048)[0]
    t1 = time.time()
    
    if is_main:
        response = tokenizer.decode(output_ids, skip_special_tokens=False)
        print(f"\nGeneration time: {t1-t0:.1f}s")
        print(f"Output tokens: {len(output_ids)}")
        print(f"Throughput: {len(output_ids)/(t1-t0):.1f} tok/s")
        print(f"\n--- Engine Response ---")
        print(response[:2000])
        if len(response) > 2000:
            print(f"\n... ({len(response)} total chars)")
        print(f"--- End Response ---")
        
        # Check if answer is correct
        from math_util import safe_math_answer_timeout
        _, extracted, reward = safe_math_answer_timeout(
            response, [gold_answer], tokenizer,
            prompt_type="chat", alg=['is_equiv', 'text'], timeout=30
        )
        print(f"\nExtracted answer: '{extracted}'")
        print(f"Gold answer: '{gold_answer}'")
        print(f"Reward: {reward}")
        print(f"Correct: {reward > 0.5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()
    test_engine(args)

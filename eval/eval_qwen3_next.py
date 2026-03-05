"""
Math Evaluation for Qwen3-Next with HybridLLMEngine.

Evaluates on MATH500, GSM8K, AIME24, and AIME25 benchmarks using the paged attention engine.
Supports tensor parallelism.

Usage:
    # Single GPU - MATH500
    python eval/eval_qwen3_next.py --model_path ./models/Qwen_Qwen3-Coder-Next/ --dataset math500

    # Single GPU - AIME24
    python eval/eval_qwen3_next.py --model_path ./models/Qwen_Qwen3-Coder-Next/ --dataset aime24

    # Single GPU - AIME25
    python eval/eval_qwen3_next.py --model_path ./models/Qwen_Qwen3-Coder-Next/ --dataset aime25

    # TP=2 - AIME24 with sampling (pass@k)
    torchrun --nproc_per_node=2 eval/eval_qwen3_next.py --model_path ./models/Qwen_Qwen3-Coder-Next/ --dataset aime24 --tensor_parallel 2 --n_rollout 8 --temperature 0.7 --top_k 50 --max_batch_size 64 --max_tokens 32768
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Add xlmlib to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xlmlib"))

from qwen3_next_engine import (
    load_qwen3_next_for_engine,
    HybridLLMEngine,
    get_tp_rank,
)
from math_util import process_math_prompt, process_math_answer, safe_math_answer_timeout


# ============================================================================
# Dataset Loading
# ============================================================================

def load_math500(data_path=None):
    """Load MATH500 evaluation dataset."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "math500_test.jsonl")
    
    if not os.path.exists(data_path):
        # Try HF datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            problems = []
            for item in ds:
                problems.append({
                    "id": str(item.get("unique_id", len(problems))),
                    "problem": item["problem"],
                    "answer": item["answer"],
                    "level": item.get("level", ""),
                    "subject": item.get("subject", ""),
                })
            return problems
        except:
            raise FileNotFoundError(f"MATH500 data not found at {data_path}. Download with: python data/download_math_datasets.py --dataset math500")
    
    problems = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "id": str(data.get("unique_id", data.get("id", len(problems)))),
                "problem": data["problem"],
                "answer": data["answer"],
                "level": data.get("level", ""),
                "subject": data.get("subject", ""),
            })
    return problems


def load_gsm8k(data_path=None):
    """Load GSM8K evaluation dataset."""
    if data_path is None:
        # Try HF datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            problems = []
            for idx, item in enumerate(ds):
                # GSM8K answer is in the format "#### <number>"
                answer_text = item["answer"]
                # Extract final number
                if "####" in answer_text:
                    final_answer = answer_text.split("####")[-1].strip()
                else:
                    final_answer = answer_text.strip()
                problems.append({
                    "id": str(idx),
                    "problem": item["question"],
                    "answer": final_answer,
                    "level": "elementary",
                    "subject": "arithmetic",
                })
            return problems
        except:
            raise FileNotFoundError("GSM8K not found. Download with: python data/download_math_datasets.py --dataset gsm8k")
    
    problems = []
    with open(data_path, "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            answer_text = data.get("answer", "")
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()
            else:
                final_answer = answer_text.strip()
            problems.append({
                "id": str(data.get("id", idx)),
                "problem": data.get("question", data.get("problem", "")),
                "answer": final_answer,
                "level": "elementary",
                "subject": "arithmetic",
            })
    return problems


def load_aime24(data_path=None):
    """Load AIME 2024 evaluation dataset."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "aime24_test.jsonl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"AIME24 data not found at {data_path}")
    
    problems = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "id": str(data.get("id", len(problems))),
                "problem": data["problem"],
                "answer": str(data["answer"]),
                "level": "olympiad",
                "subject": "competition",
            })
    return problems


def load_aime25(data_path=None):
    """Load AIME 2025 evaluation dataset."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "aime25_test.jsonl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"AIME25 data not found at {data_path}")
    
    problems = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "id": str(data.get("id", len(problems))),
                "problem": data["problem"],
                "answer": str(data["answer"]),
                "level": "olympiad",
                "subject": "competition",
            })
    return problems


DATASET_LOADERS = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
    "aime24": load_aime24,
    "aime25": load_aime25,
}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model,
    tokenizer,
    config,
    problems,
    device,
    temperature=0.0,
    top_k=0,
    max_tokens=4096,
    max_batch_size=64,
    n_rollout=1,
    prompt_type="v17",
    debug=False,
):
    """Run evaluation on a set of math problems.
    
    Uses batch-wise generation with multi-problem packing:
      - max_batch_size controls the total number of sequences per engine call.
      - Problems are packed so that (max_batch_size // n_rollout) problems run
        together, each with n_rollout copies, filling the batch.
      - e.g. max_batch_size=64, n_rollout=8 => 8 problems packed per batch.
    """
    is_main = get_tp_rank() == 0
    
    # Create engine with top-k support and batch size limit
    engine = HybridLLMEngine(model, config, str(device), temperature=temperature, top_k=top_k, max_batch_size=max_batch_size, tokenizer=tokenizer)
    
    results = {}  # id -> list of rewards
    total_response_len = 0
    total_generated = 0
    
    # Number of distinct problems per batch
    problems_per_batch = max(1, max_batch_size // max(1, n_rollout))
    
    if is_main:
        print(f"\nBatch config: max_batch_size={max_batch_size}, n_rollout={n_rollout}, "
              f"problems_per_batch={problems_per_batch}, "
              f"actual_batch_size={problems_per_batch * n_rollout}")
    
    start_time = time.time()
    
    # Process problems in chunks
    for batch_start in range(0, len(problems), problems_per_batch):
        batch_problems = problems[batch_start : batch_start + problems_per_batch]
        
        # Tokenize all problems in this chunk
        batch_prompts = []
        batch_meta = []  # (problem_index_in_chunk, rollout_index) for each prompt
        for local_idx, problem in enumerate(batch_problems):
            prompt_text = process_math_prompt(problem["problem"], prompt_type=prompt_type)
            
            # Apply chat template if available (wraps with <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n)
            if hasattr(tokenizer, 'apply_chat_template') and prompt_type in ('v_chat', 'chat'):
                messages = [{"role": "user", "content": prompt_text}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True
                )
                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            else:
                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            
            prob_idx = batch_start + local_idx
            pid = problem["id"]
            
            if is_main and prob_idx < 3:
                print(f"\n--- Problem {prob_idx} (id={pid}) ---")
                print(f"  Question: {problem['problem'][:100]}...")
                print(f"  Gold answer: {problem['answer']}")
                print(f"  Prompt tokens: {len(input_ids)}")
                print(f"  Prompt text:\n{prompt_text}")
                print(f"  Prompt token IDs (first 20): {input_ids[:20]}")
                print(f"  Prompt token IDs (last 10): {input_ids[-10:]}")
                # Decode last few tokens to verify chat template framing
                print(f"  Prompt ending (decoded): '{tokenizer.decode(input_ids[-10:], skip_special_tokens=False)}'")
            
            # Replicate for n_rollout
            for r in range(n_rollout):
                batch_prompts.append(input_ids)
                batch_meta.append((local_idx, r))
        
        # Reset cache before the batch
        engine.model_runner.cache_params.reset()
        engine.model_runner.cache_params.has_previous_state = False
        
        try:
            output_ids_list = engine.generate(batch_prompts, max_tokens=max_tokens)
        except Exception as e:
            if is_main:
                print(f"  Batch generation failed (problems {batch_start}-{batch_start+len(batch_problems)-1}): {e}")
            continue
        
        # Score all outputs
        for out_idx, (local_idx, rollout) in enumerate(batch_meta):
            problem = batch_problems[local_idx]
            pid = problem["id"]
            prob_idx = batch_start + local_idx
            output_ids = output_ids_list[out_idx]
            
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            total_response_len += len(output_ids)
            total_generated += 1
            
            _, extracted_answer, reward = safe_math_answer_timeout(
                response, [problem["answer"]], tokenizer,
                prompt_type=prompt_type, timeout=30
            )
            
            if pid not in results:
                results[pid] = []
            results[pid].append(reward)
            
            # Print sample outputs for first 5 problems (all rollouts) + every 10th problem (rollout 0)
            show_sample = is_main and (prob_idx < 5 or (prob_idx % 10 == 0 and rollout == 0))
            if show_sample:
                print(f"\n  [Problem {prob_idx} id={pid}] rollout={rollout}, reward={reward}, "
                      f"gold='{problem['answer']}', extracted='{extracted_answer}', len={len(output_ids)}")
                # Show first 500 chars and last 300 chars of response
                if len(response) > 800:
                    print(f"  Response (first 500): {response[:500]}")
                    print(f"  Response (last 300):  ...{response[-300:]}")
                else:
                    print(f"  Response: {response}")
            elif is_main and prob_idx < 10:
                print(f"  [Problem {prob_idx}] r={rollout} reward={reward} extracted='{extracted_answer}' len={len(output_ids)}")
        
        # Progress
        batch_end = batch_start + len(batch_problems)
        if is_main and (batch_end % 10 == 0 or batch_end >= len(problems)):
            elapsed = time.time() - start_time
            pass_1 = sum(np.mean(v) for v in results.values()) / len(results) if results else 0
            print(f"  Progress: {batch_end}/{len(problems)}, pass@1={pass_1:.4f}, "
                  f"elapsed={elapsed:.1f}s, avg_len={total_response_len/max(1,total_generated):.0f}")
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    if is_main:
        pass_1 = 0
        pass_n = 0
        total_count = 0
        
        for pid, rewards in results.items():
            pass_1 += np.mean(rewards)
            pass_n += min(1, np.sum(rewards))
            total_count += 1
        
        if total_count > 0:
            pass_1 /= total_count
            pass_n /= total_count
            avg_len = total_response_len / max(1, total_generated)
        else:
            pass_1 = pass_n = avg_len = 0
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"  Problems: {total_count}")
        print(f"  Rollouts per problem: {n_rollout}")
        print(f"  Temperature: {temperature}")
        print(f"  pass@1: {pass_1:.4f} ({pass_1*100:.1f}%)")
        if n_rollout > 1:
            print(f"  pass@{n_rollout}: {pass_n:.4f} ({pass_n*100:.1f}%)")
        print(f"  Avg response length: {avg_len:.0f} tokens")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Throughput: {total_generated/elapsed:.2f} samples/s")
        print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Math Evaluation for Qwen3-Next")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-Next model")
    parser.add_argument("--dataset", type=str, default="math500", 
                        choices=["math500", "gsm8k", "aime24", "aime25"],
                        help="Evaluation dataset")
    parser.add_argument("--data_path", type=str, default=None, help="Custom data file path")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="TP size")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0=disabled)")
    parser.add_argument("--max_batch_size", type=int, default=64, help="Max sequences per engine batch (e.g. 64)")
    parser.add_argument("--n_rollout", type=int, default=1, help="Number of rollouts per problem")
    parser.add_argument("--prompt_type", type=str, default="v17", help="Prompt template version")
    parser.add_argument("--max_problems", type=int, default=None, help="Max problems to evaluate (for testing)")
    parser.add_argument("--debug", action="store_true", help="Print detailed output")
    parser.add_argument("--gpu_ids", type=str, default=None, help="GPU IDs to use (e.g. '0,1' or '2,3')")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()
    
    # Set visible GPUs before any CUDA operations
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    # Load model
    model, tokenizer, config = load_qwen3_next_for_engine(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel,
    )
    device = next(model.parameters()).device
    is_main = get_tp_rank() == 0
    
    # Load dataset
    if is_main:
        print(f"\nLoading {args.dataset} dataset...")
    
    loader = DATASET_LOADERS[args.dataset]
    problems = loader(args.data_path)
    
    if args.max_problems:
        problems = problems[:args.max_problems]
    
    if is_main:
        print(f"  Loaded {len(problems)} problems")
        if problems:
            print(f"  Example: {problems[0]['problem'][:80]}...")
            print(f"  Answer: {problems[0]['answer']}")
    
    # Run evaluation
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        config=config,
        problems=problems,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        n_rollout=args.n_rollout,
        prompt_type=args.prompt_type,
        debug=args.debug,
    )
    
    # Save results
    if is_main and args.output:
        with open(args.output, "w") as f:
            json.dump({
                "dataset": args.dataset,
                "model_path": args.model_path,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_batch_size": args.max_batch_size,
                "n_rollout": args.n_rollout,
                "results": {k: v for k, v in results.items()},
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

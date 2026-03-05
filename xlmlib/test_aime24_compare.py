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
        print("TEST 1: Direct forward - first 2 tokens (ground truth)")
        print(f"{'='*60}")
    
    model_device = next(model.parameters()).device
    
    # Token 1: full prompt forward
    input_tensor = torch.tensor([input_ids], device=model_device)
    if is_main:
        print(f"  Computing token 1 ({len(input_ids)} tokens forward)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        logits1, _ = model(input_tensor)
    if is_main:
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    
    last_logits1 = logits1[0, -1, :]
    token1 = last_logits1.argmax().item()
    top5_vals1, top5_ids1 = torch.topk(last_logits1, 5)
    
    if is_main:
        print(f"Token 1 (from prompt): greedy={token1} '{tokenizer.decode([token1])}'")
        print(f"  Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(top5_vals1, top5_ids1)]}")
    
    # Token 2: append token1, forward again (no cache)
    input_tensor2 = torch.tensor([input_ids + [token1]], device=model_device)
    if is_main:
        print(f"  Computing token 2 ({len(input_ids)+1} tokens forward)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        logits2, _ = model(input_tensor2)
    if is_main:
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    
    last_logits2 = logits2[0, -1, :]
    token2 = last_logits2.argmax().item()
    top5_vals2, top5_ids2 = torch.topk(last_logits2, 5)
    
    if is_main:
        print(f"\nToken 2 (after token1={token1}): greedy={token2} '{tokenizer.decode([token2])}'")
        print(f"  Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(top5_vals2, top5_ids2)]}")
    
    # Token 3: append token1+token2, forward again (no cache)
    input_tensor3 = torch.tensor([input_ids + [token1, token2]], device=model_device)
    if is_main:
        print(f"  Computing token 3 ({len(input_ids)+2} tokens forward)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        logits3, _ = model(input_tensor3)
    if is_main:
        print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    
    last_logits3 = logits3[0, -1, :]
    token3 = last_logits3.argmax().item()
    top5_vals3, top5_ids3 = torch.topk(last_logits3, 5)
    
    if is_main:
        print(f"\nToken 3 (after token1,2={token1},{token2}): greedy={token3} '{tokenizer.decode([token3])}'")
        print(f"  Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(top5_vals3, top5_ids3)]}")
        print(f"\nGround truth first 3 tokens: '{tokenizer.decode([token1, token2, token3])}'")
    
    ground_truth_tokens = [token1, token2, token3]
    ground_truth_logits = [last_logits1, last_logits2, last_logits3]
    
    # ===== Test 2: Engine prefill + first 3 decode steps =====
    if is_main:
        print(f"\n{'='*60}")
        print("TEST 2: Engine (with KV cache) - first 3 tokens")  
        print(f"{'='*60}")
    
    config.eos_token_id = tokenizer.eos_token_id
    engine = HybridLLMEngine(
        model, config, str(model_device),
        temperature=0.0, max_batch_size=1, tokenizer=tokenizer
    )
    
    from llm_engine import Sequence
    seq = Sequence(input_ids)
    seq.max_tokens = 3
    engine.scheduler.add(seq)
    
    engine_tokens = []
    engine_logits_list = []
    step_count = 0
    while not engine.is_finished():
        seqs, is_prefill = engine.scheduler.schedule()
        
        if is_prefill:
            # Run prefill and capture the logits
            from context import get_context, reset_context
            input_ids_p, positions_p = engine.model_runner.prepare_prefill(seqs)
            logits_prefill = engine.model_runner.run_model(input_ids_p, positions_p, True)
            
            if logits_prefill.dim() == 1:
                logits_prefill = logits_prefill.unsqueeze(0)
            
            # Greedy token from prefill
            prefill_token = logits_prefill[0].argmax().item() if logits_prefill.shape[0] == 1 else logits_prefill[-1].argmax().item()
            prefill_logits = logits_prefill[0] if logits_prefill.shape[0] == 1 else logits_prefill[-1]
            
            if is_main:
                top5_e, top5_ei = torch.topk(prefill_logits, 5)
                print(f"\nPrefill token 1: engine={prefill_token} '{tokenizer.decode([prefill_token])}'")
                print(f"  Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(top5_e, top5_ei)]}")
                
                # Compare logits with ground truth
                gt_logits = ground_truth_logits[0]
                diff = (prefill_logits.float() - gt_logits.float()).abs()
                print(f"  Max logit diff vs ground truth: {diff.max().item():.6f}")
                print(f"  Mean logit diff: {diff.mean().item():.6f}")
                match = "✓" if prefill_token == ground_truth_tokens[0] else "✗"
                print(f"  Match: {match} (engine={prefill_token}, gt={ground_truth_tokens[0]})")
            
            engine_tokens.append(prefill_token)
            engine_logits_list.append(prefill_logits)
            
            reset_context()
            engine.scheduler.postprocess(seqs, [prefill_token])
            # Check stop tokens
            if engine._stop_token_ids:
                from llm_engine import SequenceStatus
                for s in seqs:
                    if not s.is_finished and prefill_token in engine._stop_token_ids:
                        s.status = SequenceStatus.FINISHED
                        engine.scheduler.block_manager.deallocate(s)
                        if s in engine.scheduler.running:
                            engine.scheduler.running.remove(s)
        else:
            # Decode step
            input_ids_t, positions_t = engine.model_runner.prepare_decode(seqs)
            logits_engine = engine.model_runner.run_model(input_ids_t, positions_t, False)
            
            if logits_engine.dim() == 1:
                logits_engine = logits_engine.unsqueeze(0)
            
            token_id = logits_engine[0].argmax().item()
            decode_logits = logits_engine[0]
            
            gt_idx = step_count + 1  # +1 because prefill was step 0
            if is_main and gt_idx < len(ground_truth_tokens):
                top5_e, top5_ei = torch.topk(decode_logits, 5)
                print(f"\nDecode token {gt_idx+1}: engine={token_id} '{tokenizer.decode([token_id])}'")
                print(f"  Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(top5_e, top5_ei)]}")
                
                gt_logits = ground_truth_logits[gt_idx]
                diff = (decode_logits.float() - gt_logits.float()).abs()
                print(f"  Max logit diff vs ground truth: {diff.max().item():.6f}")
                print(f"  Mean logit diff: {diff.mean().item():.6f}")
                # Show where the biggest differences are
                top_diff_vals, top_diff_ids = torch.topk(diff, 5)
                print(f"  Biggest diffs at tokens: {[(tid.item(), f'{v.item():.4f}', tokenizer.decode([tid.item()])) for v, tid in zip(top_diff_vals, top_diff_ids)]}")
                match = "✓" if token_id == ground_truth_tokens[gt_idx] else "✗"
                print(f"  Match: {match} (engine={token_id}, gt={ground_truth_tokens[gt_idx]})")
            
            engine_tokens.append(token_id)
            engine_logits_list.append(decode_logits)
            
            from context import reset_context
            reset_context()
            engine.scheduler.postprocess(seqs, [token_id])
            if engine._stop_token_ids:
                from llm_engine import SequenceStatus
                for s in seqs:
                    if not s.is_finished and token_id in engine._stop_token_ids:
                        s.status = SequenceStatus.FINISHED
                        engine.scheduler.block_manager.deallocate(s)
                        if s in engine.scheduler.running:
                            engine.scheduler.running.remove(s)
            step_count += 1
    
    if is_main:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Ground truth: {ground_truth_tokens} = '{tokenizer.decode(ground_truth_tokens)}'")
        print(f"Engine:       {engine_tokens} = '{tokenizer.decode(engine_tokens)}'")
        matches = sum(1 for a, b in zip(engine_tokens, ground_truth_tokens) if a == b)
        print(f"Match: {matches}/{min(len(engine_tokens), len(ground_truth_tokens))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()
    test_engine(args)

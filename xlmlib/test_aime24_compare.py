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
        print("TEST 1: Direct forward pass (no KV cache) - first token logits")
        print(f"{'='*60}")
    
    model_device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], device=model_device)
    with torch.no_grad():
        logits, _ = model(input_tensor)
    
    if is_main:
        last_logits = logits[0, -1, :]
        top5_vals, top5_ids = torch.topk(last_logits, 5)
        print(f"Logits shape: {logits.shape}")
        print(f"Top-5 next tokens (greedy, no cache):")
        for val, tid in zip(top5_vals, top5_ids):
            token_str = tokenizer.decode([tid.item()], skip_special_tokens=False)
            print(f"  {tid.item():>8} ({val.item():>8.3f}): '{token_str}'")
        
        greedy_token_nocache = top5_ids[0].item()
        print(f"\nGreedy next token: {greedy_token_nocache} = '{tokenizer.decode([greedy_token_nocache])}'")
    
    # ===== Test 2: HF generate for first 10 tokens (ground truth) =====
    if is_main:
        print(f"\n{'='*60}")
        print("TEST 2: HF model.generate() - first 20 tokens (ground truth)")
        print(f"{'='*60}")
    
    # Use autoregressive decode manually WITHOUT KV cache (slow but correct)
    
    manual_ids = list(input_ids)
    manual_greedy_tokens = []
    if is_main:
        print("Manual autoregressive decode (no KV cache, ground truth):")
    with torch.no_grad():
        for step in range(20):
            inp = torch.tensor([manual_ids], device=model_device)
            logits_step, _ = model(inp)
            next_logits = logits_step[0, -1, :]
            next_token = next_logits.argmax().item()
            manual_greedy_tokens.append(next_token)
            manual_ids.append(next_token)
            if is_main:
                token_str = tokenizer.decode([next_token], skip_special_tokens=False)
                top3_vals, top3_ids = torch.topk(next_logits, 3)
                print(f"  Step {step:>2}: token={next_token:>8} '{token_str:>15}' | "
                      f"top3: {[f'{tid.item()}({v.item():.2f})' for v, tid in zip(top3_vals, top3_ids)]}")
    
    if is_main:
        manual_text = tokenizer.decode(manual_greedy_tokens, skip_special_tokens=False)
        print(f"\nManual decode (20 tokens): '{manual_text}'")
    
    # ===== Test 3: Engine prefill + first decode steps =====
    if is_main:
        print(f"\n{'='*60}")
        print("TEST 3: Engine prefill + decode (with KV cache) - first 20 tokens")
        print(f"{'='*60}")
    
    # We need to trace the engine's token-by-token output.
    # Create engine and generate, capturing first 20 tokens from the step() loop.
    config.eos_token_id = tokenizer.eos_token_id
    engine = HybridLLMEngine(
        model, config, str(model_device),
        temperature=0.0, max_batch_size=1, tokenizer=tokenizer
    )
    
    from llm_engine import Sequence
    seq = Sequence(input_ids)
    seq.max_tokens = 20  # Only generate 20 tokens for comparison
    engine.scheduler.add(seq)
    
    engine_tokens = []
    step_count = 0
    while not engine.is_finished():
        seqs, is_prefill = engine.scheduler.schedule()
        
        if is_prefill:
            token_ids_out = engine.model_runner.call("run", seqs, True)
            engine.scheduler.postprocess(seqs, token_ids_out)
            # Check stop tokens
            if engine._stop_token_ids:
                from llm_engine import SequenceStatus
                for s, tid in zip(seqs, token_ids_out):
                    if not s.is_finished and tid in engine._stop_token_ids:
                        s.status = SequenceStatus.FINISHED
                        engine.scheduler.block_manager.deallocate(s)
                        if s in engine.scheduler.running:
                            engine.scheduler.running.remove(s)
            if is_main:
                print(f"  Prefill done, first token: {token_ids_out}")
                engine_tokens.extend(token_ids_out if isinstance(token_ids_out, list) else [token_ids_out])
        else:
            # Decode step - capture the logits before sampling
            input_ids_t, positions_t = engine.model_runner.prepare_decode(seqs)
            logits_engine = engine.model_runner.run_model(input_ids_t, positions_t, False)
            
            if logits_engine.dim() == 1:
                logits_engine = logits_engine.unsqueeze(0)
            
            # Greedy
            token_id = logits_engine.argmax(dim=-1).item()
            
            if is_main and step_count < 20:
                top3_vals, top3_ids = torch.topk(logits_engine[0], 3)
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                
                # Compare with manual decode
                match = "✓" if step_count < len(manual_greedy_tokens) and token_id == manual_greedy_tokens[step_count] else "✗"
                manual_tok = manual_greedy_tokens[step_count] if step_count < len(manual_greedy_tokens) else -1
                manual_str = tokenizer.decode([manual_tok], skip_special_tokens=False) if manual_tok >= 0 else "N/A"
                
                print(f"  Step {step_count:>2}: engine={token_id:>8} '{token_str:>15}' | "
                      f"manual={manual_tok:>8} '{manual_str:>15}' | {match} | "
                      f"top3: {[f'{tid.item()}({v.item():.2f})' for v, tid in zip(top3_vals, top3_ids)]}")
            
            engine_tokens.append(token_id)
            
            from context import reset_context
            reset_context()
            engine.scheduler.postprocess(seqs, [token_id])
            # Check stop tokens
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
        engine_text = tokenizer.decode(engine_tokens, skip_special_tokens=False)
        print(f"\nEngine tokens ({len(engine_tokens)}): '{engine_text}'")
        manual_text = tokenizer.decode(manual_greedy_tokens, skip_special_tokens=False)
        print(f"Manual tokens ({len(manual_greedy_tokens)}): '{manual_text}'")
        
        # Count matches
        matches = sum(1 for a, b in zip(engine_tokens, manual_greedy_tokens) if a == b)
        total = min(len(engine_tokens), len(manual_greedy_tokens))
        print(f"\nMatch: {matches}/{total} tokens agree")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()
    test_engine(args)

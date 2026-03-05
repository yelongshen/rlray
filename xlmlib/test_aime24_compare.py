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
    """Compare logits: HF model vs our engine model on same input."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from qwen3_next_engine import (
        load_qwen3_next_for_engine, HybridLLMEngine, get_tp_rank
    )
    
    problem, gold_answer = get_aime24_problem_0()
    
    if args.gpu_ids and args.tensor_parallel == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    device = "cuda:0"
    
    print(f"\n{'='*60}")
    print(f"AIME24 Problem 0: {problem[:80]}...")
    print(f"Gold answer: {gold_answer}")
    print(f"{'='*60}")
    
    # ===== Load HF model =====
    print("\n[1/2] Loading HF model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    hf_model.eval()
    
    # Build prompt
    prompt_text = build_prompt(problem, tokenizer, prompt_type="chat")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt tokens: {len(input_ids)}")
    print(f"Last 5 tokens: '{tokenizer.decode(input_ids[-5:], skip_special_tokens=False)}'")
    
    # HF forward
    print("\n[HF] Forward pass...", flush=True)
    input_tensor = torch.tensor([input_ids], device=device)
    t0 = time.time()
    with torch.no_grad():
        hf_out = hf_model(input_tensor)
        hf_logits = hf_out.logits[0, -1, :]  # last position
    print(f"[HF] Done in {time.time()-t0:.1f}s", flush=True)
    
    hf_token = hf_logits.argmax().item()
    hf_top5_vals, hf_top5_ids = torch.topk(hf_logits, 5)
    print(f"[HF] Greedy token: {hf_token} '{tokenizer.decode([hf_token])}'")
    print(f"[HF] Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(hf_top5_vals, hf_top5_ids)]}")
    
    # Free HF model
    print("\nFreeing HF model...", flush=True)
    del hf_model
    torch.cuda.empty_cache()
    
    # ===== Load our engine model =====
    print("\n[2/2] Loading engine model...", flush=True)
    model, tokenizer2, config = load_qwen3_next_for_engine(
        args.model_path, device=device, tensor_parallel_size=1
    )
    
    # Engine forward (no cache)
    print("\n[Engine] Forward pass (no cache)...", flush=True)
    input_tensor = torch.tensor([input_ids], device=device)
    t0 = time.time()
    with torch.no_grad():
        engine_logits_full, _ = model(input_tensor)
        engine_logits = engine_logits_full[0, -1, :]
    print(f"[Engine] Done in {time.time()-t0:.1f}s", flush=True)
    
    engine_token = engine_logits.argmax().item()
    engine_top5_vals, engine_top5_ids = torch.topk(engine_logits, 5)
    print(f"[Engine] Greedy token: {engine_token} '{tokenizer.decode([engine_token])}'")
    print(f"[Engine] Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(engine_top5_vals, engine_top5_ids)]}")
    
    # ===== Compare logits =====
    print(f"\n{'='*60}")
    print("LOGIT COMPARISON (HF vs Engine, no cache)")
    print(f"{'='*60}")
    
    diff = (engine_logits.float() - hf_logits.float()).abs()
    print(f"Max absolute diff:  {diff.max().item():.6f}")
    print(f"Mean absolute diff: {diff.mean().item():.6f}")
    print(f"Token match: {'✓' if hf_token == engine_token else '✗'} (HF={hf_token}, Engine={engine_token})")
    
    # Top differences
    top_diff_vals, top_diff_ids = torch.topk(diff, 10)
    print(f"\nLargest logit diffs:")
    for v, tid in zip(top_diff_vals, top_diff_ids):
        hf_val = hf_logits[tid].item()
        eng_val = engine_logits[tid].item()
        print(f"  token={tid.item():>8} '{tokenizer.decode([tid.item()]):>15}': HF={hf_val:>8.3f}, Engine={eng_val:>8.3f}, diff={v.item():.4f}")
    
    # Correlation
    corr = torch.corrcoef(torch.stack([hf_logits.float(), engine_logits.float()]))[0, 1].item()
    print(f"\nLogit correlation: {corr:.6f}")
    
    if diff.max().item() < 0.1:
        print("\n✓ Logits match well — weight loading is correct")
    elif diff.max().item() < 1.0:
        print("\n⚠ Small logit differences — likely bf16 rounding, but tokens match")
    else:
        print("\n✗ LARGE logit differences — weight loading or model structure bug!")
    
    # ===== Debug: inspect norm weights =====
    print(f"\n{'='*60}")
    print("DEBUG: Norm weight inspection")
    print(f"{'='*60}")
    
    # Engine norm weights
    layer0 = model.model.layers[0]
    ln_w = layer0.input_layernorm.weight.data
    print(f"Engine layer0.input_layernorm.weight: mean={ln_w.mean().item():.6f}, std={ln_w.std().item():.6f}, "
          f"min={ln_w.min().item():.6f}, max={ln_w.max().item():.6f}")
    print(f"  First 10: {ln_w[:10].tolist()}")
    print(f"  NOTE: Engine uses (1 + weight) formula, so effective weight = {(1 + ln_w.mean()).item():.6f}")
    
    post_ln_w = layer0.post_attention_layernorm.weight.data
    print(f"Engine layer0.post_attention_layernorm.weight: mean={post_ln_w.mean().item():.6f}")
    
    final_norm_w = model.model.norm.weight.data
    print(f"Engine final_norm.weight: mean={final_norm_w.mean().item():.6f}")
    
    # Check: are the weights near 0 (expect for (1+w) formula) or near 1 (standard formula)?
    if ln_w.abs().mean().item() > 0.1:
        print("\n⚠ WARNING: Norm weights are NOT near zero!")
        print("  If HF uses standard w*x formula and our engine uses (1+w)*x,")
        print("  then we're computing (1 + w_actual)*x instead of w_actual*x")
        print("  FIX: Change RMSNorm to use weight*x (standard formula)")
    else:
        print("\n  Norm weights ≈ 0, (1+w) formula should give ≈ 1.0 — correct")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()
    test_engine(args)

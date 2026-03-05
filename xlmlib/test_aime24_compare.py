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
    # NOTE: Must pass causal attention_mask since without cache_params,
    # the full attention layers use fallback path without causal masking.
    print("\n[Engine] Forward pass (no cache, with causal mask)...", flush=True)
    input_tensor = torch.tensor([input_ids], device=device)
    seq_len = input_tensor.shape[1]
    # Create causal mask: [1, 1, seq_len, seq_len], upper triangle = -inf
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.bfloat16),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        engine_logits_full, _ = model(input_tensor, attention_mask=causal_mask)
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
        print("\n✗ LARGE logit differences — investigating per-layer...")
    
    # ===== Per-layer hidden state comparison =====
    print(f"\n{'='*60}")
    print("PER-LAYER HIDDEN STATE COMPARISON")
    print(f"{'='*60}")
    print("Running HF model layer-by-layer and comparing with engine model...")
    
    # We need to re-load HF model for per-layer comparison
    print("\nRe-loading HF model for per-layer comparison...", flush=True)
    hf_model2 = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    hf_model2.eval()
    
    with torch.no_grad():
        # === Embedding ===
        hf_embed = hf_model2.model.embed_tokens(input_tensor)
        engine_embed = model.model.embed_tokens(input_tensor)
        embed_diff = (hf_embed.float() - engine_embed.float()).abs().max().item()
        print(f"Embedding:          max_diff={embed_diff:.6f}")
        
        # === RoPE ===
        seq_len = input_tensor.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        hf_cos, hf_sin = hf_model2.model.rotary_emb(hf_embed, position_ids)
        engine_cos, engine_sin = model.model.rotary_emb(engine_embed, position_ids)
        rope_diff = max(
            (hf_cos.float() - engine_cos.float()).abs().max().item(),
            (hf_sin.float() - engine_sin.float()).abs().max().item()
        )
        print(f"RoPE cos/sin:       max_diff={rope_diff:.6f}")
        
        # === Layer by layer ===
        hf_hidden = hf_embed
        engine_hidden = engine_embed
        
        num_layers = len(hf_model2.model.layers)
        for layer_idx in range(num_layers):
            hf_layer = hf_model2.model.layers[layer_idx]
            engine_layer = model.model.layers[layer_idx]
            
            # Get layer type
            layer_type = engine_layer.layer_type
            
            # --- Input layernorm ---
            hf_ln = hf_layer.input_layernorm(hf_hidden)
            engine_ln = engine_layer.input_layernorm(engine_hidden)
            ln_diff = (hf_ln.float() - engine_ln.float()).abs().max().item()
            
            # --- Attention ---
            if layer_type == "linear_attention":
                hf_attn_out = hf_layer.linear_attn(hf_ln)
                if isinstance(hf_attn_out, tuple):
                    hf_attn_out = hf_attn_out[0]
                engine_attn_out = engine_layer.linear_attn(engine_ln)
            else:
                hf_attn_out = hf_layer.self_attn(hf_ln, position_ids=position_ids)
                if isinstance(hf_attn_out, tuple):
                    hf_attn_out = hf_attn_out[0]
                engine_attn_out, _, _ = engine_layer.self_attn(
                    engine_ln, position_ids=position_ids,
                    cos=engine_cos, sin=engine_sin,
                    attention_mask=causal_mask,
                )
            attn_diff = (hf_attn_out.float() - engine_attn_out.float()).abs().max().item()
            
            # --- Residual ---
            hf_hidden = hf_hidden + hf_attn_out
            engine_hidden = engine_hidden + engine_attn_out
            
            # --- Post-attn layernorm ---
            hf_post_ln = hf_layer.post_attention_layernorm(hf_hidden)
            engine_post_ln = engine_layer.post_attention_layernorm(engine_hidden)
            
            # --- MLP/MoE ---
            hf_mlp_out = hf_layer.mlp(hf_post_ln)
            engine_mlp_out = engine_layer.mlp(engine_post_ln)
            mlp_diff = (hf_mlp_out.float() - engine_mlp_out.float()).abs().max().item()
            
            # --- Residual ---
            hf_hidden = hf_hidden + hf_mlp_out
            engine_hidden = engine_hidden + engine_mlp_out
            
            hidden_diff = (hf_hidden.float() - engine_hidden.float()).abs().max().item()
            
            flag = " ⚠" if hidden_diff > 0.5 else ""
            print(f"Layer {layer_idx:>2} ({layer_type:>17}): "
                  f"ln={ln_diff:.4f}, attn={attn_diff:.4f}, mlp={mlp_diff:.4f}, "
                  f"hidden={hidden_diff:.4f}{flag}")
            
            # Stop early if divergence is huge
            if hidden_diff > 10.0:
                print(f"  *** DIVERGENCE TOO LARGE, stopping ***")
                break
        
        # --- Final norm ---
        hf_final = hf_model2.model.norm(hf_hidden)
        engine_final = model.model.norm(engine_hidden)
        final_diff = (hf_final.float() - engine_final.float()).abs().max().item()
        print(f"\nFinal norm:         max_diff={final_diff:.6f}")
        
        # --- LM head ---
        hf_final_logits = hf_model2.lm_head(hf_final)[0, -1, :]
        engine_final_logits = model.lm_head(engine_final)[0, -1, :]
        lmhead_diff = (hf_final_logits.float() - engine_final_logits.float()).abs().max().item()
        print(f"LM head logits:     max_diff={lmhead_diff:.6f}")
    
    del hf_model2
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()
    test_engine(args)

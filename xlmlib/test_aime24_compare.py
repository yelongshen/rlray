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

def build_prompt(problem_text, tokenizer, prompt_type="v11"):
    """Build prompt exactly as eval_qwen3_next does."""        
    prompt_text = process_math_prompt(problem_text, prompt_type=prompt_type)

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def test_engine(args):
    """
    Test A vs B:
    A: Full forward [x1..xn] → xn's logits (no cache)
    B: Prefill [x1..xn-1] with cache → Decode [xn] → xn's logits (with cache)
    
    Compares per-layer hidden states and final logits to verify
    that prefill+decode produces the same result as full forward.
    """
    from transformers import AutoTokenizer, AutoConfig
    from qwen3_next_engine import (
        load_qwen3_next_for_engine, Qwen3NextCacheParams, get_tp_rank
    )
    from context import set_context, get_context, reset_context
    
    problem, gold_answer = get_aime24_problem_0()
    
    if args.gpu_ids and args.tensor_parallel == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    device = "cuda:0"
    
    print(f"\n{'='*60}")
    print(f"TEST: Full forward (A) vs Prefill+Decode (B)")
    print(f"{'='*60}")
    
    # Load engine model
    print("\nLoading engine model...", flush=True)
    model, tokenizer, config = load_qwen3_next_for_engine(
        args.model_path, device=device, tensor_parallel_size=1
    )
    
    # Build prompt
    prompt_text = build_prompt(problem, tokenizer, prompt_type="v11")
    input_ids = tokenizer.encode(prompt_text)
    n = len(input_ids)
    print(f"Prompt tokens: {n}")
    
    # ===== Case A: Full forward [x1..xn] → xn's hidden/logits =====
    print(f"\n{'='*60}")
    print(f"CASE A: Full forward [{n} tokens], no cache")
    print(f"{'='*60}")
    
    input_tensor_full = torch.tensor([input_ids], device=device)
    causal_mask = torch.triu(
        torch.full((n, n), float('-inf'), device=device, dtype=torch.bfloat16),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    # Collect per-layer states for case A
    a_states = {}
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_tensor_full)
        position_ids = torch.arange(n, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, position_ids)
        
        num_layers = len(model.model.layers)
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            ln_out = layer.input_layernorm(hidden)
            
            if layer.layer_type == "linear_attention":
                attn_out = layer.linear_attn(ln_out)
            else:
                attn_out, _, _ = layer.self_attn(
                    ln_out, position_ids=position_ids,
                    cos=cos, sin=sin, attention_mask=causal_mask,
                )
            hidden = hidden + attn_out
            post_ln = layer.post_attention_layernorm(hidden)
            mlp_out = layer.mlp(post_ln)
            hidden = hidden + mlp_out
            
            # Save last token hidden state
            a_states[layer_idx] = hidden[0, -1, :].float().cpu().clone()
        
        a_final = model.model.norm(hidden)
        a_logits = model.lm_head(a_final)[0, -1, :]
        a_states['logits'] = a_logits.float().cpu().clone()
    
    a_token = a_logits.argmax().item()
    print(f"[A] Greedy token: {a_token} '{tokenizer.decode([a_token])}'")
    
    # ===== Case B: Prefill [x1..xn-1] with cache, then Decode [xn] =====
    print(f"\n{'='*60}")
    print(f"CASE B: Prefill [{n-1} tokens] + Decode [1 token], with cache")
    print(f"{'='*60}")
    
    # Allocate cache
    cache = Qwen3NextCacheParams(
        config=config, batch_size=1,
        free_memory_budget=int(2e9),  # 2GB is plenty for 1 seq
        device=torch.device(device), block_size=256,
    )
    
    # --- Prefill: [x1..xn-1] ---
    prefill_ids = input_ids[:-1]
    prefill_len = len(prefill_ids)
    input_tensor_prefill = torch.tensor([prefill_ids], device=device)
    
    # Need to set up context for paged attention (slot_mapping for full attn layers)
    # For simplicity, compute slot_mapping manually
    block_size = 256
    num_blocks_needed = (prefill_len + block_size - 1) // block_size
    slot_mapping_prefill = list(range(prefill_len))
    cu_seqlens_q = torch.tensor([0, prefill_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, prefill_len], dtype=torch.int32, device=device)
    block_table = list(range(num_blocks_needed))
    block_tables_t = torch.tensor([block_table], dtype=torch.int32, device=device)
    slot_mapping_t = torch.tensor(slot_mapping_prefill, dtype=torch.int32, device=device)
    context_lens_t = torch.tensor([prefill_len], dtype=torch.int32, device=device)
    
    set_context(True, cu_seqlens_q, cu_seqlens_k, prefill_len, prefill_len,
                slot_mapping_t, context_lens_t, block_tables_t)
    
    print(f"  Prefill: {prefill_len} tokens...", flush=True)
    with torch.no_grad():
        prefill_logits, _ = model(
            input_ids=input_tensor_prefill,
            position_ids=torch.arange(prefill_len, device=device).unsqueeze(0),
            cache_params=cache,
        )
    cache.has_previous_state = True
    reset_context()
    print(f"  Prefill done.", flush=True)
    
    # --- Decode: [xn] ---
    last_token = input_ids[-1]
    last_pos = n - 1
    decode_slot = block_table[-1] * block_size + (prefill_len % block_size)
    # Actually the next slot after prefill
    if prefill_len % block_size == 0:
        # Need a new block
        decode_block = num_blocks_needed
        decode_slot = decode_block * block_size
    else:
        decode_block = num_blocks_needed - 1
        decode_slot = decode_block * block_size + (prefill_len % block_size)
    
    slot_mapping_decode = torch.tensor([decode_slot], dtype=torch.int32, device=device)
    context_lens_decode = torch.tensor([n], dtype=torch.int32, device=device)
    block_table_decode = block_table + ([num_blocks_needed] if prefill_len % block_size == 0 else [])
    block_tables_decode = torch.tensor([block_table_decode], dtype=torch.int32, device=device)
    
    set_context(False, slot_mapping=slot_mapping_decode,
                context_lens=context_lens_decode, block_tables=block_tables_decode)
    
    # Collect per-layer states for case B decode
    b_states = {}
    with torch.no_grad():
        input_tensor_decode = torch.tensor([[last_token]], device=device)
        pos_decode = torch.tensor([[last_pos]], device=device)
        
        hidden = model.model.embed_tokens(input_tensor_decode)
        cos, sin = model.model.rotary_emb(hidden, pos_decode)
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            ln_out = layer.input_layernorm(hidden)
            
            if layer.layer_type == "linear_attention":
                attn_out = layer.linear_attn(ln_out, cache_params=cache)
            else:
                attn_out, _, _ = layer.self_attn(
                    ln_out, position_ids=pos_decode,
                    cos=cos, sin=sin, cache_params=cache,
                )
            hidden = hidden + attn_out
            post_ln = layer.post_attention_layernorm(hidden)
            mlp_out = layer.mlp(post_ln)
            hidden = hidden + mlp_out
            
            b_states[layer_idx] = hidden[0, -1, :].float().cpu().clone()
        
        b_final = model.model.norm(hidden)
        b_logits = model.lm_head(b_final)[0, -1, :]
        b_states['logits'] = b_logits.float().cpu().clone()
    
    reset_context()
    
    b_token = b_logits.argmax().item()
    print(f"[B] Greedy token: {b_token} '{tokenizer.decode([b_token])}'")
    
    # ===== Compare A vs B =====
    print(f"\n{'='*60}")
    print("A vs B: Per-layer comparison (last token hidden)")
    print(f"{'='*60}")
    
    for layer_idx in range(num_layers):
        diff = (a_states[layer_idx] - b_states[layer_idx]).abs().max().item()
        lt = model.model.layers[layer_idx].layer_type
        flag = " ⚠" if diff > 0.5 else ""
        print(f"Layer {layer_idx:>2} ({lt:>17}): hidden_diff={diff:.6f}{flag}")
    
    logit_diff = (a_states['logits'] - b_states['logits']).abs()
    print(f"\nLogit max diff: {logit_diff.max().item():.6f}")
    print(f"Logit mean diff: {logit_diff.mean().item():.6f}")
    corr = torch.corrcoef(torch.stack([a_states['logits'], b_states['logits']]))[0, 1].item()
    print(f"Logit correlation: {corr:.6f}")
    print(f"Token match: {'✓' if a_token == b_token else '✗'} (A={a_token}, B={b_token})")
    
    if logit_diff.max().item() < 0.01:
        print("\n✓ Perfect match — prefill+decode is equivalent to full forward")
    elif logit_diff.max().item() < 1.0:
        print("\n⚠ Small differences — likely bf16 rounding in cache read/write")
    else:
        print("\n✗ LARGE differences — KV cache or recurrent state bug!")


def test_hf_vs_engine(args):
    """
    Compare HF model vs Engine model (via HybridLLMEngine) layer-by-layer.
    
    Uses HF output_hidden_states=True for HF states, and
    HybridLLMEngine.run(debug=True) for engine states (with paged attention).
    
    This tests the real engine path including paged KV cache and flash attention,
    not just a raw forward pass with causal mask.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from qwen3_next_engine import (
        load_qwen3_next_for_engine, HybridLLMEngine, get_tp_rank
    )
    
    problem, gold_answer = get_aime24_problem_0()
    
    if args.gpu_ids and args.tensor_parallel == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    device = "cuda:0"
    
    print(f"\n{'='*60}")
    print(f"TEST: HF vs Engine (HybridLLMEngine) — layer-wise comparison")
    print(f"{'='*60}")
    
    # === Step 1: HF model forward ===
    print("\nLoading HF model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    hf_model.eval()
    
    # Build prompt
    prompt_text = build_prompt(problem, tokenizer, prompt_type="v11")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)
    seq_len = len(input_ids)
    print(f"Prompt tokens: {seq_len}")
    
    # HF forward with output_hidden_states=True
    print("\n[HF] Forward with output_hidden_states=True...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        hf_out = hf_model(input_tensor, output_hidden_states=True)
    print(f"[HF] Done in {time.time()-t0:.1f}s", flush=True)
    
    hf_logits = hf_out.logits[0, -1, :]
    hf_token = hf_logits.argmax().item()
    print(f"[HF] Greedy token: {hf_token} '{tokenizer.decode([hf_token])}'")
    
    # hf_out.hidden_states: tuple of (num_layers + 1) tensors
    #   [0] = embedding output, [i] = after layer i-1 for i=1..N
    num_hf_states = len(hf_out.hidden_states)
    num_layers = num_hf_states - 1
    print(f"[HF] Hidden states: {num_hf_states} entries (embedding + {num_layers} layers)")
    
    # Save HF states to CPU (last token only)
    hf_last_token_states = []
    for hs in hf_out.hidden_states:
        hf_last_token_states.append(hs[0, -1, :].float().cpu().clone())
    hf_logits_cpu = hf_logits.float().cpu().clone()
    
    # Free HF model completely
    del hf_model, hf_out
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("[HF] Freed from GPU", flush=True)
    
    # === Step 2: Engine model via HybridLLMEngine ===
    print("\nLoading engine model via load_qwen3_next_for_engine...", flush=True)
    model, tokenizer2, config = load_qwen3_next_for_engine(
        args.model_path, device=device, tensor_parallel_size=args.tensor_parallel
    )
    
    print("\nCreating HybridLLMEngine...", flush=True)
    engine = HybridLLMEngine(
        model, config, device,
        temperature=0, max_batch_size=1, tokenizer=tokenizer2
    )
    
    # Use the controllable generation API with debug=True
    print("\n[Engine] Prefill via HybridLLMEngine (paged attention)...", flush=True)
    engine.prepare([input_ids], max_tokens=1)
    eng_input_ids, eng_position_ids = engine.schedule()
    t0 = time.time()
    engine_hidden_states = engine.run(eng_input_ids, eng_position_ids, debug=True)
    print(f"[Engine] Done in {time.time()-t0:.1f}s", flush=True)
    
    # engine_hidden_states layout:
    #   [0]   = embedding output          [1, seq_len, H]
    #   [i]   = after layer i-1           [1, seq_len, H]  for i=1..N
    #   [N+1] = after final RMSNorm       [1, seq_len, H]
    num_eng_states = len(engine_hidden_states)
    print(f"[Engine] Hidden states: {num_eng_states} entries "
          f"(embedding + {num_layers} layers + final_norm)")
    
    # Compute engine logits from the final norm output
    with torch.no_grad():
        final_hidden = engine_hidden_states[-1]  # after final norm
        last_hidden = final_hidden[:, -1:, :] if final_hidden.dim() == 3 else final_hidden[-1:, :].unsqueeze(0)
        engine_logits = model.lm_head(last_hidden).squeeze()
    
    engine_token = engine_logits.argmax().item()
    print(f"[Engine] Greedy token: {engine_token} '{tokenizer2.decode([engine_token])}'")
    
    # === Step 3: Per-layer comparison ===
    layer_types = getattr(config, 'layer_types', ['full_attention'] * num_layers)
    
    print(f"\n{'='*60}")
    print("PER-LAYER COMPARISON (last token hidden state)")
    print(f"{'='*60}")
    
    # Compare embedding (index 0)
    eng_embed_last = engine_hidden_states[0][0, -1, :].float().cpu()
    embed_diff = (eng_embed_last - hf_last_token_states[0]).abs().max().item()
    print(f"{'Embedding':>20}: max_diff={embed_diff:.6f}")
    
    # Compare each layer (index 1..N)
    for layer_idx in range(num_layers):
        lt = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        
        eng_hs = engine_hidden_states[layer_idx + 1]  # +1 because [0] is embedding
        eng_last = eng_hs[0, -1, :].float().cpu()
        hf_last = hf_last_token_states[layer_idx + 1]
        
        diff = (eng_last - hf_last).abs().max().item()
        flag = " ⚠" if diff > 0.5 else ""
        print(f"Layer {layer_idx:>2} ({lt:>17}): max_diff={diff:.6f}{flag}")
    
    # === Logit comparison ===
    engine_logits_cpu = engine_logits.float().cpu()
    logit_diff = (engine_logits_cpu - hf_logits_cpu).abs()
    
    print(f"\n{'='*60}")
    print("LOGIT COMPARISON (HF vs Engine via HybridLLMEngine)")
    print(f"{'='*60}")
    print(f"Max absolute diff:  {logit_diff.max().item():.6f}")
    print(f"Mean absolute diff: {logit_diff.mean().item():.6f}")
    corr = torch.corrcoef(torch.stack([hf_logits_cpu, engine_logits_cpu]))[0, 1].item()
    print(f"Logit correlation:  {corr:.6f}")
    print(f"Token match: {'✓' if hf_token == engine_token else '✗'} "
          f"(HF={hf_token} '{tokenizer2.decode([hf_token])}', "
          f"Engine={engine_token} '{tokenizer2.decode([engine_token])}')")
    
    if logit_diff.max().item() < 0.01:
        print("\n✓ Perfect match")
    elif logit_diff.max().item() < 1.0:
        print("\n⚠ Small differences — likely bf16 rounding across layers")
    else:
        print("\n⚠ Expected drift from bf16 accumulation across 48 layers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--test", type=str, default="ab", choices=["ab", "hf", "both"],
                        help="ab=A vs B (prefill+decode), hf=HF vs Engine layer-wise, both=run both")
    args = parser.parse_args()
    
    if args.test in ("ab", "both"):
        test_engine(args)
    if args.test in ("hf", "both"):
        test_hf_vs_engine(args)

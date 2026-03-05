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
    
    # ===== Single load: HF model → collect states → convert to engine =====
    print("\nLoading HF model (single load)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    hf_model.eval()
    
    # Build prompt
    prompt_text = build_prompt(problem, tokenizer, prompt_type="chat")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)
    seq_len = input_tensor.shape[1]
    print(f"Prompt tokens: {seq_len}")
    
    # === HF full forward (for top-level logit comparison) ===
    print("\n[HF] Full forward pass...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        hf_out = hf_model(input_tensor)
        hf_logits = hf_out.logits[0, -1, :]
    print(f"[HF] Done in {time.time()-t0:.1f}s", flush=True)
    
    hf_token = hf_logits.argmax().item()
    hf_top5_vals, hf_top5_ids = torch.topk(hf_logits, 5)
    print(f"[HF] Greedy token: {hf_token} '{tokenizer.decode([hf_token])}'")
    print(f"[HF] Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(hf_top5_vals, hf_top5_ids)]}")
    
    # Save HF logits to CPU
    hf_logits_cpu = hf_logits.float().cpu().clone()
    
    # === HF per-layer states ===
    print("\n[HF] Collecting per-layer states...", flush=True)
    hf_layer_states = {}
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    with torch.no_grad():
        hf_hidden = hf_model.model.embed_tokens(input_tensor)
        num_layers = len(hf_model.model.layers)
        
        for layer_idx in range(num_layers):
            hf_layer = hf_model.model.layers[layer_idx]
            hf_ln = hf_layer.input_layernorm(hf_hidden)
            
            if hasattr(hf_layer, 'linear_attn') and hf_layer.linear_attn is not None:
                lt = "linear_attention"
                hf_attn_out = hf_layer.linear_attn(hf_ln)
                if isinstance(hf_attn_out, tuple): hf_attn_out = hf_attn_out[0]
            else:
                lt = "full_attention"
                hf_cos, hf_sin = hf_model.model.rotary_emb(hf_ln, position_ids)
                hf_causal = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.bfloat16), diagonal=1).unsqueeze(0).unsqueeze(0)
                hf_attn_out = hf_layer.self_attn(hf_ln, position_embeddings=(hf_cos, hf_sin), attention_mask=hf_causal)
                if isinstance(hf_attn_out, tuple): hf_attn_out = hf_attn_out[0]
            
            hf_hidden = hf_hidden + hf_attn_out
            hf_post_ln = hf_layer.post_attention_layernorm(hf_hidden)
            hf_mlp_out = hf_layer.mlp(hf_post_ln)
            hf_hidden = hf_hidden + hf_mlp_out
            
            hf_layer_states[layer_idx] = {
                'type': lt,
                'ln': hf_ln[0, -1, :].float().cpu().clone(),
                'attn': hf_attn_out[0, -1, :].float().cpu().clone(),
                'mlp': hf_mlp_out[0, -1, :].float().cpu().clone(),
                'hidden': hf_hidden[0, -1, :].float().cpu().clone(),
            }
            if layer_idx % 8 == 0:
                print(f"  HF layer {layer_idx}/{num_layers}", flush=True)
        
        hf_final = hf_model.model.norm(hf_hidden)
        hf_layer_states['final_norm'] = hf_final[0, -1, :].float().cpu().clone()
    
    print(f"  HF states collected for {num_layers} layers", flush=True)
    
    # === Convert HF → Engine model (no second disk load) ===
    print("\n[Convert] HF → Engine model (move HF to CPU, create engine on GPU)...", flush=True)
    hf_model_cpu = hf_model.to("cpu")
    torch.cuda.empty_cache()
    
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    from qwen3_next_engine import Qwen3NextForLLMEngine, _copy_weights
    engine_model = Qwen3NextForLLMEngine(hf_config, use_tp=False)
    engine_model = engine_model.to(device=device, dtype=torch.bfloat16)
    _copy_weights(hf_model_cpu, engine_model, hf_config, use_tp=False, target_device=device, target_dtype=torch.bfloat16)
    engine_model.eval()
    
    del hf_model, hf_model_cpu
    torch.cuda.empty_cache()
    
    # === Engine full forward (for top-level logit comparison) ===
    print("\n[Engine] Full forward pass (no cache, with causal mask)...", flush=True)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.bfloat16),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        engine_logits_full, _ = engine_model(input_tensor, attention_mask=causal_mask)
        engine_logits = engine_logits_full[0, -1, :]
    print(f"[Engine] Done in {time.time()-t0:.1f}s", flush=True)
    
    engine_token = engine_logits.argmax().item()
    engine_top5_vals, engine_top5_ids = torch.topk(engine_logits, 5)
    print(f"[Engine] Greedy token: {engine_token} '{tokenizer.decode([engine_token])}'")
    print(f"[Engine] Top-5: {[(tid.item(), f'{v.item():.3f}', tokenizer.decode([tid.item()])) for v, tid in zip(engine_top5_vals, engine_top5_ids)]}")
    
    # === Logit comparison ===
    print(f"\n{'='*60}")
    print("LOGIT COMPARISON (HF vs Engine)")
    print(f"{'='*60}")
    
    diff = (engine_logits.float().cpu() - hf_logits_cpu).abs()
    print(f"Max absolute diff:  {diff.max().item():.6f}")
    print(f"Mean absolute diff: {diff.mean().item():.6f}")
    print(f"Token match: {'✓' if hf_token == engine_token else '✗'} (HF={hf_token}, Engine={engine_token})")
    corr = torch.corrcoef(torch.stack([hf_logits_cpu, engine_logits.float().cpu()]))[0, 1].item()
    print(f"Logit correlation: {corr:.6f}")
    
    # === Per-layer comparison (engine vs saved HF states) ===
    print(f"\n{'='*60}")
    print("PER-LAYER COMPARISON (last token hidden state):")
    print(f"{'='*60}")
    
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.bfloat16),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        engine_hidden = engine_model.model.embed_tokens(input_tensor)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        engine_cos, engine_sin = engine_model.model.rotary_emb(engine_hidden, position_ids)
        
        for layer_idx in range(num_layers):
            engine_layer = engine_model.model.layers[layer_idx]
            hf_state = hf_layer_states[layer_idx]
            lt = hf_state['type']
            
            engine_ln = engine_layer.input_layernorm(engine_hidden)
            ln_diff = (engine_ln[0, -1, :].float().cpu() - hf_state['ln']).abs().max().item()
            
            if lt == "linear_attention":
                engine_attn_out = engine_layer.linear_attn(engine_ln)
            else:
                engine_attn_out, _, _ = engine_layer.self_attn(
                    engine_ln, position_ids=position_ids,
                    cos=engine_cos, sin=engine_sin, attention_mask=causal_mask,
                )
            attn_diff = (engine_attn_out[0, -1, :].float().cpu() - hf_state['attn']).abs().max().item()
            
            engine_hidden = engine_hidden + engine_attn_out
            engine_post_ln = engine_layer.post_attention_layernorm(engine_hidden)
            engine_mlp_out = engine_layer.mlp(engine_post_ln)
            mlp_diff = (engine_mlp_out[0, -1, :].float().cpu() - hf_state['mlp']).abs().max().item()
            
            engine_hidden = engine_hidden + engine_mlp_out
            hidden_diff = (engine_hidden[0, -1, :].float().cpu() - hf_state['hidden']).abs().max().item()
            
            flag = " ⚠" if hidden_diff > 0.5 else ""
            print(f"Layer {layer_idx:>2} ({lt:>17}): "
                  f"ln={ln_diff:.4f}, attn={attn_diff:.4f}, mlp={mlp_diff:.4f}, "
                  f"hidden={hidden_diff:.4f}{flag}")
            
            if hidden_diff > 10.0:
                print(f"  *** DIVERGENCE TOO LARGE, stopping ***")
                break
        
        engine_final = engine_model.model.norm(engine_hidden)
        final_diff = (engine_final[0, -1, :].float().cpu() - hf_layer_states['final_norm']).abs().max().item()
        print(f"\nFinal norm:         max_diff={final_diff:.6f}")
    
    # === Test: Engine prefill WITH cache + 5 decode steps ===
    print(f"\n{'='*60}")
    print("ENGINE GENERATION TEST (prefill + 5 decode steps)")
    print(f"{'='*60}")
    
    from qwen3_next_engine import HybridLLMEngine
    hf_config.eos_token_id = tokenizer.eos_token_id
    
    # Collect stop tokens
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for name in ['<|im_end|>', '<|endoftext|>']:
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid is not None: stop_ids.add(tid)
        except: pass
    hf_config.stop_token_ids = stop_ids
    
    engine = HybridLLMEngine(
        engine_model, hf_config, device,
        temperature=0.0, max_batch_size=1, tokenizer=tokenizer
    )
    
    from llm_engine import Sequence
    seq = Sequence(input_ids)
    seq.max_tokens = 5
    engine.scheduler.add(seq)
    
    gen_tokens = []
    while not engine.is_finished():
        seqs, is_prefill = engine.scheduler.schedule()
        token_ids_out = engine.model_runner.call("run", seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids_out)
        
        if is_prefill:
            print(f"  Prefill: token={token_ids_out} = '{tokenizer.decode(token_ids_out, skip_special_tokens=False)}'")
            gen_tokens.extend(token_ids_out)
        else:
            for tid in token_ids_out:
                tok_str = tokenizer.decode([tid], skip_special_tokens=False)
                print(f"  Decode: token={tid} = '{tok_str}'")
                gen_tokens.append(tid)
        
        # Check stop
        if engine._stop_token_ids:
            from llm_engine import SequenceStatus
            for s, tid in zip(seqs, token_ids_out):
                if not s.is_finished and tid in engine._stop_token_ids:
                    s.status = SequenceStatus.FINISHED
                    engine.scheduler.block_manager.deallocate(s)
                    if s in engine.scheduler.running:
                        engine.scheduler.running.remove(s)
    
    print(f"\nEngine first 5 tokens: {gen_tokens}")
    print(f"Decoded: '{tokenizer.decode(gen_tokens, skip_special_tokens=False)}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Qwen_Qwen3-Coder-Next/")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()
    test_engine(args)

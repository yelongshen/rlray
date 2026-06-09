import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xlmlib.inctx_evolution_distill import (  # noqa: E402
    build_evolution_prompt,
    build_evolution_traces,
    load_jsonl,
    load_traces_jsonl,
    retrieve_relevant_traces,
    save_traces_jsonl,
)
from xlmlib.math_util import safe_math_answer_timeout  # noqa: E402


def load_eval_data(path: str) -> List[Dict]:
    data = []
    for row in load_jsonl(path):
        problem = row.get("problem", row.get("question", ""))
        answer = row.get("answer", "")
        if not problem:
            continue
        data.append(
            {
                "id": str(row.get("id", row.get("unique_id", len(data)))),
                "problem": str(problem),
                "answer": str(answer),
            }
        )
    return data


def generate_with_hf(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16000).to(
        model.device
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
    start = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][start:], skip_special_tokens=True)


def evaluate(args):
    if args.trace_path and os.path.exists(args.trace_path):
        traces = load_traces_jsonl(args.trace_path)
    else:
        events = list(load_jsonl(args.rollout_events_path))
        traces = build_evolution_traces(
            events,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            keep_only_improving_steps=not args.keep_negative_steps,
        )
        if args.trace_path:
            os.makedirs(os.path.dirname(args.trace_path), exist_ok=True)
            save_traces_jsonl(traces, args.trace_path)

    print(f"Loaded traces: {len(traces)}")
    if not traces:
        raise RuntimeError("No traces available. Check rollout events input.")

    eval_data = load_eval_data(args.eval_data)
    if args.max_samples > 0:
        eval_data = eval_data[: args.max_samples]
    print(f"Loaded eval samples: {len(eval_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    correct = 0.0
    total = 0
    t0 = time.time()
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(eval_data):
            selected = retrieve_relevant_traces(sample["problem"], traces, top_k=args.num_shots)
            prompt = build_evolution_prompt(sample["problem"], selected, prompt_type=args.prompt_type)

            response = generate_with_hf(
                model,
                tokenizer,
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

            _, extracted, reward = safe_math_answer_timeout(
                response,
                [sample["answer"]],
                tokenizer,
                prompt_type=args.prompt_type,
                timeout=args.verify_timeout,
            )

            correct += reward
            total += 1
            rec = {
                "id": sample["id"],
                "problem": sample["problem"],
                "gold_answer": sample["answer"],
                "pred_answer": extracted,
                "reward": reward,
                "num_traces": len(selected),
                "response": response,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (idx + 1) % max(1, args.log_every) == 0:
                print(f"[{idx + 1}/{len(eval_data)}] pass@1={correct / max(1, total):.4f}")

    elapsed = time.time() - t0
    metric = correct / max(1, total)
    print("=" * 70)
    print(f"In-context evolution distillation pass@1: {metric:.4f}")
    print(f"Samples: {total}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Saved detailed outputs to: {args.output_path}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="In-context evolution distillation experiments")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)

    parser.add_argument(
        "--rollout_events_path",
        type=str,
        default="",
        help="JSONL with flat RL events. Used when --trace_path does not exist.",
    )
    parser.add_argument(
        "--trace_path",
        type=str,
        default="eval/inctx_traces.jsonl",
        help="Path to built traces jsonl (load if exists, else build and save).",
    )

    parser.add_argument("--output_path", type=str, default="eval/inctx_evolution_results.jsonl")
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--prompt_type", type=str, default="v11")

    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--keep_negative_steps", action="store_true")

    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--verify_timeout", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.trace_path and not args.rollout_events_path:
        raise ValueError("Need either --trace_path or --rollout_events_path")
    if not os.path.exists(args.eval_data):
        raise FileNotFoundError(f"Eval data not found: {args.eval_data}")
    if (not args.trace_path or not os.path.exists(args.trace_path)) and not args.rollout_events_path:
        raise ValueError("Trace file does not exist and --rollout_events_path is empty")
    evaluate(args)
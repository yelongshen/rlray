import os
import sys
from collections import deque
import types
import argparse

import torch


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from qwen3_next_engine import HybridLLMEngine
from llm_engine import Sequence


class _DummyBlockManager:
    def deallocate(self, seq):
        return


class _FakeScheduler:
    def __init__(self):
        self.waiting = deque()
        self.running = deque()
        self.block_manager = _DummyBlockManager()

    def add(self, seq):
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 251 for c in text]


def _build_engine_stub():
    engine = HybridLLMEngine.__new__(HybridLLMEngine)
    engine.scheduler = _FakeScheduler()
    engine.tokenizer = _DummyTokenizer()
    engine._stop_token_ids = set()

    def _fake_schedule_mixed(self):
        if len(self.scheduler.waiting) == 0 and len(self.scheduler.running) == 0:
            return []
        return [([], False, None, None)]

    def _fake_run(self, input_ids=None, position_ids=None, debug=False):
        if len(self.scheduler.waiting) > 0:
            seq = self.scheduler.waiting.popleft()
            self.scheduler.running.append(seq)
        if len(self.scheduler.running) > 0:
            seq = self.scheduler.running.popleft()
            self._finished_outputs[seq.seq_id] = [900 + (seq.seq_id % 10)]
        return None

    engine.schedule_mixed = types.MethodType(_fake_schedule_mixed, engine)
    engine.run = types.MethodType(_fake_run, engine)
    return engine


def test_feed_run_batch_user_api():
    engine = _build_engine_stub()
    batch_prompt = [
        Sequence([1, 2]),
        Sequence([3, 4, 5]),
    ]
    batch_prompt[0].max_generated_tokens = 64
    batch_prompt[1].max_generated_tokens = 128

    engine.feed(batch_prompt)
    assert len(batch_prompt) == 2
    assert batch_prompt[0].max_generated_tokens == 64 and batch_prompt[1].max_generated_tokens == 128
    assert all(len(seq.completion_token_ids) == 0 for seq in batch_prompt), "sampled tokens should start empty"

    engine.run_batch(yield_partial=True)
    assert any(len(seq.completion_token_ids) > 0 for seq in batch_prompt), (
        "sampled tokens should be visible on sequence handles after one batch tick"
    )

    for seq in batch_prompt:
        if seq.is_finished and len(seq) <= 100000:
            seq.add_context([120, 120, 120])
            seq.max_tokens = 32

    engine.run_batch(yield_partial=False)
    assert all(seq.is_finished for seq in batch_prompt)


def run_all_tests():
    test_feed_run_batch_user_api()
    print("continuous batching controllable API unit tests passed")


def run_real_engine_test(
    model_path: str,
    prompts,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tensor_parallel: int = 1,
    max_steps: int = 5,
    with_feedback: bool = True,
):
    from qwen3_next_engine import load_qwen3_next_for_engine

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model, tokenizer, llm_config = load_qwen3_next_for_engine(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        tensor_parallel_size=tensor_parallel,
    )

    engine = HybridLLMEngine(
        model,
        llm_config,
        next(model.parameters()).device,
        temperature=0.6,
        top_k=0,
        top_p=0.95,
        max_batch_size=max(2, len(prompts)),
        prefill_one_by_one=False,
        tokenizer=tokenizer,
    )

    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    split = len(prompt_ids) // 2
    per_prompt_max_tokens = [128 if i < split else 256 for i in range(len(prompt_ids))]
    print(f"[real-test] per-prompt max_tokens={per_prompt_max_tokens}")
    batch_prompt = [
        Sequence(ids, max_generated_tokens=per_prompt_max_tokens[i])
        for i, ids in enumerate(prompt_ids)
    ]
    
    engine.feed(batch_prompt)

    feedback_round_done = set()
    ever_finished = set()
    steps = 0

    def detok(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=False)

    def validate_or_reflect(seq, text):
        if with_feedback and seq not in feedback_round_done:
            return {
                "action": "continue",
                "feed": "\nPlease verify your last answer, then provide a concise corrected final answer.",
            }
        return {"action": "done"}

    print(f"[real-test] start with {len(prompts)} prompts")
    while steps < max_steps and (not engine.is_finished()):
        steps += 1
        engine.run_batch(yield_partial=True)

        for idx, seq in enumerate(batch_prompt):
            if not seq.is_finished or seq in ever_finished:
                continue
            full_ids = seq.token_ids
            out_text = detok(full_ids)
            print(f"[real-test] step={steps} finished seq={idx} total_len={len(seq)}")
            print(f"[real-test] output preview: {out_text[:240]!r}")

            decision = validate_or_reflect(seq, out_text)
            if decision["action"] == "continue":
                seq.add_context(tokenizer.encode(decision["feed"], add_special_tokens=False))
                seq.max_tokens = 128
                feedback_round_done.add(seq)
                print(f"[real-test] seq.add_context seq={idx}")
            else:
                ever_finished.add(seq)
                print(f"[real-test] finished seq={idx}")

    expected_finished = len(batch_prompt)
    assert len(ever_finished) >= expected_finished and engine.is_finished(), (
        f"real test incomplete: finished={len(ever_finished)} < expected={expected_finished}, "
        f"steps={steps}, engine_finished={engine.is_finished()}"
    )
    print("[real-test] continuous batching controllable schema passed")


def _parse_args():
    parser = argparse.ArgumentParser(description="Continuous batching controllable API tests")
    parser.add_argument("--mode", choices=["stub", "real"], default="stub")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--no-feedback", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "stub":
        run_all_tests()
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --mode real")
        prompts = args.prompt or [
            "Solve: If 2x + 5 = 17, what is x?",
            "Compute: What is the derivative of x^3 + 2x?",
        ]
        run_real_engine_test(
            model_path=args.model_path,
            prompts=prompts,
            device=args.device,
            dtype=args.dtype,
            tensor_parallel=args.tensor_parallel,
            max_steps=args.max_steps,
            with_feedback=not args.no_feedback,
        )

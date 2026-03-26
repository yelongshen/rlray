import os
import sys
import argparse
import types
from collections import deque

import torch


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Ensure `xlmlib.*` imports work when script is run as a file path.
if 'xlmlib' not in sys.modules:
    sys.modules['xlmlib'] = types.ModuleType('xlmlib')
    sys.modules['xlmlib'].__path__ = [_SCRIPT_DIR]

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


def _set_seq_max_tokens(seq: Sequence, max_tokens: int):
    if hasattr(seq, "max_generation_tokens"):
        seq.max_generation_tokens = max_tokens
    if hasattr(seq, "max_tokens"):
        seq.max_tokens = max_tokens


def _build_engine_stub():
    engine = HybridLLMEngine.__new__(HybridLLMEngine)
    engine.scheduler = _FakeScheduler()
    engine.tokenizer = _DummyTokenizer()

    def _fake_step(self, auto_finish=True):
        outputs = []
        if len(self.scheduler.waiting) > 0:
            seq = self.scheduler.waiting.popleft()
            seq.append_token(900 + (seq.seq_id % 10))
            seq.status = type(seq.status).FINISHED
            outputs.append((seq.seq_id, seq.completion_token_ids))
            if not auto_finish:
                self.scheduler.waiting.appendleft(seq)
        return outputs

    def _fake_is_finished(self):
        return len(self.scheduler.waiting) == 0 and len(self.scheduler.running) == 0

    engine.step = _fake_step.__get__(engine, HybridLLMEngine)
    engine.is_finished = _fake_is_finished.__get__(engine, HybridLLMEngine)
    return engine


def test_feed_run_batch_user_api_stub():
    engine = _build_engine_stub()
    batch_prompt = [
        Sequence([1, 2], max_generation_tokens=64),
        Sequence([3, 4, 5], max_generation_tokens=128),
    ]
    _set_seq_max_tokens(batch_prompt[0], 64)
    _set_seq_max_tokens(batch_prompt[1], 128)

    engine.feed(batch_prompt)
    assert len(batch_prompt) == 2
    assert all(len(seq.completion_token_ids) == 0 for seq in batch_prompt)

    engine.run_batch(yield_partial=True)
    assert any(len(seq.completion_token_ids) > 0 for seq in batch_prompt)



def run_all_stub_tests():
    test_feed_run_batch_user_api_stub()
    print("stub continuous batching unit tests passed")


def run_real_engine_test(
    model_path: str,
    prompts,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tensor_parallel: int = 1,
    enable_feedback_round: bool = True,
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
        model=model,
        llm_config=llm_config,
        device=next(model.parameters()).device,
        temperature=0.6,
        top_k=0,
        top_p=0.95,
        max_batch_size=max(2, len(prompts)),
        prefill_one_by_one=False,
        tokenizer=tokenizer,
    )

    max_generation_len = 96
    max_turns = 2

    prompt_ids = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids.append(tokenizer.encode(prompt_text, add_special_tokens=False))
    seqs = []
    for ids in prompt_ids:
        seq = Sequence(ids, max_generation_tokens=max_generation_len)
        _set_seq_max_tokens(seq, max_generation_len)
        seqs.append(seq)

    engine.feed(seqs)

    def _ensure_enqueued(seq: Sequence):
        if seq in engine.scheduler.waiting or seq in engine.scheduler.running:
            return
        engine.scheduler.add(seq)

    def _finalize_sequence(seq: Sequence):
        if seq in engine.scheduler.waiting:
            engine.scheduler.waiting.remove(seq)
        if seq in engine.scheduler.running:
            engine.scheduler.running.remove(seq)
        try:
            engine.scheduler.block_manager.deallocate(seq)
        except Exception:
            pass
        try:
            engine.model_runner.release_linear_slots([seq])
        except Exception:
            pass

    def _decode_tail(seq: Sequence, n: int = 120):
        try:
            return tokenizer.decode(seq.token_ids[-n:], skip_special_tokens=False)
        except Exception:
            return "<decode-failed>"

    print(f"[real-test] start: n_prompts={len(seqs)}")

    while not engine.is_finished():
        engine.run_batch(yield_partial=True)

        for idx, seq in enumerate(seqs):
            if not seq.is_finished:
                continue

            if enable_feedback_round and seq.turn_count < max_turns:
                print(
                    f"[real-test] finished seq={idx} "
                    f"prompt_tokens={len(seq.prompt_token_ids)} completion_tokens={len(seq.completion_token_ids)}"
                )
                print(f"[real-test] tail[{idx}]: {_decode_tail(seq)!r}")
                feedback_text = "\nPlease verify the above answer and provide a concise corrected final answer."
                feedback_messages = [{"role": "user", "content": feedback_text}]
                feedback_prompt_text = tokenizer.apply_chat_template(
                    feedback_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                feedback_ids = tokenizer.encode(feedback_prompt_text, add_special_tokens=False)
                seq.add_context(feedback_ids)
                _set_seq_max_tokens(seq, max_generation_len)
                _ensure_enqueued(seq)
                print(f"[real-test] seq={idx} add_context(feedback_round)")
            else:
                _finalize_sequence(seq)
    for seq in seqs:
        if not seq.is_finished:
            _ensure_enqueued(seq)

    if not engine.is_finished():
        print("[real-test] forcing full drain with run_batch(yield_partial=False)")
        engine.run_batch(yield_partial=False)

    all_finished = all(seq.is_finished for seq in seqs)
    all_have_completion = all(len(seq.completion_token_ids) > 0 for seq in seqs)

    assert all_finished, "not all sequences finished"
    assert all_have_completion, "some sequences have empty completion"

    print("[real-test] continuous batching real-engine test passed")
    for idx, seq in enumerate(seqs):
        try:
            decoded_text = tokenizer.decode(seq.token_ids, skip_special_tokens=False)
        except Exception:
            decoded_text = "<decode-failed>"
        print(
            f"[real-test] final seq={idx}: "
            f"prompt_token_ids={seq.prompt_token_ids}, completion_token_ids={seq.completion_token_ids}, "
            f"decoded_text={decoded_text!r}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Continuous batching unit tests")
    parser.add_argument("--mode", choices=["stub", "real"], default="stub")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--no-feedback", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "stub":
        run_all_stub_tests()
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --mode real")
        prompts = args.prompt or [
            "Solve: If 2x + 5 = 17, what is x?",
            "Compute: What is the derivative of x^3 + 2x?",
            "Give a short explanation of Bayes rule.",
        ]
        run_real_engine_test(
            model_path=args.model_path,
            prompts=prompts,
            device=args.device,
            dtype=args.dtype,
            tensor_parallel=args.tensor_parallel,
            enable_feedback_round=not args.no_feedback,
        )

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
    engine._run_calls = []

    def _fake_run_schedule_once(self, force_decode=False):
        self._run_calls.append(force_decode)
        if not force_decode and len(self.scheduler.waiting) > 0:
            seq = self.scheduler.waiting.popleft()
            self.scheduler.running.append(seq)
            return
        if len(self.scheduler.running) > 0:
            seq = self.scheduler.running.popleft()
            self._finished_outputs[seq.seq_id] = [900 + (seq.seq_id % 10)]

    engine._run_schedule_once = types.MethodType(_fake_run_schedule_once, engine)
    return engine


def test_continuous_batching_partial_generation():
    engine = _build_engine_stub()
    prompt_list = [[11, 12], [21, 22, 23]]
    engine.prepare(prompt_list, max_tokens=32)

    ids_1, outs_1 = engine.generate_partial(yield_finished=True, continuous_batching=True)
    assert engine._run_calls == [False, True], f"unexpected run call order: {engine._run_calls}"
    assert len(ids_1) == 1 and len(outs_1) == 1, "first partial should finish exactly one request"
    assert ids_1[0] == 0, f"expected first external id=0, got {ids_1[0]}"

    ids_2, outs_2 = engine.generate_partial(yield_finished=True, continuous_batching=True)
    assert engine._run_calls[-2:] == [False, True], "second partial should also do prefill+decode"
    assert len(ids_2) == 1 and len(outs_2) == 1, "second partial should finish exactly one request"
    assert ids_2[0] == 1, f"expected second external id=1, got {ids_2[0]}"

    ids_3, outs_3 = engine.generate_partial(yield_finished=True, continuous_batching=True)
    assert ids_3 == [] and outs_3 == [], "no outputs expected after all requests are finished"


def test_stream_feed_and_done_schema():
    engine = _build_engine_stub()
    engine.prepare([[1, 2, 3]], max_tokens=16)

    finished_ids, finished_outs = engine.generate_partial(yield_finished=True, continuous_batching=True)
    assert finished_ids == [0], f"expected finished external id [0], got {finished_ids}"
    assert len(finished_outs) == 1 and len(finished_outs[0]) > 0, "finished output should be non-empty"

    before_len = len(engine._external_histories[0])
    engine.stream_feed([0], ["feedback + new question"])
    after_len = len(engine._external_histories[0])
    assert after_len > before_len, "history should extend after stream_feed"
    assert 0 in engine._active_seq_by_external, "stream_feed should reactivate external id"
    assert len(engine.scheduler.waiting) > 0, "stream_feed should enqueue a new waiting sequence"

    engine.done([0])
    assert 0 not in engine._active_seq_by_external, "done() should remove active sequence"
    assert 0 in engine._closed_external_ids, "done() should mark external id closed"

    waiting_before = len(engine.scheduler.waiting)
    engine.stream_feed([0], ["should be ignored because closed"])
    waiting_after = len(engine.scheduler.waiting)
    assert waiting_after == waiting_before, "closed external id must not be re-enqueued"


def test_per_prompt_max_tokens_schema():
    engine = _build_engine_stub()
    engine.prepare([[10], [20, 21]], max_tokens=[3, 7])

    seq0 = engine._active_seq_by_external[0]
    seq1 = engine._active_seq_by_external[1]
    assert seq0.max_tokens == 3, f"expected prompt0 max_tokens=3, got {seq0.max_tokens}"
    assert seq1.max_tokens == 7, f"expected prompt1 max_tokens=7, got {seq1.max_tokens}"

    finished_ids, _ = engine.generate_partial(yield_finished=True, continuous_batching=True)
    assert finished_ids == [0], f"expected first finished external id [0], got {finished_ids}"

    engine.stream_feed([0], ["retry answer"])
    seq0_reactivated = engine._active_seq_by_external[0]
    assert seq0_reactivated.max_tokens == 3, (
        f"reactivated external id should keep max_tokens=3, got {seq0_reactivated.max_tokens}"
    )

    finished_ids_2, _ = engine.generate_partial(yield_finished=True, continuous_batching=True)
    assert finished_ids_2 == [1], f"expected second finished external id [1], got {finished_ids_2}"

    engine.stream_feed([1], [[101, 102]])
    seq1_reactivated = engine._active_seq_by_external[1]
    assert seq1_reactivated.max_tokens == 7, (
        f"reactivated external id should keep max_tokens=7, got {seq1_reactivated.max_tokens}"
    )


def test_per_prompt_max_tokens_validation():
    engine = _build_engine_stub()

    try:
        engine.prepare([[1], [2]], max_tokens=[5])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched max_tokens length")

    try:
        engine.prepare([[1]], max_tokens=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive global max_tokens")


def test_step_continuous_and_multiturn_api():
    engine = _build_engine_stub()
    engine.prepare([[5, 6], [7, 8]], max_tokens=[128, 256])

    finished_1 = engine.step_continuous()
    assert len(finished_1) == 1, f"expected 1 finished item at step1, got {len(finished_1)}"
    external_id_1, out_tokens_1 = finished_1[0]
    assert external_id_1 == 0, f"expected first external id=0, got {external_id_1}"
    assert len(out_tokens_1) > 0, "finished tokens should be non-empty"

    activated = engine.add_multiturn_from_finished(
        followup_prompts=[[101, 102]],
        base_seq_ids=[external_id_1],
        max_tokens=128,
        inherit_max_tokens=False,
    )
    assert activated == [external_id_1], f"expected activated [0], got {activated}"
    resumed_seq = engine._active_seq_by_external[external_id_1]
    assert resumed_seq.max_tokens == 128, f"expected resumed max_tokens=128, got {resumed_seq.max_tokens}"

    finished_2 = engine.step_continuous()
    assert len(finished_2) == 1, f"expected 1 finished item at step2, got {len(finished_2)}"
    external_id_2, _ = finished_2[0]
    assert external_id_2 == 1, f"expected second external id=1, got {external_id_2}"

    try:
        engine.add_multiturn_from_finished(
            followup_prompts=[[]],
            base_seq_ids=[external_id_2],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty followup prompt")


def run_all_tests():
    test_continuous_batching_partial_generation()
    test_stream_feed_and_done_schema()
    test_per_prompt_max_tokens_schema()
    test_per_prompt_max_tokens_validation()
    test_step_continuous_and_multiturn_api()
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
    engine.prepare(prompt_ids, max_tokens=per_prompt_max_tokens)

    feedback_round_done = set()
    closed_ids = set()
    steps = 0

    def tok(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def detok(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=False)

    def validate_or_reflect(seq_id, text):
        if with_feedback and seq_id not in feedback_round_done:
            return {
                "action": "continue",
                "feed": "\nPlease verify your last answer, then provide a concise corrected final answer.",
            }
        return {"action": "done"}

    print(f"[real-test] start with {len(prompts)} prompts")
    while steps < max_steps and (not engine.is_finished()):
        steps += 1
        finished = engine.step_continuous()

        if not finished:
            continue

        for seq_id, out_tokens in finished:
            out_text = detok(out_tokens)
            print(f"[real-test] step={steps} finished id={seq_id} out_len={len(out_tokens)}")
            print(f"[real-test] output preview: {out_text[:240]!r}")

            decision = validate_or_reflect(seq_id, out_text)
            if decision["action"] == "continue":
                followup = tok(decision["feed"])
                engine.add_multiturn_from_finished(
                    followup_prompts=[followup],
                    base_seq_ids=[seq_id],
                    max_tokens=128,
                    inherit_max_tokens=False,
                )
                feedback_round_done.add(seq_id)
                print(f"[real-test] add_multiturn_from_finished id={seq_id}")
            else:
                engine.done([seq_id])
                closed_ids.add(seq_id)
                print(f"[real-test] done id={seq_id}")

    expected_min_closed = len(prompts)
    assert len(closed_ids) >= expected_min_closed, (
        f"real test incomplete: closed={len(closed_ids)} < expected={expected_min_closed}, "
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

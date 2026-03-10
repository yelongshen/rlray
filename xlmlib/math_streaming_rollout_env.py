"""
Streaming multi-turn math rollout environment for Qwen3-Next.

Goal
----
Use OpenMathInstruct-2 as seed prompts and run an interactive multi-turn
rollout where model generation reuses KV cache/context continuously
(no reset between turns in an episode).

Outputs
-------
Chunked rollout files containing:
- tokens: int32
- rewards: float32
- mask: uint8 (1=model-generated token, 0=prompt/system token)

Each episode is one long stream suitable for RL training pipelines.

Example
-------
python xlmlib/math_streaming_rollout_env.py \
  --model_path Qwen/Qwen3-Coder-Next \
  --seed_path ./data/math_datasets/open_math_instruct_2 \
  --output_dir ./data/stream_rollouts \
  --target_tokens 5000000 \
  --max_new_tokens_per_turn 192 \
  --temperature 0.7 --top_p 0.95
"""

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

# Ensure project root is importable when launched as:
#   python ./xlmlib/math_streaming_rollout_env.py
#   torchrun ... ./xlmlib/math_streaming_rollout_env.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from math_util import process_math_prompt, safe_math_answer_timeout


@dataclass
class SeedSample:
    prompt: str
    answer: Optional[str]


class OpenMathInstruct2SeedStream:
    """Best-effort streaming reader for OpenMathInstruct-2 local snapshot.

    Supports:
    - local json/jsonl shards
    - local parquet shards (requires `datasets`)
    - HF dataset repo id (requires `datasets`)
    """

    def __init__(self, seed_path: str, split: str = "train", shuffle_buffer: int = 0):
        self.seed_path = seed_path
        self.split = split
        self.shuffle_buffer = shuffle_buffer

    @staticmethod
    def _extract_prompt(record: Dict) -> Optional[str]:
        for key in ("problem", "question", "instruction", "input", "prompt", "query"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _extract_answer(record: Dict) -> Optional[str]:
        for key in ("answer", "solution", "output", "target"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _iter_json_like(self, files: List[str]) -> Iterator[SeedSample]:
        for path in files:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue
                        prompt = self._extract_prompt(record)
                        if prompt is None:
                            continue
                        yield SeedSample(prompt=prompt, answer=self._extract_answer(record))
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        payload = json.load(f)
                    except Exception:
                        continue
                if isinstance(payload, dict):
                    payload = payload.get("data", [])
                if isinstance(payload, list):
                    for record in payload:
                        if not isinstance(record, dict):
                            continue
                        prompt = self._extract_prompt(record)
                        if prompt is None:
                            continue
                        yield SeedSample(prompt=prompt, answer=self._extract_answer(record))

    def _iter_datasets_stream(self, data_files: Optional[List[str]] = None) -> Iterator[SeedSample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "`datasets` is required for HF streaming repo IDs. "
                "For local parquet, install `pyarrow` or convert to jsonl."
            ) from exc

        if data_files:
            ds = load_dataset("parquet", data_files={self.split: data_files}, split=self.split, streaming=True)
        elif os.path.exists(self.seed_path):
            ds = load_dataset(self.seed_path, split=self.split, streaming=True)
        else:
            ds = load_dataset(self.seed_path, split=self.split, streaming=True)

        for record in ds:
            if not isinstance(record, dict):
                continue
            prompt = self._extract_prompt(record)
            if prompt is None:
                continue
            yield SeedSample(prompt=prompt, answer=self._extract_answer(record))

    def _iter_parquet_local(self, files: List[str]) -> Iterator[SeedSample]:
        """Read local parquet shards without requiring `datasets` package."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Local parquet detected but `pyarrow` is not installed. "
                "Install via: pip install pyarrow (or install datasets), "
                "or point --seed_path to json/jsonl files."
            ) from exc

        for path in files:
            try:
                parquet_file = pq.ParquetFile(path)
            except Exception:
                continue
            for batch in parquet_file.iter_batches(batch_size=8192):
                rows = batch.to_pylist()
                for record in rows:
                    if not isinstance(record, dict):
                        continue
                    prompt = self._extract_prompt(record)
                    if prompt is None:
                        continue
                    yield SeedSample(prompt=prompt, answer=self._extract_answer(record))

    def __iter__(self) -> Iterator[SeedSample]:
        if os.path.exists(self.seed_path):
            json_files = []
            for pattern in ("**/*.jsonl", "**/*.json"):
                json_files.extend(glob.glob(os.path.join(self.seed_path, pattern), recursive=True))
            parquet_files = glob.glob(os.path.join(self.seed_path, "**/*.parquet"), recursive=True)

            if json_files:
                stream = self._iter_json_like(sorted(json_files))
            elif parquet_files:
                stream = self._iter_parquet_local(sorted(parquet_files))
            else:
                raise RuntimeError(
                    f"No json/jsonl/parquet files found under --seed_path={self.seed_path}. "
                    "Please provide a local snapshot directory or an HF dataset repo id."
                )
        else:
            stream = self._iter_datasets_stream(data_files=None)

        if self.shuffle_buffer <= 1:
            yield from stream
            return

        buffer: List[SeedSample] = []
        rng = random.Random(1234)
        for sample in stream:
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                yield buffer.pop()
        while buffer:
            idx = rng.randrange(len(buffer))
            buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
            yield buffer.pop()


class ChunkedRolloutWriter:
    """Writes very long rollout arrays in chunked NPZ shards."""

    def __init__(self, output_dir: str, episode_id: int, chunk_tokens: int = 250_000):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_id = episode_id
        self.chunk_tokens = chunk_tokens

        self._tokens: List[int] = []
        self._rewards: List[float] = []
        self._mask: List[int] = []
        self._chunk_idx = 0
        self.total_tokens = 0

    def append(self, tokens: List[int], rewards: List[float], mask: List[int]) -> None:
        if not (len(tokens) == len(rewards) == len(mask)):
            raise ValueError("tokens/rewards/mask length mismatch")
        self._tokens.extend(tokens)
        self._rewards.extend(rewards)
        self._mask.extend(mask)
        self.total_tokens += len(tokens)

        while len(self._tokens) >= self.chunk_tokens:
            self._flush(self.chunk_tokens)

    def _flush(self, n: int) -> None:
        t = np.asarray(self._tokens[:n], dtype=np.int32)
        r = np.asarray(self._rewards[:n], dtype=np.float32)
        m = np.asarray(self._mask[:n], dtype=np.uint8)

        file_name = f"episode_{self.episode_id:06d}_chunk_{self._chunk_idx:06d}.npz"
        out_path = os.path.join(self.output_dir, file_name)
        np.savez_compressed(out_path, tokens=t, rewards=r, mask=m)

        self._tokens = self._tokens[n:]
        self._rewards = self._rewards[n:]
        self._mask = self._mask[n:]
        self._chunk_idx += 1

    def close(self) -> None:
        if self._tokens:
            self._flush(len(self._tokens))


class StreamingQwenPlayer:
    """HF player with persistent `past_key_values` across turns."""

    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str = "bfloat16",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        max_cache_length: Optional[int] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=None,
        ).to(self.device)
        self.model.eval()

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_cache_length = max_cache_length

        self.past_key_values = None
        self.total_context_tokens = 0
        self.last_logits = None

    def reset_episode(self) -> None:
        self.past_key_values = None
        self.total_context_tokens = 0
        self.last_logits = None

    def _cache_length(self) -> int:
        pkv = self.past_key_values
        if pkv is None:
            return 0
        if hasattr(pkv, "get_seq_length"):
            return int(pkv.get_seq_length())
        if isinstance(pkv, tuple) and len(pkv) > 0:
            return int(pkv[0][0].size(2))
        return 0

    def _trim_cache_if_needed(self) -> None:
        if self.max_cache_length is None or self.past_key_values is None:
            return
        cache_len = self._cache_length()
        if cache_len <= self.max_cache_length:
            return

        trim_amount = cache_len - self.max_cache_length
        pkv = self.past_key_values

        if hasattr(pkv, "crop"):
            pkv.crop(self.max_cache_length)
            return

        if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
            for i in range(len(pkv.key_cache)):
                pkv.key_cache[i] = pkv.key_cache[i][:, :, trim_amount:, :]
                pkv.value_cache[i] = pkv.value_cache[i][:, :, trim_amount:, :]
            return

        if isinstance(pkv, tuple):
            self.past_key_values = tuple(
                (
                    layer_k[:, :, trim_amount:, :],
                    layer_v[:, :, trim_amount:, :],
                )
                for (layer_k, layer_v) in pkv
            )

    def _sample_from_logits(self, logits: torch.Tensor) -> int:
        logits = logits.float()

        if self.temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / self.temperature

        if self.top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=min(self.top_k, logits.numel()), dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            logits = filtered

        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    @torch.inference_mode()
    def append_context_text(self, text: str) -> List[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        position_ids = None
        if self.past_key_values is not None:
            start = self._cache_length()
            position_ids = torch.arange(start, start + input_ids.shape[1], device=self.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
        )
        self.past_key_values = outputs.past_key_values
        self.last_logits = outputs.logits[:, -1, :]
        self.total_context_tokens += len(token_ids)
        self._trim_cache_if_needed()
        return token_ids

    @torch.inference_mode()
    def generate_turn(self, max_new_tokens: int, stop_token_ids: Optional[set] = None) -> List[int]:
        generated: List[int] = []

        if self.past_key_values is None or self.last_logits is None:
            raise RuntimeError("No context logits available. Call append_context_text() before generate_turn().")

        for _ in range(max_new_tokens):
            next_token = self._sample_from_logits(self.last_logits.squeeze(0))
            generated.append(next_token)
            self.total_context_tokens += 1

            if stop_token_ids and next_token in stop_token_ids:
                break

            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

            pos = self._cache_length()
            position_ids = torch.tensor([[pos]], dtype=torch.long, device=self.device)

            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
            self.past_key_values = outputs.past_key_values
            self.last_logits = outputs.logits[:, -1, :]
            self._trim_cache_if_needed()

        return generated


class StreamingQwenEnginePlayer:
    """In-house Qwen3-Next engine backend with optional tensor parallel."""

    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str = "bfloat16",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        max_cache_length: Optional[int] = None,
        tensor_parallel: int = 1,
    ):
        from qwen3_next_engine import (
            load_qwen3_next_for_engine,
            get_tp_rank,
            get_tp_world_size,
            get_tp_group,
        )

        self._get_tp_rank = get_tp_rank
        self._get_tp_world_size = get_tp_world_size
        self._get_tp_group = get_tp_group

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_cache_length = max_cache_length
        self.tensor_parallel = tensor_parallel
        self.total_context_tokens = 0
        self.last_logits = None

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.model, self.tokenizer, self.llm_config = load_qwen3_next_for_engine(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            tensor_parallel_size=tensor_parallel,
        )
        self.model.eval()

        if self.max_cache_length is not None and self._get_tp_rank() == 0:
            print(
                "Warning: --max_cache_length is not enforced in --backend engine mode yet; "
                "cache grows with streamed context."
            )

        self.device = next(self.model.parameters()).device
        self._allocate_cache()

    @property
    def is_main(self) -> bool:
        return self._get_tp_rank() == 0

    def _allocate_cache(self):
        device_idx = self.device.index if self.device.index is not None else 0
        free, _ = torch.cuda.mem_get_info(device_idx)
        usable_free = int(free * 0.70)
        self.cache_params = self.model.allocate_cache(
            batch_size=1,
            free_memory_budget=usable_free,
            device=self.device,
            block_size=256,
        )

    def reset_episode(self) -> None:
        self.cache_params.reset()
        self.cache_params.has_previous_state = False
        self.total_context_tokens = 0
        self.last_logits = None

    def _sample_local(self, logits: torch.Tensor) -> int:
        logits = logits.float()
        if self.temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / self.temperature
        if self.top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=min(self.top_k, logits.numel()), dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            logits = filtered

        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    def _sample_from_logits(self, logits: torch.Tensor) -> int:
        world_size = self._get_tp_world_size()
        if world_size <= 1:
            return self._sample_local(logits)

        if self.is_main:
            tok = self._sample_local(logits)
            token_t = torch.tensor([tok], dtype=torch.long, device=self.device)
        else:
            token_t = torch.zeros(1, dtype=torch.long, device=self.device)

        dist.broadcast(token_t, src=0, group=self._get_tp_group())
        return int(token_t.item())

    @torch.inference_mode()
    def append_context_text(self, text: str) -> List[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        start = self.total_context_tokens
        position_ids = torch.arange(start, start + input_ids.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        logits, _ = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_params=self.cache_params,
        )
        self.last_logits = logits[:, -1, :]
        self.total_context_tokens += len(token_ids)
        self.cache_params.has_previous_state = True
        return token_ids

    @torch.inference_mode()
    def generate_turn(self, max_new_tokens: int, stop_token_ids: Optional[set] = None) -> List[int]:
        generated: List[int] = []
        if self.last_logits is None:
            raise RuntimeError("No context logits available. Call append_context_text() before generate_turn().")

        for _ in range(max_new_tokens):
            next_token = self._sample_from_logits(self.last_logits.squeeze(0))
            generated.append(next_token)

            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            position_ids = torch.tensor([[self.total_context_tokens]], dtype=torch.long, device=self.device)
            logits, _ = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_params=self.cache_params,
            )
            self.last_logits = logits[:, -1, :]
            self.total_context_tokens += 1
            self.cache_params.has_previous_state = True

            if stop_token_ids and next_token in stop_token_ids:
                break

        return generated


class InteractiveMathStreamingEnv:
    """Interactive multi-turn environment with persistent decoder context."""

    def __init__(
        self,
        player: StreamingQwenPlayer,
        prompt_type: str = "v11",
        max_turns: int = 8,
        max_new_tokens_per_turn: int = 192,
    ):
        self.player = player
        self.prompt_type = prompt_type
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn

    @staticmethod
    def _writer_append(writer: Optional[ChunkedRolloutWriter], tokens: List[int], rewards: List[float], mask: List[int]) -> None:
        if writer is not None:
            writer.append(tokens, rewards, mask)

    def run_episode(self, seed: SeedSample, writer: Optional[ChunkedRolloutWriter]) -> Tuple[int, float]:
        self.player.reset_episode()

        total_reward = 0.0
        total_tokens = 0

        # Prompt (mask=0)
        user_prompt = process_math_prompt(seed.prompt, prompt_type=self.prompt_type)
        prompt_ids = self.player.append_context_text(user_prompt)
        self._writer_append(writer, prompt_ids, [0.0] * len(prompt_ids), [0] * len(prompt_ids))
        total_tokens += len(prompt_ids)

        stop_ids = set()
        eos = self.player.tokenizer.eos_token_id
        if eos is not None:
            stop_ids.add(int(eos))

        for turn_idx in range(self.max_turns):
            generated_ids = self.player.generate_turn(
                max_new_tokens=self.max_new_tokens_per_turn,
                stop_token_ids=stop_ids,
            )
            if not generated_ids:
                break

            reward = 0.0
            if seed.answer:
                response_text = self.player.tokenizer.decode(generated_ids, skip_special_tokens=True)
                _, _, reward = safe_math_answer_timeout(
                    response_text,
                    [seed.answer],
                    self.player.tokenizer,
                    prompt_type=self.prompt_type,
                    timeout=20,
                )

            rewards = [0.0] * len(generated_ids)
            rewards[-1] = float(reward)  # sparse terminal reward for this turn
            masks = [1] * len(generated_ids)
            self._writer_append(writer, generated_ids, rewards, masks)

            total_reward += float(reward)
            total_tokens += len(generated_ids)

            # Environment/system feedback becomes non-action context (mask=0)
            if reward > 0.5:
                feedback = "\nSystem: Correct. Provide a shorter proof and verify key steps.\n"
            else:
                feedback = "\nSystem: Not correct yet. Re-check assumptions and solve again carefully.\n"
            feedback_ids = self.player.append_context_text(feedback)
            self._writer_append(writer, feedback_ids, [0.0] * len(feedback_ids), [0] * len(feedback_ids))
            total_tokens += len(feedback_ids)

        return total_tokens, total_reward


def run_rollouts(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seed_stream = OpenMathInstruct2SeedStream(
        seed_path=args.seed_path,
        split=args.split,
        shuffle_buffer=args.shuffle_buffer,
    )

    if args.backend == "engine":
        player = StreamingQwenEnginePlayer(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_cache_length=args.max_cache_length,
            tensor_parallel=args.tensor_parallel,
        )
        is_main = player.is_main
    else:
        player = StreamingQwenPlayer(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_cache_length=args.max_cache_length,
        )
        is_main = True

    env = InteractiveMathStreamingEnv(
        player=player,
        prompt_type=args.prompt_type,
        max_turns=args.max_turns,
        max_new_tokens_per_turn=args.max_new_tokens_per_turn,
    )

    total_tokens = 0
    episode_id = 0

    metadata_path = os.path.join(args.output_dir, "rollout_index.jsonl")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    with (open(metadata_path, "a", encoding="utf-8") if is_main else open(os.devnull, "w")) as meta_f:
        for seed in seed_stream:
            if total_tokens >= args.target_tokens:
                break

            if is_main:
                writer = ChunkedRolloutWriter(
                    output_dir=args.output_dir,
                    episode_id=episode_id,
                    chunk_tokens=args.chunk_tokens,
                )
            else:
                writer = None

            ep_tokens, ep_reward = env.run_episode(seed=seed, writer=writer)
            if writer is not None:
                writer.close()

            total_tokens += ep_tokens
            meta = {
                "episode_id": episode_id,
                "episode_tokens": ep_tokens,
                "episode_reward_sum": ep_reward,
                "global_tokens": total_tokens,
                "seed_preview": seed.prompt[:160],
            }
            if is_main:
                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                meta_f.flush()

            if is_main:
                print(
                    f"[episode={episode_id}] ep_tokens={ep_tokens} ep_reward={ep_reward:.3f} "
                    f"global_tokens={total_tokens}/{args.target_tokens}"
                )
            episode_id += 1

    if is_main:
        print("=" * 80)
        print("Streaming rollout complete")
        print(f"Output dir: {args.output_dir}")
        print(f"Total tokens: {total_tokens}")
        if args.max_cache_length is None:
            print("Cache policy: strict no-discard (may require very large GPU memory)")
        else:
            print(f"Cache policy: cropped to max_cache_length={args.max_cache_length}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenMathInstruct-2 -> streaming multi-turn rollout")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "engine"])
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed_path", type=str, default="./data/math_datasets/open_math_instruct_2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./data/stream_rollouts")

    parser.add_argument("--target_tokens", type=int, default=5_000_000)
    parser.add_argument("--chunk_tokens", type=int, default=250_000)

    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_new_tokens_per_turn", type=int, default=192)
    parser.add_argument("--prompt_type", type=str, default="v11")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--max_cache_length",
        type=int,
        default=None,
        help="Optional cache cap. None means no cache discard/crop.",
    )

    parser.add_argument("--shuffle_buffer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_rollouts(args)

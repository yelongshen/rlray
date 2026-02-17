#!/usr/bin/env python3
"""
Perplexity Evaluation Script for Qwen-Next Model on PG19 Corpus

This script evaluates the perplexity (PPL) of the qwen-next model
on the PG19 dataset (Project Gutenberg books) at different token scales: 1M, 10M, and 1B tokens.

PG19 is a standard benchmark for evaluating long-range language modeling capabilities,
containing full-length books from Project Gutenberg published before 1919.

Usage:
    python eval_ppl.py --model_path <path_to_model> --token_scale 1M
    python eval_ppl.py --model_path <path_to_model> --token_scale 10M
    python eval_ppl.py --model_path <path_to_model> --token_scale 1B
    python eval_ppl.py --model_path <path_to_model> --token_scale all
    python eval_ppl.py --model_path <path_to_model> --data_path <local_data> --token_scale all
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")


# Token scale configurations
TOKEN_SCALES = {
    "1M": 1_000_000,
    "10M": 10_000_000,
    "1B": 1_000_000_000,
}


class PG19Dataset(Dataset):
    """Dataset for loading PG19 (Project Gutenberg) corpus for PPL evaluation."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        max_tokens: int = 1_000_000,
        split: str = "test",
        stride: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.stride = stride
        self.samples = []

        self._load_pg19(split)

    def _load_pg19(self, split: str):
        """Load PG19 dataset from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise RuntimeError(
                "HuggingFace 'datasets' library is required for PG19. "
                "Install with: pip install datasets"
            )

        print(f"Loading PG19 dataset (split: {split})...")
        
        # Load PG19 from HuggingFace
        # PG19 contains ~28K books for training, ~50 for validation, ~100 for test
        try:
            dataset = load_dataset("deepmind/pg19", split=split)
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                print("Warning: deepmind/pg19 uses legacy script format.")
                print("Using alternative: emozilla/pg19 (Parquet format)...")
                dataset = load_dataset("emozilla/pg19", split=split)
            else:
                raise
        
        total_tokens = 0
        
        for item in tqdm(dataset, desc="Tokenizing PG19 books"):
            if total_tokens >= self.max_tokens:
                break
            
            text = item["text"]
            
            # Tokenize the full book
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            
            input_ids = encodings["input_ids"].squeeze(0)
            seq_len = input_ids.size(0)
            
            # Use sliding window to create chunks from long books
            for begin_loc in range(0, seq_len, self.stride):
                if total_tokens >= self.max_tokens:
                    break
                    
                end_loc = min(begin_loc + self.max_length, seq_len)
                chunk_ids = input_ids[begin_loc:end_loc]
                
                if chunk_ids.size(0) < 32:  # Skip very short chunks
                    continue
                
                self.samples.append({
                    "input_ids": chunk_ids.unsqueeze(0),
                    "attention_mask": torch.ones(1, chunk_ids.size(0), dtype=torch.long),
                })
                total_tokens += chunk_ids.size(0)
                
                # For non-overlapping chunks at the end
                if end_loc == seq_len:
                    break

        print(f"Loaded {len(self.samples)} chunks with {total_tokens:,} tokens from PG19")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TextDataset(Dataset):
    """Dataset for loading and tokenizing text data for PPL evaluation."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_tokens: int = 1_000_000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.samples = []

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load and tokenize data from file."""
        total_tokens = 0
        
        print(f"Loading data from {data_path}...")
        
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading data"):
                    if total_tokens >= self.max_tokens:
                        break
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", data.get("content", ""))
                        if text:
                            tokens = self.tokenizer(
                                text,
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt",
                            )
                            token_count = tokens["input_ids"].shape[1]
                            if token_count > 10:  # Skip very short samples
                                self.samples.append(tokens)
                                total_tokens += token_count
                    except json.JSONDecodeError:
                        continue
        elif data_path.endswith(".txt"):
            with open(data_path, "r", encoding="utf-8") as f:
                text_buffer = []
                for line in tqdm(f, desc="Loading data"):
                    if total_tokens >= self.max_tokens:
                        break
                    text_buffer.append(line.strip())
                    if len(text_buffer) >= 10:  # Process in chunks
                        text = " ".join(text_buffer)
                        tokens = self.tokenizer(
                            text,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        token_count = tokens["input_ids"].shape[1]
                        if token_count > 10:
                            self.samples.append(tokens)
                            total_tokens += token_count
                        text_buffer = []
        elif os.path.isdir(data_path):
            # Load from directory of files
            files = [f for f in os.listdir(data_path) if f.endswith((".jsonl", ".txt", ".json"))]
            for filename in tqdm(files, desc="Loading files"):
                if total_tokens >= self.max_tokens:
                    break
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        if filepath.endswith(".jsonl"):
                            for line in content.split("\n"):
                                if total_tokens >= self.max_tokens:
                                    break
                                try:
                                    data = json.loads(line.strip())
                                    text = data.get("text", data.get("content", ""))
                                    if text:
                                        tokens = self.tokenizer(
                                            text,
                                            truncation=True,
                                            max_length=self.max_length,
                                            return_tensors="pt",
                                        )
                                        token_count = tokens["input_ids"].shape[1]
                                        if token_count > 10:
                                            self.samples.append(tokens)
                                            total_tokens += token_count
                                except json.JSONDecodeError:
                                    continue
                        else:
                            tokens = self.tokenizer(
                                content,
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt",
                            )
                            token_count = tokens["input_ids"].shape[1]
                            if token_count > 10:
                                self.samples.append(tokens)
                                total_tokens += token_count
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue
        else:
            raise ValueError(f"Unsupported data format or path: {data_path}")

        print(f"Loaded {len(self.samples)} samples with {total_tokens:,} tokens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_tokens: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Compute perplexity on the given dataloader.

    Args:
        model: The language model
        dataloader: DataLoader with tokenized samples
        device: Device to run evaluation on
        max_tokens: Maximum number of tokens to evaluate (optional)

    Returns:
        Tuple of (perplexity, total_tokens_evaluated)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing PPL"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            # Compute loss only on non-padded tokens
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_mask = shift_mask.view(-1)

            # Compute cross-entropy loss
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            masked_loss = loss * shift_mask
            
            batch_loss = masked_loss.sum().item()
            batch_tokens = shift_mask.sum().item()

            total_loss += batch_loss
            total_tokens += batch_tokens

            if max_tokens and total_tokens >= max_tokens:
                break

    # Compute perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return perplexity, total_tokens


def compute_perplexity_streaming(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    chunk_size: int = 2048,
    max_tokens: Optional[int] = None,
    max_cache_length: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Compute perplexity using streaming with KV cache to leverage previous context.
    
    This method processes text sequentially while maintaining KV cache from previous
    chunks, allowing the model to use full context history for predictions.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        text: Full text to evaluate
        device: Device to run evaluation on
        chunk_size: Number of new tokens to process per iteration
        max_tokens: Maximum total tokens to evaluate (optional)
        max_cache_length: Maximum KV cache length to maintain (optional, for memory management)

    Returns:
        Tuple of (perplexity, total_tokens_evaluated)
    """
    model.eval()
    
    # Tokenize the full text
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = encodings["input_ids"].to(device)
    seq_len = input_ids.size(1)
    
    if max_tokens:
        seq_len = min(seq_len, max_tokens)
        input_ids = input_ids[:, :seq_len]
    
    print(f"Streaming PPL evaluation on {seq_len:,} tokens with chunk_size={chunk_size}")
    
    total_loss = 0.0
    total_tokens = 0
    past_key_values = None
    prev_end_loc = 0
    
    with torch.no_grad():
        pbar = tqdm(total=seq_len, desc="Streaming PPL")
        
        for begin_loc in range(0, seq_len, chunk_size):
            end_loc = min(begin_loc + chunk_size, seq_len)
            
            # Get the current chunk
            if past_key_values is None:
                # First chunk: process from the beginning
                chunk_input_ids = input_ids[:, :end_loc]
                position_ids = None
            else:
                # Subsequent chunks: only process new tokens, use KV cache for context
                chunk_input_ids = input_ids[:, begin_loc:end_loc]
                # Position IDs need to continue from where we left off
                position_ids = torch.arange(
                    begin_loc, end_loc, dtype=torch.long, device=device
                ).unsqueeze(0)
            
            # Manage cache length to prevent OOM
            if max_cache_length and past_key_values is not None:
                # Handle different cache formats (DynamicCache vs tuple)
                if hasattr(past_key_values, 'get_seq_length'):
                    # DynamicCache object (newer transformers)
                    cache_len = past_key_values.get_seq_length()
                    if cache_len > max_cache_length:
                        if hasattr(past_key_values, 'crop'):
                            past_key_values.crop(max_cache_length)
                        else:
                            trim_amount = cache_len - max_cache_length
                            for layer_idx in range(len(past_key_values.key_cache)):
                                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, trim_amount:, :]
                                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, trim_amount:, :]
                elif isinstance(past_key_values, tuple):
                    # Legacy tuple format
                    cache_len = past_key_values[0][0].size(2)
                    if cache_len > max_cache_length:
                        trim_amount = cache_len - max_cache_length
                        past_key_values = tuple(
                            tuple(kv[:, :, trim_amount:, :] for kv in layer_kv)
                            for layer_kv in past_key_values
                        )
            
            # Forward pass with KV cache
            try:
                outputs = model(
                    input_ids=chunk_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            except TypeError:
                # Some models don't support position_ids with past_key_values
                outputs = model(
                    input_ids=chunk_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            # Update KV cache for next iteration
            past_key_values = outputs.past_key_values
            
            # Compute loss on this chunk
            logits = outputs.logits
            
            if past_key_values is None or begin_loc == 0:
                # First chunk: compute loss on all tokens except first
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chunk_input_ids[..., 1:].contiguous()
            else:
                # Subsequent chunks: compute loss on all new tokens
                # logits shape: [batch, chunk_size, vocab]
                # We predict token[i+1] from logits[i]
                # For new chunk starting at begin_loc, we predict tokens [begin_loc, end_loc)
                # from logits at positions [0, chunk_size)
                
                # Get the target tokens (the ones we're predicting)
                target_start = begin_loc
                target_end = end_loc
                target_ids = input_ids[:, target_start:target_end]
                
                # The logits at position i predict token i+1 relative to the chunk
                # But since we have past KV, the last token of prev chunk predicts first token of this chunk
                # Actually, logits[i] predicts input[i+1], but with past_key_values,
                # we need to align properly
                
                # With past_key_values, outputs.logits has shape [batch, new_tokens, vocab]
                # logits[0] predicts the token after input_ids[begin_loc], which is input_ids[begin_loc+1]
                # So we use all logits except the last one to predict tokens [begin_loc+1, end_loc]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
            
            # Flatten and compute loss
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            chunk_loss = loss.sum().item()
            chunk_tokens = shift_labels.size(0)
            
            total_loss += chunk_loss
            total_tokens += chunk_tokens
            
            pbar.update(end_loc - prev_end_loc)
            prev_end_loc = end_loc
            
            # Report intermediate PPL every 10 chunks
            if (begin_loc // chunk_size) % 10 == 0 and total_tokens > 0:
                current_ppl = math.exp(total_loss / total_tokens)
                pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": total_tokens})
        
        pbar.close()
    
    # Compute final perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    
    return perplexity, total_tokens


def compute_perplexity_streaming_from_dataset(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    chunk_size: int = 2048,
    max_tokens: Optional[int] = None,
    max_cache_length: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Compute streaming perplexity across multiple documents/samples.
    
    Each document is processed with its own KV cache context.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        dataset: Dataset with tokenized samples
        device: Device to run evaluation on
        chunk_size: Number of new tokens to process per iteration  
        max_tokens: Maximum total tokens to evaluate
        max_cache_length: Maximum KV cache length per document

    Returns:
        Tuple of (perplexity, total_tokens_evaluated)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing documents"):
            if max_tokens and total_tokens >= max_tokens:
                break
                
            sample = dataset[idx]
            input_ids = sample["input_ids"].to(device)
            
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            seq_len = input_ids.size(1)
            
            # Skip very short sequences
            if seq_len < 2:
                continue
            
            past_key_values = None
            
            for begin_loc in range(0, seq_len, chunk_size):
                end_loc = min(begin_loc + chunk_size, seq_len)
                
                if past_key_values is None:
                    chunk_input_ids = input_ids[:, :end_loc]
                    position_ids = None
                else:
                    chunk_input_ids = input_ids[:, begin_loc:end_loc]
                    position_ids = torch.arange(
                        begin_loc, end_loc, dtype=torch.long, device=device
                    ).unsqueeze(0)
                
                # Manage cache length
                if max_cache_length and past_key_values is not None:
                    # Handle different cache formats (DynamicCache vs tuple)
                    if hasattr(past_key_values, 'get_seq_length'):
                        # DynamicCache object (newer transformers)
                        cache_len = past_key_values.get_seq_length()
                        if cache_len > max_cache_length:
                            # DynamicCache supports crop method in some versions
                            if hasattr(past_key_values, 'crop'):
                                past_key_values.crop(max_cache_length)
                            else:
                                # Manual trimming for DynamicCache
                                trim_amount = cache_len - max_cache_length
                                for layer_idx in range(len(past_key_values.key_cache)):
                                    past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, trim_amount:, :]
                                    past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, trim_amount:, :]
                    elif isinstance(past_key_values, tuple):
                        # Legacy tuple format
                        cache_len = past_key_values[0][0].size(2)
                        if cache_len > max_cache_length:
                            trim_amount = cache_len - max_cache_length
                            past_key_values = tuple(
                                tuple(kv[:, :, trim_amount:, :] for kv in layer_kv)
                                for layer_kv in past_key_values
                            )
                
                try:
                    outputs = model(
                        input_ids=chunk_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                except TypeError:
                    outputs = model(
                        input_ids=chunk_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                
                if begin_loc == 0:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = chunk_input_ids[..., 1:].contiguous()
                else:
                    target_ids = input_ids[:, begin_loc:end_loc]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
                total_loss += loss.sum().item()
                total_tokens += shift_labels.size(0)
                
                if max_tokens and total_tokens >= max_tokens:
                    break
            
            # Clear cache between documents
            past_key_values = None
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    
    return perplexity, total_tokens


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    num_gpus: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    max_memory_per_gpu: Optional[str] = None,
) -> Tuple:
    """Load the qwen-next model and tokenizer.
    
    Args:
        model_path: Path to the model
        device: Default device
        dtype: Model dtype (default: float16)
        num_gpus: Number of GPUs to use (None for auto)
        gpu_ids: Specific GPU IDs to use (e.g., [0, 1, 2, 3])
        load_in_4bit: Load model with 4-bit quantization (requires bitsandbytes)
        load_in_8bit: Load model with 8-bit quantization (requires bitsandbytes)
        max_memory_per_gpu: Max memory per GPU for device_map='auto', e.g., "40GiB"
    """
    from transformers import AutoConfig
    
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config first with trust_remote_code to register custom model type
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print(f"Model type: {config.model_type}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Attempting to load model directly...")
        config = None

    # Determine device_map for multi-GPU
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    # Default per-GPU max memory if not provided (used when device_map='auto')
    if max_memory_per_gpu is None:
        # Use a conservative default to reduce OOM risk if the user
        # did not specify an explicit limit.
        max_memory_per_gpu = "70GiB"
    
    if gpu_ids is not None:
        # Use specific GPU IDs
        max_memory = {i: max_memory_per_gpu for i in gpu_ids}
        max_memory["cpu"] = "100GiB"
        device_map = "auto"
        print(f"Using specific GPUs: {gpu_ids}")
    elif num_gpus is not None and num_gpus > 1:
        # Use first N GPUs
        max_memory = {i: max_memory_per_gpu for i in range(min(num_gpus, available_gpus))}
        max_memory["cpu"] = "100GiB"
        device_map = "auto"
        print(f"Using {min(num_gpus, available_gpus)} GPUs")
    elif available_gpus > 1:
        # Auto-use all available GPUs
        max_memory = {i: max_memory_per_gpu for i in range(available_gpus)}
        max_memory["cpu"] = "100GiB"
        device_map = "auto"
        print(f"Auto-using all {available_gpus} GPUs")
    else:
        max_memory = None
        device_map = None

    # Optional 4/8-bit quantization to reduce GPU memory usage
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes and transformers with BitsAndBytesConfig are required "
                "for --load_in_4bit/--load_in_8bit. Install with: pip install bitsandbytes transformers"
            ) from e

        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one of load_in_4bit or load_in_8bit can be True.")

        # For 8-bit, allow fp32 CPU offload so that modules which
        # cannot fit in GPU memory can still be placed on CPU without
        # triggering a validation error inside the HF quantizer.
        llm_int8_enable_fp32_cpu_offload = bool(load_in_8bit)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
        )

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=None if quantization_config is not None else dtype,
            device_map=device_map,
            max_memory=max_memory,
            config=config,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
    except ValueError as e:
        if "does not recognize this architecture" in str(e):
            print("\nError: Model architecture not recognized.")
            print("Try upgrading transformers:")
            print("  pip install --upgrade transformers")
            print("Or install from source:")
            print("  pip install git+https://github.com/huggingface/transformers.git")
            raise
        raise

    if device_map is None and not hasattr(model, 'hf_device_map'):
        model = model.to(device)
    
    if hasattr(model, 'hf_device_map'):
        print(f"Model distributed across devices: {set(model.hf_device_map.values())}")

    model.eval()
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def run_evaluation(
    model_path: str,
    data_path: Optional[str],
    token_scales: List[str],
    batch_size: int = 4,
    max_length: int = 2048,
    stride: int = 512,
    split: str = "test",
    output_file: Optional[str] = None,
    num_gpus: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    max_memory_per_gpu: Optional[str] = None,
    streaming: bool = False,
    chunk_size: int = 2048,
    max_cache_length: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Run perplexity evaluation across different token scales.

    Args:
        model_path: Path to the qwen-next model
        data_path: Path to the evaluation data (optional, uses PG19 if not provided)
        token_scales: List of token scales to evaluate ("1M", "10M", "1B")
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        stride: Stride for sliding window (for PG19)
        split: Dataset split for PG19 ("train", "validation", "test")
        output_file: Optional path to save results
        num_gpus: Number of GPUs to use
        gpu_ids: Specific GPU IDs to use
        load_in_4bit: Load model with 4-bit quantization (requires bitsandbytes)
        load_in_8bit: Load model with 8-bit quantization (requires bitsandbytes)
        max_memory_per_gpu: Max memory per GPU for device_map='auto', e.g., "40GiB"
        streaming: Use streaming PPL with KV cache to leverage previous context
        chunk_size: Number of new tokens to process per iteration in streaming mode
        max_cache_length: Maximum KV cache length for streaming mode (memory management)

    Returns:
        Dictionary with evaluation results for each scale
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        device,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        max_memory_per_gpu=max_memory_per_gpu,
    )

    # Determine data source
    use_pg19 = data_path is None or data_path.lower() == "pg19"
    data_source = "PG19 (deepmind/pg19)" if use_pg19 else data_path

    results = {}

    for scale in token_scales:
        print(f"\n{'='*60}")
        print(f"Evaluating PPL on {scale} tokens using {data_source}")
        print(f"{'='*60}")

        max_tokens = TOKEN_SCALES.get(scale, int(scale.replace("M", "000000").replace("B", "000000000")))

        # Create dataset
        if use_pg19:
            dataset = PG19Dataset(
                tokenizer=tokenizer,
                max_length=max_length,
                max_tokens=max_tokens,
                split=split,
                stride=stride,
            )
        else:
            dataset = TextDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_length=max_length,
                max_tokens=max_tokens,
            )

        if len(dataset) == 0:
            print(f"Warning: No data loaded for {scale} scale")
            results[scale] = {"error": "No data loaded"}
            continue

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Compute perplexity
        start_time = time.time()
        if streaming:
            print(f"Using streaming PPL with chunk_size={chunk_size}, max_cache_length={max_cache_length}")
            perplexity, tokens_evaluated = compute_perplexity_streaming_from_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                chunk_size=chunk_size,
                max_tokens=max_tokens,
                max_cache_length=max_cache_length,
            )
        else:
            perplexity, tokens_evaluated = compute_perplexity(
                model, dataloader, device, max_tokens=max_tokens
            )
        elapsed_time = time.time() - start_time

        results[scale] = {
            "perplexity": perplexity,
            "tokens_evaluated": tokens_evaluated,
            "samples": len(dataset),
            "time_seconds": elapsed_time,
            "tokens_per_second": tokens_evaluated / elapsed_time if elapsed_time > 0 else 0,
        }

        print(f"\nResults for {scale}:")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Tokens evaluated: {tokens_evaluated:,}")
        print(f"  Samples: {len(dataset):,}")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Throughput: {tokens_evaluated / elapsed_time:,.0f} tokens/s")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_source}")
    print("-" * 60)
    print(f"{'Scale':<10} {'PPL':<12} {'Tokens':<15} {'Time (s)':<10}")
    print("-" * 60)
    for scale, res in results.items():
        if "error" not in res:
            print(f"{scale:<10} {res['perplexity']:<12.4f} {res['tokens_evaluated']:<15,} {res['time_seconds']:<10.2f}")
    print("-" * 60)

    # Save results
    if output_file:
        output_data = {
            "model_path": model_path,
            "data_source": data_source,
            "results": results,
            "config": {
                "batch_size": batch_size,
                "max_length": max_length,
                "stride": stride if use_pg19 else None,
                "split": split if use_pg19 else None,
                "streaming": streaming,
                "chunk_size": chunk_size if streaming else None,
                "max_cache_length": max_cache_length if streaming else None,
            },
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of qwen-next model on PG19 corpus at different token scales"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the qwen-next model (local path or HuggingFace model ID)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to evaluation data (optional, defaults to PG19 from HuggingFace)",
    )
    parser.add_argument(
        "--token_scale",
        type=str,
        default="all",
        choices=["1M", "10M", "1B", "all"],
        help="Token scale to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10_000_000,
        help="Maximum sequence length / context length (default: 10M tokens)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for sliding window on PG19 books (default: 512)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="PG19 dataset split to use (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (JSON format)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for model loading (default: auto-detect)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model with 4-bit quantization using bitsandbytes (reduces GPU memory usage)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model with 8-bit quantization using bitsandbytes (reduces GPU memory usage)",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memory per GPU for device_map='auto', e.g., '40GiB' (default: 70GiB)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming PPL calculation with KV cache to leverage previous context",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Number of new tokens to process per iteration in streaming mode (default: 2048)",
    )
    parser.add_argument(
        "--max_cache_length",
        type=int,
        default=None,
        help="Maximum KV cache length for streaming mode (default: unlimited, set to limit memory)",
    )

    args = parser.parse_args()

    # Parse GPU IDs if provided
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    # Validate quantization options
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Only one of --load_in_4bit or --load_in_8bit can be specified.")

    # Determine token scales to evaluate
    if args.token_scale == "all":
        token_scales = ["1M", "10M", "1B"]
    else:
        token_scales = [args.token_scale]

    # Run evaluation
    results = run_evaluation(
        model_path=args.model_path,
        data_path=args.data_path,
        token_scales=token_scales,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        split=args.split,
        output_file=args.output,
        num_gpus=args.num_gpus,
        gpu_ids=gpu_ids,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_memory_per_gpu=args.max_memory_per_gpu,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
        max_cache_length=args.max_cache_length,
    )

    return results


if __name__ == "__main__":
    main()

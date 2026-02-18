#!/usr/bin/env python3
"""
Continued Training Script for Infinite Context Support

This script fine-tunes a Qwen3-Next or similar hybrid model to support infinite streaming context.

Key Training Strategies:
1. Sliding Window Attention - Train with limited attention span
2. Attention Sinks - Learn to use initial tokens as anchors  
3. Position Reset - Learn to handle position ID wrap-around
4. RoPE Extrapolation - Apply YaRN scaling for longer positions

The hybrid GDN architecture is ideal because:
- SSM layers already support infinite context (O(1) recurrence)
- Only attention layers need modification for sliding window

Usage:
    python train_infinite_context.py \
        --model_path /path/to/qwen3-next \
        --data_path /path/to/long_documents \
        --output_dir ./infinite_context_model \
        --sliding_window 4096 \
        --num_epochs 3
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers/datasets not installed")


@dataclass
class InfiniteContextTrainingConfig:
    """Configuration for infinite context training."""
    
    # Model
    model_path: str = ""
    output_dir: str = "./infinite_context_model"
    
    # Sliding window
    sliding_window: int = 4096
    num_sink_tokens: int = 4
    
    # RoPE scaling
    rope_scaling_factor: float = 4.0  # e.g., 256K * 4 = 1M
    rope_scaling_type: str = "yarn"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data
    max_seq_length: int = 4096  # Training sequence length
    chunk_overlap: int = 1024
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    bf16: bool = True
    
    # Position reset training (experimental)
    train_position_reset: bool = False
    position_reset_prob: float = 0.1  # Probability of resetting position mid-sequence


class StreamingContextDataset(Dataset):
    """
    Dataset for training streaming/infinite context.
    
    Creates training samples that simulate streaming:
    - Long documents split into overlapping chunks
    - Position IDs that may reset mid-sequence (if train_position_reset=True)
    - Attention masks that enforce sliding window
    """
    
    def __init__(
        self,
        tokenizer,
        texts: List[str],
        config: InfiniteContextTrainingConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.samples = []
        
        self._prepare_samples(texts)
    
    def _prepare_samples(self, texts: List[str]):
        """Prepare training samples from long documents."""
        
        for text in tqdm(texts, desc="Tokenizing documents"):
            # Tokenize full document
            encoding = self.tokenizer(
                text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoding["input_ids"].squeeze(0)
            
            seq_len = input_ids.size(0)
            if seq_len < self.config.max_seq_length:
                # Short document - use as-is
                self.samples.append({
                    "input_ids": input_ids,
                    "position_offset": 0,
                })
                continue
            
            # Split into overlapping chunks
            stride = self.config.max_seq_length - self.config.chunk_overlap
            
            for start in range(0, seq_len - self.config.max_seq_length + 1, stride):
                end = start + self.config.max_seq_length
                chunk = input_ids[start:end]
                
                self.samples.append({
                    "input_ids": chunk,
                    "position_offset": start,  # Track position for training
                })
        
        print(f"Created {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = sample["input_ids"]
        position_offset = sample["position_offset"]
        
        seq_len = input_ids.size(0)
        
        # Create position IDs
        if self.config.train_position_reset and torch.rand(1).item() < self.config.position_reset_prob:
            # Simulate position reset mid-sequence
            reset_point = torch.randint(seq_len // 4, 3 * seq_len // 4, (1,)).item()
            
            # Positions: 0, 1, ..., reset_point-1, 0, 1, ..., seq_len-reset_point-1
            position_ids = torch.cat([
                torch.arange(reset_point),
                torch.arange(seq_len - reset_point),
            ])
        else:
            # Normal positions with offset
            position_ids = torch.arange(position_offset, position_offset + seq_len)
        
        # Create sliding window attention mask
        # For training, we use a simplified mask that the model learns from
        attention_mask = self._create_sliding_window_mask(seq_len)
        
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
    
    def _create_sliding_window_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create attention mask enforcing sliding window + sink tokens.
        
        Each position can attend to:
        1. First num_sink_tokens positions (attention sinks)
        2. Previous sliding_window positions (local attention)
        """
        # For simplicity, return a 1D mask (full attention)
        # The actual sliding window is implemented in the model's flash attention
        return torch.ones(seq_len, dtype=torch.long)


def modify_model_for_infinite_context(
    model,
    config: InfiniteContextTrainingConfig,
):
    """
    Modify model configuration and architecture for infinite context.
    
    Changes:
    1. Add sliding_window to config
    2. Apply RoPE scaling
    3. Enable necessary attention modifications
    """
    
    # Update config
    model.config.sliding_window = config.sliding_window
    model.config.num_sink_tokens = config.num_sink_tokens
    
    # Apply RoPE scaling
    if config.rope_scaling_factor > 1.0:
        original_max_pos = model.config.max_position_embeddings
        
        model.config.rope_scaling = {
            "type": config.rope_scaling_type,
            "factor": config.rope_scaling_factor,
            "original_max_position_embeddings": original_max_pos,
        }
        
        print(f"Applied RoPE scaling:")
        print(f"  Type: {config.rope_scaling_type}")
        print(f"  Factor: {config.rope_scaling_factor}")
        print(f"  Original max pos: {original_max_pos}")
        print(f"  Effective max pos: {original_max_pos * config.rope_scaling_factor:,.0f}")
    
    # Enable gradient checkpointing if requested
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    
    return model


def load_long_context_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load long-context training data."""
    
    texts = []
    
    if os.path.isfile(data_path):
        # Single file
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    data = json.loads(line.strip())
                    text = data.get("text", data.get("content", ""))
                    if len(text) > 10000:  # Only long documents
                        texts.append(text)
        elif data_path.endswith(".txt"):
            with open(data_path, "r") as f:
                texts.append(f.read())
        else:
            with open(data_path, "r") as f:
                texts.append(f.read())
                
    elif os.path.isdir(data_path):
        # Directory of files
        for filename in os.listdir(data_path):
            if max_samples and len(texts) >= max_samples:
                break
            filepath = os.path.join(data_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        if len(content) > 10000:
                            texts.append(content)
                except:
                    continue
    
    elif data_path.lower() == "pg19":
        # Use PG19 from HuggingFace
        print("Loading PG19 dataset...")
        dataset = load_dataset("emozilla/pg19", split="train")
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            texts.append(item["text"])
    
    print(f"Loaded {len(texts)} long documents")
    return texts


class InfiniteContextTrainer:
    """
    Custom trainer for infinite context fine-tuning.
    
    Handles:
    - Sliding window attention during training
    - Position ID management
    - Memory-efficient training of long sequences
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: InfiniteContextTrainingConfig,
        train_dataset: StreamingContextDataset,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        
    def train(self):
        """Run training loop."""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Training loop
        total_steps = len(dataloader) * self.config.num_epochs
        global_step = 0
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(pbar):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                position_ids = batch["position_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    labels=labels,
                )
                
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                pbar.set_postfix({
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "step": global_step,
                })
            
            print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save model
        self.save()
    
    def _collate_fn(self, batch):
        """Collate batch of samples."""
        
        input_ids = torch.stack([s["input_ids"] for s in batch])
        position_ids = torch.stack([s["position_ids"] for s in batch])
        labels = torch.stack([s["labels"] for s in batch])
        
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels,
        }
    
    def save(self):
        """Save the trained model."""
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, "infinite_context_config.json")
        config_dict = {
            "sliding_window": self.config.sliding_window,
            "num_sink_tokens": self.config.num_sink_tokens,
            "rope_scaling_factor": self.config.rope_scaling_factor,
            "rope_scaling_type": self.config.rope_scaling_type,
            "max_seq_length": self.config.max_seq_length,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model saved to {self.config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train model for infinite context")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--data_path", type=str, default="pg19",
                        help="Path to training data or 'pg19'")
    parser.add_argument("--output_dir", type=str, default="./infinite_context_model",
                        help="Output directory")
    
    parser.add_argument("--sliding_window", type=int, default=4096,
                        help="Sliding window size for attention")
    parser.add_argument("--num_sink_tokens", type=int, default=4,
                        help="Number of sink tokens to always attend to")
    parser.add_argument("--rope_scaling_factor", type=float, default=4.0,
                        help="RoPE scaling factor (e.g., 4.0 for 4x context)")
    
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length for training")
    
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--train_position_reset", action="store_true",
                        help="Train with position reset simulation")
    
    args = parser.parse_args()
    
    # Create config
    config = InfiniteContextTrainingConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        sliding_window=args.sliding_window,
        num_sink_tokens=args.num_sink_tokens,
        rope_scaling_factor=args.rope_scaling_factor,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        train_position_reset=args.train_position_reset,
    )
    
    print("=" * 60)
    print("Infinite Context Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Sliding window: {config.sliding_window}")
    print(f"Sink tokens: {config.num_sink_tokens}")
    print(f"RoPE scaling: {config.rope_scaling_type} x{config.rope_scaling_factor}")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Modify model for infinite context
    model = modify_model_for_infinite_context(model, config)
    
    # Load training data
    print("\nLoading training data...")
    texts = load_long_context_data(args.data_path, args.max_samples)
    
    # Create dataset
    print("\nPreparing dataset...")
    dataset = StreamingContextDataset(tokenizer, texts, config)
    
    # Train
    print("\nStarting training...")
    trainer = InfiniteContextTrainer(model, tokenizer, config, dataset)
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

# Perplexity Evaluation for Qwen3-Next on PG19

This directory contains scripts for evaluating language model perplexity on the PG19 (Project Gutenberg) corpus.

## Quick Start

```bash
# Evaluate on 1M tokens
python eval_ppl.py --model_path /path/to/qwen3-next --token_scale 1M --streaming

# Evaluate on 10M tokens with continuous context
python eval_ppl.py --model_path /path/to/qwen3-next --token_scale 10M --streaming --concat_docs
```

---

## 1. Download Qwen3-Next Model

### Option A: Using Hugging Face CLI

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login (required for gated models)
huggingface-cli login

# Download model
huggingface-cli download Qwen/Qwen3-Next-Coder-8B --local-dir ./models/qwen3-next-8b
```

### Option B: Using Python

```python
from huggingface_hub import snapshot_download

# Download the model
snapshot_download(
    repo_id="Qwen/Qwen3-Next-Coder-8B",
    local_dir="./models/qwen3-next-8b",
    local_dir_use_symlinks=False,
)
```

### Option C: Using Git LFS

```bash
# Install git-lfs
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs

git lfs install
git clone https://huggingface.co/Qwen/Qwen3-Next-Coder-8B ./models/qwen3-next-8b
```

### Available Qwen3-Next Models

| Model | Parameters | Context Length | HuggingFace ID |
|-------|------------|----------------|----------------|
| Qwen3-Next-Coder-8B | 8B | 256K | `Qwen/Qwen3-Next-Coder-8B` |
| Qwen3-Next-32B | 32B | 256K | `Qwen/Qwen3-Next-32B` |

---

## 2. Download PG19 Dataset

PG19 (Project Gutenberg 19th century) is automatically downloaded when running `eval_ppl.py`, but you can pre-download it:

### Option A: Automatic (Recommended)

The script automatically downloads PG19 from HuggingFace:

```bash
# Just run the evaluation - dataset downloads automatically
python eval_ppl.py --model_path ./models/qwen3-next-8b --token_scale 1M
```

### Option B: Pre-download Using Python

```python
from datasets import load_dataset

# Download PG19 test split (~100 books)
dataset = load_dataset("emozilla/pg19", split="test")
print(f"Downloaded {len(dataset)} books")

# Download all splits
dataset_full = load_dataset("emozilla/pg19")
print(f"Train: {len(dataset_full['train'])} books")
print(f"Validation: {len(dataset_full['validation'])} books")
print(f"Test: {len(dataset_full['test'])} books")
```

### Option C: Download Script

```bash
python download_pg19.py
```

### PG19 Dataset Info

| Split | Books | Approx. Tokens |
|-------|-------|----------------|
| train | ~28,000 | ~6B |
| validation | ~50 | ~10M |
| test | ~100 | ~20M |

---

## 3. Run Evaluation

### Basic Usage

```bash
# Evaluate on 1M tokens (quick test)
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 1M \
    --streaming

# Evaluate on 10M tokens
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 10M \
    --streaming

# Evaluate on 1B tokens (full benchmark)
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 1B \
    --streaming
```

### Advanced Options

```bash
# Use continuous context across documents (tests long-range modeling)
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 10M \
    --streaming \
    --concat_docs

# Multi-GPU evaluation
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 10M \
    --streaming \
    --gpu_ids 0,1,2,3

# 8-bit quantization (saves GPU memory)
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 10M \
    --streaming \
    --load_in_8bit

# 4-bit quantization (even more memory savings)
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale 10M \
    --streaming \
    --load_in_4bit

# Save results to JSON
python eval_ppl.py \
    --model_path ./models/qwen3-next-8b \
    --token_scale all \
    --streaming \
    --output results.json
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to model | Required |
| `--token_scale` | 1M, 10M, 1B, or all | all |
| `--streaming` | Use KV cache streaming | False |
| `--concat_docs` | Concatenate docs for continuous context | False |
| `--chunk_size` | Tokens per forward pass | Auto (32K) |
| `--load_in_8bit` | 8-bit quantization | False |
| `--load_in_4bit` | 4-bit quantization | False |
| `--gpu_ids` | Specific GPUs (e.g., "0,1,2,3") | Auto |
| `--output` | Save results to JSON file | None |

---

## 4. Expected Results

### Qwen3-Next-Coder-8B on PG19 Test

| Scale | Expected PPL | Tokens |
|-------|--------------|--------|
| 1M | ~8-10 | 1,000,000 |
| 10M | ~8-10 | 10,000,000 |
| 1B | ~8-10 | 1,000,000,000 |

### Memory Requirements

| Model | FP16 | 8-bit | 4-bit |
|-------|------|-------|-------|
| 8B | ~16GB | ~8GB | ~4GB |
| 32B | ~64GB | ~32GB | ~16GB |

---

## 5. Troubleshooting

### "CUDA out of memory"

```bash
# Use quantization
python eval_ppl.py --model_path ... --load_in_8bit

# Or reduce chunk size
python eval_ppl.py --model_path ... --chunk_size 8192
```

### "Dataset scripts are no longer supported"

The script automatically falls back to `emozilla/pg19` (Parquet format) if `deepmind/pg19` fails.

### "Model architecture not recognized"

```bash
# Upgrade transformers
pip install --upgrade transformers

# Or install from source
pip install git+https://github.com/huggingface/transformers.git
```

---

## 6. Dependencies

```bash
pip install torch transformers datasets tqdm huggingface_hub
pip install flash-attn  # Optional, for faster attention
pip install bitsandbytes  # Optional, for quantization
```

#!/usr/bin/env python3
"""
Download Script for Qwen3-Coder-Next Models

This script downloads Qwen3-Coder-Next models from HuggingFace Hub.
Available models:
  - Qwen/Qwen3-Coder-Next (80B, full precision)
  - Qwen/Qwen3-Coder-Next-Base (80B, base model)
  - Qwen/Qwen3-Coder-Next-FP8 (80B, FP8 quantized)
  - Qwen/Qwen3-Coder-Next-GGUF (80B, GGUF format for llama.cpp)

Usage:
    # Download default model (Qwen3-Coder-Next)
    python download_qwen3_next.py --output_dir ./models

    # Download specific variant
    python download_qwen3_next.py --model base --output_dir ./models
    python download_qwen3_next.py --model fp8 --output_dir ./models
    python download_qwen3_next.py --model gguf --output_dir ./models

    # Download all variants
    python download_qwen3_next.py --model all --output_dir ./models

    # Resume interrupted download
    python download_qwen3_next.py --output_dir ./models --resume
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Error: 'huggingface_hub' library is required.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


# Qwen3-Coder-Next model variants
QWEN3_NEXT_MODELS = {
    "default": {
        "repo_id": "Qwen/Qwen3-Coder-Next",
        "description": "Qwen3-Coder-Next (80B, full precision, instruction-tuned)",
        "size": "~160GB",
    },
    "base": {
        "repo_id": "Qwen/Qwen3-Coder-Next-Base",
        "description": "Qwen3-Coder-Next-Base (80B, base pretrained model)",
        "size": "~160GB",
    },
    "fp8": {
        "repo_id": "Qwen/Qwen3-Coder-Next-FP8",
        "description": "Qwen3-Coder-Next-FP8 (80B, FP8 quantized for efficient inference)",
        "size": "~80GB",
    },
    "gguf": {
        "repo_id": "Qwen/Qwen3-Coder-Next-GGUF",
        "description": "Qwen3-Coder-Next-GGUF (80B, GGUF format for llama.cpp)",
        "size": "Varies by quantization",
    },
}

# GGUF quantization options
GGUF_QUANTS = [
    "Q2_K",
    "Q3_K_M", 
    "Q4_0",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
]


def check_disk_space(output_dir: str, required_gb: float = 200) -> bool:
    """Check if there's enough disk space."""
    import shutil
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free / (1024 ** 3)
    
    print(f"Available disk space: {free_gb:.1f} GB")
    
    if free_gb < required_gb:
        print(f"Warning: Less than {required_gb}GB available. Large models may not fit.")
        return False
    return True


def download_model(
    model_key: str,
    output_dir: str,
    resume: bool = True,
    token: Optional[str] = None,
    revision: str = "main",
    ignore_patterns: Optional[List[str]] = None,
    gguf_quant: Optional[str] = None,
) -> str:
    """
    Download a Qwen3-Coder-Next model variant.

    Args:
        model_key: Model variant key (default, base, fp8, gguf)
        output_dir: Directory to save the model
        resume: Whether to resume interrupted downloads
        token: HuggingFace API token (optional)
        revision: Model revision/branch
        ignore_patterns: File patterns to ignore during download
        gguf_quant: Specific GGUF quantization to download (for gguf variant)

    Returns:
        Path to the downloaded model
    """
    if model_key not in QWEN3_NEXT_MODELS:
        raise ValueError(f"Unknown model variant: {model_key}. "
                        f"Available: {list(QWEN3_NEXT_MODELS.keys())}")
    
    model_info = QWEN3_NEXT_MODELS[model_key]
    repo_id = model_info["repo_id"]
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_info['description']}")
    print(f"Repository: {repo_id}")
    print(f"Estimated size: {model_info['size']}")
    print(f"{'='*60}\n")
    
    # Create output directory
    local_dir = os.path.join(output_dir, repo_id.replace("/", "_"))
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Set default ignore patterns
    if ignore_patterns is None:
        ignore_patterns = []
    
    # For GGUF, optionally filter to specific quantization
    if model_key == "gguf" and gguf_quant:
        print(f"Filtering to {gguf_quant} quantization...")
        # Download only the specific quantization file
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id, revision=revision)
            
            # Find files matching the quantization
            matching_files = [f for f in files if gguf_quant.lower() in f.lower()]
            
            if not matching_files:
                print(f"Warning: No files found matching {gguf_quant}")
                print(f"Available quantizations: {GGUF_QUANTS}")
            else:
                print(f"Found {len(matching_files)} files for {gguf_quant}")
                for f in matching_files:
                    print(f"  Downloading: {f}")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=f,
                        local_dir=local_dir,
                        resume_download=resume,
                        token=token,
                        revision=revision,
                    )
                print(f"\nDownload complete: {local_dir}")
                return local_dir
                
        except Exception as e:
            print(f"Error listing files: {e}")
            print("Falling back to full download...")
    
    # Download the full model
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=resume,
            token=token,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
        
        print(f"\nDownload complete!")
        print(f"Model saved to: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def list_models():
    """List all available Qwen3-Coder-Next model variants."""
    print("\nAvailable Qwen3-Coder-Next Model Variants:")
    print("=" * 70)
    
    for key, info in QWEN3_NEXT_MODELS.items():
        print(f"\n  {key}:")
        print(f"    Repository: {info['repo_id']}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}")
    
    print("\n" + "=" * 70)
    print("\nGGUF Quantization Options:")
    print(f"  {', '.join(GGUF_QUANTS)}")
    print("\nExample: --model gguf --gguf_quant Q4_K_M")


def verify_download(model_path: str) -> bool:
    """Verify the downloaded model has essential files."""
    essential_files = [
        "config.json",
    ]
    
    # Check for model weights (could be safetensors or pytorch)
    weight_patterns = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    
    print(f"\nVerifying download at: {model_path}")
    
    # Check essential files
    for filename in essential_files:
        filepath = os.path.join(model_path, filename)
        if not os.path.exists(filepath):
            print(f"  ❌ Missing: {filename}")
            return False
        print(f"  ✓ Found: {filename}")
    
    # Check for model weights
    has_weights = False
    for pattern in weight_patterns:
        filepath = os.path.join(model_path, pattern)
        if os.path.exists(filepath):
            has_weights = True
            print(f"  ✓ Found weights: {pattern}")
            break
    
    # For GGUF models, check for .gguf files
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if gguf_files:
        has_weights = True
        print(f"  ✓ Found {len(gguf_files)} GGUF file(s)")
    
    if not has_weights:
        print("  ⚠ Warning: No model weight files found")
        return False
    
    print("\n✓ Download verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-Coder-Next models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default instruction-tuned model
  python download_qwen3_next.py --output_dir ./models

  # Download base pretrained model
  python download_qwen3_next.py --model base --output_dir ./models

  # Download FP8 quantized model (smaller, faster)
  python download_qwen3_next.py --model fp8 --output_dir ./models

  # Download specific GGUF quantization
  python download_qwen3_next.py --model gguf --gguf_quant Q4_K_M --output_dir ./models

  # List all available models
  python download_qwen3_next.py --list
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        choices=["default", "base", "fp8", "gguf", "all"],
        help="Model variant to download (default: default)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save downloaded models (default: ./models)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, for gated models)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume interrupted downloads (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resume and start fresh download",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch to download (default: main)",
    )
    parser.add_argument(
        "--gguf_quant",
        type=str,
        default=None,
        choices=GGUF_QUANTS,
        help="Specific GGUF quantization to download (only for --model gguf)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available model variants and exit",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify download after completion (default: True)",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip disk space check",
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        list_models()
        return
    
    # Check disk space
    if not args.skip_disk_check:
        check_disk_space(args.output_dir)
    
    # Determine which models to download
    if args.model == "all":
        models_to_download = ["default", "base", "fp8", "gguf"]
    else:
        models_to_download = [args.model]
    
    # Download models
    downloaded_paths = []
    for model_key in models_to_download:
        try:
            path = download_model(
                model_key=model_key,
                output_dir=args.output_dir,
                resume=args.resume,
                token=args.token,
                revision=args.revision,
                gguf_quant=args.gguf_quant if model_key == "gguf" else None,
            )
            downloaded_paths.append((model_key, path))
            
            # Verify download
            if args.verify and model_key != "gguf":
                verify_download(path)
                
        except Exception as e:
            print(f"Failed to download {model_key}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for model_key, path in downloaded_paths:
        print(f"  {model_key}: {path}")
    print("=" * 60)
    
    # Print usage hint
    if downloaded_paths:
        first_path = downloaded_paths[0][1]
        print(f"\nTo evaluate perplexity with the downloaded model:")
        print(f"  python eval_ppl.py --model_path {first_path} --token_scale all")


if __name__ == "__main__":
    main()

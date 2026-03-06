#!/usr/bin/env python3
"""
Download Qwen3-Next-80B-A3B-Thinking from Hugging Face Hub.

Usage:
    # Basic download
    python eval/download_qwen3_next_thinking.py --output_dir ./models

    # With token (recommended for gated/private repos)
    python eval/download_qwen3_next_thinking.py --output_dir ./models --token <HF_TOKEN>

    # Custom revision
    python eval/download_qwen3_next_thinking.py --output_dir ./models --revision main
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: missing dependency 'huggingface_hub'.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


DEFAULT_REPO_ID = "Qwen/Qwen3-Next-80B-A3B-Thinking"


def check_disk_space(output_dir: str, required_gb: float) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free / (1024 ** 3)
    print(f"Available disk space: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(
            f"Warning: only {free_gb:.1f} GB free (< {required_gb:.1f} GB requested). "
            "Download may fail due to insufficient space."
        )


def verify_download(model_dir: str) -> bool:
    required = ["config.json"]
    weight_candidates = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]

    print(f"\nVerifying files in: {model_dir}")
    for name in required:
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            print(f"  ❌ Missing required file: {name}")
            return False
        print(f"  ✓ Found: {name}")

    has_weights = any(os.path.exists(os.path.join(model_dir, p)) for p in weight_candidates)
    if not has_weights:
        print("  ⚠ No recognized model weight index/file found.")
        return False

    print("  ✓ Found model weight file/index")
    print("Verification passed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-Next-80B-A3B-Thinking model")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                        help=f"HF repo id (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Directory to store model files")
    parser.add_argument("--revision", type=str, default="main",
                        help="Model revision/branch/tag")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (or use HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume; redownload from scratch")
    parser.add_argument("--local-dir-name", type=str, default=None,
                        help="Optional custom folder name under output_dir")
    parser.add_argument("--skip-disk-check", action="store_true",
                        help="Skip disk-space warning check")
    parser.add_argument("--required-gb", type=float, default=180.0,
                        help="Minimum recommended free GB for warning")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify essential files after download (default: True)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not args.skip_disk_check:
        check_disk_space(args.output_dir, required_gb=args.required_gb)

    print("=" * 70)
    print("Downloading Qwen3-Next-80B-A3B-Thinking")
    print(f"Repo:     {args.repo_id}")
    print(f"Revision: {args.revision}")
    print("=" * 70)

    folder_name = args.local_dir_name or args.repo_id.replace("/", "_")
    local_dir = os.path.join(args.output_dir, folder_name)
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=local_dir,
            token=token,
            resume_download=not args.no_resume,
        )
    except Exception as exc:
        print(f"Download failed for repo '{args.repo_id}'.")
        print("Check: repo id, internet access, and HF token permissions.")
        raise RuntimeError(str(exc)) from exc

    print(f"\nDownload complete: {downloaded_path}")

    if args.verify:
        ok = verify_download(downloaded_path)
        if not ok:
            sys.exit(2)

    print("\nDone.")


if __name__ == "__main__":
    main()

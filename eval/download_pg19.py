#!/usr/bin/env python3
"""
Download Script for PG19 Dataset (Project Gutenberg)

PG19 is a standard benchmark for evaluating long-range language modeling,
containing full-length books from Project Gutenberg published before 1919.

Dataset Statistics:
  - Train: ~28,000 books
  - Validation: ~50 books  
  - Test: ~100 books
  - Total tokens: ~2B tokens (train), ~3M tokens (test)

Usage:
    # Download all splits
    python download_pg19.py --output_dir ./data/pg19

    # Download specific split
    python download_pg19.py --split test --output_dir ./data/pg19

    # Download and save as JSONL
    python download_pg19.py --output_dir ./data/pg19 --format jsonl

    # Stream download (low memory usage)
    python download_pg19.py --output_dir ./data/pg19 --streaming
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    from tqdm import tqdm
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Error: Required libraries not installed.")
    print("Install with: pip install datasets tqdm")
    sys.exit(1)


PG19_REPO = "deepmind/pg19"

SPLIT_INFO = {
    "train": {
        "description": "Training set with ~28,000 books",
        "approx_books": 28000,
        "approx_tokens": "~2B",
    },
    "validation": {
        "description": "Validation set with ~50 books",
        "approx_books": 50,
        "approx_tokens": "~3M",
    },
    "test": {
        "description": "Test set with ~100 books",
        "approx_books": 100,
        "approx_tokens": "~6M",
    },
}


def download_pg19(
    output_dir: str,
    split: Optional[str] = None,
    format: str = "arrow",
    streaming: bool = False,
    max_books: Optional[int] = None,
) -> dict:
    """
    Download PG19 dataset from HuggingFace.

    Args:
        output_dir: Directory to save the dataset
        split: Specific split to download (train, validation, test) or None for all
        format: Output format ('arrow', 'jsonl', 'txt')
        streaming: Use streaming mode for low memory usage
        max_books: Maximum number of books to download (for testing)

    Returns:
        Dictionary with download statistics
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    splits_to_download = [split] if split else ["train", "validation", "test"]
    stats = {}
    
    for current_split in splits_to_download:
        print(f"\n{'='*60}")
        print(f"Downloading PG19 {current_split} split")
        print(f"  {SPLIT_INFO[current_split]['description']}")
        print(f"  Approx tokens: {SPLIT_INFO[current_split]['approx_tokens']}")
        print(f"{'='*60}\n")
        
        try:
            if streaming:
                dataset = load_dataset(
                    PG19_REPO,
                    split=current_split,
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    PG19_REPO,
                    split=current_split,
                )
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                print(f"Warning: {PG19_REPO} uses legacy script format.")
                print("Trying alternative: emozilla/pg19 (Parquet format)...")
                alt_repo = "emozilla/pg19"
                if streaming:
                    dataset = load_dataset(
                        alt_repo,
                        split=current_split,
                        streaming=True,
                    )
                else:
                    dataset = load_dataset(
                        alt_repo,
                        split=current_split,
                    )
            else:
                raise
        
        split_dir = os.path.join(output_dir, current_split)
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        
        if format == "arrow":
            # Save in Arrow format (HuggingFace native)
            if not streaming:
                arrow_path = os.path.join(split_dir, "data.arrow")
                dataset.save_to_disk(split_dir)
                print(f"Saved to: {split_dir}")
                stats[current_split] = {
                    "format": "arrow",
                    "path": split_dir,
                    "num_books": len(dataset),
                }
            else:
                print("Warning: Arrow format not supported with streaming. Using JSONL.")
                format = "jsonl"
        
        if format == "jsonl":
            # Save as JSONL
            jsonl_path = os.path.join(split_dir, f"pg19_{current_split}.jsonl")
            book_count = 0
            total_chars = 0
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                iterator = tqdm(dataset, desc=f"Saving {current_split}")
                for i, item in enumerate(iterator):
                    if max_books and i >= max_books:
                        break
                    
                    record = {
                        "id": item.get("short_book_title", f"book_{i}"),
                        "text": item["text"],
                        "url": item.get("url", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    book_count += 1
                    total_chars += len(item["text"])
            
            print(f"Saved to: {jsonl_path}")
            stats[current_split] = {
                "format": "jsonl",
                "path": jsonl_path,
                "num_books": book_count,
                "total_chars": total_chars,
            }
        
        elif format == "txt":
            # Save each book as separate txt file
            txt_dir = os.path.join(split_dir, "books")
            Path(txt_dir).mkdir(parents=True, exist_ok=True)
            
            book_count = 0
            total_chars = 0
            
            iterator = tqdm(dataset, desc=f"Saving {current_split}")
            for i, item in enumerate(iterator):
                if max_books and i >= max_books:
                    break
                
                # Create safe filename
                title = item.get("short_book_title", f"book_{i}")
                safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
                safe_title = safe_title[:100]  # Limit length
                
                txt_path = os.path.join(txt_dir, f"{i:05d}_{safe_title}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(item["text"])
                
                book_count += 1
                total_chars += len(item["text"])
            
            print(f"Saved {book_count} books to: {txt_dir}")
            stats[current_split] = {
                "format": "txt",
                "path": txt_dir,
                "num_books": book_count,
                "total_chars": total_chars,
            }
    
    return stats


def verify_download(output_dir: str, split: str = "test") -> bool:
    """Verify the downloaded dataset."""
    split_dir = os.path.join(output_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"❌ Split directory not found: {split_dir}")
        return False
    
    # Check for arrow format
    arrow_file = os.path.join(split_dir, "dataset_info.json")
    if os.path.exists(arrow_file):
        print(f"✓ Found Arrow format dataset in {split_dir}")
        return True
    
    # Check for JSONL
    jsonl_files = list(Path(split_dir).glob("*.jsonl"))
    if jsonl_files:
        print(f"✓ Found {len(jsonl_files)} JSONL file(s) in {split_dir}")
        return True
    
    # Check for TXT
    txt_dir = os.path.join(split_dir, "books")
    if os.path.exists(txt_dir):
        txt_files = list(Path(txt_dir).glob("*.txt"))
        print(f"✓ Found {len(txt_files)} TXT files in {txt_dir}")
        return True
    
    print(f"❌ No valid dataset files found in {split_dir}")
    return False


def print_dataset_info():
    """Print information about the PG19 dataset."""
    print("\n" + "=" * 60)
    print("PG19 Dataset Information")
    print("=" * 60)
    print(f"\nRepository: {PG19_REPO}")
    print("\nPG19 contains full-length books from Project Gutenberg")
    print("published before 1919. It's a standard benchmark for")
    print("evaluating long-range language modeling capabilities.")
    print("\nSplits:")
    for split, info in SPLIT_INFO.items():
        print(f"  {split}:")
        print(f"    - {info['description']}")
        print(f"    - Approx tokens: {info['approx_tokens']}")
    print("\nCitation:")
    print("  Rae et al., 'Compressive Transformers for Long-Range")
    print("  Sequence Modelling', ICLR 2020")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download PG19 dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all splits in Arrow format
  python download_pg19.py --output_dir ./data/pg19

  # Download only test split
  python download_pg19.py --split test --output_dir ./data/pg19

  # Download as JSONL (easier to inspect)
  python download_pg19.py --format jsonl --output_dir ./data/pg19

  # Download as separate text files
  python download_pg19.py --format txt --output_dir ./data/pg19

  # Stream download (low memory, for large splits)
  python download_pg19.py --split train --streaming --format jsonl --output_dir ./data/pg19

  # Download first 100 books only (for testing)
  python download_pg19.py --max_books 100 --output_dir ./data/pg19
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/pg19",
        help="Directory to save the dataset (default: ./data/pg19)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="Specific split to download (default: all splits)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="arrow",
        choices=["arrow", "jsonl", "txt"],
        help="Output format: arrow (HF native), jsonl, or txt (default: arrow)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for low memory usage (recommended for train split)",
    )
    parser.add_argument(
        "--max_books",
        type=int,
        default=None,
        help="Maximum number of books to download (for testing)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information and exit",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing download",
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info()
        return
    
    if args.verify:
        split = args.split or "test"
        success = verify_download(args.output_dir, split)
        sys.exit(0 if success else 1)
    
    # Download the dataset
    print_dataset_info()
    
    stats = download_pg19(
        output_dir=args.output_dir,
        split=args.split,
        format=args.format,
        streaming=args.streaming,
        max_books=args.max_books,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for split, info in stats.items():
        print(f"\n{split}:")
        print(f"  Format: {info['format']}")
        print(f"  Path: {info['path']}")
        print(f"  Books: {info['num_books']}")
        if "total_chars" in info:
            print(f"  Total characters: {info['total_chars']:,}")
    print("=" * 60)
    
    # Usage hint
    print("\nTo evaluate PPL with this data:")
    print(f"  python eval_ppl.py --model_path <model> --token_scale all")
    print("\n  (PG19 is loaded automatically from HuggingFace)")


if __name__ == "__main__":
    main()

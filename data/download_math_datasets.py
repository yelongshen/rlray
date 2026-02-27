"""
Download Math Domain Datasets for Training and Evaluation.

Covers:
- Elementary/Middle: GSM8K, SVAMP
- High School: MATH (all levels), AMC/AIME
- Competition: NuminaMath, Olympiad Bench
- College+: MMLU-STEM, GPQA
- Training: MetaMathQA, DeepSeek-Math, ORCA-Math, MathInstruct
- Process Reward: PRM800K

Usage:
    python data/download_math_datasets.py --all
    python data/download_math_datasets.py --eval-only
    python data/download_math_datasets.py --dataset gsm8k math
    python data/download_math_datasets.py --list
"""

import os
import argparse
from pathlib import Path

# Try to import datasets library
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = {
    # === Evaluation Datasets ===
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "level": "elementary/middle",
        "area": "arithmetic, word problems",
        "size": "~8.5K",
        "description": "Grade school math word problems",
        "category": "eval",
    },
    "math": {
        "hf_path": "hendrycks/competition_math",
        "hf_name": None,
        "level": "high school / competition",
        "area": "algebra, geometry, number theory, counting, probability",
        "size": "~12.5K",
        "description": "MATH benchmark (Hendrycks et al.) - competition math problems levels 1-5",
        "category": "eval",
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_name": None,
        "level": "high school / competition",
        "area": "mixed math",
        "size": "500",
        "description": "Curated 500 hard MATH problems for evaluation",
        "category": "eval",
    },
    "aime24": {
        "hf_path": "AI-MO/aimo-validation-aime",
        "hf_name": None,
        "level": "olympiad",
        "area": "competition math",
        "size": "90",
        "description": "AIME 2024 competition problems",
        "category": "eval",
    },
    "svamp": {
        "hf_path": "ChilleD/SVAMP",
        "hf_name": None,
        "level": "elementary",
        "area": "arithmetic word problems",
        "size": "~1K",
        "description": "Simple Variations on Arithmetic Math word Problems",
        "category": "eval",
    },
    "mmlu_math": {
        "hf_path": "cais/mmlu",
        "hf_name": "abstract_algebra",
        "level": "college",
        "area": "abstract algebra, college math",
        "size": "~1K",
        "description": "MMLU abstract algebra subset",
        "category": "eval",
    },
    "gpqa": {
        "hf_path": "Idavidrein/gpqa",
        "hf_name": "gpqa_main",
        "level": "graduate/PhD",
        "area": "science + math",
        "size": "448",
        "description": "Graduate-level science questions",
        "category": "eval",
    },
    "minerva_math": {
        "hf_path": "meta-llama/Llama-3.1-8B-evals",
        "hf_name": None,
        "level": "high school / college",
        "area": "mixed math",
        "size": "~272",
        "description": "Minerva math evaluation (via Llama evals, or use MATH instead)",
        "category": "eval",
        "skip": True,  # Hard to get standalone, use MATH instead
    },
    "olympiad_bench": {
        "hf_path": "Hothan/OlympiadBench",
        "hf_name": None,
        "level": "olympiad",
        "area": "competition math + physics",
        "size": "~8.5K",
        "description": "Olympic-level math and physics problems",
        "category": "eval",
    },

    # === Training Datasets (SFT) ===
    "metamathqa": {
        "hf_path": "meta-math/MetaMathQA",
        "hf_name": None,
        "level": "middle / high school",
        "area": "augmented GSM8K + MATH",
        "size": "~395K",
        "description": "Augmented math QA for fine-tuning (bootstrapped from GSM8K + MATH)",
        "category": "train",
    },
    "numina_math_cot": {
        "hf_path": "AI-MO/NuminaMath-CoT",
        "hf_name": None,
        "level": "mixed",
        "area": "competition math with chain-of-thought",
        "size": "~860K",
        "description": "Competition math with step-by-step solutions",
        "category": "train",
    },
    "numina_math_tir": {
        "hf_path": "AI-MO/NuminaMath-TIR",
        "hf_name": None,
        "level": "mixed",
        "area": "competition math with tool-integrated reasoning",
        "size": "~72K",
        "description": "Math problems with Python code verification",
        "category": "train",
    },
    "orca_math": {
        "hf_path": "microsoft/orca-math-word-problems-200k",
        "hf_name": None,
        "level": "elementary / middle",
        "area": "word problems",
        "size": "~200K",
        "description": "GPT-4 generated math word problems with solutions",
        "category": "train",
    },
    "mathinstruct": {
        "hf_path": "TIGER-Lab/MathInstruct",
        "hf_name": None,
        "level": "mixed",
        "area": "diverse math instructions",
        "size": "~260K",
        "description": "Compiled math instruction dataset from multiple sources",
        "category": "train",
    },
    "deepseek_math": {
        "hf_path": "deepseek-ai/deepseek-math",
        "hf_name": None,
        "level": "mixed",
        "area": "web-crawled math corpus",
        "size": "~500K",
        "description": "High-quality math training data from DeepSeek (may require auth)",
        "category": "train",
        "skip": True,  # Gated/private repo
    },
    "open_math_instruct_2": {
        "hf_path": "nvidia/OpenMathInstruct-2",
        "hf_name": None,
        "level": "mixed",
        "area": "math instructions",
        "size": "~14M",
        "description": "NVIDIA's large-scale math instruction dataset",
        "category": "train",
    },
    "kpmath": {
        "hf_path": "KPMath/KPMath-Plus",
        "hf_name": None,
        "level": "mixed",
        "area": "diverse math",
        "size": "~1M",
        "description": "Knowledge-Preserving Math dataset (may require auth)",
        "category": "train",
        "skip": True,  # May not be publicly available
    },

    # === Process Reward / RL Datasets ===
    "prm800k": {
        "hf_path": "tasksource/prm800k",
        "hf_name": None,
        "level": "high school",
        "area": "step-level math verification",
        "size": "~800K steps",
        "description": "Process Reward Model training data with step-level labels",
        "category": "reward",
    },
    "math_shepherd": {
        "hf_path": "peiyi9979/Math-Shepherd",
        "hf_name": None,
        "level": "mixed",
        "area": "step-level process supervision",
        "size": "~400K",
        "description": "Step-level process supervision for math reasoning",
        "category": "reward",
    },

    # === Pretraining (Large) ===
    "openwebmath": {
        "hf_path": "open-web-math/open-web-math",
        "hf_name": None,
        "level": "mixed",
        "area": "web math text",
        "size": "~14.7B tokens",
        "description": "Large-scale math web text for pretraining (WARNING: very large)",
        "category": "pretrain",
    },
    "proof_pile_2": {
        "hf_path": "EleutherAI/proof-pile-2",
        "hf_name": None,
        "level": "mixed",
        "area": "arXiv, textbooks, code",
        "size": "~55B tokens",
        "description": "Math pretraining corpus (WARNING: extremely large)",
        "category": "pretrain",
    },
}


def list_datasets():
    """Print all available datasets."""
    print("\n" + "=" * 100)
    print("Available Math Domain Datasets")
    print("=" * 100)
    
    for category in ["eval", "train", "reward", "pretrain"]:
        cat_name = {
            "eval": "Evaluation Datasets",
            "train": "Training Datasets (SFT)",
            "reward": "Process Reward / RL Datasets",
            "pretrain": "Pretraining Datasets (Large)",
        }[category]
        
        print(f"\n--- {cat_name} ---")
        print(f"{'Name':<25} {'Level':<25} {'Size':<15} {'Description'}")
        print("-" * 100)
        
        for name, info in DATASETS.items():
            if info["category"] == category:
                print(f"{name:<25} {info['level']:<25} {info['size']:<15} {info['description']}")
    
    print()


def download_dataset(name: str, output_dir: str, max_samples: int = None):
    """Download a single dataset."""
    if not HF_DATASETS_AVAILABLE:
        print(f"ERROR: 'datasets' library required. Install with: pip install datasets")
        return False
    
    if name not in DATASETS:
        print(f"ERROR: Unknown dataset '{name}'. Use --list to see available datasets.")
        return False
    
    info = DATASETS[name]
    
    # Skip datasets marked as unavailable
    if info.get("skip"):
        print(f"  SKIP: {name} - {info.get('description', 'unavailable')}")
        return True
    
    save_path = os.path.join(output_dir, name)
    
    if os.path.exists(save_path):
        print(f"  SKIP: {name} already exists at {save_path}")
        return True
    
    print(f"\n  Downloading: {name}")
    print(f"    Source: {info['hf_path']}")
    print(f"    Level: {info['level']}")
    print(f"    Size: {info['size']}")
    print(f"    Description: {info['description']}")
    
    try:
        kwargs = {}
        if info["hf_name"]:
            kwargs["name"] = info["hf_name"]
        
        try:
            ds = load_dataset(info["hf_path"], **kwargs)
        except Exception as e1:
            # Fallback: try with explicit split download if '**' pattern error
            if "'**'" in str(e1) or "pattern" in str(e1).lower():
                print(f"    Retrying with explicit splits...")
                try:
                    # Try downloading train/test splits explicitly
                    splits = {}
                    for split_name in ["train", "test", "validation"]:
                        try:
                            splits[split_name] = load_dataset(info["hf_path"], split=split_name, **kwargs)
                        except:
                            pass
                    if splits:
                        from datasets import DatasetDict
                        ds = DatasetDict(splits)
                    else:
                        raise e1
                except Exception:
                    # Final fallback: use snapshot_download
                    print(f"    Retrying with snapshot_download...")
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id=info["hf_path"],
                        repo_type="dataset",
                        local_dir=save_path,
                    )
                    print(f"    Saved raw files to: {save_path}")
                    return True
            else:
                raise e1
        
        if max_samples:
            # Truncate each split
            for split in ds:
                if len(ds[split]) > max_samples:
                    ds[split] = ds[split].select(range(max_samples))
                    print(f"    Truncated {split} to {max_samples} samples")
        
        ds.save_to_disk(save_path)
        print(f"    Saved to: {save_path}")
        
        # Print split info
        for split in ds:
            print(f"    {split}: {len(ds[split])} samples")
        
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Math Domain Datasets")
    parser.add_argument("--output-dir", type=str, default="./data/math_datasets",
                        help="Directory to save datasets")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets (excluding pretrain)")
    parser.add_argument("--eval-only", action="store_true", help="Download only evaluation datasets")
    parser.add_argument("--train-only", action="store_true", help="Download only training datasets")
    parser.add_argument("--reward-only", action="store_true", help="Download only reward datasets")
    parser.add_argument("--dataset", nargs="+", type=str, help="Download specific datasets by name")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for testing)")
    parser.add_argument("--include-pretrain", action="store_true",
                        help="Include large pretraining datasets (WARNING: very large)")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    # Determine which datasets to download
    to_download = []
    
    if args.dataset:
        to_download = args.dataset
    elif args.eval_only:
        to_download = [n for n, d in DATASETS.items() if d["category"] == "eval"]
    elif args.train_only:
        to_download = [n for n, d in DATASETS.items() if d["category"] == "train"]
    elif args.reward_only:
        to_download = [n for n, d in DATASETS.items() if d["category"] == "reward"]
    elif args.all:
        categories = {"eval", "train", "reward"}
        if args.include_pretrain:
            categories.add("pretrain")
        to_download = [n for n, d in DATASETS.items() if d["category"] in categories]
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python data/download_math_datasets.py --list")
        print("  python data/download_math_datasets.py --eval-only")
        print("  python data/download_math_datasets.py --dataset gsm8k math math500")
        print("  python data/download_math_datasets.py --all")
        print("  python data/download_math_datasets.py --dataset metamathqa --max-samples 1000")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nDownloading {len(to_download)} datasets to {args.output_dir}")
    print("=" * 60)
    
    success = 0
    failed = 0
    skipped = 0
    
    for name in to_download:
        result = download_dataset(name, args.output_dir, args.max_samples)
        if result:
            success += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Done: {success} downloaded, {failed} failed")
    print(f"Datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

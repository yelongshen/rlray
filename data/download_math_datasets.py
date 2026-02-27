"""
Download Math Domain Datasets for Training and Evaluation.

Uses huggingface_hub.snapshot_download directly — no 'datasets' library needed.

Usage:
    python data/download_math_datasets.py --all
    python data/download_math_datasets.py --eval-only
    python data/download_math_datasets.py --dataset gsm8k math math500
    python data/download_math_datasets.py --list
"""

import os
import argparse
from huggingface_hub import snapshot_download

DATASETS = {
    # === Evaluation ===
    "gsm8k":            {"hf_path": "openai/gsm8k",                      "category": "eval",    "size": "~8.5K",     "description": "Grade school math word problems"},
    "math":             {"hf_path": "hendrycks/competition_math",         "category": "eval",    "size": "~12.5K",    "description": "MATH benchmark - competition math levels 1-5"},
    "math500":          {"hf_path": "HuggingFaceH4/MATH-500",            "category": "eval",    "size": "500",       "description": "Curated 500 hard MATH problems"},
    "aime24":           {"hf_path": "AI-MO/aimo-validation-aime",         "category": "eval",    "size": "90",        "description": "AIME 2024 competition problems"},
    "aime25":           {"hf_path": "yentinglin/aime_2025",               "category": "eval",    "size": "30",        "description": "AIME 2025 competition problems"},
    "aime26":           {"hf_path": "yentinglin/aime_2026",               "category": "eval",    "size": "30",        "description": "AIME 2026 competition problems"},
    "svamp":            {"hf_path": "ChilleD/SVAMP",                      "category": "eval",    "size": "~1K",       "description": "Arithmetic word problems"},
    "mmlu_math":        {"hf_path": "cais/mmlu",                          "category": "eval",    "size": "~1K",       "description": "MMLU abstract algebra subset"},
    "gpqa":             {"hf_path": "Idavidrein/gpqa",                    "category": "eval",    "size": "448",       "description": "Graduate-level science questions"},
    "olympiad_bench":   {"hf_path": "Hothan/OlympiadBench",              "category": "eval",    "size": "~8.5K",     "description": "Olympic-level math + physics"},

    # === Training (SFT) ===
    "metamathqa":       {"hf_path": "meta-math/MetaMathQA",               "category": "train",   "size": "~395K",     "description": "Augmented GSM8K + MATH for fine-tuning"},
    "numina_math_cot":  {"hf_path": "AI-MO/NuminaMath-CoT",              "category": "train",   "size": "~860K",     "description": "Competition math with chain-of-thought"},
    "numina_math_tir":  {"hf_path": "AI-MO/NuminaMath-TIR",              "category": "train",   "size": "~72K",      "description": "Math with Python code verification"},
    "orca_math":        {"hf_path": "microsoft/orca-math-word-problems-200k", "category": "train", "size": "~200K",   "description": "GPT-4 generated math word problems"},
    "mathinstruct":     {"hf_path": "TIGER-Lab/MathInstruct",             "category": "train",   "size": "~260K",     "description": "Compiled math instruction dataset"},
    "open_math_instruct_2": {"hf_path": "nvidia/OpenMathInstruct-2",      "category": "train",   "size": "~14M",      "description": "NVIDIA large-scale math instructions"},

    # === Process Reward / RL ===
    "prm800k":          {"hf_path": "tasksource/prm800k",                 "category": "reward",  "size": "~800K",     "description": "Process Reward Model step-level labels"},
    "math_shepherd":    {"hf_path": "peiyi9979/Math-Shepherd",            "category": "reward",  "size": "~400K",     "description": "Step-level process supervision"},

    # === Pretraining (Large) ===
    "openwebmath":      {"hf_path": "open-web-math/open-web-math",        "category": "pretrain","size": "~14.7B tok","description": "Web math text (WARNING: very large)"},
    "proof_pile_2":     {"hf_path": "EleutherAI/proof-pile-2",            "category": "pretrain","size": "~55B tok",  "description": "arXiv + textbooks (WARNING: huge)"},
}


def list_datasets():
    print("\n" + "=" * 90)
    print("Available Math Domain Datasets")
    print("=" * 90)
    for cat, label in [("eval","Evaluation"), ("train","Training (SFT)"), ("reward","Process Reward / RL"), ("pretrain","Pretraining")]:
        print(f"\n--- {label} ---")
        print(f"{'Name':<24} {'Size':<12} {'Description'}")
        print("-" * 90)
        for name, info in DATASETS.items():
            if info["category"] == cat:
                print(f"{name:<24} {info['size']:<12} {info['description']}")
    print()


def download_dataset(name: str, output_dir: str):
    if name not in DATASETS:
        print(f"  ERROR: Unknown dataset '{name}'")
        return False

    info = DATASETS[name]
    save_path = os.path.join(output_dir, name)

    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"  SKIP: {name} (already exists)")
        return True

    print(f"\n  Downloading: {name} ({info['size']})")
    print(f"    From: {info['hf_path']}")

    try:
        snapshot_download(
            repo_id=info["hf_path"],
            repo_type="dataset",
            local_dir=save_path,
        )
        file_count = sum(1 for _, _, files in os.walk(save_path) for _ in files)
        print(f"    OK: {save_path} ({file_count} files)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Math Datasets (via huggingface_hub)")
    parser.add_argument("--output-dir", type=str, default="./data/math_datasets")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true", help="All except pretrain")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--reward-only", action="store_true")
    parser.add_argument("--dataset", nargs="+", type=str)
    parser.add_argument("--include-pretrain", action="store_true")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

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
        cats = {"eval", "train", "reward"}
        if args.include_pretrain:
            cats.add("pretrain")
        to_download = [n for n, d in DATASETS.items() if d["category"] in cats]
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python data/download_math_datasets.py --list")
        print("  python data/download_math_datasets.py --eval-only")
        print("  python data/download_math_datasets.py --dataset gsm8k math500 aime25")
        print("  python data/download_math_datasets.py --all")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nDownloading {len(to_download)} datasets to {args.output_dir}")
    print("=" * 60)

    ok = fail = 0
    for name in to_download:
        if download_dataset(name, args.output_dir):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}\nDone: {ok} ok, {fail} failed\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()

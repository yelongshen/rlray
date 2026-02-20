"""
AIME24 Evaluation with Qwen-Next

This script evaluates Qwen-Next model on AIME24 math competition problems.

Supports multiple inference backends:
- HuggingFace generate() - Simple, works with any HF model
- vLLM API - Fast inference via vLLM server (OpenAI-compatible API)
- In-house LLMEngine - Custom engine with paged attention

Usage:
    # Option 1: Direct HuggingFace inference
    python aime24_qwen_eval.py \
        --model_path Qwen/Qwen3-Coder-Next \
        --data_path ../eval/aime24_test.jsonl
    
    # Option 2: vLLM server (faster)
    # First start vLLM server:
    #   vllm serve Qwen/Qwen3-Coder-Next --port 8000 --tensor-parallel-size 2
    # Then run evaluation:
    python aime24_qwen_eval.py \
        --use_vllm \
        --vllm_url http://localhost:8000/v1 \
        --model_path Qwen/Qwen3-Coder-Next \
        --data_path ../eval/aime24_test.jsonl
    
    # Option 3: In-house LLMEngine (custom paged attention)
    python aime24_qwen_eval.py \
        --use_engine \
        --model_path Qwen/Qwen3-Coder-Next \
        --data_path ../eval/aime24_test.jsonl
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

# Add xlmlib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_util import process_math_prompt, safe_math_answer_timeout

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AIME24Problem:
    """Single AIME problem."""
    id: int
    problem: str
    answer: str
    solution: str = ""
    url: str = ""

@dataclass
class AIME24Result:
    """Result for a single AIME problem."""
    id: int
    problem: str
    gold_answer: str
    predicted_answer: str
    response: str
    reward: float
    correct: bool

# ============================================================================
# Data Loading
# ============================================================================

def load_aime24_dataset(data_path: str) -> List[AIME24Problem]:
    """Load AIME24 problems from JSONL file."""
    problems = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            problems.append(AIME24Problem(
                id=data.get('id', 0),
                problem=data.get('problem', data.get('question', '')),
                answer=str(data.get('answer', '')),
                solution=data.get('solution', ''),
                url=data.get('url', '')
            ))
    return problems

# ============================================================================
# AIME24 Evaluator using vLLM API (Fast Inference)
# ============================================================================

class AIME24VLLMEvaluator:
    """
    Fast evaluator using vLLM server via OpenAI-compatible API.
    
    Start vLLM server first:
        vllm serve Qwen/Qwen3-Coder-Next --port 8000 --tensor-parallel-size 2
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        vllm_url: str = "http://localhost:8000/v1",
        max_tokens: int = 1024,
        temperature: float = 0.6,
        prompt_type: str = "v11",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = OpenAI(base_url=vllm_url, api_key="dummy")
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_type = prompt_type
        
        # Load tokenizer for answer verification
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"Connected to vLLM server at {vllm_url}")
        print(f"Model: {model_path}")
    
    def solve(self, problem: AIME24Problem) -> AIME24Result:
        """Solve single problem using vLLM API."""
        prompt = process_math_prompt(problem.problem, prompt_type=self.prompt_type)
        
        print(f"  [DEBUG] Calling vLLM API...", flush=True)
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
        )
        
        gen_time = time.time() - start_time
        response_text = response.choices[0].message.content
        tokens_generated = response.usage.completion_tokens if response.usage else len(response_text) // 4
        
        print(f"  [DEBUG] vLLM response in {gen_time:.1f}s. ~{tokens_generated} tokens ({tokens_generated/gen_time:.1f} tok/s)", flush=True)
        
        # Verify answer
        _, predicted_answer, reward = safe_math_answer_timeout(
            response_text,
            [problem.answer],
            self.tokenizer,
            prompt_type=self.prompt_type,
            alg=['is_equiv', 'text'],
            timeout=30
        )
        
        return AIME24Result(
            id=problem.id,
            problem=problem.problem,
            gold_answer=problem.answer,
            predicted_answer=predicted_answer,
            response=response_text,
            reward=reward,
            correct=(reward > 0.5)
        )

# ============================================================================
# AIME24 Evaluator using HuggingFace
# ============================================================================

class AIME24Evaluator:
    """Evaluator using HuggingFace generate() for Qwen-Next models."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        device: str = "cuda",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        prompt_type: str = "v11",
        gpu_ids: Optional[str] = None
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_type = prompt_type
        
        # Handle device mapping
        if gpu_ids is not None:
            # Use specific GPUs with device_map
            gpu_list = [int(g.strip()) for g in gpu_ids.split(',')]
            if len(gpu_list) == 1:
                self.device = f"cuda:{gpu_list[0]}"
                device_map = {"": self.device}
            else:
                self.device = "cuda:0"  # First GPU for inputs
                device_map = "auto"  # Let transformers handle multi-GPU
        else:
            self.device = device
            device_map = "auto"
        
        print(f"Loading model from {model_path}...")
        print(f"Device: {self.device}, Device map: {device_map}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded!")
    
    def solve(self, problem: AIME24Problem) -> AIME24Result:
        """Solve single problem using HF generate."""
        import sys
        
        print(f"  [DEBUG] Building prompt...", flush=True)
        prompt = process_math_prompt(problem.problem, prompt_type=self.prompt_type)
        
        # Apply chat template for instruction-tuned models
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        print(f"  [DEBUG] Tokenizing (prompt len: {len(text)} chars)...", flush=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        print(f"  [DEBUG] Input tokens: {inputs['input_ids'].shape[1]}", flush=True)
        
        print(f"  [DEBUG] Starting generation (max_new_tokens={self.max_new_tokens})...", flush=True)
        import time
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Explicit KV cache for faster generation
            )
        gen_time = time.time() - start_time
        new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        print(f"  [DEBUG] Generation done in {gen_time:.1f}s. New tokens: {new_tokens} ({new_tokens/gen_time:.1f} tok/s)", flush=True)
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        print(f"  [DEBUG] Response length: {len(response)} chars", flush=True)
        
        print(f"  [DEBUG] Verifying answer...", flush=True)
        # Use simpler algorithms that don't require external math_verify package
        # AIME answers are integers 0-999, so is_equiv and text matching suffice
        _, predicted_answer, reward = safe_math_answer_timeout(
            response,
            [problem.answer],
            self.tokenizer,
            prompt_type=self.prompt_type,
            alg=['is_equiv', 'text'],  # Skip math_verify which requires external package
            timeout=30
        )
        print(f"  [DEBUG] Verification done. Predicted: {predicted_answer}, Reward: {reward}", flush=True)
        
        return AIME24Result(
            id=problem.id,
            problem=problem.problem,
            gold_answer=problem.answer,
            predicted_answer=predicted_answer,
            response=response,
            reward=reward,
            correct=(reward > 0.5)
        )

# ============================================================================
# AIME24 Evaluator using In-house LLMEngine
# ============================================================================

class AIME24LLMEngineEvaluator:
    """
    Fast evaluator using in-house LLMEngine with paged attention.
    
    This uses the custom Qwen3-Next adapter that implements the LLMEngine interface.
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        device: str = "cuda",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        prompt_type: str = "v11",
        gpu_ids: Optional[str] = None
    ):
        # Import engine components
        from llm_engine import LLMEngine
        from qwen3_next_engine import load_qwen3_next_for_engine
        from transformers import AutoTokenizer
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_type = prompt_type
        
        # Handle device
        if gpu_ids is not None:
            gpu_list = [int(g.strip()) for g in gpu_ids.split(',')]
            self.device = f"cuda:{gpu_list[0]}"
        else:
            self.device = device
        
        print(f"Loading Qwen3-Next model with LLMEngine adapter...")
        print(f"Device: {self.device}")
        
        # Load custom model adapter
        self.model, self.tokenizer, self.config = load_qwen3_next_for_engine(
            model_path, device=self.device
        )
        
        # Set EOS token ID in config
        self.config.eos_token_id = self.tokenizer.eos_token_id
        
        # Initialize LLMEngine
        self.engine = LLMEngine(self.model, self.config, self.device)
        self.engine.model_runner.temperature = self.temperature
        print("LLMEngine initialized!")
    
    def solve(self, problem: AIME24Problem) -> AIME24Result:
        """Solve single problem using LLMEngine."""
        import time
        
        print(f"  [DEBUG] Building prompt...", flush=True)
        prompt = process_math_prompt(problem.problem, prompt_type=self.prompt_type)
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        print(f"  [DEBUG] Tokenizing (prompt len: {len(text)} chars)...", flush=True)
        input_ids = self.tokenizer.encode(text)  # Returns list of ints
        print(f"  [DEBUG] Input tokens: {len(input_ids)}", flush=True)
        
        print(f"  [DEBUG] Starting LLMEngine generation (max_tokens={self.max_new_tokens})...", flush=True)
        start_time = time.time()
        
        # Use LLMEngine for generation
        # LLMEngine.generate() expects List[List[int]] and returns List[List[int]]
        output_ids_list = self.engine.generate([input_ids])
        output_ids = output_ids_list[0]  # Get first (and only) result
        
        gen_time = time.time() - start_time
        new_tokens = len(output_ids) - len(input_ids)
        print(f"  [DEBUG] Generation done in {gen_time:.1f}s. New tokens: {new_tokens} ({new_tokens/gen_time:.1f} tok/s)", flush=True)
        
        response = self.tokenizer.decode(
            output_ids[len(input_ids):],
            skip_special_tokens=True
        )
        print(f"  [DEBUG] Response length: {len(response)} chars", flush=True)
        
        print(f"  [DEBUG] Verifying answer...", flush=True)
        _, predicted_answer, reward = safe_math_answer_timeout(
            response,
            [problem.answer],
            self.tokenizer,
            prompt_type=self.prompt_type,
            alg=['is_equiv', 'text'],
            timeout=30
        )
        print(f"  [DEBUG] Verification done. Predicted: {predicted_answer}, Reward: {reward}", flush=True)
        
        return AIME24Result(
            id=problem.id,
            problem=problem.problem,
            gold_answer=problem.answer,
            predicted_answer=predicted_answer,
            response=response,
            reward=reward,
            correct=(reward > 0.5)
        )

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AIME24 Evaluation with Qwen-Next")
    
    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    
    # Inference backend
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM server for fast inference (requires running vLLM server)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL (default: http://localhost:8000/v1)")
    parser.add_argument("--use_engine", action="store_true",
                       help="Use in-house LLMEngine with paged attention")
    
    # Data
    parser.add_argument("--data_path", type=str, default="../eval/aime24_test.jsonl")
    parser.add_argument("--output_path", type=str, default="./aime24_results.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1, 
                       help="Max samples to evaluate (-1 for all)")
    
    # Generation
    parser.add_argument("--max_gen_len", type=int, default=1024,
                       help="Max new tokens to generate (1024 is usually plenty for math)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--prompt_type", type=str, default="v11",
                       choices=["v8", "v9", "v10", "v11", "v12"])
    
    args = parser.parse_args()
    
    # Set GPU devices if specified (only for HuggingFace mode)
    if args.gpu_ids is not None and not args.use_vllm:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    # Load dataset
    print(f"\nLoading AIME24 dataset from {args.data_path}...")
    problems = load_aime24_dataset(args.data_path)
    if args.max_samples > 0:
        problems = problems[:args.max_samples]
    print(f"Loaded {len(problems)} problems")
    
    # Initialize evaluator
    if args.use_vllm:
        print(f"\nUsing vLLM server at {args.vllm_url}...")
        evaluator = AIME24VLLMEvaluator(
            model_path=args.model_path,
            vllm_url=args.vllm_url,
            max_tokens=args.max_gen_len,
            temperature=args.temperature,
            prompt_type=args.prompt_type
        )
    elif args.use_engine:
        print("\nUsing In-house LLMEngine with paged attention...")
        evaluator = AIME24LLMEngineEvaluator(
            model_path=args.model_path,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
            prompt_type=args.prompt_type,
            gpu_ids=args.gpu_ids
        )
    else:
        print("\nUsing HuggingFace Evaluator...")
        evaluator = AIME24Evaluator(
            model_path=args.model_path,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
            prompt_type=args.prompt_type,
            gpu_ids=args.gpu_ids
        )
    
    # Evaluate
    results = []
    correct_count = 0
    
    print(f"\nStarting AIME24 Evaluation...")
    print("="*70)
    
    with open(args.output_path, 'w') as f:
        for i, problem in enumerate(tqdm(problems, desc="Evaluating")):
            start_time = time.time()
            
            try:
                result = evaluator.solve(problem)
                results.append(result)
                
                if result.correct:
                    correct_count += 1
                
                elapsed = time.time() - start_time
                accuracy = correct_count / len(results) * 100
                
                # Log progress
                print(f"\n[{i+1}/{len(problems)}] Problem ID: {problem.id}")
                print(f"  Problem:")
                print("-" * 50)
                print(problem.problem)
                print("-" * 50)
                print(f"  Gold Answer: {problem.answer}")
                print(f"  Predicted: {result.predicted_answer}")
                print(f"  Reward: {result.reward:.4f}")
                print(f"  Correct: {result.correct}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Running Accuracy: {accuracy:.2f}%")
                print(f"  Response:")
                print("-" * 50)
                print(result.response)
                print("-" * 50)
                
                # Save result
                f.write(json.dumps({
                    'id': result.id,
                    'problem': result.problem,
                    'gold_answer': result.gold_answer,
                    'predicted_answer': result.predicted_answer,
                    'response': result.response,
                    'reward': result.reward,
                    'correct': result.correct
                }, ensure_ascii=False) + '\n')
                f.flush()
                
            except Exception as e:
                print(f"\nError on problem {problem.id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("AIME24 EVALUATION COMPLETE")
    print("="*70)
    print(f"Total problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count/len(results)*100:.2f}%")
    print(f"Results saved to: {args.output_path}")
    
    # Print detailed breakdown
    print(f"\nDetailed Results:")
    for r in results:
        status = "✓" if r.correct else "✗"
        print(f"  [{status}] ID {r.id}: Gold={r.gold_answer}, Pred={r.predicted_answer}, Reward={r.reward:.4f}")

if __name__ == "__main__":
    main()

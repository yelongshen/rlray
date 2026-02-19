"""
AIME24 Evaluation with Qwen-Next using In-house LLM Engine

This script uses the in-house llm_engine.py for efficient inference
with Qwen-Next model on AIME24 math competition problems.

Usage:
    # Run evaluation with nohup (keeps running after terminal close)
    nohup python aime24_qwen_eval.py \
        --model_path Qwen/Qwen3-Coder-Next \
        --data_path ../eval/aime24_test.jsonl \
        --output_path ./aime24_results.jsonl \
        > aime24_eval.log 2>&1 &
    
    # Monitor progress
    tail -f aime24_eval.log
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

from llm_engine import LLMEngine, Sequence
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
# Qwen-Next Model Wrapper for LLM Engine
# ============================================================================

class QwenNextForLLMEngine:
    """
    Wrapper class to make Qwen-Next compatible with the in-house LLMEngine.
    
    The LLMEngine expects the model to have:
    - model.layers with self_attn having k_cache and v_cache attributes
    - Forward pass returning (_, logits, _, _)
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        device: str = "cuda",
        torch_dtype = torch.bfloat16
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading Qwen-Next model from {model_path}...")
        self._load_model(model_path)
        print("Model loaded successfully!")
    
    def _load_model(self, model_path: str):
        """Load Qwen model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        # Load config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Adapt config for LLMEngine compatibility
        self._adapt_config()
    
    def _adapt_config(self):
        """Adapt the config to be compatible with LLMEngine."""
        # Create a config object that LLMEngine expects
        class LLMConfig:
            pass
        
        self.llm_config = LLMConfig()
        self.llm_config.hidden_size = self.config.hidden_size
        self.llm_config.num_attention_heads = self.config.num_attention_heads
        self.llm_config.num_key_value_heads = getattr(
            self.config, 'num_key_value_heads', 
            self.config.num_attention_heads
        )
        self.llm_config.num_hidden_layers = self.config.num_hidden_layers
        self.llm_config.eos_token_id = self.tokenizer.eos_token_id
        
    def get_model(self):
        return self.model
    
    def get_config(self):
        return self.llm_config
    
    def get_tokenizer(self):
        return self.tokenizer

# ============================================================================
# AIME24 Evaluator
# ============================================================================

class AIME24Evaluator:
    """Evaluator for AIME24 problems using in-house LLM Engine."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        device: str = "cuda",
        max_gen_len: int = 4096,
        temperature: float = 0.6,
        prompt_type: str = "v11"  # boxed format for math
    ):
        self.device = device
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.prompt_type = prompt_type
        
        # Load model
        self.qwen_wrapper = QwenNextForLLMEngine(model_path, device)
        self.model = self.qwen_wrapper.get_model()
        self.config = self.qwen_wrapper.get_config()
        self.tokenizer = self.qwen_wrapper.get_tokenizer()
        
        # Initialize LLM Engine
        print("Initializing LLM Engine...")
        self.engine = LLMEngine(self.model, self.config, device)
        self.engine.model_runner.temperature = temperature
        print("LLM Engine initialized!")
    
    def _build_prompt(self, problem: str) -> str:
        """Build prompt for AIME problem."""
        # Use math prompt formatting
        return process_math_prompt(problem, prompt_type=self.prompt_type)
    
    def _tokenize(self, prompt: str) -> List[int]:
        """Tokenize prompt."""
        return self.tokenizer.encode(prompt, add_special_tokens=True)
    
    def _decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def solve(self, problem: AIME24Problem) -> AIME24Result:
        """Solve a single AIME problem."""
        # Build prompt
        prompt = self._build_prompt(problem.problem)
        input_ids = self._tokenize(prompt)
        
        # Limit input length
        max_input_len = 2048
        if len(input_ids) > max_input_len:
            input_ids = input_ids[:max_input_len]
        
        # Generate response using LLM Engine
        outputs = self.engine.generate([input_ids])
        output_ids = outputs[0] if outputs else []
        
        # Decode response
        response = self._decode(output_ids)
        
        # Extract answer and verify
        _, predicted_answer, reward = safe_math_answer_timeout(
            response,
            [problem.answer],
            self.tokenizer,
            prompt_type=self.prompt_type,
            timeout=30
        )
        
        return AIME24Result(
            id=problem.id,
            problem=problem.problem,
            gold_answer=problem.answer,
            predicted_answer=predicted_answer,
            response=response,
            correct=(reward > 0.5)
        )
    
    def solve_batch(self, problems: List[AIME24Problem], batch_size: int = 4) -> List[AIME24Result]:
        """Solve a batch of AIME problems."""
        results = []
        
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i+batch_size]
            
            # Build prompts for batch
            prompts = [self._build_prompt(p.problem) for p in batch]
            input_ids_list = [self._tokenize(p) for p in prompts]
            
            # Truncate if needed
            max_input_len = 2048
            input_ids_list = [ids[:max_input_len] if len(ids) > max_input_len else ids 
                             for ids in input_ids_list]
            
            # Generate responses
            outputs = self.engine.generate(input_ids_list)
            
            # Process results
            for j, (problem, output_ids) in enumerate(zip(batch, outputs)):
                response = self._decode(output_ids)
                
                _, predicted_answer, reward = safe_math_answer_timeout(
                    response,
                    [problem.answer],
                    self.tokenizer,
                    prompt_type=self.prompt_type,
                    timeout=30
                )
                
                results.append(AIME24Result(
                    id=problem.id,
                    problem=problem.problem,
                    gold_answer=problem.answer,
                    predicted_answer=predicted_answer,
                    response=response,
                    correct=(reward > 0.5)
                ))
        
        return results

# ============================================================================
# Alternative: Simple HuggingFace Generation (if LLMEngine doesn't work)
# ============================================================================

class AIME24SimpleEvaluator:
    """
    Simple evaluator using HuggingFace generate() directly.
    Use this if LLMEngine has compatibility issues.
    """
    
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
        prompt = process_math_prompt(problem.problem, prompt_type=self.prompt_type)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        _, predicted_answer, reward = safe_math_answer_timeout(
            response,
            [problem.answer],
            self.tokenizer,
            prompt_type=self.prompt_type,
            timeout=30
        )
        
        return AIME24Result(
            id=problem.id,
            problem=problem.problem,
            gold_answer=problem.answer,
            predicted_answer=predicted_answer,
            response=response,
            correct=(reward > 0.5)
        )

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AIME24 Evaluation with Qwen-Next")
    
    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--use_simple", action="store_true", 
                       help="Use simple HF evaluator instead of LLMEngine")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    
    # Data
    parser.add_argument("--data_path", type=str, default="../eval/aime24_test.jsonl")
    parser.add_argument("--output_path", type=str, default="./aime24_results.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1, 
                       help="Max samples to evaluate (-1 for all)")
    
    # Generation
    parser.add_argument("--max_gen_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--prompt_type", type=str, default="v11",
                       choices=["v8", "v9", "v10", "v11", "v12"])
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    # Set GPU devices if specified
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    # Load dataset
    print(f"\nLoading AIME24 dataset from {args.data_path}...")
    problems = load_aime24_dataset(args.data_path)
    if args.max_samples > 0:
        problems = problems[:args.max_samples]
    print(f"Loaded {len(problems)} problems")
    
    # Initialize evaluator
    if args.use_simple:
        print("\nUsing Simple HuggingFace Evaluator...")
        evaluator = AIME24SimpleEvaluator(
            model_path=args.model_path,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
            prompt_type=args.prompt_type,
            gpu_ids=args.gpu_ids
        )
    else:
        print("\nUsing In-house LLM Engine Evaluator...")
        evaluator = AIME24Evaluator(
            model_path=args.model_path,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            prompt_type=args.prompt_type
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
                print(f"  Gold Answer: {problem.answer}")
                print(f"  Predicted: {result.predicted_answer}")
                print(f"  Correct: {result.correct}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Running Accuracy: {accuracy:.2f}%")
                
                # Save result
                f.write(json.dumps({
                    'id': result.id,
                    'problem': result.problem,
                    'gold_answer': result.gold_answer,
                    'predicted_answer': result.predicted_answer,
                    'response': result.response,
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
        print(f"  [{status}] ID {r.id}: Gold={r.gold_answer}, Pred={r.predicted_answer}")

if __name__ == "__main__":
    main()

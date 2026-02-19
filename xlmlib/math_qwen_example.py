"""
Math Task Evaluation with Qwen-Next Model

Usage:
    # Single GPU evaluation
    python math_qwen_example.py --model_path Qwen/Qwen3-Coder-Next --data_path ../eval/math500_test.jsonl

    # Multi-GPU with nohup (keep running after terminal close)
    nohup python math_qwen_example.py --model_path Qwen/Qwen3-Coder-Next > math_eval.log 2>&1 &
    tail -f math_eval.log
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xlmlib.math_util import (
    process_math_prompt, 
    process_math_answer,
    safe_math_answer_timeout,
    compare_math_answers
)

@dataclass
class MathProblem:
    id: str
    problem: str
    answer: str
    level: str = ""
    subject: str = ""

@dataclass  
class MathResult:
    id: str
    problem: str
    gold_answer: str
    predicted_answer: str
    response: str
    reward: float
    correct: bool

def load_math_dataset(data_path: str) -> List[MathProblem]:
    """Load math problems from JSONL file."""
    problems = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            problems.append(MathProblem(
                id=data.get('unique_id', str(idx)),
                problem=data['problem'],
                answer=data['answer'],
                level=data.get('level', ''),
                subject=data.get('subject', '')
            ))
    return problems

class QwenMathAgent:
    """Qwen-based Math Problem Solver."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        device: str = "cuda",
        max_length: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prompt_type: str = "v11"  # boxed format
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_type = prompt_type
        
        print(f"Loading model from {model_path}...")
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load Qwen model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded successfully!")
        
    def solve(self, problem: str) -> str:
        """Generate solution for a math problem."""
        # Format prompt with instruction
        prompt = process_math_prompt(problem, prompt_type=self.prompt_type)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response (only the generated part)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response
    
    def evaluate(self, problem: MathProblem) -> MathResult:
        """Solve and evaluate a single math problem."""
        response = self.solve(problem.problem)
        
        # Extract answer and compute reward
        _, extracted_answer, reward = safe_math_answer_timeout(
            response, 
            [problem.answer], 
            self.tokenizer,
            prompt_type=self.prompt_type,
            timeout=30
        )
        
        return MathResult(
            id=problem.id,
            problem=problem.problem,
            gold_answer=problem.answer,
            predicted_answer=extracted_answer,
            response=response,
            reward=reward,
            correct=(reward > 0.5)
        )

def main():
    parser = argparse.ArgumentParser(description="Math evaluation with Qwen model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--data_path", type=str, default="../eval/math500_test.jsonl")
    parser.add_argument("--output_path", type=str, default="./math_results.jsonl")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prompt_type", type=str, default="v11", 
                       choices=["v8", "v9", "v10", "v11", "v12"])
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    problems = load_math_dataset(args.data_path)
    if args.max_samples > 0:
        problems = problems[:args.max_samples]
    print(f"Loaded {len(problems)} problems")
    
    # Initialize agent
    agent = QwenMathAgent(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_type=args.prompt_type
    )
    
    # Evaluate
    results = []
    correct_count = 0
    
    with open(args.output_path, 'w') as f:
        for problem in tqdm(problems, desc="Evaluating"):
            result = agent.evaluate(problem)
            results.append(result)
            
            if result.correct:
                correct_count += 1
            
            # Write result
            f.write(json.dumps({
                'id': result.id,
                'problem': result.problem,
                'gold_answer': result.gold_answer,
                'predicted_answer': result.predicted_answer,
                'response': result.response,
                'correct': result.correct
            }) + '\n')
            f.flush()
            
            # Print progress
            accuracy = correct_count / len(results) * 100
            print(f"\n[{len(results)}/{len(problems)}] Accuracy: {accuracy:.2f}%")
            print(f"  Problem: {problem.problem[:100]}...")
            print(f"  Gold: {problem.answer}")
            print(f"  Pred: {result.predicted_answer}")
            print(f"  Correct: {result.correct}")
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Total problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count/len(results)*100:.2f}%")
    print(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()

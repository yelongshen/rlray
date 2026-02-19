"""
RL Training for Math Tasks with Qwen-Next Model

This example shows how to:
1. Use Qwen-Next as the foundation model
2. Generate math solutions (actor)
3. Score with math verifier (reward)
4. Train with policy gradient (learner)

Usage:
    # Single node with nohup
    nohup python math_rl_qwen.py --model_path Qwen/Qwen3-Coder-Next > rl_train.log 2>&1 &
    
    # Check progress
    tail -f rl_train.log
"""

import os
import sys
import json
import torch
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import deque
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xlmlib.math_util import (
    process_math_prompt,
    safe_math_answer_timeout
)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MathExperience:
    """Single RL experience for math task."""
    prompt: str
    response: str
    answer: str
    reward: float
    log_prob: float = 0.0

class ReplayBuffer:
    """Thread-safe replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def push(self, experience: MathExperience):
        with self.lock:
            self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[MathExperience]:
        with self.lock:
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Math Environment
# ============================================================================

class MathEnv:
    """Math problem environment with reward based on correctness."""
    
    def __init__(
        self, 
        data_path: str,
        prompt_type: str = "v11",
        tokenizer = None
    ):
        self.problems = self._load_data(data_path)
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.current_idx = 0
        
    def _load_data(self, path: str) -> List[Dict]:
        problems = []
        with open(path, 'r') as f:
            for line in f:
                problems.append(json.loads(line))
        return problems
    
    def reset(self) -> str:
        """Get next problem as prompt."""
        self.current_problem = self.problems[self.current_idx % len(self.problems)]
        self.current_idx += 1
        return process_math_prompt(
            self.current_problem['problem'], 
            prompt_type=self.prompt_type
        )
    
    def reward(self, response: str) -> float:
        """Compute reward based on answer correctness."""
        _, _, reward = safe_math_answer_timeout(
            response,
            [self.current_problem['answer']],
            self.tokenizer,
            prompt_type=self.prompt_type,
            timeout=10
        )
        return reward
    
    def get_answer(self) -> str:
        return self.current_problem['answer']

# ============================================================================
# Actor (Generation)
# ============================================================================

class MathActor:
    """Actor that generates math solutions."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        device: str = "cuda:0",
        use_vllm: bool = True
    ):
        self.device = device
        self.use_vllm = use_vllm
        
        if use_vllm:
            self._init_vllm(model_path)
        else:
            self._init_hf(model_path)
    
    def _init_vllm(self, model_path: str):
        """Initialize with vLLM for fast inference."""
        from vllm import LLM, SamplingParams
        
        print(f"Loading vLLM model: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048
        )
        print("vLLM model loaded!")
    
    def _init_hf(self, model_path: str):
        """Initialize with HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading HF model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("HF model loaded!")
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for batch of prompts."""
        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [o.outputs[0].text for o in outputs]
        else:
            responses = []
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                ).to(self.device)
                
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True
                    )
                response = self.tokenizer.decode(
                    output[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)
            return responses
    
    def get_tokenizer(self):
        if self.use_vllm:
            return self.llm.get_tokenizer()
        return self.tokenizer

# ============================================================================
# Learner (Training)
# ============================================================================

class MathLearner:
    """Learner that trains the model with policy gradient."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Coder-Next",
        learning_rate: float = 1e-5,
        device: str = "cuda:0"
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.optim import AdamW
        
        self.device = device
        
        print(f"Loading learner model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Use LoRA for efficient training
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Using LoRA for efficient training")
        except ImportError:
            print("PEFT not available, training full model")
        
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()
        print("Learner ready!")
    
    def train_step(self, experiences: List[MathExperience]) -> float:
        """Single training step with policy gradient."""
        total_loss = 0.0
        
        for exp in experiences:
            # Tokenize prompt + response
            full_text = exp.prompt + exp.response
            
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Get prompt length to mask loss
            prompt_tokens = self.tokenizer(
                exp.prompt, return_tensors="pt"
            )['input_ids'].shape[1]
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            
            # Weight loss by reward (policy gradient)
            # Positive reward -> reinforce, negative -> discourage
            reward_weight = exp.reward * 2 - 1  # Scale to [-1, 1]
            loss = outputs.loss * reward_weight
            
            # Backward
            loss.backward()
            total_loss += loss.item()
        
        # Update
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss / len(experiences)
    
    def save(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

# ============================================================================
# Main Training Loop
# ============================================================================

def train_math_rl(args):
    """Main training loop combining actor, environment, and learner."""
    
    # Initialize components
    print("Initializing Math RL training...")
    
    actor = MathActor(
        model_path=args.model_path,
        use_vllm=args.use_vllm
    )
    
    env = MathEnv(
        data_path=args.data_path,
        prompt_type=args.prompt_type,
        tokenizer=actor.get_tokenizer()
    )
    
    learner = MathLearner(
        model_path=args.model_path,
        learning_rate=args.learning_rate
    )
    
    buffer = ReplayBuffer(capacity=args.buffer_size)
    
    # Training stats
    total_reward = 0.0
    correct_count = 0
    
    print("\nStarting training loop...")
    
    for iteration in range(args.num_iterations):
        # ===== Actor Phase: Generate experiences =====
        prompts = []
        for _ in range(args.batch_size):
            prompts.append(env.reset())
        
        responses = actor.generate(prompts)
        
        # Score and store experiences
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            reward = env.reward(response)
            
            experience = MathExperience(
                prompt=prompt,
                response=response,
                answer=env.get_answer(),
                reward=reward
            )
            buffer.push(experience)
            
            total_reward += reward
            if reward > 0.5:
                correct_count += 1
        
        # ===== Learner Phase: Train on buffer =====
        if len(buffer) >= args.min_buffer_size:
            batch = buffer.sample(args.train_batch_size)
            loss = learner.train_step(batch)
        else:
            loss = 0.0
        
        # Logging
        if (iteration + 1) % args.log_interval == 0:
            samples = (iteration + 1) * args.batch_size
            accuracy = correct_count / samples * 100
            avg_reward = total_reward / samples
            
            print(f"\n[Iter {iteration+1}/{args.num_iterations}]")
            print(f"  Samples: {samples}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Buffer Size: {len(buffer)}")
        
        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0:
            save_path = f"{args.output_dir}/checkpoint_{iteration+1}"
            os.makedirs(save_path, exist_ok=True)
            learner.save(save_path)
    
    # Final save
    learner.save(f"{args.output_dir}/final")
    print("\nTraining complete!")

def main():
    parser = argparse.ArgumentParser(description="Math RL training with Qwen")
    
    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for fast generation")
    
    # Data
    parser.add_argument("--data_path", type=str, default="../eval/math500_test.jsonl")
    parser.add_argument("--prompt_type", type=str, default="v11")
    
    # Training
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--min_buffer_size", type=int, default=32)
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="./math_rl_output")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_math_rl(args)

if __name__ == "__main__":
    main()

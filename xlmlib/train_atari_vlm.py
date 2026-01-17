"""
Training script for VLM-based RL agents on Atari games.

Supports:
- Multiple VLM backends (GPT-4V, Claude, Qwen-VL, etc.)
- Different training paradigms (in-context learning, fine-tuning, RL)
- Experience replay and trajectory optimization
- Evaluation and logging
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from atari_vlm_env import (
    AtariVLMEnvironment,
    VLMAgent,
    GPT4VAgent,
    ClaudeAgent,
    QwenVLAgent,
    print_vlm_recommendations
)


class ExperienceBuffer:
    """Buffer to store and sample game experiences."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self.successful_trajectories = []
    
    def add(self, transition: Dict):
        """Add transition to buffer."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def add_trajectory(self, trajectory: List[Dict], total_reward: float):
        """Store successful trajectory for in-context learning."""
        self.successful_trajectories.append({
            'trajectory': trajectory,
            'reward': total_reward
        })
        
        # Keep only top K trajectories
        self.successful_trajectories = sorted(
            self.successful_trajectories,
            key=lambda x: x['reward'],
            reverse=True
        )[:20]
    
    def get_best_trajectories(self, k: int = 3) -> List[Dict]:
        """Get top K best trajectories."""
        return self.successful_trajectories[:k]
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch from buffer."""
        if len(self.buffer) < batch_size:
            return self.buffer
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class VLMTrainer:
    """Trainer for VLM-based Atari agents."""
    
    def __init__(
        self,
        game_name: str,
        vlm_type: str,
        api_key: Optional[str] = None,
        prompt_type: str = "cot",
        use_in_context_learning: bool = True,
        save_dir: str = "./atari_vlm_results",
    ):
        """
        Args:
            game_name: Atari game name
            vlm_type: Type of VLM ("gpt-4o", "claude", "qwen-vl")
            api_key: API key for commercial VLMs
            prompt_type: Type of prompt ("basic", "cot", "expert")
            use_in_context_learning: Whether to use ICL with best trajectories
            save_dir: Directory to save results
        """
        self.game_name = game_name
        self.vlm_type = vlm_type
        self.prompt_type = prompt_type
        self.use_in_context_learning = use_in_context_learning
        self.save_dir = save_dir
        
        # Create environment
        self.env = AtariVLMEnvironment(
            game_name=game_name,
            frame_stack=4,
            resize_shape=(224, 224),
        )
        
        # Create VLM agent
        self.agent = self._create_agent(vlm_type, api_key)
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=1000)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def _create_agent(self, vlm_type: str, api_key: Optional[str]) -> VLMAgent:
        """Create VLM agent based on type."""
        if vlm_type == "gpt-4o":
            if not api_key:
                raise ValueError("API key required for GPT-4V")
            return GPT4VAgent(api_key=api_key)
        
        elif vlm_type == "claude":
            if not api_key:
                raise ValueError("API key required for Claude")
            return ClaudeAgent(api_key=api_key)
        
        elif vlm_type == "qwen-vl":
            return QwenVLAgent()
        
        else:
            raise ValueError(f"Unknown VLM type: {vlm_type}")
    
    def run_episode(
        self,
        max_steps: int = 1000,
        render: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Run single episode with VLM agent.
        
        Returns:
            episode_info: Dict with reward, length, trajectory
        """
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        trajectory = []
        
        done = False
        while not done and episode_length < max_steps:
            # Get prompt
            base_prompt = self.env.get_vlm_prompt(self.prompt_type)
            
            # Add in-context examples if enabled
            if self.use_in_context_learning:
                prompt = self._add_in_context_examples(base_prompt)
            else:
                prompt = base_prompt
            
            # Get action from VLM
            try:
                obs['num_actions'] = self.env.num_actions
                obs['game_name'] = self.game_name
                action = self.agent.select_action(obs, prompt)
            except Exception as e:
                if verbose:
                    print(f"Error getting action: {e}")
                action = np.random.randint(0, self.env.num_actions)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            transition = {
                'observation': obs['text'],
                'action': action,
                'action_name': self.env.action_meanings[action],
                'reward': reward,
                'next_observation': next_obs['text'],
                'done': done,
                'step': episode_length
            }
            
            trajectory.append(transition)
            self.experience_buffer.add(transition)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if verbose and episode_length % 100 == 0:
                print(f"  Step {episode_length}: Reward={episode_reward:.1f}")
        
        # Store successful trajectory
        if episode_reward > 0:
            self.experience_buffer.add_trajectory(trajectory, episode_reward)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'trajectory': trajectory
        }
    
    def _add_in_context_examples(self, base_prompt: str) -> str:
        """Add best trajectory examples to prompt for in-context learning."""
        best_trajectories = self.experience_buffer.get_best_trajectories(k=2)
        
        if not best_trajectories:
            return base_prompt
        
        examples = "\n\nExamples of successful plays:\n"
        examples += "="*50 + "\n"
        
        for i, traj_data in enumerate(best_trajectories):
            trajectory = traj_data['trajectory']
            reward = traj_data['reward']
            
            examples += f"\nExample {i+1} (Total Reward: {reward}):\n"
            
            # Show first few steps
            for step in trajectory[:5]:
                examples += f"  Step {step['step']}: "
                examples += f"Action={step['action_name']}, "
                examples += f"Reward={step['reward']}\n"
            
            if len(trajectory) > 5:
                examples += f"  ... ({len(trajectory)} total steps)\n"
        
        examples += "\n" + "="*50 + "\n"
        examples += "Now play the current game:\n\n"
        
        return examples + base_prompt
    
    def train(
        self,
        num_episodes: int = 100,
        eval_interval: int = 10,
        save_interval: int = 20
    ):
        """
        Train VLM agent on Atari game.
        
        Args:
            num_episodes: Number of episodes to run
            eval_interval: Evaluate every N episodes
            save_interval: Save results every N episodes
        """
        print(f"\n{'='*70}")
        print(f"TRAINING VLM AGENT ON {self.game_name}")
        print(f"{'='*70}\n")
        print(f"VLM: {self.vlm_type}")
        print(f"Prompt Type: {self.prompt_type}")
        print(f"In-Context Learning: {self.use_in_context_learning}")
        print(f"Episodes: {num_episodes}\n")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print("-" * 50)
            
            start_time = time.time()
            episode_info = self.run_episode(verbose=(episode % eval_interval == 0))
            episode_time = time.time() - start_time
            
            # Store stats
            self.episode_rewards.append(episode_info['reward'])
            self.episode_lengths.append(episode_info['length'])
            
            # Print stats
            print(f"Reward: {episode_info['reward']:.1f}")
            print(f"Length: {episode_info['length']}")
            print(f"Time: {episode_time:.1f}s")
            
            # Print running average
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Avg Reward (last 10): {avg_reward:.1f}")
            
            # Save results periodically
            if (episode + 1) % save_interval == 0:
                self.save_results()
        
        # Final save
        self.save_results()
        print(f"\nTraining complete! Results saved to {self.save_dir}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate agent performance.
        
        Returns:
            eval_stats: Dict with mean/std of rewards and lengths
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING VLM AGENT")
        print(f"{'='*70}\n")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            episode_info = self.run_episode(verbose=False)
            eval_rewards.append(episode_info['reward'])
            eval_lengths.append(episode_info['length'])
        
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
        }
        
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"  Mean Length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
        print(f"  Min/Max Reward: {eval_stats['min_reward']:.1f} / {eval_stats['max_reward']:.1f}")
        
        return eval_stats
    
    def save_results(self):
        """Save training results and statistics."""
        results = {
            'game_name': self.game_name,
            'vlm_type': self.vlm_type,
            'prompt_type': self.prompt_type,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_trajectories': [
                {
                    'reward': traj['reward'],
                    'length': len(traj['trajectory'])
                }
                for traj in self.experience_buffer.get_best_trajectories(k=5)
            ]
        }
        
        save_path = os.path.join(
            self.save_dir,
            f"{self.game_name}_{self.vlm_type}_{self.prompt_type}.json"
        )
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train VLM-based RL agent on Atari games"
    )
    
    # Environment args
    parser.add_argument(
        '--game',
        type=str,
        default='Pong-v5',
        choices=[
            'Pong-v5', 'Breakout-v5', 'SpaceInvaders-v5',
            'MsPacman-v5', 'Seaquest-v5', 'Qbert-v5',
            'Asteroids-v5', 'BeamRider-v5', 'Enduro-v5'
        ],
        help='Atari game to play'
    )
    
    # VLM args
    parser.add_argument(
        '--vlm',
        type=str,
        default='qwen-vl',
        choices=['gpt-4o', 'claude', 'qwen-vl'],
        help='VLM to use'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for commercial VLMs'
    )
    parser.add_argument(
        '--prompt-type',
        type=str,
        default='cot',
        choices=['basic', 'cot', 'expert'],
        help='Type of prompt to use'
    )
    
    # Training args
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of episodes to train'
    )
    parser.add_argument(
        '--use-icl',
        action='store_true',
        help='Use in-context learning with best trajectories'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation (no training)'
    )
    parser.add_argument(
        '--num-eval-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    
    # Other args
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./atari_vlm_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--show-recommendations',
        action='store_true',
        help='Show VLM recommendations and exit'
    )
    
    args = parser.parse_args()
    
    # Show recommendations if requested
    if args.show_recommendations:
        print_vlm_recommendations()
        return
    
    # Get API key from environment if not provided
    if args.api_key is None:
        if args.vlm == 'gpt-4o':
            args.api_key = os.getenv('OPENAI_API_KEY')
        elif args.vlm == 'claude':
            args.api_key = os.getenv('ANTHROPIC_API_KEY')
    
    # Create trainer
    trainer = VLMTrainer(
        game_name=args.game,
        vlm_type=args.vlm,
        api_key=args.api_key,
        prompt_type=args.prompt_type,
        use_in_context_learning=args.use_icl,
        save_dir=args.save_dir,
    )
    
    # Run evaluation or training
    if args.eval_only:
        trainer.evaluate(num_episodes=args.num_eval_episodes)
    else:
        trainer.train(num_episodes=args.num_episodes)


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("ATARI VLM TRAINING EXAMPLES")
    print("="*70 + "\n")
    
    print("1. Train with open-source Qwen-VL (local, free):")
    print("   python train_atari_vlm.py --game Pong-v5 --vlm qwen-vl --use-icl\n")
    
    print("2. Train with GPT-4V (requires API key):")
    print("   python train_atari_vlm.py --game Breakout-v5 --vlm gpt-4o --api-key YOUR_KEY\n")
    
    print("3. Train with Claude (requires API key):")
    print("   python train_atari_vlm.py --game SpaceInvaders-v5 --vlm claude --api-key YOUR_KEY\n")
    
    print("4. Show VLM recommendations:")
    print("   python train_atari_vlm.py --show-recommendations\n")
    
    print("5. Evaluate trained agent:")
    print("   python train_atari_vlm.py --game Pong-v5 --vlm qwen-vl --eval-only\n")
    
    # Uncomment to run main
    main()

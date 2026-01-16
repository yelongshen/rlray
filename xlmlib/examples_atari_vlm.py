"""
Example script demonstrating Atari VLM environment usage.

This script shows:
1. Basic environment usage
2. Different VLM backends
3. Prompt engineering examples
4. Performance comparison
5. Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from atari_vlm_env import (
    AtariVLMEnvironment,
    print_vlm_recommendations
)


def example_basic_usage():
    """Example 1: Basic environment usage."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Environment Usage")
    print("="*70 + "\n")
    
    # Create environment
    env = AtariVLMEnvironment(
        game_name="Pong-v5",
        frame_stack=4,
        resize_shape=(224, 224),
        frame_skip=4
    )
    
    # Reset
    obs = env.reset()
    
    print("Observation keys:", list(obs.keys()))
    print("\nImage shape:", np.array(obs['image']).shape)
    print("Stacked frames shape:", obs['frames'].shape)
    print("\nText description:")
    print(obs['text'])
    print("\nAction space:")
    print(obs['action_space'])
    
    # Run a few steps
    print("\nRunning 5 random steps:")
    for step in range(5):
        action = np.random.randint(0, env.num_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={env.action_meanings[action]}, "
              f"Reward={reward}, Done={terminated or truncated}")
    
    env.close()


def example_prompt_comparison():
    """Example 2: Compare different prompt types."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Prompt Type Comparison")
    print("="*70 + "\n")
    
    env = AtariVLMEnvironment(game_name="Breakout-v5")
    obs = env.reset()
    
    prompt_types = ["basic", "cot", "expert"]
    
    for prompt_type in prompt_types:
        print(f"\n{prompt_type.upper()} PROMPT:")
        print("-" * 70)
        prompt = env.get_vlm_prompt(prompt_type)
        print(prompt)
        print()
    
    env.close()


def example_game_comparison():
    """Example 3: Compare different Atari games."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Different Atari Games")
    print("="*70 + "\n")
    
    games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5", "MsPacman-v5"]
    
    for game_name in games:
        print(f"\n{game_name}:")
        print("-" * 50)
        
        env = AtariVLMEnvironment(game_name=game_name)
        obs = env.reset()
        
        print(f"Action space: {env.num_actions} actions")
        print(f"Actions: {env.action_meanings}")
        
        # Run 10 random steps
        total_reward = 0
        for _ in range(10):
            action = np.random.randint(0, env.num_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"10-step random play reward: {total_reward}")
        
        env.close()


def example_single_episode():
    """Example 4: Complete episode walkthrough."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Episode with Random Agent")
    print("="*70 + "\n")
    
    env = AtariVLMEnvironment(
        game_name="Pong-v5",
        max_episode_steps=500
    )
    
    obs = env.reset(seed=42)
    
    episode_reward = 0
    episode_length = 0
    rewards_history = []
    
    done = False
    while not done:
        # Random action (replace with VLM in practice)
        action = np.random.randint(0, env.num_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        rewards_history.append(reward)
        
        # Print every 50 steps
        if episode_length % 50 == 0:
            print(f"Step {episode_length}: Total Reward = {episode_reward}")
    
    print(f"\nEpisode finished!")
    print(f"Total Reward: {episode_reward}")
    print(f"Episode Length: {episode_length}")
    print(f"Non-zero Rewards: {np.count_nonzero(rewards_history)}")
    
    env.close()


def example_vlm_simulation():
    """Example 5: Simulate VLM decision-making (without actual VLM calls)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Simulated VLM Agent")
    print("="*70 + "\n")
    
    env = AtariVLMEnvironment(game_name="Pong-v5")
    obs = env.reset()
    
    # Simulate VLM reasoning process
    print("Simulating VLM decision-making process:\n")
    
    for step in range(5):
        print(f"\nStep {step+1}:")
        print("-" * 50)
        
        # Show what VLM would see
        print("Game State:")
        print(obs['text'])
        
        # Show prompt
        prompt = env.get_vlm_prompt("cot")
        print("\nPrompt to VLM:")
        print(prompt[:200] + "...\n")
        
        # Simulate VLM response (in reality, call VLM API here)
        print("VLM Response (simulated):")
        print("1. Observe: I see the Pong game screen with paddles and ball")
        print("2. Analyze: The ball is moving towards my paddle")
        print("3. Plan: I should position my paddle to intercept")
        print("4. Decide: Move paddle UP")
        print("ACTION: 2\n")
        
        # Take action (using random for simulation)
        action = np.random.randint(0, env.num_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action Taken: {env.action_meanings[action]}")
        print(f"Reward: {reward}")
        
        if terminated or truncated:
            break
    
    env.close()


def example_performance_tracking():
    """Example 6: Track performance over multiple episodes."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Performance Tracking")
    print("="*70 + "\n")
    
    env = AtariVLMEnvironment(
        game_name="Pong-v5",
        max_episode_steps=1000
    )
    
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    
    print(f"Running {num_episodes} episodes with random agent...\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = np.random.randint(0, env.num_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, "
              f"Length={episode_length}")
    
    # Print statistics
    print(f"\nStatistics over {num_episodes} episodes:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min/Max Reward: {np.min(episode_rewards):.1f} / {np.max(episode_rewards):.1f}")
    
    env.close()


def example_action_analysis():
    """Example 7: Analyze action distribution."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Action Distribution Analysis")
    print("="*70 + "\n")
    
    env = AtariVLMEnvironment(game_name="SpaceInvaders-v5")
    obs = env.reset()
    
    # Track action usage
    action_counts = {action: 0 for action in range(env.num_actions)}
    
    print("Collecting action statistics over 200 steps...\n")
    
    for _ in range(200):
        action = np.random.randint(0, env.num_actions)
        action_counts[action] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs = env.reset()
    
    # Print action distribution
    print("Action Distribution:")
    print("-" * 50)
    for action, count in action_counts.items():
        percentage = (count / 200) * 100
        bar = "█" * int(percentage / 2)
        print(f"{env.action_meanings[action]:15s} | {bar} {percentage:.1f}% ({count})")
    
    env.close()


def example_frame_visualization():
    """Example 8: Visualize frame preprocessing."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Frame Preprocessing Visualization")
    print("="*70 + "\n")
    
    # Create environment with different settings
    env_color = AtariVLMEnvironment(
        game_name="Breakout-v5",
        resize_shape=(224, 224),
        grayscale=False
    )
    
    env_gray = AtariVLMEnvironment(
        game_name="Breakout-v5",
        resize_shape=(224, 224),
        grayscale=True
    )
    
    # Get observations
    obs_color = env_color.reset()
    obs_gray = env_gray.reset()
    
    print("Frame preprocessing comparison:")
    print(f"\nColor Image:")
    print(f"  Shape: {np.array(obs_color['image']).shape}")
    print(f"  Size: {obs_color['image'].size}")
    
    print(f"\nGrayscale Image:")
    print(f"  Shape: {np.array(obs_gray['image']).shape}")
    print(f"  Size: {obs_gray['image'].size}")
    
    print(f"\nFrame Stack:")
    print(f"  Color shape: {obs_color['frames'].shape}")
    print(f"  Gray shape: {obs_gray['frames'].shape}")
    
    # Note: In practice, you would use matplotlib to visualize
    print("\nNote: Use matplotlib to visualize images:")
    print("  plt.imshow(obs['image'])")
    print("  plt.show()")
    
    env_color.close()
    env_gray.close()


def example_best_practices():
    """Example 9: Best practices for VLM-based RL."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Best Practices Summary")
    print("="*70 + "\n")
    
    print("🎯 GAME SELECTION:")
    print("-" * 50)
    print("Start with simple games:")
    print("  ✓ Pong: Best for initial testing")
    print("  ✓ Breakout: Good for strategy learning")
    print("  ✓ Space Invaders: Pattern recognition")
    print("  ✗ Montezuma's Revenge: Too complex for VLMs")
    print()
    
    print("🤖 VLM SELECTION:")
    print("-" * 50)
    print("API-based (Fast, Expensive):")
    print("  1. GPT-4o: Best overall performance")
    print("  2. Claude 3.5: Best reasoning")
    print("  3. Gemini 1.5 Pro: Best for video")
    print()
    print("Open-source (Slower, Free):")
    print("  1. Qwen2-VL-7B: Best efficiency")
    print("  2. LLaVA-1.6-34B: Best reasoning")
    print("  3. CogVLM2: Best for real-time")
    print()
    
    print("📝 PROMPT ENGINEERING:")
    print("-" * 50)
    print("  • Start with 'cot' (chain-of-thought) prompts")
    print("  • Use 'expert' prompts for strategic games")
    print("  • Enable in-context learning (--use-icl)")
    print("  • Include action history in prompts")
    print()
    
    print("⚡ OPTIMIZATION:")
    print("-" * 50)
    print("  • Reduce image size: 112x112 instead of 224x224")
    print("  • Increase frame skip: 8 instead of 4")
    print("  • Use quantization for local models (4-bit)")
    print("  • Batch multiple games in parallel")
    print()
    
    print("📊 EVALUATION:")
    print("-" * 50)
    print("  • Run at least 10 episodes for statistics")
    print("  • Compare against random and human baselines")
    print("  • Track both reward and episode length")
    print("  • Monitor API costs (if using commercial VLMs)")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ATARI VLM ENVIRONMENT - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    # Print VLM recommendations first
    print_vlm_recommendations()
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Prompt Comparison", example_prompt_comparison),
        ("Game Comparison", example_game_comparison),
        ("Single Episode", example_single_episode),
        ("VLM Simulation", example_vlm_simulation),
        ("Performance Tracking", example_performance_tracking),
        ("Action Analysis", example_action_analysis),
        ("Frame Visualization", example_frame_visualization),
        ("Best Practices", example_best_practices),
    ]
    
    print("\n" + "="*70)
    print("AVAILABLE EXAMPLES:")
    print("="*70)
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print()
    
    choice = input("Enter example number to run (1-9, or 'all' for all): ").strip()
    
    if choice.lower() == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
    else:
        print("Invalid choice. Running Example 1 (Basic Usage)...")
        example_basic_usage()


if __name__ == "__main__":
    # Run main interactive menu
    # main()
    
    # Or run specific examples directly:
    example_basic_usage()
    example_prompt_comparison()
    example_best_practices()

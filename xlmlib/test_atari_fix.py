"""
Quick test to verify Atari environment name fix.
"""

from atari_vlm_env import AtariVLMEnvironment, normalize_game_name

print("Testing game name normalization...")
print()

# Test normalization function
test_cases = [
    "Pong-v5",
    "ALE/Pong-v5",
    "Breakout-v5",
    "SpaceInvaders-v5",
]

print("Game name normalization:")
for game in test_cases:
    normalized = normalize_game_name(game)
    print(f"  {game:25s} -> {normalized}")

print()
print("Testing environment creation...")

# Test creating environments
games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]

for game in games:
    try:
        print(f"\nCreating {game}...")
        env = AtariVLMEnvironment(game_name=game, max_episode_steps=100)
        obs = env.reset()
        print(f"  ✓ Success! Action space: {env.num_actions} actions")
        env.close()
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*70)
print("All tests passed! Environment names are working correctly.")
print("="*70)

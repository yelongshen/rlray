"""
Check and fix Atari environment installation.

This script checks if Atari environments are properly installed and
provides instructions to fix any issues.
"""

import sys

def check_gymnasium():
    """Check if Gymnasium is installed."""
    try:
        import gymnasium as gym
        print(f"✓ Gymnasium installed (version {gym.__version__})")
        return True
    except ImportError:
        print("✗ Gymnasium not installed")
        print("  Install with: pip install gymnasium")
        return False


def check_ale():
    """Check if ALE is installed."""
    try:
        import ale_py
        print(f"✓ ALE (Arcade Learning Environment) installed")
        return True
    except ImportError:
        print("✗ ALE not installed")
        print("  Install with: pip install ale-py")
        return False


def check_atari_roms():
    """Check if Atari ROMs are available."""
    try:
        import ale_py
        import gymnasium as gym
        
        # Register ALE environments
        gym.register_envs(ale_py)
        
        # Try to create a simple Atari environment
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env.close()
        print("✓ Atari ROMs installed and working")
        return True
    except Exception as e:
        print(f"✗ Atari ROMs not available: {e}")
        return False


def print_installation_instructions():
    """Print installation instructions."""
    print("\n" + "="*70)
    print("INSTALLATION INSTRUCTIONS")
    print("="*70)
    print("\nOption 1: Quick install (with ROM license acceptance):")
    print("  pip install 'gymnasium[atari,accept-rom-license]'")
    
    print("\nOption 2: Step by step:")
    print("  pip install gymnasium ale-py")
    print("  pip install autorom")
    print("  AutoROM --accept-license")
    
    print("\nOption 3: Manual install:")
    print("  pip install gymnasium ale-py")
    print("  # Then download ROMs from: https://github.com/Farama-Foundation/AutoROM")
    
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print("After installation, run this script again to verify:")
    print("  python check_atari_install.py")
    print("="*70 + "\n")


def main():
    """Main check routine."""
    print("\n" + "="*70)
    print("ATARI ENVIRONMENT INSTALLATION CHECK")
    print("="*70 + "\n")
    
    all_ok = True
    
    # Check Gymnasium
    if not check_gymnasium():
        all_ok = False
    
    # Check ALE
    if not check_ale():
        all_ok = False
    
    # Check ROMs
    if not check_atari_roms():
        all_ok = False
    
    print()
    
    if all_ok:
        print("="*70)
        print("✓ ALL CHECKS PASSED!")
        print("="*70)
        print("\nYou can now run Atari VLM environments:")
        print("  python examples_atari_vlm.py")
        print("  python train_atari_vlm.py --game Pong-v5 --vlm qwen-vl")
        print("="*70 + "\n")
        return 0
    else:
        print("="*70)
        print("✗ SOME CHECKS FAILED")
        print("="*70)
        print_installation_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())

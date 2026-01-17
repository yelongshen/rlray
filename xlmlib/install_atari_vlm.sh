#!/bin/bash
# Quick installation script for Atari VLM environment

echo "========================================================================"
echo "ATARI VLM ENVIRONMENT - INSTALLATION SCRIPT"
echo "========================================================================"
echo ""

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ Success"
    else
        echo "✗ Failed"
        exit 1
    fi
}

# Step 1: Install Gymnasium with Atari support
echo "Step 1/5: Installing Gymnasium with Atari support..."
pip install 'gymnasium[atari,accept-rom-license]' --upgrade
check_status
echo ""

# Step 2: Install ALE
echo "Step 2/5: Installing ALE (Arcade Learning Environment)..."
pip install ale-py
check_status
echo ""

# Step 3: Install core dependencies
echo "Step 3/5: Installing core dependencies..."
pip install opencv-python pillow numpy tqdm
check_status
echo ""

# Step 4: Install ML dependencies
echo "Step 4/5: Installing machine learning dependencies..."
pip install torch transformers accelerate
check_status
echo ""

# Step 5: Install optional API clients
echo "Step 5/5: Installing optional API clients (for commercial VLMs)..."
pip install openai anthropic google-generativeai
check_status
echo ""

# Verify installation
echo "========================================================================"
echo "VERIFYING INSTALLATION"
echo "========================================================================"
echo ""

python -c "
import sys
try:
    import gymnasium as gym
    print('✓ Gymnasium installed')
    
    import ale_py
    print('✓ ALE installed')
    
    gym.register_envs(ale_py)
    print('✓ ALE environments registered')
    
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    env.close()
    print('✓ Atari ROMs available')
    
    import cv2
    print('✓ OpenCV installed')
    
    import torch
    print('✓ PyTorch installed')
    
    import transformers
    print('✓ Transformers installed')
    
    print('')
    print('='*70)
    print('✓ ALL INSTALLATIONS SUCCESSFUL!')
    print('='*70)
    print('')
    print('You can now run:')
    print('  python examples_atari_vlm.py')
    print('  python train_atari_vlm.py --game Pong-v5 --vlm qwen-vl')
    print('='*70)
    
except Exception as e:
    print(f'✗ Error: {e}')
    print('')
    print('Please run: python check_atari_install.py')
    print('for detailed diagnostics.')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "Installation complete!"
else
    echo ""
    echo "Installation had issues. Run: python check_atari_install.py"
    exit 1
fi

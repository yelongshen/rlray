# Quick installation script for Atari VLM environment (Windows PowerShell)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "ATARI VLM ENVIRONMENT - INSTALLATION SCRIPT" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

function Check-Status {
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Success" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed" -ForegroundColor Red
        exit 1
    }
}

# Step 1: Install Gymnasium with Atari support
Write-Host "Step 1/5: Installing Gymnasium with Atari support..." -ForegroundColor Yellow
pip install 'gymnasium[atari,accept-rom-license]' --upgrade
Check-Status
Write-Host ""

# Step 2: Install ALE
Write-Host "Step 2/5: Installing ALE (Arcade Learning Environment)..." -ForegroundColor Yellow
pip install ale-py
Check-Status
Write-Host ""

# Step 3: Install core dependencies
Write-Host "Step 3/5: Installing core dependencies..." -ForegroundColor Yellow
pip install opencv-python pillow numpy tqdm
Check-Status
Write-Host ""

# Step 4: Install ML dependencies
Write-Host "Step 4/5: Installing machine learning dependencies..." -ForegroundColor Yellow
pip install torch transformers accelerate
Check-Status
Write-Host ""

# Step 5: Install optional API clients
Write-Host "Step 5/5: Installing optional API clients (for commercial VLMs)..." -ForegroundColor Yellow
pip install openai anthropic google-generativeai
Check-Status
Write-Host ""

# Verify installation
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "VERIFYING INSTALLATION" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

python -c @"
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
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Installation complete!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Installation had issues. Run: python check_atari_install.py" -ForegroundColor Red
    exit 1
}

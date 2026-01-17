"""
Atari Game RL Environment for Pretrained Vision-Language Models (VLMs)

This module provides an environment wrapper for Atari games that works with
pretrained VLMs for decision-making and policy learning.

Supported VLM Candidates:
1. GPT-4V / GPT-4o (OpenAI) - Best overall, supports video understanding
2. Claude 3.5 Sonnet (Anthropic) - Excellent reasoning, vision capabilities
3. Gemini 1.5 Pro (Google) - Long context, native video support
4. Qwen-VL / Qwen2-VL (Alibaba) - Open-source, strong game understanding
5. LLaVA-1.6 (34B) - Open-source, good visual reasoning
6. CogVLM2 (THUDM) - Optimized for visual grounding
7. Idefics2 (Hugging Face) - Open multimodal model
8. Video-LLaVA - Specialized for video understanding
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Union
import cv2
from PIL import Image
import base64
import io
import torch


def normalize_game_name(game_name: str) -> str:
    """Convert game name to proper Gymnasium format.
    
    Handles both old format (Pong-v5) and new format (ALE/Pong-v5).
    Tries multiple formats to ensure compatibility across Gymnasium versions.
    """
    if game_name.startswith("ALE/"):
        return game_name
    
    # Add ALE/ prefix if not present
    return f"ALE/{game_name}"


def check_atari_installation():
    """Check if Atari environments are properly installed."""
    try:
        import ale_py
        gym.register_envs(ale_py)
        return True
    except ImportError:
        return False


def try_create_env(game_name: str, render_mode: str):
    """Try to create Atari environment with fallback strategies."""
    errors = []
    
    # Strategy 1: Try with ALE/ prefix
    try:
        normalized = normalize_game_name(game_name)
        env = gym.make(normalized, render_mode=render_mode)
        return env
    except Exception as e:
        errors.append(f"ALE format ({normalized}): {str(e)}")
    
    # Strategy 2: Try without ALE/ prefix (older Gymnasium versions)
    try:
        base_name = game_name.replace("ALE/", "")
        env = gym.make(base_name, render_mode=render_mode)
        return env
    except Exception as e:
        errors.append(f"Base format ({base_name}): {str(e)}")
    
    # Strategy 3: Try NoFrameskip version
    try:
        base_name = game_name.replace("ALE/", "").replace("-v5", "NoFrameskip-v4")
        env = gym.make(base_name, render_mode=render_mode)
        return env
    except Exception as e:
        errors.append(f"NoFrameskip format ({base_name}): {str(e)}")
    
    # All strategies failed - provide helpful error message
    error_msg = "Failed to create Atari environment. Tried multiple formats:\n"
    for err in errors:
        error_msg += f"  - {err}\n"
    
    error_msg += "\n" + "="*70 + "\n"
    error_msg += "SOLUTION: Install Atari dependencies:\n"
    error_msg += "="*70 + "\n"
    error_msg += "1. Install ALE (Arcade Learning Environment):\n"
    error_msg += "   pip install gymnasium[atari]\n"
    error_msg += "   pip install ale-py\n\n"
    error_msg += "2. Accept ROM license and install ROMs:\n"
    error_msg += "   pip install gymnasium[accept-rom-license]\n\n"
    error_msg += "   OR manually:\n"
    error_msg += "   pip install autorom\n"
    error_msg += "   AutoROM --accept-license\n\n"
    error_msg += "3. Verify installation:\n"
    error_msg += "   python -c 'import ale_py; import gymnasium as gym; gym.register_envs(ale_py)'\n"
    error_msg += "="*70 + "\n"
    
    raise RuntimeError(error_msg)


class AtariVLMEnvironment:
    """
    Atari environment wrapper designed for VLM-based RL agents.
    
    Features:
    - Frame stacking and preprocessing for VLMs
    - Action space description generation
    - Observation to text/image conversion
    - Reward shaping and episode management
    """
    
    def __init__(
        self,
        game_name: str = "Pong-v5",
        render_mode: str = "rgb_array",
        frame_stack: int = 4,
        frame_skip: int = 4,
        resize_shape: Tuple[int, int] = (224, 224),
        grayscale: bool = False,
        max_episode_steps: int = 10000,
    ):
        """
        Args:
            game_name: Atari game name (e.g., "Pong-v5", "Breakout-v5")
            render_mode: Rendering mode for the environment
            frame_stack: Number of frames to stack as observation
            frame_skip: Number of frames to skip between actions
            resize_shape: Target size for VLM input (224x224 for most VLMs)
            grayscale: Whether to convert to grayscale
            max_episode_steps: Maximum steps per episode
        """
        self.game_name = game_name
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.resize_shape = resize_shape
        self.grayscale = grayscale
        self.max_episode_steps = max_episode_steps
        
        # Check if ALE is installed
        if not check_atari_installation():
            print("\n" + "="*70)
            print("WARNING: ALE (Arcade Learning Environment) not detected!")
            print("="*70)
            print("Attempting to register ALE environments...")
            try:
                import ale_py
                gym.register_envs(ale_py)
                print("✓ Successfully registered ALE environments")
            except ImportError:
                print("✗ ALE not installed. Please install with:")
                print("  pip install gymnasium[atari] ale-py")
                print("  pip install gymnasium[accept-rom-license]")
            print("="*70 + "\n")
        
        # Create Atari environment with fallback strategies
        self.env = try_create_env(game_name, render_mode)
        
        # Frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_history = []
        
        # Action space
        self.action_space = self.env.action_space
        self.num_actions = self.action_space.n
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        
        print(f"Initialized {game_name}")
        print(f"Action space: {self.num_actions} actions")
        print(f"Actions: {self.action_meanings}")
    
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset environment and return initial observation."""
        obs, info = self.env.reset(seed=seed)
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_history = []
        
        # Initialize frame stack
        processed_frame = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action in environment with frame skipping.
        
        Returns:
            observation: Dict with 'image', 'text', etc.
            reward: Reward from the action
            terminated: Whether episode ended (game over)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        total_reward = 0
        terminated = False
        truncated = False
        
        # Frame skipping: repeat action for frame_skip steps
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Update frame stack
        processed_frame = self._preprocess_frame(obs)
        self.frames.append(processed_frame)
        
        # Update episode tracking
        self.episode_step += 1
        self.episode_reward += total_reward
        
        # Check max episode steps
        if self.episode_step >= self.max_episode_steps:
            truncated = True
        
        # Store transition in history
        self.episode_history.append({
            'action': action,
            'reward': total_reward,
            'step': self.episode_step
        })
        
        observation = self._get_observation()
        info['episode_reward'] = self.episode_reward
        info['episode_step'] = self.episode_step
        
        return observation, total_reward, terminated, truncated, info
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for VLM input."""
        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.expand_dims(frame, axis=-1)
        
        # Resize to target shape
        frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _get_observation(self) -> Dict:
        """
        Get current observation in VLM-friendly format.
        
        Returns dict with:
        - 'image': PIL Image or numpy array
        - 'text': Text description of current state
        - 'frames': Stacked frames
        - 'action_space': Description of available actions
        """
        # Stack frames
        stacked_frames = np.array(list(self.frames))  # [frame_stack, H, W, C]
        
        # Get latest frame as PIL Image for VLM
        latest_frame = stacked_frames[-1]
        if self.grayscale:
            pil_image = Image.fromarray(latest_frame.squeeze(), mode='L')
        else:
            pil_image = Image.fromarray(latest_frame.astype(np.uint8), mode='RGB')
        
        # Generate text description
        text_description = self._generate_text_description()
        
        return {
            'image': pil_image,
            'frames': stacked_frames,
            'text': text_description,
            'action_space': self.get_action_space_description(),
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
        }
    
    def _generate_text_description(self) -> str:
        """Generate text description of current game state."""
        description = f"Game: {self.game_name}\n"
        description += f"Step: {self.episode_step}\n"
        description += f"Total Reward: {self.episode_reward}\n"
        
        if self.episode_history:
            last_action = self.action_meanings[self.episode_history[-1]['action']]
            last_reward = self.episode_history[-1]['reward']
            description += f"Last Action: {last_action}\n"
            description += f"Last Reward: {last_reward}\n"
        
        return description
    
    def get_action_space_description(self) -> str:
        """Get human-readable description of action space."""
        description = f"Available actions ({self.num_actions}):\n"
        for i, action_name in enumerate(self.action_meanings):
            description += f"  {i}: {action_name}\n"
        return description
    
    def render(self) -> np.ndarray:
        """Render current frame."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def get_vlm_prompt(self, prompt_type: str = "basic") -> str:
        """
        Generate prompts for VLM to make decisions.
        
        Args:
            prompt_type: Type of prompt ("basic", "cot", "expert")
        """
        if prompt_type == "basic":
            return self._get_basic_prompt()
        elif prompt_type == "cot":
            return self._get_cot_prompt()
        elif prompt_type == "expert":
            return self._get_expert_prompt()
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    def _get_basic_prompt(self) -> str:
        """Basic prompt for VLM."""
        prompt = f"""You are playing the Atari game: {self.game_name}.

Current State:
{self._generate_text_description()}

{self.get_action_space_description()}

Based on the current game screen, what action should you take?
Respond with ONLY the action number (0-{self.num_actions-1})."""
        return prompt
    
    def _get_cot_prompt(self) -> str:
        """Chain-of-thought prompt for better reasoning."""
        prompt = f"""You are an expert Atari game player playing: {self.game_name}.

Current State:
{self._generate_text_description()}

{self.get_action_space_description()}

Analyze the game screen and decide the best action using this process:
1. Observe: What do you see in the current game state?
2. Analyze: What is the current situation? What are the threats/opportunities?
3. Plan: What is your strategy for the next move?
4. Decide: Which action will best achieve your goal?

Think step by step, then provide your final answer as:
ACTION: [number]"""
        return prompt
    
    def _get_expert_prompt(self) -> str:
        """Expert-level prompt with game-specific strategies."""
        game_strategies = self._get_game_specific_strategies()
        
        prompt = f"""You are a world-class Atari {self.game_name} player.

Current State:
{self._generate_text_description()}

{self.get_action_space_description()}

Game-Specific Strategies:
{game_strategies}

Analyze the screen carefully and select the optimal action.
Consider:
- Immediate rewards vs long-term strategy
- Risk assessment
- Timing and positioning
- Pattern recognition

Provide your action as: ACTION: [number]"""
        return prompt
    
    def _get_game_specific_strategies(self) -> str:
        """Get game-specific strategy hints."""
        strategies = {
            "Pong": """
- Position paddle to intercept the ball
- Predict ball trajectory
- Stay centered when possible
- React quickly to opponent's moves""",
            
            "Breakout": """
- Aim for tunneling through to top layers
- Keep paddle under the ball
- Create gaps in brick formations
- Maximize ball speed with edge hits""",
            
            "SpaceInvaders": """
- Use shields strategically
- Clear columns systematically
- Time shots carefully
- Dodge alien projectiles""",
            
            "MsPacman": """
- Plan routes to collect all dots
- Avoid ghosts unless powered up
- Use power pellets strategically
- Corner turning techniques""",
            
            "Seaquest": """
- Manage oxygen levels
- Collect divers efficiently
- Avoid enemies and obstacles
- Surface when needed""",
        }
        
        for game_key, strategy in strategies.items():
            if game_key.lower() in self.game_name.lower():
                return strategy
        
        return "- Maximize score\n- Avoid losing lives\n- Learn patterns"
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API calls."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str


class VLMAgent:
    """Base class for VLM-based Atari agents."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Args:
            model_name: Name of VLM to use
            api_key: API key for the model (if needed)
        """
        self.model_name = model_name
        self.api_key = api_key
    
    def select_action(self, observation: Dict, prompt: str) -> int:
        """
        Select action based on observation and prompt.
        
        Args:
            observation: Dict with 'image', 'text', etc.
            prompt: Text prompt for the VLM
        
        Returns:
            action: Action index to take
        """
        raise NotImplementedError("Subclass must implement select_action")
    
    def parse_action_from_response(self, response: str, num_actions: int) -> int:
        """Parse action number from VLM response."""
        # Try to extract number from response
        import re
        
        # Look for "ACTION: X" pattern
        match = re.search(r'ACTION:\s*(\d+)', response, re.IGNORECASE)
        if match:
            action = int(match.group(1))
            if 0 <= action < num_actions:
                return action
        
        # Look for any number in response
        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in numbers:
            action = int(num_str)
            if 0 <= action < num_actions:
                return action
        
        # Default to random action if parsing fails
        return np.random.randint(0, num_actions)


class GPT4VAgent(VLMAgent):
    """GPT-4V/GPT-4o agent for Atari games."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(model_name=model, api_key=api_key)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def select_action(self, observation: Dict, prompt: str) -> int:
        """Select action using GPT-4V."""
        # Convert image to base64
        env = AtariVLMEnvironment(observation.get('game_name', 'Pong-v5'))
        image_base64 = env.image_to_base64(observation['image'])
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.7,
        )
        
        response_text = response.choices[0].message.content
        action = self.parse_action_from_response(
            response_text,
            observation.get('num_actions', 6)
        )
        
        return action


class ClaudeAgent(VLMAgent):
    """Claude 3.5 Sonnet agent for Atari games."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name=model, api_key=api_key)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def select_action(self, observation: Dict, prompt: str) -> int:
        """Select action using Claude."""
        # Convert image to base64
        env = AtariVLMEnvironment(observation.get('game_name', 'Pong-v5'))
        image_base64 = env.image_to_base64(observation['image'])
        
        # Make API call
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        response_text = response.content[0].text
        action = self.parse_action_from_response(
            response_text,
            observation.get('num_actions', 6)
        )
        
        return action


class QwenVLAgent(VLMAgent):
    """Qwen-VL agent (open-source, can run locally)."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        super().__init__(model_name=model_name)
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")
    
    def select_action(self, observation: Dict, prompt: str) -> int:
        """Select action using Qwen-VL."""
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": observation['image']},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[observation['image']],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7
            )
        
        response_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]
        
        action = self.parse_action_from_response(
            response_text,
            observation.get('num_actions', 6)
        )
        
        return action


# List of recommended VLMs for Atari games
RECOMMENDED_VLMS = {
    "gpt-4o": {
        "provider": "OpenAI",
        "type": "API",
        "strengths": "Best overall, excellent reasoning, fast",
        "cost": "High",
        "setup": "pip install openai"
    },
    "claude-3-5-sonnet": {
        "provider": "Anthropic",
        "type": "API",
        "strengths": "Excellent reasoning, good vision understanding",
        "cost": "High",
        "setup": "pip install anthropic"
    },
    "gemini-1.5-pro": {
        "provider": "Google",
        "type": "API",
        "strengths": "Long context, native video support",
        "cost": "Medium",
        "setup": "pip install google-generativeai"
    },
    "qwen2-vl-7b": {
        "provider": "Alibaba",
        "type": "Open-source",
        "strengths": "Good performance, runs locally, free",
        "cost": "Free (compute only)",
        "setup": "pip install transformers torch"
    },
    "llava-1.6-34b": {
        "provider": "Hugging Face",
        "type": "Open-source",
        "strengths": "Strong visual reasoning, open-source",
        "cost": "Free (compute only)",
        "setup": "pip install transformers torch"
    },
    "cogvlm2": {
        "provider": "THUDM",
        "type": "Open-source",
        "strengths": "Visual grounding, efficient",
        "cost": "Free (compute only)",
        "setup": "pip install transformers torch"
    },
    "video-llava": {
        "provider": "Research",
        "type": "Open-source",
        "strengths": "Specialized for video/temporal understanding",
        "cost": "Free (compute only)",
        "setup": "Custom installation required"
    },
}


def print_vlm_recommendations():
    """Print recommended VLMs for Atari games."""
    print("\n" + "="*70)
    print("RECOMMENDED VISION-LANGUAGE MODELS FOR ATARI GAMES")
    print("="*70 + "\n")
    
    for model_name, info in RECOMMENDED_VLMS.items():
        print(f"Model: {model_name}")
        print(f"  Provider: {info['provider']}")
        print(f"  Type: {info['type']}")
        print(f"  Strengths: {info['strengths']}")
        print(f"  Cost: {info['cost']}")
        print(f"  Setup: {info['setup']}")
        print()


if __name__ == "__main__":
    # Print VLM recommendations
    print_vlm_recommendations()
    
    # Example usage
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Create environment
    env = AtariVLMEnvironment(
        game_name="Pong-v5",
        frame_stack=4,
        resize_shape=(224, 224)
    )
    
    # Reset and get observation
    obs = env.reset()
    
    print("Observation keys:", obs.keys())
    print("\nAction space description:")
    print(obs['action_space'])
    
    print("\nBasic prompt:")
    print(env.get_vlm_prompt("basic"))
    
    print("\nChain-of-thought prompt:")
    print(env.get_vlm_prompt("cot"))
    
    # Example episode (random actions)
    print("\n" + "="*70)
    print("RUNNING EXAMPLE EPISODE (Random Actions)")
    print("="*70 + "\n")
    
    episode_reward = 0
    for step in range(10):
        action = np.random.randint(0, env.num_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        print(f"Step {step}: Action={env.action_meanings[action]}, "
              f"Reward={reward}, Total={episode_reward}")
        
        if terminated or truncated:
            break
    
    env.close()
    print(f"\nEpisode finished! Total reward: {episode_reward}")

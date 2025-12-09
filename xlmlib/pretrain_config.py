"""
Configuration file for minimal pretraining
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # For GQA, None means MHA
    max_seq_length: int = 512
    rms_norm_eps: float = 1e-6
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Batch size
    micro_batch_size: int = 4
    global_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Training steps
    max_steps: int = 10000
    warmup_ratio: float = 0.05
    
    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Paths
    output_dir: str = "./checkpoints"
    data_dir: Optional[str] = None


@dataclass
class ParallelConfig:
    """Parallelism configuration"""
    # Data parallel (automatically determined by world_size / (tp * pp))
    data_parallel_size: int = 1
    
    # Tensor parallel
    tensor_parallel_size: int = 1
    
    # Pipeline parallel
    pipeline_parallel_size: int = 1
    num_microbatches: int = 1
    pipeline_schedule: str = "1f1b"  # "gpipe" or "1f1b"
    
    def validate(self, world_size: int):
        """Validate parallel configuration"""
        total_parallel = self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
        assert total_parallel == world_size, \
            f"DP({self.data_parallel_size}) * TP({self.tensor_parallel_size}) * PP({self.pipeline_parallel_size}) = {total_parallel} != world_size({world_size})"
        
        assert self.pipeline_schedule in ["gpipe", "1f1b"], \
            f"Invalid pipeline schedule: {self.pipeline_schedule}"


# ====================== Predefined Configurations ======================

def get_small_config() -> tuple:
    """Small model (similar to GPT-2 small: ~125M params)"""
    model = ModelConfig(
        vocab_size=50257,
        hidden_size=768,
        intermediate_size=3072,
        num_layers=12,
        num_attention_heads=12,
        max_seq_length=1024,
    )
    
    training = TrainingConfig(
        micro_batch_size=8,
        global_batch_size=64,
        max_steps=100000,
        learning_rate=6e-4,
    )
    
    parallel = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    
    return model, training, parallel


def get_medium_config() -> tuple:
    """Medium model (similar to GPT-2 medium: ~350M params)"""
    model = ModelConfig(
        vocab_size=50257,
        hidden_size=1024,
        intermediate_size=4096,
        num_layers=24,
        num_attention_heads=16,
        max_seq_length=1024,
    )
    
    training = TrainingConfig(
        micro_batch_size=4,
        global_batch_size=64,
        max_steps=100000,
        learning_rate=3e-4,
    )
    
    parallel = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    
    return model, training, parallel


def get_large_config() -> tuple:
    """Large model (similar to GPT-2 large: ~774M params)"""
    model = ModelConfig(
        vocab_size=50257,
        hidden_size=1280,
        intermediate_size=5120,
        num_layers=36,
        num_attention_heads=20,
        max_seq_length=1024,
    )
    
    training = TrainingConfig(
        micro_batch_size=2,
        global_batch_size=64,
        max_steps=100000,
        learning_rate=2.5e-4,
    )
    
    parallel = ParallelConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        num_microbatches=4,
    )
    
    return model, training, parallel


def get_xl_config() -> tuple:
    """XL model (similar to GPT-2 XL: ~1.5B params)"""
    model = ModelConfig(
        vocab_size=50257,
        hidden_size=1600,
        intermediate_size=6400,
        num_layers=48,
        num_attention_heads=25,
        max_seq_length=1024,
    )
    
    training = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=64,
        max_steps=100000,
        learning_rate=2e-4,
    )
    
    parallel = ParallelConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        num_microbatches=8,
    )
    
    return model, training, parallel


def get_config(config_name: str) -> tuple:
    """Get configuration by name"""
    configs = {
        "small": get_small_config,
        "medium": get_medium_config,
        "large": get_large_config,
        "xl": get_xl_config,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]()


def save_config(model_config: ModelConfig, training_config: TrainingConfig, 
                parallel_config: ParallelConfig, path: str):
    """Save configuration to JSON"""
    import json
    from dataclasses import asdict
    
    config_dict = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "parallel": asdict(parallel_config),
    }
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str) -> tuple:
    """Load configuration from JSON"""
    import json
    
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    model_config = ModelConfig(**config_dict["model"])
    training_config = TrainingConfig(**config_dict["training"])
    parallel_config = ParallelConfig(**config_dict["parallel"])
    
    return model_config, training_config, parallel_config

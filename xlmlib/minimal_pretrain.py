"""
Minimal Language Model Pretraining with Data Parallel, Tensor Parallel, and Pipeline Parallel
"""
import os
import sys
import time
import math
import argparse
from datetime import datetime
from typing import Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Import our modules
from minimal_model import MinimalTransformer
from moe_layers import MoETransformer
from pipeline_parallel import PipelineParallel, partition_model_for_pipeline
from parallel_initialization import (
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)


class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration - replace with PackedDataset in production"""
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 512, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random token sequences
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {
            'input_ids': tokens,
            'labels': tokens.clone(),
        }


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return rank, world_size, local_rank, device


def setup_model_parallel(tensor_parallel_size: int, pipeline_parallel_size: int):
    """Initialize tensor and pipeline parallel groups"""
    if tensor_parallel_size > 1 or pipeline_parallel_size > 1:
        if not model_parallel_is_initialized():
            initialize_model_parallel(
                model_parallel_size_=tensor_parallel_size,
                pipeline_length=pipeline_parallel_size,
            )


def create_model(args, device):
    """Create model with appropriate parallelism"""
    use_tensor_parallel = args.tensor_parallel_size > 1
    use_expert_parallel = args.expert_parallel_size > 1
    
    # Create MoE model if num_experts > 0
    if args.num_experts > 0:
        model = MoETransformer(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.max_seq_length,
            num_experts=args.num_experts,
            num_experts_per_token=args.num_experts_per_token,
            moe_layer_interval=args.moe_layer_interval,
            use_tensor_parallel=use_tensor_parallel,
            use_expert_parallel=use_expert_parallel,
            router_aux_loss_coef=args.router_aux_loss_coef,
            router_z_loss_coef=args.router_z_loss_coef,
        )
    else:
        model = MinimalTransformer(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.max_seq_length,
            use_tensor_parallel=use_tensor_parallel,
        )
    
    model = model.to(device)
    
    # Wrap with pipeline parallel if needed
    if args.pipeline_parallel_size > 1:
        model = PipelineParallel(
            model,
            num_stages=args.pipeline_parallel_size,
            num_microbatches=args.num_microbatches,
            schedule_type=args.pipeline_schedule,
        )
        # For pipeline parallel, we don't use DDP on the whole model
        return model, None
    
    # Wrap with DDP for data parallel
    if args.data_parallel_size > 1 and not use_tensor_parallel:
        # Standard DDP
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
        return model, None
    elif args.data_parallel_size > 1 and use_tensor_parallel:
        # DDP with tensor parallel - use process group
        dp_group = get_data_parallel_group()
        model = DDP(model, process_group=dp_group)
        return model, dp_group
    
    return model, None


def create_optimizer(model, args):
    """Create optimizer"""
    # Get parameters based on model type
    if isinstance(model, PipelineParallel):
        params = model.stage_module.parameters()
    elif isinstance(model, DDP):
        params = model.module.parameters()
    else:
        params = model.parameters()
    
    optimizer = AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    
    return optimizer


def create_scheduler(optimizer, args, num_training_steps):
    """Create learning rate scheduler"""
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=args.min_lr,
    )
    
    return scheduler


def train_step(model, batch, optimizer, scheduler, args, device):
    """Single training step"""
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    if isinstance(model, PipelineParallel):
        # Pipeline parallel forward-backward
        loss, logits = model.forward_backward(
            input_ids=input_ids,
            labels=labels,
        )
        aux_loss = None
    else:
        # Regular forward
        if args.num_experts > 0:
            # MoE model returns (logits, loss, aux_loss)
            logits, loss, aux_loss = model(input_ids=input_ids, labels=labels)
        else:
            # Regular model returns (logits, loss)
            logits, loss = model(input_ids=input_ids, labels=labels)
            aux_loss = None
    
    # Backward pass (if not pipeline parallel)
    if not isinstance(model, PipelineParallel):
        loss.backward()
    
    # Gradient clipping
    if args.grad_clip > 0:
        if isinstance(model, PipelineParallel):
            nn.utils.clip_grad_norm_(model.stage_module.parameters(), args.grad_clip)
        elif isinstance(model, DDP):
            nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    
    return loss.item() if loss is not None else 0.0, aux_loss.item() if aux_loss is not None else 0.0


def save_checkpoint(model, optimizer, scheduler, step, args, rank):
    """Save checkpoint"""
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get model state dict
        if isinstance(model, PipelineParallel):
            model_state = model.stage_module.state_dict()
        elif isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args),
        }
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def print_training_info(args, rank, world_size):
    """Print training configuration"""
    if rank == 0:
        print("=" * 80)
        print("Training Configuration")
        print("=" * 80)
        print(f"World Size: {world_size}")
        print(f"Data Parallel Size: {args.data_parallel_size}")
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
        print(f"Pipeline Parallel Size: {args.pipeline_parallel_size}")
        if args.num_experts > 0:
            print(f"Expert Parallel Size: {args.expert_parallel_size}")
            print(f"Num Experts: {args.num_experts}")
            print(f"Num Experts Per Token: {args.num_experts_per_token}")
            print(f"MoE Layer Interval: {args.moe_layer_interval}")
        print(f"Micro Batch Size: {args.micro_batch_size}")
        print(f"Global Batch Size: {args.global_batch_size}")
        print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Max Steps: {args.max_steps}")
        print(f"Model Parameters:")
        print(f"  - Hidden Size: {args.hidden_size}")
        print(f"  - Num Layers: {args.num_layers}")
        print(f"  - Num Attention Heads: {args.num_attention_heads}")
        print(f"  - Vocab Size: {args.vocab_size}")
        print(f"  - Max Seq Length: {args.max_seq_length}")
        print("=" * 80)


def get_parallel_info():
    """Get parallel configuration info"""
    if model_parallel_is_initialized():
        dp_rank = get_data_parallel_rank()
        dp_world_size = get_data_parallel_world_size()
        tp_rank = get_model_parallel_rank()
        tp_world_size = get_model_parallel_world_size()
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
    else:
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        dp_world_size = dist.get_world_size() if dist.is_initialized() else 1
        tp_rank = 0
        tp_world_size = 1
        pp_rank = 0
        pp_world_size = 1
    
    return dp_rank, dp_world_size, tp_rank, tp_world_size, pp_rank, pp_world_size


def main():
    parser = argparse.ArgumentParser(description='Minimal LLM Pretraining')
    
    # Model args
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--intermediate-size', type=int, default=3072)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--num-attention-heads', type=int, default=12)
    parser.add_argument('--num-key-value-heads', type=int, default=None)
    parser.add_argument('--max-seq-length', type=int, default=512)
    
    # Training args
    parser.add_argument('--micro-batch-size', type=int, default=4)
    parser.add_argument('--global-batch-size', type=int, default=32)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    
    # Parallelism args
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', type=int, default=1)
    parser.add_argument('--expert-parallel-size', type=int, default=1,
                       help='Expert parallel size for MoE models')
    parser.add_argument('--num-microbatches', type=int, default=1,
                       help='Number of microbatches for pipeline parallelism')
    parser.add_argument('--pipeline-schedule', type=str, default='1f1b',
                       choices=['gpipe', '1f1b'])
    
    # MoE args
    parser.add_argument('--num-experts', type=int, default=0,
                       help='Number of experts (0 = no MoE, use dense model)')
    parser.add_argument('--num-experts-per-token', type=int, default=2,
                       help='Number of experts to route each token to')
    parser.add_argument('--moe-layer-interval', type=int, default=2,
                       help='Use MoE every N layers (0 = all layers, 1 = no MoE)')
    parser.add_argument('--router-aux-loss-coef', type=float, default=0.01,
                       help='Coefficient for router auxiliary loss')
    parser.add_argument('--router-z-loss-coef', type=float, default=0.001,
                       help='Coefficient for router z-loss')
    
    # Data args
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--num-train-samples', type=int, default=10000,
                       help='Number of training samples (for dummy data)')
    
    # Logging and checkpointing
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    
    # Calculate data parallel size
    # Note: For MoE models with expert parallel, EP size is same as TP size
    if args.num_experts > 0 and args.expert_parallel_size > 1:
        args.expert_parallel_size = args.tensor_parallel_size
    else:
        args.expert_parallel_size = 1
    
    args.data_parallel_size = world_size // (args.tensor_parallel_size * args.pipeline_parallel_size)
    
    # Setup model parallel
    setup_model_parallel(args.tensor_parallel_size, args.pipeline_parallel_size)
    
    # Print training info
    print_training_info(args, rank, world_size)
    
    # Create model
    model, dp_group = create_model(args, device)
    
    # Print model size
    if rank == 0:
        if isinstance(model, PipelineParallel):
            total_params = sum(p.numel() for p in model.stage_module.parameters())
        elif isinstance(model, DDP):
            total_params = model.module.get_num_params()
        else:
            total_params = model.get_num_params()
        print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, args.max_steps)
    
    # Create dataset and dataloader
    if args.data_dir:
        # Use PackedDataset (requires packed_dataset.py)
        from packed_dataset import PackedDataset
        import glob
        filenames = sorted(glob.glob(f'{args.data_dir}/train*'))
        dataset = PackedDataset(
            filenames,
            n_chunks=8,
            block_size=args.max_seq_length,
            shuffle=True,
            seed=42 + rank,
            num_processes=args.data_parallel_size,
            process_rank=get_data_parallel_rank() if model_parallel_is_initialized() else rank,
        )
        dataloader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, pin_memory=True)
    else:
        # Use dummy data
        dataset = SimpleTextDataset(
            num_samples=args.num_train_samples,
            seq_length=args.max_seq_length,
            vocab_size=args.vocab_size,
        )
        # Use DistributedSampler for data parallel
        if args.data_parallel_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=args.data_parallel_size,
                rank=get_data_parallel_rank() if model_parallel_is_initialized() else rank,
                shuffle=True,
            )
            dataloader = DataLoader(dataset, batch_size=args.micro_batch_size, sampler=sampler, pin_memory=True)
        else:
            dataloader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=True, pin_memory=True)
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    total_aux_loss = 0
    start_time = time.time()
    
    dp_rank, dp_world_size, tp_rank, tp_world_size, pp_rank, pp_world_size = get_parallel_info()
    
    print(f"Rank {rank} | DP: {dp_rank}/{dp_world_size} | TP: {tp_rank}/{tp_world_size} | PP: {pp_rank}/{pp_world_size}")
    
    data_iter = iter(dataloader)
    
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Training step
        loss, aux_loss = train_step(model, batch, optimizer, scheduler, args, device)
        total_loss += loss
        total_aux_loss += aux_loss
        global_step += 1
        
        # Logging
        if global_step % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            avg_aux_loss = total_aux_loss / args.log_interval
            elapsed = time.time() - start_time
            throughput = args.log_interval * args.micro_batch_size / elapsed
            
            if rank == 0 or (pp_rank == pp_world_size - 1 and dp_rank == 0 and tp_rank == 0):
                log_msg = f"Step {global_step} | Loss: {avg_loss:.4f}"
                if args.num_experts > 0:
                    log_msg += f" | Aux Loss: {avg_aux_loss:.4f}"
                log_msg += f" | LR: {scheduler.get_last_lr()[0]:.2e} | Throughput: {throughput:.2f} samples/s"
                print(log_msg)
            
            total_loss = 0
            total_aux_loss = 0
            start_time = time.time()
        
        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, args, rank)
    
    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, args, rank)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        print("Training complete!")


if __name__ == '__main__':
    main()

"""
Pipeline Parallel Implementation for Transformer Models
Supports interleaved and non-interleaved pipeline schedules
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Tuple, Union, Any
from collections import deque
import queue


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group."""
    from parallel_initialization import get_pipeline_parallel_group
    return get_pipeline_parallel_group()


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    group = get_pipeline_model_parallel_group()
    return dist.get_world_size(group=group)


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    group = get_pipeline_model_parallel_group()
    return dist.get_rank(group=group)


def get_pipeline_model_parallel_ranks():
    """Return all ranks in the pipeline model parallel group."""
    from parallel_initialization import get_pipeline_parallel_ranks
    return get_pipeline_parallel_ranks()


class PipelineStage:
    """Wrapper for a model partition in pipeline parallelism"""
    
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
    ):
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        
        # Get pipeline parallel group
        self.pp_group = get_pipeline_model_parallel_group()
        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_world_size = get_pipeline_model_parallel_world_size()
        self.pp_ranks = get_pipeline_model_parallel_ranks()
        
    def send_forward(self, tensor: torch.Tensor) -> None:
        """Send activation to next stage"""
        if not self.is_last_stage:
            next_rank = self.pp_ranks[self.pp_rank + 1]
            dist.send(tensor.contiguous(), dst=next_rank)
    
    def recv_forward(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Receive activation from previous stage"""
        if not self.is_first_stage:
            prev_rank = self.pp_ranks[self.pp_rank - 1]
            tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
            dist.recv(tensor, src=prev_rank)
            return tensor
        return None
    
    def send_backward(self, tensor: torch.Tensor) -> None:
        """Send gradient to previous stage"""
        if not self.is_first_stage:
            prev_rank = self.pp_ranks[self.pp_rank - 1]
            dist.send(tensor.contiguous(), dst=prev_rank)
    
    def recv_backward(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Receive gradient from next stage"""
        if not self.is_last_stage:
            next_rank = self.pp_ranks[self.pp_rank + 1]
            tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tensor, src=next_rank)
            return tensor
        return None


class PipelineSchedule:
    """Base class for pipeline schedules"""
    
    def __init__(
        self,
        num_microbatches: int,
        num_stages: int,
        stage_id: int,
    ):
        self.num_microbatches = num_microbatches
        self.num_stages = num_stages
        self.stage_id = stage_id
        
    def get_schedule(self) -> List[Tuple[str, int]]:
        """
        Returns a schedule as a list of (operation, microbatch_id) tuples.
        Operations: 'forward', 'backward'
        """
        raise NotImplementedError


class GPipeSchedule(PipelineSchedule):
    """GPipe: naive pipeline schedule with bubble overhead"""
    
    def get_schedule(self) -> List[Tuple[str, int]]:
        schedule = []
        
        # Forward passes for all microbatches
        for i in range(self.num_microbatches):
            schedule.append(('forward', i))
        
        # Backward passes for all microbatches (in reverse order)
        for i in range(self.num_microbatches - 1, -1, -1):
            schedule.append(('backward', i))
        
        return schedule


class OneFOneBSchedule(PipelineSchedule):
    """1F1B: One forward, one backward interleaved schedule"""
    
    def get_schedule(self) -> List[Tuple[str, int]]:
        schedule = []
        
        # Warmup phase: fill pipeline with forward passes
        num_warmup_microbatches = self.num_stages - self.stage_id - 1
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        
        for i in range(num_warmup_microbatches):
            schedule.append(('forward', i))
        
        # 1F1B phase: alternate forward and backward
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches
        for i in range(num_microbatches_remaining):
            schedule.append(('forward', num_warmup_microbatches + i))
            schedule.append(('backward', i))
        
        # Cooldown phase: remaining backward passes
        for i in range(num_warmup_microbatches):
            schedule.append(('backward', num_microbatches_remaining + i))
        
        return schedule


class PipelineParallel:
    """
    Pipeline Parallel wrapper for transformer models.
    Splits the model across multiple devices/stages.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_microbatches: int = 1,
        schedule_type: str = "1f1b",  # "gpipe" or "1f1b"
    ):
        """
        Args:
            model: The model to be split into pipeline stages
            num_stages: Number of pipeline stages
            num_microbatches: Number of microbatches to split each batch into
            schedule_type: Type of pipeline schedule ("gpipe" or "1f1b")
        """
        self.model = model
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.schedule_type = schedule_type
        
        # Get pipeline parallel info
        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_world_size = get_pipeline_model_parallel_world_size()
        
        assert self.pp_world_size == num_stages, \
            f"Number of stages ({num_stages}) must match PP world size ({self.pp_world_size})"
        
        self.is_first_stage = (self.pp_rank == 0)
        self.is_last_stage = (self.pp_rank == self.num_stages - 1)
        
        # Partition the model
        self.stage_module = self._partition_model()
        
        # Create pipeline stage wrapper
        self.stage = PipelineStage(
            self.stage_module,
            self.pp_rank,
            self.num_stages,
            self.is_first_stage,
            self.is_last_stage,
        )
        
        # Create schedule
        if schedule_type == "gpipe":
            self.schedule = GPipeSchedule(num_microbatches, num_stages, self.pp_rank)
        elif schedule_type == "1f1b":
            self.schedule = OneFOneBSchedule(num_microbatches, num_stages, self.pp_rank)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def _partition_model(self) -> nn.Module:
        """
        Partition the model into stages.
        For a transformer, we split the layers evenly across stages.
        """
        if not hasattr(self.model, 'layers'):
            raise ValueError("Model must have a 'layers' attribute for pipeline partitioning")
        
        num_layers = len(self.model.layers)
        layers_per_stage = (num_layers + self.num_stages - 1) // self.num_stages
        
        start_layer = self.pp_rank * layers_per_stage
        end_layer = min((self.pp_rank + 1) * layers_per_stage, num_layers)
        
        # Create a module for this stage
        stage_module = nn.Module()
        
        if self.is_first_stage:
            # First stage has embedding
            stage_module.embed_tokens = self.model.embed_tokens
        
        # Add layers for this stage
        stage_module.layers = nn.ModuleList(self.model.layers[start_layer:end_layer])
        
        if self.is_last_stage:
            # Last stage has norm and lm_head
            stage_module.norm = self.model.norm
            stage_module.lm_head = self.model.lm_head
        
        return stage_module
    
    def forward_backward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Execute forward and backward passes with pipeline parallelism.
        
        Returns:
            loss: Loss tensor (only on last stage)
            logits: Output logits (only on last stage)
        """
        # Storage for intermediate activations and gradients
        activations = {}
        
        # Get schedule
        schedule_ops = self.schedule.get_schedule()
        
        # Split batch into microbatches
        if self.is_first_stage:
            microbatch_size = input_ids.size(0) // self.num_microbatches
            input_ids_list = torch.chunk(input_ids, self.num_microbatches, dim=0)
            if labels is not None:
                labels_list = torch.chunk(labels, self.num_microbatches, dim=0)
            else:
                labels_list = [None] * self.num_microbatches
        
        total_loss = None
        output_logits = []
        
        # Execute schedule
        for op, mb_id in schedule_ops:
            if op == 'forward':
                loss, logits = self._forward_microbatch(
                    mb_id,
                    input_ids_list[mb_id] if self.is_first_stage else None,
                    attention_mask,
                    labels_list[mb_id] if self.is_first_stage else None,
                    activations,
                )
                
                if self.is_last_stage:
                    output_logits.append(logits)
                    if loss is not None:
                        if total_loss is None:
                            total_loss = loss / self.num_microbatches
                        else:
                            total_loss = total_loss + loss / self.num_microbatches
            
            elif op == 'backward':
                self._backward_microbatch(mb_id, activations)
        
        # Concatenate outputs
        if self.is_last_stage and output_logits:
            output_logits = torch.cat(output_logits, dim=0)
        else:
            output_logits = None
        
        return total_loss, output_logits
    
    def _forward_microbatch(
        self,
        mb_id: int,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        activations: dict,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for a single microbatch"""
        
        if self.is_first_stage:
            # First stage: embedding
            hidden_states = self.stage_module.embed_tokens(input_ids)
        else:
            # Receive from previous stage
            # We need to know the shape - assume we can derive it from model config
            batch_size = input_ids.size(0) if input_ids is not None else 1
            seq_len = input_ids.size(1) if input_ids is not None else 1
            hidden_size = self.model.hidden_size
            
            # Receive activation
            shape = (batch_size, seq_len, hidden_size)
            hidden_states = self.stage.recv_forward(
                shape, 
                dtype=torch.float32,
                device=next(self.stage_module.parameters()).device
            )
        
        # Store input for backward
        activations[f'input_{mb_id}'] = hidden_states
        
        # Forward through layers
        for layer in self.stage_module.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Store output for backward
        activations[f'output_{mb_id}'] = hidden_states
        
        loss = None
        logits = None
        
        if self.is_last_stage:
            # Last stage: norm and lm_head
            hidden_states = self.stage_module.norm(hidden_states)
            logits = self.stage_module.lm_head(hidden_states)
            
            # Compute loss if labels provided
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1)
                )
                activations[f'loss_{mb_id}'] = loss
        else:
            # Send to next stage
            self.stage.send_forward(hidden_states)
        
        return loss, logits
    
    def _backward_microbatch(self, mb_id: int, activations: dict):
        """Backward pass for a single microbatch"""
        
        output_tensor = activations[f'output_{mb_id}']
        
        if self.is_last_stage:
            # Last stage: backward from loss
            loss = activations[f'loss_{mb_id}']
            loss.backward()
            grad_output = activations[f'output_{mb_id}'].grad
        else:
            # Receive gradient from next stage
            grad_output = self.stage.recv_backward(
                output_tensor.shape,
                dtype=output_tensor.dtype,
                device=output_tensor.device
            )
        
        # Backward through layers (automatically via autograd)
        output_tensor.backward(grad_output)
        
        if not self.is_first_stage:
            # Send gradient to previous stage
            input_tensor = activations[f'input_{mb_id}']
            if input_tensor.grad is not None:
                self.stage.send_backward(input_tensor.grad)
        
        # Clean up activations to save memory
        del activations[f'input_{mb_id}']
        del activations[f'output_{mb_id}']
        if f'loss_{mb_id}' in activations:
            del activations[f'loss_{mb_id}']


def partition_model_for_pipeline(
    model: nn.Module,
    num_stages: int,
    stage_id: int,
) -> nn.Module:
    """
    Partition a transformer model for pipeline parallelism.
    
    Args:
        model: The full model
        num_stages: Total number of pipeline stages
        stage_id: Current stage ID (0-indexed)
    
    Returns:
        A module containing only the layers for this stage
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model must have a 'layers' attribute for pipeline partitioning")
    
    num_layers = len(model.layers)
    layers_per_stage = (num_layers + num_stages - 1) // num_stages
    
    start_layer = stage_id * layers_per_stage
    end_layer = min((stage_id + 1) * layers_per_stage, num_layers)
    
    is_first_stage = (stage_id == 0)
    is_last_stage = (stage_id == num_stages - 1)
    
    # Create a new module for this stage
    stage_module = nn.Module()
    
    if is_first_stage:
        stage_module.embed_tokens = model.embed_tokens
    
    stage_module.layers = nn.ModuleList(model.layers[start_layer:end_layer])
    
    if is_last_stage:
        stage_module.norm = model.norm
        stage_module.lm_head = model.lm_head
    
    return stage_module

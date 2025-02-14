# coding=utf-8
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" PyTorch Samba model."""

import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
import copy
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache

from transformers.utils import is_torchdynamo_compiling

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from einops import rearrange, repeat

#from .configuration_phi3_samba import SambaConfig

logger = logging.get_logger(__name__)

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

if not _flash_supports_window_size:
    raise ValueError("Please update flash-attention to support window size.")

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
import causal_conv1d_cuda
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

from torch.cuda.amp import custom_bwd, custom_fwd

# monkey patch to add support for our cache
def _prepare_cache_for_generation(
    self,
    generation_config,
    model_kwargs: Dict,
    assistant_model: "PreTrainedModel",
    batch_size: int,
    max_cache_length: int,
    device: torch.device,
) -> bool:
    """
    Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
    instantiated, writes it to `model_kwargs`, under the name expected by the model.
    """
    cache_name = "past_key_values" 
    # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
    # `generation_config.validate()`)
    if generation_config.use_cache is False:
        return
    # Otherwise we NEED to prepare a cache, based on `generation_config.cache_implementation`
    # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
    # which is only supported in dynamic caches atm
    if assistant_model is not None:
        logger.warning_once(
            "An assistant model is provided, using a dynamic cache instead of a cache of type="
            f"'{generation_config.cache_implementation}'."
        )
        model_kwargs[cache_name] = DynamicCache()
        return
    model_kwargs[cache_name] = self._get_cache(
        cache_implementation="sambay",
        batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
        max_cache_len=max_cache_length,
        device=device,
        model_kwargs=model_kwargs,
    )

def _get_cache(
    self, cache_implementation: str, batch_size: int, max_cache_len: int, device: torch.device, model_kwargs
) -> Cache:
    """
    Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
    new `generate` call requires a larger cache or uses a different batch size.
    Returns the resulting cache object.
    """
    cache_cls: Cache = SambaYCache
    requires_cross_attention_cache = (
        self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
    )
    if hasattr(self, "_cache"):
        cache_to_check = self._cache.self_attention_cache if requires_cross_attention_cache else self._cache
    if cache_implementation == "sliding_window":
        max_cache_len = min(self.config.sliding_window, max_cache_len)
    need_new_cache = (
        not hasattr(self, "_cache")
        or (not isinstance(cache_to_check, cache_cls))
        or cache_to_check.batch_size != batch_size
    )
    if cache_implementation != "mamba":
        need_new_cache = need_new_cache or cache_to_check.max_cache_len < max_cache_len
    if requires_cross_attention_cache and hasattr(self, "_cache"):
        need_new_cache = (
            need_new_cache
            or self._cache.cross_attention_cache.max_cache_len != model_kwargs["encoder_outputs"][0].shape[1]
        )

    if need_new_cache:
        if hasattr(self.config, "_pre_quantization_dtype"):
            cache_dtype = self.config._pre_quantization_dtype
        else:
            if not is_torchdynamo_compiling():
                cache_dtype = self.dtype
            else:
                # NOTE: self.dtype is not compatible with torch.compile, as it calls `self.parameters()`.
                # Workaround: trust the lm_head, whose attribute name is somewhat consistent across generative
                # models. May cause trobles with non-text modalities.
                cache_dtype = self.get_output_embeddings().weight.dtype

        def get_layer_device_map(execution_device_map: Optional[dict] = None):
            if execution_device_map is None:
                return None
            elif len(execution_device_map) == 1 and "" in execution_device_map:
                return {idx: execution_device_map[""] for idx in range(self.config.num_hidden_layers)}
            layer_device_map = {}
            for layer in execution_device_map:
                for idx in range(self.config.num_hidden_layers):
                    if f".{idx}." in f"{layer}.":
                        layer_device_map[idx] = execution_device_map[layer]
                        break
            for idx in range(self.config.num_hidden_layers):
                if idx not in layer_device_map:
                    raise RuntimeError(f"layer {idx} has not been mapped to a device.")
            return layer_device_map

        execution_device_map = None
        # Taken from dispatch_model from accelerate.
        # This is needed here if we don't want to make changes in accelerate in order to save execution_device
        # For offloaded case, we need to get the execution device, not just the device where it is offloaded
        if hasattr(self, "hf_device_map"):
            main_device = [d for d in self.hf_device_map.values() if d not in ["cpu", "disk"]][0]
            execution_device_map = {
                name: main_device if device in ["cpu", "disk"] else device
                for name, device in self.hf_device_map.items()
            }
        layer_device_map = get_layer_device_map(execution_device_map)
        cache_kwargs = {
            "config": self.config.get_text_config(),
            "batch_size": batch_size,
            "max_cache_len": max_cache_len,
            "device": device,
            "dtype": cache_dtype,
            "layer_device_map": layer_device_map,
        }
        self._cache = cache_cls(**cache_kwargs)
    else:
        self._cache.reset()
    return self._cache

GenerationMixin._prepare_cache_for_generation = _prepare_cache_for_generation
GenerationMixin._get_cache = _get_cache

class SambaYCache(Cache):
    """
    A dynamic cache that can handle both the sliding window attention cache, one full attention cache and the mamba cache
    (which has a constant shape regardless of seq_len).
    """
    def __init__(self,
        config: SambaConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device: Union[torch.device, str] = "cuda",
        dtype: torch.dtype = torch.float16,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_kernel_size = conv_kernel_size
    
        if batch_size is not None:
            logger.warning_once(
                f"The 'batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.49. Use the more precisely named 'max_batch_size' argument instead."
            )
            
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size or max_batch_size
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.global_attn_idx = config.num_hidden_layers//2 + 1
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        global_cache_shape = (self.max_batch_size, self.num_key_value_heads, max_cache_len, self.head_dim)
        sliding_cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            min(config.sliding_window, max_cache_len),
            self.head_dim,
        )
        conv_cache_shape = (self.max_batch_size, intermediate_size, conv_kernel_size)
        ssm_cache_shape = (self.max_batch_size, intermediate_size, ssm_state_size)
        for i in range(config.num_hidden_layers//2 + 2):
            if layer_device_map is not None:
                layer_device = layer_device_map[i]
            else:
                layer_device = device
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            if i == self.global_attn_idx:
                key_cache_shape = value_cache_shape = global_cache_shape
            elif i % 2 == 0:
                key_cache_shape = conv_cache_shape
                value_cache_shape = ssm_cache_shape
            else:
                key_cache_shape = value_cache_shape =  sliding_cache_shape
            new_layer_key_cache = torch.zeros(key_cache_shape, dtype=dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(value_cache_shape, dtype=dtype, device=layer_device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def _sliding_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        if cache_position.shape[0] > max_cache_len:
            k_out = key_states[:, :, -max_cache_len:, :]
            v_out = value_states[:, :, -max_cache_len:, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(max_cache_len, dtype=torch.long, device=value_states.device).cumsum(0)
        cache_position = cache_position.clamp(0, max_cache_len - 1)
        to_shift = cache_position >= max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % max_cache_len
        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out 
        self.value_cache[layer_idx] += v_out  
        return k_out, v_out

    def _static_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out
        return k_out, v_out
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        if layer_idx == self.global_attn_idx:
            update_fn = self._static_update
        elif layer_idx % 2 == 1:
            update_fn = self._sliding_update

        return update_fn(
            cache_position,
            layer_idx,
            key_states,
            value_states,
            k_out,
            v_out,
            k_out.shape[2],
        )
    
    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[self.global_attn_idx][0, 0].any(dim=-1)).sum()

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def batch_size(self):
        logger.warning_once(
            f"The 'batch_size' attribute of {self.__class__.__name__} is deprecated and will be removed in "
            "v4.49. Use the more precisely named 'self.max_batch_size' attribute instead."
        )
        return self.max_batch_size
    
swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""

swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)

class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)
    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)
swiglu = SwiGLUFunction.apply


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Samba
class _SambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        SambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

PHI_NORM_CLASS = nn.LayerNorm #SambaRMSNorm if SambaFlashRMSNorm is None else SambaFlashRMSNorm


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class _SambaMLP(nn.Module):
    """Gated Linear Unit.
    Reference:
        Language Modeling with Gated Convolutional Networks.
        https://arxiv.org/pdf/1612.08083v3.pdf.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        y = self.fc1(hidden_states)
        # Special case for SwiGLU
        if self.config.hidden_act == "silu" and swiglu is not None:
            gate, y = y.chunk(2, dim=-1)
            y = swiglu(gate, y)
        else:
            gate, y = y.chunk(2, dim=-1)
            y = y * self.activation_fn(gate)
        return self.fc2(y)


class MambaInnerFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, mask=None, delta_softplus=True, checkpoint_lvl=1,):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        if mask is not None:
            conv1d_out = conv1d_out * mask.unsqueeze(1)
        # print(mask[0,:])
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None, None)



def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, mask=None, delta_softplus=True
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, mask, delta_softplus)
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def split_heads(x):
    # split by num_heads, the stripe pattern is friendly to tensor parallel.
    x = rearrange(x, "... (H two) D -> ... H two D", two=2)
    x1 = x[..., 0, :]
    x2 = x[..., 1, :]
    return x1, x2


class _FlashDiffCustomAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        head_dim: The dimension of the heads.
        depth: The layer id, starting from 0.
    """
    def __init__(
        self,
        head_dim,
        depth,
        fa_og = True,
    ):
        super().__init__()
        assert flash_attn_varlen_func is not None, "FlashAttention is not installed"
        assert flash_attn_func is not None, "FlashAttention is not installed"
        self.head_dim = head_dim
        self.fa_og = fa_og # turning it to false needs customized flash attention https://github.com/xiayuqing0622/flex_head_fa
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = _SambaRMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(
        self,
        q,
        k,
        v,
        dropout_p = 0.0,
        cu_seqlens_q=None,
        max_seqlen_q=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
        softmax_scale=None,
        window_size=(-1, -1),
        causal=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensors containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then each has shape (B, S, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then each has shape
                (total, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and k.is_cuda and v.is_cuda
        #causal = self.causal if causal is None else causal
        unpadded = cu_seqlens_q is not None
        #print("qkv",q,k,v )
        q1, q2 = split_heads(q)
        k1, k2 = split_heads(k)
        if self.fa_og:
            v1, v2 = split_heads(v)
        else:
            v = rearrange(v, "... (H two) D -> ... H (two D)", two=2)
        kwargs = {
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
        }

        if unpadded:
            assert cu_seqlens_q.dtype == torch.int32
            assert max_seqlen_q is not None
            assert isinstance(max_seqlen_q, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen_k, int)

            kwargs.update({
                "cu_seqlens_q": cu_seqlens_q,
                "max_seqlen_q": max_seqlen_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_k": max_seqlen_k,
            })
            attn_func = flash_attn_varlen_func
        else:
            attn_func = flash_attn_func

        if self.fa_og:
            attn11 = attn_func(q1, k1, v1, **kwargs)
            attn12 = attn_func(q1, k1, v2, **kwargs)
            attn1 = torch.cat([attn11, attn12], dim=-1)
            attn21 = attn_func(q2, k2, v1, **kwargs)
            attn22 = attn_func(q2, k2, v2, **kwargs)
            attn2 = torch.cat([attn21, attn22], dim=-1)
        else:
            attn1 = attn_func(q1, k1, v, **kwargs)
            attn2 = attn_func(q2, k2, v, **kwargs)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        # reshape back to 2 * num_head
        attn = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)
        return attn
      
class _SambaFlashAttention2(nn.Module):
    """
    Samba flash attention module. This module inherits from `SambaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, config, layer_idx: Optional[int] = None, yoco_cross: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.yoco_cross = yoco_cross
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        if yoco_cross:
            self.Wqkv =  nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=True)
            
        self.inner_cross_attn = _FlashDiffCustomAttention(self.head_dim, self.layer_idx,)       
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        yoco_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # SambaFlashAttention2 attention does not support output_attentions
        output_attentions = False
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()
      
        if self.yoco_cross:
            q = self.Wqkv(hidden_states)
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim).transpose(1,2)
            key_states, value_states  = yoco_key_values   
            query_states = q
            use_sliding_windows = False
        else:
            qkv = self.Wqkv(hidden_states)
            query_pos = self.num_heads * self.head_dim
            query_states = qkv[..., :query_pos]
            key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
            value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

            # Flash attention requires the input to have the shape
            # batch_size x seq_length x head_dim x hidden_dim
            # therefore we just need to keep the original shape
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            use_sliding_windows = getattr(self.config, "sliding_window", None) is not None

            #print(self.layer_idx, kv_seq_len)
            if past_key_value is not None:

                cache_kwargs = {"cache_position": cache_position}# Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        yoco_key_values = key_states, value_states

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.Wqkv.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        # -> b,q,h,d
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if attention_mask is not None:
            key_states = key_states[:, :attention_mask.shape[-1]]
            value_states = value_states[:, :attention_mask.shape[-1]]
        #print("q_pre_attn,", query_states, query_states.max(), query_states.mean())
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        #print("attn_o,",attn_output)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        #print("post_attn:", attn_output, attn_output.max(), attn_output.mean())
        return attn_output, attn_weights, yoco_key_values

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        causal = self.is_causal
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = self.inner_cross_attn(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = self.inner_cross_attn(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window -1,
                        self.config.sliding_window -1,
                    ),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = self.inner_cross_attn(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = self.inner_cross_attn(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window -1,
                        self.config.sliding_window -1,
                    ),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, -1, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )



class _Phi3Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        yoco_cross=False,
        yoco_kv=False,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        if self.yoco_cross:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:   
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.activation = "silu"
            self.act = nn.SiLU()

            self.x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # S4D real initialization
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_log = nn.Parameter(A_log)

            # D "skip" parameter
            self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32

            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, mask= None, yoco_key_values = None, cache_position = None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        if self.yoco_cross:
            out = self.in_proj(hidden_states)
            out = swiglu(out, yoco_key_values)
            out = self.out_proj(out)
            return out, yoco_key_values 
        
        batch, seqlen, _ = hidden_states.shape
        # print(hidden_states.shape,mask)
        # print(inference_params)
        conv_state, ssm_state = None, None
      
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params)
            if cache_position[0] > 0: #inference_params.get_seq_length(self.layer_idx) > 0:
                # The states are updated inplace
                out, _, _, yoco_key_values = self.step(hidden_states, conv_state, ssm_state, yoco_key_values)
                return out, yoco_key_values

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states.to(dtype = self.in_proj.weight.dtype), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # if not self.training:
        #     xz = xz.to(torch.float32)
        #print("xz,",xz,xz.max())
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (not self.yoco_kv) and self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                mask=mask,
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            if self.yoco_kv:
                z = z.transpose(-1,-2).contiguous()
            # #print(x.shape,mask.shape)
            if mask is not None:
                x = x * mask.unsqueeze(1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            if mask is not None:
                x = x * mask.unsqueeze(1)
            # print(mask[0,:])
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z= None if self.yoco_kv else z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            if self.yoco_kv:
                yoco_key_values = y
                y = swiglu(z, y)
            out = self.out_proj(y)
        #print("m_out,",out,out.max())
        return out, yoco_key_values

    def step(self, hidden_states, conv_state, ssm_state, yoco_key_values):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.to(dtype = self.in_proj.weight.dtype).squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z= None if self.yoco_kv else z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )
        if self.yoco_kv:
            yoco_key_values = y.unsqueeze(1)
            y = swiglu(z, y)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state, yoco_key_values

    def _get_states_from_cache(self, inference_params):
        conv_state, ssm_state = inference_params.key_cache[self.layer_idx], inference_params.value_cache[self.layer_idx]           
        return conv_state, ssm_state


class _SambaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config= config   

        self.mlp = _SambaMLP(config)
        self.input_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
        
        self.yoco_kv = False
        self.yoco_cross = False
        self.yoco_mb = False
        assert config.num_hidden_layers % 4 == 0, 'n_layer should be divisible by 4 for samba + yoco'
      
        if layer_idx >= config.num_hidden_layers // 2:
            self.yoco_mb = True
            self.yoco_kv = (layer_idx >= (config.num_hidden_layers//2 +1))
            self.yoco_cross = (layer_idx >= (config.num_hidden_layers//2 +2))
            if (layer_idx >= (config.num_hidden_layers//2 +1)):
                config = copy.deepcopy(config)
                config.sliding_window = None     
          
        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            factory_kwargs = {"d_conv": config.mamba_d_conv, "d_state": config.mamba_d_state, "expand": config.mamba_expand , "dtype": None}
            self.attn = _Phi3Mamba(config.hidden_size, layer_idx=layer_idx, yoco_cross=self.yoco_cross, yoco_kv=self.yoco_mb, **factory_kwargs)
        else:
            #self.input_layernorm2 = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
            self.attn = _SambaFlashAttention2(config, layer_idx=layer_idx, yoco_cross=self.yoco_cross)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        ssm_output: Optional[torch.Tensor] = None,
        yoco_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))

        #print("post_ln:",hidden_states, hidden_states.max(),hidden_states.mean(),hidden_states.std())
        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(
                hidden_states, inference_params=past_key_value, 
                mask = attention_mask, yoco_key_values = ssm_output,
                cache_position=cache_position,
            )
            residual = residual.to(torch.float32) 
            self_attn_weights = None
        else:
            if self.config.sliding_window is not None and attention_mask is not None:  # efficient SDPA and no padding
                if past_key_value is not None and cache_position[0] > 0:  # when decoding
                    attention_mask = attention_mask[:, -self.config.sliding_window:]
            #hidden_states = self.input_layernorm2(hidden_states.to(dtype=self.input_layernorm2.weight.dtype))
            # Self Attention
            attn_outputs, self_attn_weights, yoco_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                yoco_key_values = yoco_key_values,
            )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)
        outputs += (ssm_output,)    
        outputs += (yoco_key_values,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs

class _SambaPreTrainedModel(nn.Module):
    def __init__(self, config): # SambaConfig
        super().__init__()
        self.base_model_prefix = "model"
        self.supports_gradient_checkpointing = True
        self._no_split_modules = ["SambaDecoderLayer"]
        self._skip_keys_device_placement = "past_key_values"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = False
        self._supports_cache_class = True
        self._version = "0.1.0"
        self.config = config
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class _SambaModel(_SambaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SambaDecoderLayer`]
    Args:
        config: SambaConfig
    """
    def __init__(self, config): # : SambaConfig
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [_SambaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = PHI_NORM_CLASS(config.hidden_size, eps=config.layer_norm_eps)
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ): 
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # print(use_cache)
        # # use_cache = False
        # print(input_ids)
        # print(attention_mask)
        # print(position_ids)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = SambaYCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                device=self.device,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache and not self.training:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Samba. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        ssm_output = None
        yoco_key_values = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            #print("layer_input,", hidden_states, hidden_states.max(), hidden_states.mean())
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    ssm_output,
                    yoco_key_values,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position = cache_position,
                    ssm_output = ssm_output,
                    yoco_key_values = yoco_key_values,
                )

            hidden_states = layer_outputs[0]
            ssm_output = layer_outputs[1]
            yoco_key_values = layer_outputs[2] 

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        hidden_states = self.final_layernorm(hidden_states.to(dtype=self.final_layernorm.weight.dtype))

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class _SambaForCausalLM(_SambaPreTrainedModel):

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Samba,bias=False->bias=True
    def __init__(self, config):
        super().__init__(config)
        self.config = config
          
        self.model = _SambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        #self.post_init()
        self._tied_weights_keys = ["lm_head.weight"]

        # model archiecture: connect from intermedia layer possibily.
        self.critic_head = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.xavier_normal_(self.critic_head.weight)
        nn.init.constant_(self.critic_head.bias, -8.0)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)


    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, SambaForCausalLM
        >>> model = SambaForCausalLM.from_pretrained("microsoft/phi-1")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n\n\n\nfrom typing import List\n\ndef find_most_common_letter(words: List[str'
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position = cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

## support batched generation
import selective_scan_cuda

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)
          
    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)




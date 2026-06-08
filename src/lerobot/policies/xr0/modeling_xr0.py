#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""XR0 VLA policy: Qwen3-VL backbone + DiT head with rectified flow matching.

Architecture (based on Xiaomi Robotics XR0):
1. Qwen3-VL encodes images + language, producing a KV-cache
2. DiT (Diffusion Transformer) cross-attends to the VLM KV-cache
3. Rectified flow matching: predict velocity v = action - noise

Training:
- VLM is frozen (optional), only DiT + projectors are trained
- Flow matching: sample t ~ Beta(1.5, 1.0), interpolate z_t = (1-t)*noise + t*action
- Predict velocity v_t, MSE loss

Inference:
- Run VLM once to get KV-cache
- Denoise via Euler integration (default 5 steps)

ACP support:
- ACP (Advantage-Conditioned Prompting) works automatically via the ACPPromptHook
- The hook prepends "Advantage: positive" / "Advantage: negative" tags to the task text
- These tags flow through the Qwen3-VL processor into the VLM, providing guidance signals
"""

import math
import logging
from collections import deque
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta, LogisticNormal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.xr0.configuration_xr0 import XR0Config
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


# ============================================================
# Helper functions (ported from XR0.py)
# ============================================================


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads for GQA: (B, n_kv_heads, S, D) → (B, n_q_heads, S, D)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return hidden_states.repeat_interleave(n_rep, dim=1)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to new_dim with zeros."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# ============================================================
# DiT components (ported from XR0.py)
# ============================================================


class MLPProjector(nn.Module):
    """Multi-layer perceptron projector with optional GELU activation."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, output_dim, bias=bias)]
        for _ in range(1, num_layers):
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(output_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t: Tensor) -> Tensor:
        """Embed timestep t, return (B, 1, hidden_size)."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb[:, None]


class DiTAttention(nn.Module):
    """Multi-head attention with GQA, QK-RMSNorm, and VLM KV-cache for DiT."""

    def __init__(self, hidden_size: int = 768, head_dim: int = 128, kv_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_state: Tensor,
        past_key_values: Tuple[Tensor, Tensor],
        position_embeds: Tuple[Tensor, Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, q_len, _ = hidden_state.size()

        qkv = self.qkv_proj(hidden_state)
        qkv = qkv.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeds
        if cos.ndim == 4:
            cos = cos[0]
            sin = sin[0]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Prepend cached KV from VLM
        k_cache, v_cache = past_key_values
        k_cache = repeat_kv(k_cache, self.kv_group)
        v_cache = repeat_kv(v_cache, self.kv_group)

        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DiTMLP(nn.Module):
    """SwiGLU MLP used in DiT decoder layers."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DecoderLayer(nn.Module):
    """DiT decoder layer with AdaLN modulation conditioned on diffusion timestep."""

    def __init__(self, hidden_size: int = 768, head_dim: int = 128, kv_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = DiTAttention(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads)
        self.mlp = DiTMLP(hidden_size=hidden_size)

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-06)
        self.middle_layernorm = nn.RMSNorm(hidden_size, eps=1e-06)
        self.post_layernorm = nn.RMSNorm(hidden_size, eps=1e-06)
        self.final_layernorm = nn.RMSNorm(hidden_size, eps=1e-06)

        # AdaLN: 6 modulation params per layer (shift/scale/gate × attn + ffn)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values: Tuple[Tensor, Tensor],
        position_embeds: Tuple[Tensor, Tensor],
        t_embeds: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + t_embeds
        ).chunk(6, dim=1)

        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        hidden_states = self.attn(hidden_states, past_key_values, position_embeds, attn_mask=attn_mask)
        hidden_states = residual + gate_msa * hidden_states
        hidden_states = self.middle_layernorm(hidden_states)

        # FFN block
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class DiT(nn.Module):
    """Diffusion Transformer that cross-attends to VLM KV-cache with AdaLN timestep conditioning.

    DiT layers align with the *tail* of the VLM's KV-cache so deeper DiT layers
    attend to deeper VLM layers.
    """

    def __init__(self, hidden_size: int = 768, layer_num: int = 16, head_dim: int = 128, kv_heads: int = 8):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads) for _ in range(layer_num)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values: list,
        attn_mask: Tensor,
        position_embeds: Tuple[Tensor, Tensor],
        t_embeds: Tensor,
    ) -> Tensor:
        # Align DiT layers with tail of VLM KV-cache
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start_idx + i],
                position_embeds,
                t_embeds,
                attn_mask=attn_mask,
            )
        return hidden_states


# ============================================================
# XR0 Model (VLM + DiT + Flow Matching)
# ============================================================


class XR0Model(nn.Module):
    """Vision-Language-Action model: Qwen3-VL encodes vision+language, DiT decodes actions via rectified flow.

    Args:
        config: XR0Config with all hyperparameters.
    """

    def __init__(self, config: XR0Config):
        super().__init__()
        self.config = config

        # Determine dtype
        self._dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

        # Action shape: (chunk_size, max_action_dim)
        self.action_shape = (config.chunk_size, config.max_action_dim)
        # State shape: (1, max_state_dim)
        self.state_shape = (1, config.max_state_dim)

        # Flow sampling distributions
        self.logistic_normal = LogisticNormal(0.0, 1.0)
        self.beta = Beta(config.time_sampling_beta_alpha, config.time_sampling_beta_beta)

        self._build_model()

    def _build_model(self) -> None:
        """Instantiate all sub-modules."""
        config = self.config

        # ---- Qwen3-VL processor (for image + text processing) ----
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(config.qwen_variant)
            logger.info(f"Loaded Qwen3-VL processor from {config.qwen_variant}")
        except Exception as e:
            logger.warning(f"Failed to load Qwen3-VL processor: {e}. Will require pre-processed VLM inputs.")
            self.processor = None

        # ---- VLM backbone (Qwen3-VL) ----
        try:
            from transformers import Qwen3VLForConditionalGeneration

            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                config.qwen_variant,
                attn_implementation="sdpa",
                torch_dtype=self._dtype,
            )
        except ImportError:
            # Fallback: use AutoModel
            from transformers import AutoModelForVision2Seq

            self.vlm = AutoModelForVision2Seq.from_pretrained(
                config.qwen_variant,
                attn_implementation="sdpa",
                torch_dtype=self._dtype,
                trust_remote_code=True,
            )

        # Freeze VLM if requested
        if config.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
            # Enable gradient checkpointing for vision encoder (memory efficient)
            if hasattr(self.vlm, "visual") and hasattr(self.vlm.visual, "gradient_checkpointing_enable"):
                self.vlm.visual.gradient_checkpointing_enable()

        # ---- DiT head ----
        self.dit = DiT(
            hidden_size=config.dit_hidden_size,
            layer_num=config.dit_num_layers,
            kv_heads=8,
        )

        # ---- Projectors ----
        self.state_projector = MLPProjector(
            input_dim=config.max_state_dim, output_dim=config.dit_hidden_size, num_layers=2
        )
        self.action_projector = MLPProjector(
            input_dim=config.max_action_dim, output_dim=config.dit_hidden_size, num_layers=2
        )
        self.action_output_layer = MLPProjector(
            input_dim=config.dit_hidden_size, output_dim=config.max_action_dim, num_layers=2
        )

        # ---- Timestep embedding ----
        self.t_embedder = TimestepEmbedder(config.dit_hidden_size, dtype=self._dtype)
        self.t_projector = MLPProjector(
            input_dim=config.dit_hidden_size, output_dim=6 * config.dit_hidden_size, bias=True
        )

        # ---- RoPE for DiT ----
        self.rotary_emb = None
        try:
            from transformers import Qwen3VLTextConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

            text_config = Qwen3VLTextConfig.from_pretrained(config.qwen_variant)
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config)
            logger.info("Loaded Qwen3-VL MRoPE for DiT")
        except Exception as e:
            logger.warning(
                f"Could not load Qwen3-VL MRoPE: {e}. "
                f"Falling back to simple sinusoidal RoPE (may affect performance)."
            )

        # ---- Sink token (prepended to DiT input) ----
        self.sink = nn.Embedding(1, config.dit_hidden_size)

        # ---- Local causal mask (sink + state + action) ----
        self._build_causal_mask()

        # ---- Gradient checkpointing for DiT (memory efficient) ----
        if config.gradient_checkpointing:
            self.dit.gradient_checkpointing_enable = True
            # Wrap each decoder layer with checkpoint
            for layer in self.dit.layers:
                layer.gradient_checkpointing = True

        # Cast to dtype
        self.to(self._dtype)

    def _build_causal_mask(self) -> None:
        """Build local causal mask for DiT tokens [sink, state, action]."""
        local_window = 4
        s_len = self.state_shape[0] + 1  # +1 for sink token
        a_len = self.action_shape[0]

        mask_ss = torch.tril(torch.ones(s_len, s_len))
        mask_sa = torch.zeros(s_len, a_len)
        mask_as = torch.ones(a_len, s_len)
        mask_aa = torch.tril(torch.ones(a_len, a_len))
        mask_aa = mask_aa * torch.triu(torch.ones(a_len, a_len), diagonal=-local_window)

        top = torch.cat([mask_ss, mask_sa], dim=1)
        bottom = torch.cat([mask_as, mask_aa], dim=1)
        full_mask = torch.cat([top, bottom], dim=0)
        self.register_buffer("saved_causal_mask", full_mask.unsqueeze(0).int(), persistent=False)

    # --------------------------------------------------------
    # Rectified flow methods
    # --------------------------------------------------------

    @torch.no_grad()
    def _sample_timestep(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample random timesteps for rectified flow training."""
        config = self.config
        if config.flow_sampling == "logit_normal":
            u = self.logistic_normal.sample((batch_size,))[:, 0].to(device)
        elif config.flow_sampling == "beta":
            u = self.beta.sample((batch_size,)).to(device)
            u = (1 - u) * 0.999
        else:
            u = torch.rand(size=(batch_size,), device=device)
        return u.to(self._dtype)

    @torch.no_grad()
    def _flow_interpolate(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Linear interpolation: z_t = (1-t)*x0 + t*x1."""
        return (1 - t) * x0 + t * x1

    @torch.no_grad()
    def _flow_velocity_target(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Velocity target: v = x1 - x0."""
        return x1 - x0

    @torch.no_grad()
    def _flow_generate(
        self,
        z: Tensor,
        state_embed: Tensor,
        action_mask: Tensor,
        position_embeds: Tuple[Tensor, Tensor],
        past_key_values: list,
        attn_mask: Tensor,
    ) -> Tensor:
        """Euler integration: denoise from noise to action."""
        config = self.config
        dt = 1.0 / config.num_inference_steps
        for step in range(config.num_inference_steps):
            t_val = step / config.num_inference_steps
            t = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * t_val
            v = self._dit_forward(z, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask)
            z = z + v * dt
        return z

    # --------------------------------------------------------
    # DiT forward
    # --------------------------------------------------------

    def _dit_forward(
        self,
        noisy_action: Tensor,
        t: Tensor,
        action_mask: Tensor,
        state_embed: Tensor,
        position_embeds: Tuple[Tensor, Tensor],
        past_key_values: list,
        attn_mask: Tensor,
    ) -> Tensor:
        """Single forward pass of DiT.

        Returns predicted velocity (training) or action update (inference).
        Shape: (B, action_len, action_dim).
        """
        # Embed timestep → 6 AdaLN modulation params per layer
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        # Project noisy action to DiT hidden dim
        noisy_action = noisy_action * action_mask
        noisy_action_h = self.action_projector(noisy_action)

        # Concatenate: [sink, state, noisy_action]
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action_h], dim=1).contiguous()

        # DiT forward
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        # Extract action tokens → project to action dim
        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        output = self.action_output_layer(hidden_states)
        return output

    def _make_local_causal_mask(
        self,
        batch_size: int,
        state_length: int,
        action_length: int,
        device: torch.device,
    ) -> Tensor:
        expected_s_len = self.state_shape[0]
        expected_a_len = self.action_shape[0]
        if state_length == expected_s_len and action_length == expected_a_len:
            return self.saved_causal_mask.expand(batch_size, -1, -1)

        s_len = state_length + 1  # +1 for sink
        a_len = action_length
        local_window = 4
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        mask_aa = mask_aa * torch.triu(torch.ones(a_len, a_len, device=device), diagonal=-local_window)

        top = torch.cat([mask_ss, mask_sa], dim=1)
        bottom = torch.cat([mask_as, mask_aa], dim=1)
        return torch.cat([top, bottom], dim=0).unsqueeze(0).expand(batch_size, -1, -1)

    # --------------------------------------------------------
    # Main forward / inference
    # --------------------------------------------------------

    def get_rotary_embeddings(self, action: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Get (cos, sin) rotary embeddings for DiT.

        Uses Qwen3-VL MRoPE when available, falls back to simple sinusoidal RoPE.
        MRoPE uses 3D position IDs (temporal, height, width) vs standard 1D.
        """
        if self.rotary_emb is not None:
            try:
                return self.rotary_emb(action, position_ids)
            except Exception as e:
                logger.warning(f"MRoPE failed, falling back to sinusoidal RoPE: {e}")

        # Sinusoidal RoPE fallback
        device = action.device
        head_dim = 128  # Default head_dim used by DiT

        # Position indices from position_ids
        if position_ids.ndim == 3:
            pos = position_ids[0].float()  # MRoPE: use temporal dimension
        else:
            pos = position_ids.float()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        sinusoid_inp = torch.einsum("bs,d->bsd", pos, inv_freq)
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        return cos, sin

    def _get_image_keys(self) -> list[str]:
        """Get sorted list of image feature keys from config."""
        return sorted(self.config.image_features.keys())

    def _extract_state(self, batch: dict, action_device: torch.device) -> Tensor:
        """Extract and pad state from batch, returning (B, 1, max_state_dim)."""
        state = None
        for key in ("observation.state", OBS_STATE, "state"):
            if key in batch:
                state = batch[key]
                break
        if state is None:
            state = torch.zeros(1, 1, self.config.max_state_dim, device=action_device, dtype=self._dtype)
        else:
            state = pad_vector(state, self.config.max_state_dim)

        if state.ndim == 2:
            state = state.unsqueeze(1)  # (B, D) → (B, 1, D)
        return state

    def _run_vlm(self, vlm_kwargs: dict, grad_enabled: bool = False) -> tuple:
        """Run VLM forward pass and extract past_key_values and position_ids.

        Handles different return types across transformers versions.

        Returns:
            Tuple of (past_key_values_as_list, position_ids_tensor).
        """
        with torch.set_grad_enabled(grad_enabled):
            vlm_outputs = self.vlm(**vlm_kwargs, use_cache=True)

        # past_key_values can be DynamicCache (newer transformers) or tuple of tuples (older)
        if hasattr(vlm_outputs.past_key_values, "to_legacy_cache"):
            # HuggingFace DynamicCache (transformers >= 4.53)
            past_key_values = list(vlm_outputs.past_key_values)
        elif isinstance(vlm_outputs.past_key_values, (list, tuple)):
            past_key_values = list(vlm_outputs.past_key_values)
        else:
            past_key_values = list(vlm_outputs.past_key_values)

        # position_ids: Qwen3-VL uses MRoPE with 3D position ids shape (3, B, L)
        position_ids = getattr(vlm_outputs, "position_ids", None)
        if position_ids is None:
            # For models that don't return position_ids, create default (1, B, L)
            seq_len = vlm_kwargs["input_ids"].shape[1]
            bs = vlm_kwargs["input_ids"].shape[0]
            position_ids = torch.arange(0, seq_len, device=vlm_kwargs["input_ids"].device).unsqueeze(0).expand(bs, -1)

        return past_key_values, position_ids

    def _prepare_vlm_inputs(self, batch: dict) -> dict:
        """Build Qwen3-VL inputs from raw images and task text in the batch.

        This is called during both training and inference. It:
        1. Extracts raw images from batch (using self.config.image_features keys)
        2. Extracts task text from batch["task"]
        3. Converts images to numpy format for Qwen3-VL processor
        4. Builds chat messages and applies Qwen3-VL processor
        5. Returns a dict with input_ids, attention_mask, pixel_values, image_grid_thw

        Args:
            batch: Flat batch dict from LeRobot. Contains image tensors at keys
                   like "observation.images.camera1" and task string at "task".

        Returns:
            Dict of tensors suitable for self.vlm(**kwargs).
        """
        # If batch already has VLM inputs (e.g., from a custom data pipeline), use directly
        if "input_ids" in batch and "pixel_values" in batch:
            device = next(self.parameters()).device
            return {
                k: batch[k].to(device) if isinstance(batch[k], Tensor) else batch[k]
                for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
                if k in batch
            }

        # We need to do Qwen3-VL processing ourselves
        if self.processor is None:
            raise RuntimeError(
                "Qwen3-VL processor not loaded. "
                "Batch must already contain VLM inputs (input_ids, pixel_values, etc.), "
                "or the processor must be available at model init time."
            )

        device = next(self.parameters()).device

        # Get image keys from config
        image_keys = self._get_image_keys()
        if not image_keys:
            raise ValueError(
                "No image features configured. XR0 requires at least one image input."
            )

        # Determine batch size from first image
        first_img = batch[image_keys[0]]
        if first_img.ndim == 4:
            batch_size = first_img.shape[0]
        else:
            batch_size = 1

        # Get task texts (ACP-compatible: tags from ACPPromptHook flow through here)
        tasks = batch.get("task", None)
        if tasks is None:
            tasks = ["Execute the robot action."] * batch_size
        elif isinstance(tasks, str):
            tasks = [tasks]
        elif isinstance(tasks, (list, tuple)) and len(tasks) < batch_size:
            tasks = list(tasks) + [tasks[-1]] * (batch_size - len(tasks))

        # Build messages for each sample in the batch
        messages_list = []
        for i in range(batch_size):
            content = []
            # Add images (multiple cameras supported naturally by Qwen3-VL)
            for img_key in image_keys:
                img_tensor = batch[img_key]
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[i]  # (C, H, W) from batch
                img_np = self._tensor_to_image(img_tensor)
                content.append({"type": "image", "image": img_np})
            # Add text (may include ACP tags: "Advantage: positive\nTask: ...")
            task = tasks[i] if i < len(tasks) else tasks[0]
            content.append({"type": "text", "text": task})
            messages_list.append([{"role": "user", "content": content}])

        # Apply Qwen3-VL processor
        try:
            # Apply chat template to get formatted prompts
            texts = self.processor.apply_chat_template(
                messages_list, tokenize=False, add_generation_prompt=True
            )

            # Collect all images from all messages (in order)
            image_inputs = []
            for messages in messages_list:
                for msg_content in messages[0]["content"]:
                    if msg_content["type"] == "image":
                        image_inputs.append(msg_content["image"])

            # Run processor with all images
            vlm_inputs = self.processor(
                text=texts if isinstance(texts, list) else [texts],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning(
                f"Qwen3-VL processor failed with images: {e}. "
                f"Trying text-only fallback (may reduce performance)."
            )
            # Text-only fallback
            texts = []
            for i in range(batch_size):
                task = tasks[i] if i < len(tasks) else tasks[0]
                texts.append(f"Task: {task}")
            vlm_inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            )

        # Move to correct device
        vlm_inputs = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in vlm_inputs.items()}

        return vlm_inputs

    def _tensor_to_image(self, img_tensor: Tensor) -> np.ndarray:
        """Convert a LeRobot image tensor to numpy format for Qwen3-VL processor.

        Input: (C, H, W) or (H, W, C) tensor in [0, 1] range (float32).
        Output: (H, W, C) numpy uint8 array in [0, 255].
        """
        img = img_tensor.detach().cpu().float().numpy()

        # Detect format: if first dim is 3 (channels-first), transpose to channels-last
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # (C,H,W) → (H,W,C)
        elif len(img.shape) == 4 and img.shape[1] == 3:
            # (1, C, H, W) → (H, W, C)
            img = img[0]
            img = np.transpose(img, (1, 2, 0))

        # Clamp to [0, 1] and convert to uint8
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

        return img

    def forward(self, batch: dict) -> Tensor:
        """Training forward: compute per-element MSE losses.

        Args:
            batch: Dict with raw images + task + action + state.

        Returns:
            losses: (B, chunk_size, max_action_dim) per-element MSE losses.
        """
        # Prepare VLM inputs from raw images + text
        vlm_kwargs = self._prepare_vlm_inputs(batch)

        # Extract and pad action
        action = batch[ACTION].to(self._dtype)
        action = pad_vector(action, self.config.max_action_dim)
        action_bs, action_length, _ = action.shape

        # Extract and pad state
        state = self._extract_state(batch, action.device)
        state_length = state.shape[1]

        # Action mask
        action_mask = torch.ones_like(action)

        # ---- VLM forward ----
        past_key_values, vlm_pos_ids = self._run_vlm(
            vlm_kwargs, grad_enabled=not self.config.freeze_vlm
        )

        # ---- Build attention mask & position embeddings ----
        attn_mask, position_embeds = self._build_attention_and_position_ids(
            vlm_kwargs, vlm_pos_ids,
            action_bs, state_length, action_length, action.device,
        )

        # ---- Project state ----
        state_embed = self.state_projector(state.to(self._dtype))

        # ---- Rectified flow training ----
        noise = torch.randn_like(action)
        t = self._sample_timestep(action_bs, device=action.device)
        t = t.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)

        noisy_action = self._flow_interpolate(noise, action, t)
        target = self._flow_velocity_target(noise, action)

        pred = self._dit_forward(
            noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask
        )

        # Per-element MSE loss (no reduction)
        losses = F.mse_loss(pred, target, reduction="none")
        return losses

    def _build_attention_and_position_ids(
        self,
        vlm_kwargs: dict,
        vlm_pos_ids: Tensor,
        batch_size: int,
        state_length: int,
        action_length: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Build attention mask and rotary position embeddings for DiT forward.

        Returns:
            (attn_mask, position_embeds) tuple.
        """
        q_len = action_length + state_length + 1  # +1 for sink token

        # ---- Extend position IDs from VLM's last position ----
        if vlm_pos_ids.ndim == 3:
            # MRoPE: (3, B, L_vlm)
            vlm_pos_max = vlm_pos_ids.max(dim=-1).values[..., None]  # (3, B, 1)
            extended_pos_ids = (
                torch.arange(0, q_len, device=device).view(1, 1, -1).repeat(3, batch_size, 1)
                + vlm_pos_max + 1
            )
        else:
            # Standard 1D: (B, L_vlm)
            vlm_pos_max = vlm_pos_ids.max(dim=-1).values[..., None]  # (B, 1)
            extended_pos_ids = (
                torch.arange(0, q_len, device=device).unsqueeze(0).expand(batch_size, -1)
                + vlm_pos_max + 1
            )

        # ---- Attention mask: [VLM padding mask | DiT local causal mask] ----
        attn_vlm = vlm_kwargs.get("attention_mask", torch.ones(batch_size, 1, device=device))
        cache_mask = attn_vlm[:, None, :].expand(-1, q_len, -1)
        causal_mask = self._make_local_causal_mask(batch_size, state_length, action_length, device)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        # ---- RoPE embeddings ----
        dummy_action = torch.zeros(
            batch_size, action_length, self.config.max_action_dim, device=device, dtype=self._dtype
        )
        position_embeds = self.get_rotary_embeddings(dummy_action, extended_pos_ids)

        return attn_mask, position_embeds

    def predict_action_chunk(self, batch: dict) -> Tensor:
        """Inference: predict action chunk via flow matching denoising.

        Args:
            batch: Dict with raw images + task + state.

        Returns:
            action_chunk: (B, chunk_size, max_action_dim).
        """
        device = next(self.parameters()).device

        # Prepare VLM inputs from raw images + text
        vlm_kwargs = self._prepare_vlm_inputs(batch)

        # Extract state
        state = self._extract_state(batch, device)
        bs = state.shape[0]
        state_length = state.shape[1]
        action_length = self.config.chunk_size

        # Action mask
        action_mask = torch.ones(bs, action_length, self.config.max_action_dim, device=device, dtype=self._dtype)

        # ---- VLM forward (once, with KV cache) ----
        past_key_values, vlm_pos_ids = self._run_vlm(vlm_kwargs, grad_enabled=False)

        # ---- Build attention mask & position embeddings ----
        attn_mask, position_embeds = self._build_attention_and_position_ids(
            vlm_kwargs, vlm_pos_ids,
            bs, state_length, action_length, device,
        )

        # ---- Project state ----
        state_embed = self.state_projector(state.to(self._dtype))

        # ---- Denoise via Euler integration ----
        z = torch.randn(bs, action_length, self.config.max_action_dim, device=device, dtype=self._dtype)
        result = self._flow_generate(z, state_embed, action_mask, position_embeds, past_key_values, attn_mask)
        return result


# ============================================================
# LeRobot PreTrainedPolicy wrapper
# ============================================================


class XR0Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for XR0: Qwen3-VL + DiT + Rectified Flow.

    Integrates the XR0 VLA architecture into the LeRobot ecosystem,
    supporting training, ACP training, and evaluation.
    """

    config_class = XR0Config
    name = "xr0"

    def __init__(self, config: XR0Config, **kwargs):
        super().__init__(config)
        config.validate_features()

        self.model = XR0Model(config)
        self.reset()

        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset action queue (called when environment resets)."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def prepare_action(self, batch: dict) -> Tensor:
        """Pad action to max_action_dim."""
        actions = batch[ACTION]
        return pad_vector(actions, self.config.max_action_dim)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training forward: compute flow matching loss.

        Args:
            batch: Training batch with VLM inputs, action, state.
            reduction: "mean" for scalar loss, "none" for per-sample losses.

        Returns:
            (loss, loss_dict) tuple.
        """
        # Run the model to get per-element losses
        # (action is extracted and padded internally by XR0Model.forward)
        losses = self.model.forward(batch)

        # Truncate losses to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Uses action chunk queue: when empty, runs full inference and fills queue.
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # (B, n_action_steps, D) → transpose to fill queue as (n_action_steps, B, D)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a full chunk of actions given environment observations.

        Returns:
            action_chunk: (B, chunk_size, max_action_dim).
        """
        self.eval()

        # Run model inference (flow matching denoising)
        actions = self.model.predict_action_chunk(batch)

        # Truncate to original action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def _get_default_peft_targets(self) -> dict:
        """Default PEFT target modules for XR0 fine-tuning."""
        target_modules = (
            r"(.*\.dit\..*\.self_attn\.(q|v)_proj"
            r"|.*\.action_projector.*"
            r"|.*\.action_output_layer.*"
            r"|.*\.state_projector.*"
            r"|.*\.t_embedder.*"
            r"|.*\.t_projector.*)"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

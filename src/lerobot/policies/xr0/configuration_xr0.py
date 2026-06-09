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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("xr0")
@dataclass
class XR0Config(PreTrainedConfig):
    """Configuration for XR0 VLA policy (Qwen3-VL + DiT + Rectified Flow).

    This policy uses Qwen3-VL as the vision-language backbone and a DiT
    (Diffusion Transformer) head with rectified flow matching for action
    prediction, based on the Xiaomi Robotics XR0 architecture.
    """

    # === Qwen3-VL VLM backbone ===
    qwen_variant: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"

    # === LeRobot standard fields ===
    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # === DiT head ===
    dit_num_layers: int = 16
    dit_hidden_size: int = 1024
    training_repeat: int = 4
    enable_freq: bool = True

    # === Flow matching ===
    num_inference_steps: int = 5
    flow_sampling: str = "beta"  # "beta", "logit_normal", or "uniform"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    prefix_mask_prob: float = 0.5
    async_train: bool = False

    # === Image processing ===
    # Qwen3-VL handles image resizing internally via its processor.
    # This is the image resolution we resize to before passing to the processor.
    image_min_pixels: int = 256 * 28 * 28
    image_max_pixels: int = 1280 * 28 * 28

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # === Training settings ===
    freeze_vlm: bool = True  # Freeze VLM backbone, train only DiT + projectors
    gradient_checkpointing: bool = False

    # === Native XR0 integration ===
    # Official Xiaomi checkpoint converted by xr0/tools/weight_convert.py.
    xr0_pretrained_path: str | None = None
    # Optional stats file with action mean/std shaped (chunk_size, 32).
    xr0_stats_path: str | None = None
    action_layout: str = "aloha14"  # "aloha14" or "xr0_32"
    actions_are_delta: bool = False

    # === Optimizer settings (same as PI05) ===
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # === Scheduler settings (same as PI05) ===
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be 'bfloat16' or 'float32'.")

        if self.qwen_attn_implementation not in ["flash_attention_2", "sdpa", "eager"]:
            raise ValueError(
                "qwen_attn_implementation must be 'flash_attention_2', 'sdpa', or 'eager'."
            )

        if self.flow_sampling not in ["beta", "logit_normal", "uniform"]:
            raise ValueError(
                f"Invalid flow_sampling: {self.flow_sampling}. Must be 'beta', 'logit_normal', or 'uniform'."
            )

        if self.action_layout not in ["aloha14", "xr0_32"]:
            raise ValueError("action_layout must be 'aloha14' or 'xr0_32'.")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Convert raw dict entries to PolicyFeature objects (needed when loading from JSON)
        for key, ft in list(self.input_features.items()):
            if isinstance(ft, dict):
                self.input_features[key] = PolicyFeature(**ft)
        for key, ft in list(self.output_features.items()):
            if isinstance(ft, dict):
                self.output_features[key] = PolicyFeature(**ft)

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LeRobot wrapper for native Xiaomi XR0.

The native XR0 architecture lives in ``lerobot.policies.xr0.native`` and is kept
as close as possible to Xiaomi's implementation so official checkpoints remain
load-compatible. This file only adapts LeRobot batches, processors and policy
APIs around that native model.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xr0.configuration_xr0 import XR0Config
from lerobot.policies.xr0.native import XR0 as NativeXR0
from lerobot.policies.xr0.processor_xr0 import _xr0_action_mask, load_xr0_action_stats
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_DESCRIPTIONS = {
    "observation.images.left_wrist": "left wrist RGB camera view",
    "observation.images.left_ego": "ego RGB camera view",
    "observation.images.right_wrist": "right wrist RGB camera view",
    "observation.images.left_tactile": (
        "left wrist tactile heatmap showing contact force distribution"
    ),
    "observation.images.right_tactile": (
        "right gripper tactile heatmap showing contact force distribution"
    ),
}

DEFAULT_IMAGE_KEY_ORDER = (
    "observation.images.left_ego",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.left_tactile",
    "observation.images.right_tactile",
)

DEFAULT_IMAGE_VIEW_HEADINGS = {
    "observation.images.left_ego": "Ego View",
    "observation.images.left_wrist": "Left-Wrist View",
    "observation.images.right_wrist": "Right-Wrist View",
    "observation.images.left_tactile": "Left-Tactile View",
    "observation.images.right_tactile": "Right-Gripper Tactile View",
}

DESCRIPTION_TO_NATIVE_VIEW_HEADING = {
    "base ego rgb camera": "Ego View",
    "base or front rgb camera view": "Ego View",
    "ego rgb camera view": "Ego View",
    "ego view": "Ego View",
    "base view": "Ego View",
    "left wrist rgb camera": "Left-Wrist View",
    "left wrist rgb camera view": "Left-Wrist View",
    "left-wrist view": "Left-Wrist View",
    "right wrist rgb camera": "Right-Wrist View",
    "right wrist rgb camera view": "Right-Wrist View",
    "right-wrist view": "Right-Wrist View",
    "left fingertip tactile heatmap": "Left-Tactile View",
    "left wrist tactile heatmap showing contact force distribution": "Left-Tactile View",
    "left-tactile view": "Left-Tactile View",
    "right fingertip tactile heatmap": "Right-Gripper Tactile View",
    "right wrist tactile heatmap showing contact force distribution": "Right-Gripper Tactile View",
    "right gripper tactile heatmap showing contact force distribution": "Right-Gripper Tactile View",
    "right-tactile view": "Right-Gripper Tactile View",
    "right-gripper tactile view": "Right-Gripper Tactile View",
}


def _remap_official_checkpoint_key(key: str) -> str:
    """Map Xiaomi's converted checkpoint keys to the native LeRobot XR0 module."""

    for prefix in ("_forward_module.model.", "module.model."):
        if key.startswith(prefix):
            key = "model." + key.removeprefix(prefix)
            break

    if key.startswith("model.model."):
        return "vlm.model." + key.removeprefix("model.model.")
    if key.startswith("model."):
        return key.removeprefix("model.")
    return key


class XR0Policy(PreTrainedPolicy):
    """LeRobot policy adapter for the official XR0 VLA model."""

    config_class = XR0Config
    name = "xr0"

    def __init__(self, config: XR0Config, **kwargs):
        super().__init__(config)
        config.validate_features()

        self.model = NativeXR0(
            state_shape=(1, config.max_state_dim),
            action_shape=(config.chunk_size, config.max_action_dim),
            dit_num_layers=config.dit_num_layers,
            dit_hidden_size=config.dit_hidden_size,
            num_steps=config.num_inference_steps,
            flow_sampling=config.flow_sampling,
            training_repeat=config.training_repeat,
            enable_freq=config.enable_freq,
            prefix_mask_prob=config.prefix_mask_prob,
            async_train=config.async_train,
            qwen_variant=config.qwen_variant,
            qwen_attn_implementation=config.qwen_attn_implementation,
            dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
        )

        if config.freeze_vlm:
            for param in self.model.vlm.parameters():
                param.requires_grad = False

        self.processor = self._load_processor(config.qwen_variant)
        self.register_buffer(
            "_xr0_action_mean",
            torch.zeros(config.chunk_size, config.max_action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_xr0_action_std",
            torch.ones(config.chunk_size, config.max_action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("_xr0_action_stats_loaded", torch.tensor(0, dtype=torch.uint8), persistent=True)
        self._maybe_load_xr0_action_stats()
        self._maybe_load_official_checkpoint()
        self.reset()

    @staticmethod
    def _load_processor(qwen_variant: str):
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(qwen_variant)
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.padding_side = "right"
            return processor
        except Exception as exc:
            logger.warning("Failed to load Qwen processor from %s: %s", qwen_variant, exc)
            return None

    def _maybe_load_xr0_action_stats(self) -> None:
        if not self.config.xr0_stats_path:
            return
        stats = load_xr0_action_stats(
            self.config.xr0_stats_path,
            horizon=self.config.chunk_size,
            dim=self.config.max_action_dim,
        )[ACTION]
        self._xr0_action_mean.copy_(stats["mean"])
        self._xr0_action_std.copy_(stats["std"])
        self._xr0_action_stats_loaded.fill_(1)
        logger.info("Loaded XR0 32D action stats from %s", self.config.xr0_stats_path)

    def _maybe_load_official_checkpoint(self) -> None:
        if not self.config.xr0_pretrained_path:
            return
        # When loading a LeRobot checkpoint, safetensors will load immediately
        # after __init__; avoid doing the expensive official load first.
        if self.config.pretrained_path:
            return

        path = Path(self.config.xr0_pretrained_path).expanduser()
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "module" in payload:
            state_dict = payload["module"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported XR0 checkpoint payload type: {type(payload)}")

        remapped = {}
        for key, value in state_dict.items():
            remapped[_remap_official_checkpoint_key(key)] = value

        info = self.model.load_state_dict(remapped, strict=False)
        logger.info("Loaded official XR0 checkpoint from %s: %s", path, info)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _get_image_keys(self) -> list[str]:
        available_keys = set(self.config.image_features.keys())
        if self.config.image_key_order:
            missing_keys = [key for key in self.config.image_key_order if key not in available_keys]
            if missing_keys:
                raise ValueError(
                    "XR0 image_key_order contains keys not found in dataset features: "
                    f"{missing_keys}. Available image keys: {sorted(available_keys)}"
                )
            unused_keys = sorted(available_keys - set(self.config.image_key_order))
            if unused_keys:
                logger.warning("XR0 will ignore image keys not listed in image_key_order: %s", unused_keys)
            return list(self.config.image_key_order)
        ordered_keys = [key for key in DEFAULT_IMAGE_KEY_ORDER if key in available_keys]
        ordered_keys.extend(sorted(available_keys - set(ordered_keys)))
        return ordered_keys

    def _get_image_description(self, image_key: str) -> str:
        return self.config.image_key_descriptions.get(
            image_key,
            DEFAULT_IMAGE_DESCRIPTIONS.get(image_key, image_key),
        )

    def _get_image_view_heading(self, image_key: str) -> str:
        custom_description = self.config.image_key_descriptions.get(image_key)
        if custom_description:
            normalized = custom_description.strip().rstrip(".")
            return DESCRIPTION_TO_NATIVE_VIEW_HEADING.get(normalized.lower(), normalized)
        return DEFAULT_IMAGE_VIEW_HEADINGS.get(image_key, self._get_image_description(image_key).rstrip("."))

    @staticmethod
    def _format_task_for_native_prompt(task: str) -> str:
        task = task.strip()
        if task.endswith("/no_cot"):
            return task
        return f"{task} /no_cot"

    def _tensor_to_image(self, image: Tensor) -> np.ndarray:
        image = image.detach().cpu()
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        array = image.float().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.max(initial=0) > 1.5:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)
        return (array * 255).astype(np.uint8)

    def _tasks_from_batch(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get("task") or batch.get("subtask")
        if tasks is None:
            return ["Execute the robot action."] * batch_size
        if isinstance(tasks, str):
            return [tasks] * batch_size
        if isinstance(tasks, Tensor):
            return [str(item) for item in tasks.detach().cpu().tolist()]
        tasks = list(tasks)
        if not tasks:
            return ["Execute the robot action."] * batch_size
        if len(tasks) < batch_size:
            tasks.extend([tasks[-1]] * (batch_size - len(tasks)))
        return [str(task) for task in tasks[:batch_size]]

    def _prepare_vlm_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "input_ids" in batch and "attention_mask" in batch:
            return {
                key: value.to(self.device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
                if key
                in {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                }
            }

        if self.processor is None:
            raise RuntimeError(
                "XR0 requires either pre-tokenized Qwen inputs or an available Qwen AutoProcessor."
            )

        image_keys = self._get_image_keys()
        if not image_keys:
            raise ValueError("XR0 requires at least one visual input feature.")

        first_image = batch[image_keys[0]]
        batch_size = first_image.shape[0] if isinstance(first_image, Tensor) and first_image.ndim == 4 else 1
        tasks = self._tasks_from_batch(batch, batch_size)

        messages = []
        for batch_index in range(batch_size):
            content = [
                {
                    "type": "text",
                    "text": "The following observations are captured from multiple views.\n",
                }
            ]
            for image_key in image_keys:
                image = batch[image_key]
                if isinstance(image, Tensor) and image.ndim == 4:
                    image = image[batch_index]
                heading = self._get_image_view_heading(image_key)
                content.append({"type": "text", "text": f"# {heading}\n"})
                content.append({"type": "image", "image": self._tensor_to_image(image)})
                content.append({"type": "text", "text": "\n"})
            task = self._format_task_for_native_prompt(tasks[batch_index])
            content.append({"type": "text", "text": f"Generate robot actions for the task:\n{task}"})
            messages.append(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
                ]
            )

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                images_kwargs={"do_resize": False},
            )
        except TypeError:
            texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            images = [
                item["image"]
                for message in messages
                for item in message[0]["content"]
                if item["type"] == "image"
            ]
            inputs = self.processor(
                text=texts if isinstance(texts, list) else [texts],
                images=images,
                padding=True,
                return_tensors="pt",
            )

        return {key: value.to(self.device) if isinstance(value, Tensor) else value for key, value in inputs.items()}

    def _prepare_native_batch(self, batch: dict[str, Any], *, include_action: bool) -> dict[str, Any]:
        native_batch = dict(self._prepare_vlm_inputs(batch))

        if OBS_STATE in batch:
            native_batch["state"] = batch[OBS_STATE].to(self.device)
        elif "state" in batch:
            native_batch["state"] = batch["state"].to(self.device)

        if include_action and ACTION in batch:
            native_batch["action"] = batch[ACTION].to(self.device)
            action_mask = batch.get("action_mask")
            if action_mask is None:
                action_mask = self._make_action_mask(
                    native_batch["action"].shape[0],
                    native_batch["action"].shape[1],
                    device=self.device,
                    dtype=torch.int32,
                )
            native_batch["action_mask"] = action_mask.to(self.device)
        elif not include_action:
            batch_size = (
                native_batch["state"].shape[0]
                if "state" in native_batch
                else native_batch["input_ids"].shape[0]
            )
            native_batch["action"] = torch.zeros(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=self.device,
                dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32,
            )
            native_batch["action_mask"] = self._make_action_mask(
                batch_size,
                self.config.chunk_size,
                device=self.device,
                dtype=torch.int32,
            )

        if "prefix_length" in batch:
            native_batch["prefix_length"] = batch["prefix_length"]

        return native_batch

    def _make_action_mask(
        self,
        batch_size: int,
        horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return _xr0_action_mask(
            batch_size,
            horizon,
            action_layout=self.config.action_layout,
            controlled_arms=self.config.controlled_arms,
            device=device,
            dtype=dtype,
        )

    def _unnormalize_native_action(self, action: Tensor) -> Tensor:
        if not bool(self._xr0_action_stats_loaded.item()):
            return action
        mean = self._xr0_action_mean.to(device=action.device, dtype=action.dtype)
        std = self._xr0_action_std.to(device=action.device, dtype=action.dtype)
        return action * (std + 1e-6) + mean

    def _restore_absolute_action(self, action: Tensor, state: Tensor) -> Tensor:
        if self.config.actions_are_delta:
            return action
        if state.ndim == 2:
            state = state.unsqueeze(1)
        return action + state.to(device=action.device, dtype=action.dtype)

    def _hold_uncontrolled_action_dims(self, action: Tensor, state: Tensor) -> Tensor:
        control_mask = self._make_action_mask(
            action.shape[0],
            action.shape[1],
            device=action.device,
            dtype=torch.bool,
        )
        if torch.all(control_mask):
            return action

        if self.config.actions_are_delta:
            fallback = torch.zeros_like(action)
        else:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            fallback = state.to(device=action.device, dtype=action.dtype).expand(-1, action.shape[1], -1)
        return torch.where(control_mask, action, fallback)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        native_batch = self._prepare_native_batch(batch, include_action=True)
        loss_dict = self.model(native_batch, return_loss=True)
        loss = loss_dict["loss"]

        if reduction == "none":
            # Native XR0 returns a scalar masked loss. Keep LeRobot's RA-BC path
            # functional by returning a batch-sized vector with the same value.
            batch_size = batch[ACTION].shape[0]
            per_sample = loss.expand(batch_size)
            return per_sample, {"loss": float(loss.detach().cpu())}

        output_dict = {
            key: float(value.detach().cpu()) if isinstance(value, Tensor) and value.numel() == 1 else value
            for key, value in loss_dict.items()
        }
        return loss, output_dict

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        return self.select_action(batch)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        native_batch = self._prepare_native_batch(batch, include_action=False)
        state = native_batch["state"]
        actions = self.model.generate(native_batch)
        actions = self._unnormalize_native_action(actions)
        actions = self._restore_absolute_action(actions, state)
        actions = self._hold_uncontrolled_action_dims(actions, state)
        return actions[:, :, : self.config.max_action_dim]

    def _get_default_peft_targets(self) -> dict:
        target_modules = (
            r"(.*\.dit\..*\.self_attn\.(q|v)_proj"
            r"|.*\.action_projector.*"
            r"|.*\.action_output_layer.*"
            r"|.*\.state_projector.*"
            r"|.*\.t_embedder.*"
            r"|.*\.t_projector.*)"
        )
        return {"target_modules": target_modules}

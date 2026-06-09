#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""XR0 policy processors.

XR0 keeps the official 32-dimensional internal action/state layout so Xiaomi's
pretrained weights stay compatible. ALOHA-style 14D data is packed into that
layout before the native XR0 model sees it, and unpacked again for robot control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.xr0.configuration_xr0 import XR0Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

XR0_DIM = 32
ALOHA_DIM = 14

LEFT_GRIPPER = slice(6, 7)
LEFT_JOINTS = slice(7, 13)
RIGHT_GRIPPER = slice(20, 21)
RIGHT_JOINTS = slice(21, 27)

XR0_CONTROL_SLICES = (LEFT_JOINTS, LEFT_GRIPPER, RIGHT_JOINTS, RIGHT_GRIPPER)
XR0_LEFT_ARM_SLICES = (LEFT_JOINTS, LEFT_GRIPPER)
XR0_RIGHT_ARM_SLICES = (RIGHT_JOINTS, RIGHT_GRIPPER)
CONTROLLED_ARMS = ("both", "left", "right")


def _as_tensor(value: Any, *, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def _ensure_batched_state(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim == 3 and state.shape[1] == 1:
        state = state[:, 0]
    if state.ndim != 2:
        raise ValueError(f"XR0 state must have shape (B,D), got {tuple(state.shape)}")
    return state


def _ensure_batched_action(action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"XR0 action must have shape (B,T,D), got {tuple(action.shape)}")
    return action


def _fit_horizon(action: torch.Tensor, horizon: int) -> torch.Tensor:
    if action.shape[1] == horizon:
        return action
    if action.shape[1] > horizon:
        return action[:, :horizon]
    pad = action[:, -1:].expand(-1, horizon - action.shape[1], -1)
    return torch.cat([action, pad], dim=1)


def _fit_horizon_mask(mask: torch.Tensor, horizon: int) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"XR0 valid-step mask must have shape (B,T), got {tuple(mask.shape)}")
    mask = mask.bool()
    if mask.shape[1] == horizon:
        return mask
    if mask.shape[1] > horizon:
        return mask[:, :horizon]
    pad = torch.zeros(mask.shape[0], horizon - mask.shape[1], device=mask.device, dtype=torch.bool)
    return torch.cat([mask, pad], dim=1)


def _aloha14_to_xr032(values: torch.Tensor, *, is_action: bool = True) -> torch.Tensor:
    if values.shape[-1] == XR0_DIM:
        return values
    if values.shape[-1] != ALOHA_DIM:
        raise ValueError(f"Expected ALOHA 14D or XR0 32D values, got last dim={values.shape[-1]}")

    out = values.new_zeros(*values.shape[:-1], XR0_DIM)
    out[..., LEFT_JOINTS] = values[..., 0:6]
    out[..., LEFT_GRIPPER] = values[..., 6:7]
    out[..., RIGHT_JOINTS] = values[..., 7:13]
    out[..., RIGHT_GRIPPER] = values[..., 13:14]
    return out


def _xr032_to_aloha14(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] == ALOHA_DIM:
        return values
    if values.shape[-1] != XR0_DIM:
        raise ValueError(f"Expected XR0 32D values, got last dim={values.shape[-1]}")
    return torch.cat(
        [
            values[..., LEFT_JOINTS],
            values[..., LEFT_GRIPPER],
            values[..., RIGHT_JOINTS],
            values[..., RIGHT_GRIPPER],
        ],
        dim=-1,
    )


def _xr0_action_mask(
    batch_size: int,
    horizon: int,
    *,
    action_layout: str,
    controlled_arms: str = "both",
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if controlled_arms not in CONTROLLED_ARMS:
        raise ValueError("controlled_arms must be 'both', 'left', or 'right'.")
    if action_layout not in {"aloha14", "xr0_32"}:
        raise ValueError("action_layout must be 'aloha14' or 'xr0_32'.")

    if action_layout == "xr0_32" and controlled_arms == "both":
        return torch.ones(batch_size, horizon, XR0_DIM, device=device, dtype=dtype)

    control_slices = {
        "both": XR0_CONTROL_SLICES,
        "left": XR0_LEFT_ARM_SLICES,
        "right": XR0_RIGHT_ARM_SLICES,
    }[controlled_arms]
    mask = torch.zeros(batch_size, horizon, XR0_DIM, device=device, dtype=dtype)
    for slc in control_slices:
        mask[..., slc] = 1
    return mask


def _valid_xr0_stats(stats: dict[str, dict[str, Any]] | None, horizon: int) -> bool:
    if not stats or ACTION not in stats:
        return False
    action_stats = stats[ACTION]
    mean = action_stats.get("mean")
    std = action_stats.get("std")
    if mean is None or std is None:
        return False
    return tuple(torch.as_tensor(mean).shape[-2:]) == (horizon, XR0_DIM) and tuple(
        torch.as_tensor(std).shape[-2:]
    ) == (horizon, XR0_DIM)


def _coerce_stats_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "stats" in payload and isinstance(payload["stats"], dict):
        payload = payload["stats"]
    if isinstance(payload, dict) and ACTION in payload:
        return payload[ACTION]
    if isinstance(payload, dict) and {"mean", "std"}.issubset(payload):
        return payload
    raise ValueError(
        "XR0 stats file must contain either {'action': {'mean', 'std'}} or a top-level {'mean', 'std'}."
    )


def load_xr0_action_stats(path: str | Path, *, horizon: int, dim: int = XR0_DIM) -> dict[str, dict[str, torch.Tensor]]:
    """Load XR0 action stats used for native 32D action normalization."""

    stats_path = Path(path).expanduser()
    if not stats_path.exists():
        raise FileNotFoundError(f"XR0 stats file not found: {stats_path}")

    if stats_path.suffix.lower() == ".json":
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        payload = torch.load(stats_path, map_location="cpu")

    action_stats = _coerce_stats_payload(payload)
    mean = _as_tensor(action_stats["mean"])
    std = _as_tensor(action_stats["std"])

    if mean.shape == (dim,):
        mean = mean.unsqueeze(0).expand(horizon, dim).clone()
    if std.shape == (dim,):
        std = std.unsqueeze(0).expand(horizon, dim).clone()

    if tuple(mean.shape[-2:]) != (horizon, dim) or tuple(std.shape[-2:]) != (horizon, dim):
        raise ValueError(
            f"XR0 action stats must have shape ({horizon}, {dim}) or ({dim},), "
            f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )

    stats: dict[str, dict[str, torch.Tensor]] = {ACTION: {"mean": mean.float(), "std": std.float()}}
    if isinstance(action_stats, dict) and "count" in action_stats:
        stats[ACTION]["count"] = _as_tensor(action_stats["count"]).float()
    return stats


def resolve_xr0_action_stats(
    config: XR0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    if config.xr0_stats_path:
        return load_xr0_action_stats(config.xr0_stats_path, horizon=config.chunk_size, dim=config.max_action_dim)
    if _valid_xr0_stats(dataset_stats, config.chunk_size):
        return dataset_stats
    return None


@dataclass
@ProcessorStepRegistry.register(name="xr0_pack_action_layout")
class PackXR0ActionLayoutStep(ProcessorStep):
    """Pack LeRobot/ALOHA batches into native XR0 state/action tensors."""

    action_layout: str = "aloha14"
    horizon: int = 30
    controlled_arms: str = "both"
    actions_are_delta: bool = False
    stats: dict[str, dict[str, Any]] | None = None
    eps: float = 1e-6

    _mean: torch.Tensor | None = field(default=None, init=False, repr=False)
    _std: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.action_layout not in {"aloha14", "xr0_32"}:
            raise ValueError("action_layout must be 'aloha14' or 'xr0_32'.")
        if self.controlled_arms not in CONTROLLED_ARMS:
            raise ValueError("controlled_arms must be 'both', 'left', or 'right'.")
        self._load_stats(self.stats)

    def _load_stats(self, stats: dict[str, dict[str, Any]] | None) -> None:
        self.stats = stats
        if _valid_xr0_stats(stats, self.horizon):
            self._mean = _as_tensor(stats[ACTION]["mean"])
            self._std = _as_tensor(stats[ACTION]["std"])
        else:
            self._mean = None
            self._std = None

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            return action
        mean = self._mean.to(device=action.device, dtype=action.dtype)
        std = self._std.to(device=action.device, dtype=action.dtype)
        return (action - mean) / (std + self.eps)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = dict(new_transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if OBS_STATE not in observation:
            raise ValueError(f"XR0 requires `{OBS_STATE}` in the batch.")

        state_in = _ensure_batched_state(_as_tensor(observation[OBS_STATE]))
        state32 = _aloha14_to_xr032(state_in, is_action=False)
        observation[OBS_STATE] = state32.unsqueeze(1)

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            action_in = _fit_horizon(_ensure_batched_action(_as_tensor(action)), self.horizon)
            action32 = _aloha14_to_xr032(action_in, is_action=True)
            if not self.actions_are_delta:
                action32 = action32 - state32.unsqueeze(1)

            valid_steps = torch.ones(
                action_in.shape[:2],
                device=action32.device,
                dtype=torch.bool,
            )
            action_is_pad = complementary.get(f"{ACTION}_is_pad")
            if action_is_pad is not None:
                valid_steps = ~_fit_horizon_mask(
                    _as_tensor(action_is_pad, dtype=None).to(action32.device),
                    self.horizon,
                )
            else:
                valid_steps = _fit_horizon_mask(valid_steps, self.horizon)

            mask = _xr0_action_mask(
                action32.shape[0],
                self.horizon,
                action_layout=self.action_layout,
                controlled_arms=self.controlled_arms,
                device=action32.device,
                dtype=action32.dtype,
            )
            mask = mask * valid_steps.unsqueeze(-1).to(dtype=mask.dtype)
            new_transition[TransitionKey.ACTION] = self._normalize_action(action32)
            complementary["action_mask"] = mask

        new_transition[TransitionKey.OBSERVATION] = observation
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "action_layout": self.action_layout,
            "horizon": self.horizon,
            "controlled_arms": self.controlled_arms,
            "actions_are_delta": self.actions_are_delta,
            "eps": self.eps,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._mean is None or self._std is None:
            return {}
        return {"action.mean": self._mean.cpu(), "action.std": self._std.cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "action.mean" in state and "action.std" in state:
            self._mean = state["action.mean"]
            self._std = state["action.std"]
            self.stats = {ACTION: {"mean": self._mean, "std": self._std}}


@dataclass
@ProcessorStepRegistry.register(name="xr0_unpack_action_layout")
class UnpackXR0ActionLayoutStep(ProcessorStep):
    """Convert native XR0 actions back to the requested robot action layout."""

    action_layout: str = "aloha14"
    horizon: int = 30
    stats: dict[str, dict[str, Any]] | None = None
    eps: float = 1e-6

    _mean: torch.Tensor | None = field(default=None, init=False, repr=False)
    _std: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_stats(self.stats)

    def _load_stats(self, stats: dict[str, dict[str, Any]] | None) -> None:
        self.stats = stats
        if _valid_xr0_stats(stats, self.horizon):
            self._mean = _as_tensor(stats[ACTION]["mean"])
            self._std = _as_tensor(stats[ACTION]["std"])
        else:
            self._mean = None
            self._std = None

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            return action
        mean = self._mean.to(device=action.device, dtype=action.dtype)
        std = self._std.to(device=action.device, dtype=action.dtype)
        return action * (std + self.eps) + mean

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        action_tensor = self._unnormalize_action(_as_tensor(action, dtype=None))
        if self.action_layout == "aloha14":
            action_tensor = _xr032_to_aloha14(action_tensor)
        new_transition[TransitionKey.ACTION] = action_tensor
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"action_layout": self.action_layout, "horizon": self.horizon, "eps": self.eps}

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._mean is None or self._std is None:
            return {}
        return {"action.mean": self._mean.cpu(), "action.std": self._std.cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "action.mean" in state and "action.std" in state:
            self._mean = state["action.mean"]
            self._std = state["action.std"]
            self.stats = {ACTION: {"mean": self._mean, "std": self._std}}


def make_xr0_pre_post_processors(
    config: XR0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    xr0_action_stats = resolve_xr0_action_stats(config, dataset_stats)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        PackXR0ActionLayoutStep(
            action_layout=config.action_layout,
            horizon=config.chunk_size,
            controlled_arms=config.controlled_arms,
            actions_are_delta=config.actions_are_delta,
            stats=xr0_action_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        UnpackXR0ActionLayoutStep(
            action_layout=config.action_layout,
            horizon=config.chunk_size,
            stats=None,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

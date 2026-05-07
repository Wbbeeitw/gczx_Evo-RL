#!/usr/bin/env python

from __future__ import annotations

import numpy as np


def normalize_values(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("'values' must contain only finite numbers.")

    if mode == "none":
        return values.astype(np.float32, copy=True)
    if mode == "clip":
        return np.clip(values, -1.0, 0.0).astype(np.float32)
    if mode == "episode_minmax":
        if values.size == 0:
            return values.astype(np.float32, copy=True)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        span = max_value - min_value
        if span <= 1e-8:
            return np.clip(values, -1.0, 0.0).astype(np.float32)
        scaled = (values - min_value) / span
        return np.clip(scaled - 1.0, -1.0, 0.0).astype(np.float32)
    raise ValueError("'mode' must be one of {'clip', 'episode_minmax', 'none'}.")


def smooth_values(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if window <= 1 or values.size == 0:
        return values.astype(np.float32, copy=True)

    radius = window // 2
    smoothed = np.empty_like(values, dtype=np.float32)
    for idx in range(values.size):
        left = max(0, idx - radius)
        right = min(values.size, idx + radius + 1)
        smoothed[idx] = float(np.mean(values[left:right]))
    return smoothed


def values_to_completion(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.clip(values + 1.0, 0.0, 1.0).astype(np.float32)


def completion_to_stages(completion: np.ndarray, num_stages: int) -> np.ndarray:
    if num_stages <= 0:
        raise ValueError("'num_stages' must be > 0.")
    completion = np.asarray(completion, dtype=np.float32)
    stages = np.floor(completion * float(num_stages)).astype(np.int64)
    return np.minimum(num_stages - 1, np.maximum(0, stages))


def values_to_stages(values: np.ndarray, num_stages: int) -> tuple[np.ndarray, np.ndarray]:
    completion = values_to_completion(values)
    stages = completion_to_stages(completion, num_stages)
    return completion, stages

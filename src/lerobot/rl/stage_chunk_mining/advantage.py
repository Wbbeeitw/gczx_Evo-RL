#!/usr/bin/env python

from __future__ import annotations

import numpy as np


def compute_chunk_advantage(
    chunk_values: np.ndarray | list[float],
    chunk_size: int,
    l_max: float,
    lam: float = 0.95,
) -> float:
    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    if l_max <= 0:
        raise ValueError("'l_max' must be > 0.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("'lam' must be within [0, 1].")

    values = np.asarray(chunk_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"'chunk_values' must be rank-1, got shape={tuple(values.shape)}.")
    expected_length = int(chunk_size) + 1
    if values.shape[0] != expected_length:
        raise ValueError(
            f"'chunk_values' must have length chunk_size + 1 ({expected_length}), "
            f"got {values.shape[0]}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("'chunk_values' must contain only finite numbers.")

    rho = 1.0 / float(l_max)
    deltas = -rho + values[1:] - values[:-1]
    powers = float(lam) ** np.arange(int(chunk_size), dtype=np.float64)
    return float(np.dot(powers, deltas))


def compute_chunk_advantages_batch(
    values: np.ndarray,
    chunk_size: int,
    l_max: float,
    lam: float = 0.95,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    if l_max <= 0:
        raise ValueError("'l_max' must be > 0.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("'lam' must be within [0, 1].")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("'values' must contain only finite numbers.")
    if values.shape[0] < int(chunk_size) + 1:
        return np.empty((0,), dtype=np.float64)

    rho = 1.0 / float(l_max)
    deltas = -rho + values[1:] - values[:-1]
    powers = float(lam) ** np.arange(int(chunk_size), dtype=np.float64)
    return np.convolve(deltas, powers[::-1], mode="valid").astype(np.float64, copy=False)


def safe_l_max(value: float, chunk_size: int) -> float:
    return float(max(float(value), float(chunk_size + 1)))


def percentile_or_max(lengths: list[int], percentile: float) -> float:
    if not lengths:
        raise ValueError("Cannot compute L_max from an empty length list.")
    arr = np.asarray(lengths, dtype=np.float32)
    if percentile >= 100.0:
        return float(np.max(arr))
    return float(np.percentile(arr, percentile))

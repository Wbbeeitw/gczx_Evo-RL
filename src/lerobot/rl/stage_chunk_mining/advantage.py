#!/usr/bin/env python

from __future__ import annotations

import numpy as np


def compute_chunk_advantage(
    start_value: float,
    bootstrap_value: float,
    chunk_size: int,
    l_max: float,
) -> float:
    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    if l_max <= 0:
        raise ValueError("'l_max' must be > 0.")
    return float(bootstrap_value) - float(start_value) - float(chunk_size) / float(l_max)


def safe_l_max(value: float, chunk_size: int) -> float:
    return float(max(float(value), float(chunk_size + 1)))


def percentile_or_max(lengths: list[int], percentile: float) -> float:
    if not lengths:
        raise ValueError("Cannot compute L_max from an empty length list.")
    arr = np.asarray(lengths, dtype=np.float32)
    if percentile >= 100.0:
        return float(np.max(arr))
    return float(np.percentile(arr, percentile))


#!/usr/bin/env python

from __future__ import annotations

import numpy as np


def find_forward_boundaries(stages: np.ndarray) -> np.ndarray:
    stages = np.asarray(stages, dtype=np.int64)
    if stages.ndim != 1:
        raise ValueError(f"'stages' must be rank-1, got shape={tuple(stages.shape)}.")
    if stages.size <= 1:
        return np.empty((0,), dtype=np.int64)
    return np.asarray([idx for idx in range(1, stages.size) if stages[idx - 1] < stages[idx]], dtype=np.int64)


def find_unique_stage_boundaries(stages: np.ndarray) -> np.ndarray:
    """Return one canonical boundary for each newly reached higher stage.

    Raw framewise stages can jitter around a threshold, e.g. 3,4,3,4. For
    stage-aware chunk mining, those repeated local crossings describe the same
    semantic stage boundary. This function follows the monotonic stage envelope
    and records only the first frame where the episode reaches a stage above all
    previous stages.
    """

    stages = np.asarray(stages, dtype=np.int64)
    if stages.ndim != 1:
        raise ValueError(f"'stages' must be rank-1, got shape={tuple(stages.shape)}.")
    if stages.size <= 1:
        return np.empty((0,), dtype=np.int64)

    boundaries: list[int] = []
    running_max = int(stages[0])
    for idx in range(1, stages.size):
        stage = int(stages[idx])
        if stage > running_max:
            boundaries.append(idx)
            running_max = stage

    return np.asarray(boundaries, dtype=np.int64)


def boundary_candidate_starts(
    boundary_index: int,
    episode_length: int,
    chunk_size: int,
) -> np.ndarray:
    # t < b < t + K, and t + K must exist for bootstrapping.
    first = max(0, boundary_index - chunk_size + 1)
    last = min(boundary_index - 1, episode_length - chunk_size - 1)
    if last < first:
        return np.empty((0,), dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)

#!/usr/bin/env python

from __future__ import annotations

import numpy as np


def valid_chunk_start_count(episode_length: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    # A K-action chunk a[t:t+K] bootstraps from value[t+K].
    return max(episode_length - chunk_size, 0)


def iter_episode_slices(episode_indices: np.ndarray):
    episode_indices = np.asarray(episode_indices)
    if episode_indices.ndim != 1:
        raise ValueError(f"'episode_indices' must be rank-1, got shape={tuple(episode_indices.shape)}.")
    if episode_indices.size == 0:
        return

    start = 0
    current = episode_indices[0]
    for idx in range(1, episode_indices.size):
        if episode_indices[idx] != current:
            yield int(current), slice(start, idx)
            start = idx
            current = episode_indices[idx]
    yield int(current), slice(start, episode_indices.size)


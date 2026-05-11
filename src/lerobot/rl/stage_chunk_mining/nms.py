#!/usr/bin/env python

from __future__ import annotations

import numpy as np


def temporal_iou(start_a: int, start_b: int, length: int) -> float:
    if length <= 0:
        raise ValueError("'length' must be > 0.")
    end_a = start_a + length
    end_b = start_b + length
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(inter) / float(union) if union > 0 else 0.0


def temporal_overlap_ratio(start_a: int, start_b: int, length: int) -> float:
    if length <= 0:
        raise ValueError("'length' must be > 0.")
    end_a = start_a + length
    end_b = start_b + length
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    return float(inter) / float(length)


def temporal_nms(
    starts: np.ndarray,
    scores: np.ndarray,
    length: int,
    threshold: float,
    top_k: int = 0,
    metric: str = "iou",
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if starts.ndim != 1 or scores.ndim != 1:
        raise ValueError("'starts' and 'scores' must be rank-1 arrays.")
    if starts.shape[0] != scores.shape[0]:
        raise ValueError("'starts' and 'scores' must have the same length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("'threshold' must be within [0, 1].")
    if metric not in {"iou", "overlap_ratio"}:
        raise ValueError("'metric' must be one of ['iou', 'overlap_ratio'].")
    if starts.size == 0:
        return np.empty((0,), dtype=np.int64)

    overlap_fn = temporal_iou if metric == "iou" else temporal_overlap_ratio
    order = np.argsort(-scores, kind="mergesort")
    kept: list[int] = []
    for candidate_idx in order:
        candidate_start = int(starts[candidate_idx])
        if all(overlap_fn(candidate_start, int(starts[kept_idx]), length) <= threshold for kept_idx in kept):
            kept.append(int(candidate_idx))
            if top_k > 0 and len(kept) >= top_k:
                break
    return np.asarray(kept, dtype=np.int64)

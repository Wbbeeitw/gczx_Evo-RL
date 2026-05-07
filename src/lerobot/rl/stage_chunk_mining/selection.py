#!/usr/bin/env python

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil

import numpy as np

from lerobot.rl.stage_chunk_mining.advantage import compute_chunk_advantage, safe_l_max
from lerobot.rl.stage_chunk_mining.boundary import (
    boundary_candidate_starts,
    find_forward_boundaries,
    find_unique_stage_boundary_events,
)
from lerobot.rl.stage_chunk_mining.chunks import iter_episode_slices, valid_chunk_start_count
from lerobot.rl.stage_chunk_mining.nms import temporal_nms
from lerobot.rl.stage_chunk_mining.stages import normalize_values, smooth_values, values_to_stages

CHUNK_TYPE_INVALID = 0
CHUNK_TYPE_INTRA_STAGE = 1
CHUNK_TYPE_FORWARD_TRANSITION = 2
CHUNK_TYPE_REGRESSION = 3

SELECTION_NONE = 0
SELECTION_INTRA_STAGE_TOP = 1
SELECTION_BOUNDARY_TRANSITION = 2


@dataclass
class StageChunkMiningResult:
    normalized_value: np.ndarray
    completion: np.ndarray
    stage: np.ndarray
    chunk_advantage: np.ndarray
    chunk_type: np.ndarray
    chunk_stage: np.ndarray
    boundary_id: np.ndarray
    indicator: np.ndarray
    weight: np.ndarray
    selection_role: np.ndarray
    report: dict

    def as_columns(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}.normalized_value": self.normalized_value.astype(np.float32, copy=False),
            f"{prefix}.completion": self.completion.astype(np.float32, copy=False),
            f"{prefix}.stage": self.stage.astype(np.int64, copy=False),
            f"{prefix}.chunk_advantage": self.chunk_advantage.astype(np.float32, copy=False),
            f"{prefix}.chunk_type": self.chunk_type.astype(np.int64, copy=False),
            f"{prefix}.chunk_stage": self.chunk_stage.astype(np.int64, copy=False),
            f"{prefix}.boundary_id": self.boundary_id.astype(np.int64, copy=False),
            f"{prefix}.indicator": self.indicator.astype(np.int64, copy=False),
            f"{prefix}.weight": self.weight.astype(np.float32, copy=False),
            f"{prefix}.selection_role": self.selection_role.astype(np.int64, copy=False),
        }


def _validate_inputs(
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
) -> None:
    sizes = {arr.shape[0] for arr in [values, episode_indices, frame_indices, task_indices]}
    if len(sizes) != 1:
        raise ValueError(
            "values, episode_indices, frame_indices, and task_indices must have the same length."
        )
    for name, arr in {
        "values": values,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "task_indices": task_indices,
    }.items():
        if arr.ndim != 1:
            raise ValueError(f"'{name}' must be rank-1, got shape={tuple(arr.shape)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("'values' must contain only finite numbers.")


def _validate_episode_slice(
    *,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    ep_slice: slice,
) -> None:
    ep_frames = frame_indices[ep_slice]
    if ep_frames.size > 1 and np.any(np.diff(ep_frames) < 0):
        raise ValueError("Frames must be sorted by frame_index within each episode.")

    ep_tasks = task_indices[ep_slice]
    if np.unique(ep_tasks).size > 1:
        raise ValueError("Each episode must contain exactly one task_index.")


def _stage_keep_count(num_candidates: int, stage_top_ratio: float, stage_top_k: int, min_candidates: int) -> int:
    if num_candidates <= 0:
        return 0
    if stage_top_k > 0:
        return min(stage_top_k, num_candidates)
    if stage_top_ratio <= 0.0:
        return 0
    return min(num_candidates, max(min_candidates, int(ceil(num_candidates * stage_top_ratio))))


def _episode_success_value(episode_success: dict[int, bool] | np.ndarray | None, episode_id: int) -> bool:
    if episode_success is None:
        return True
    if isinstance(episode_success, dict):
        return bool(episode_success.get(int(episode_id), True))
    success_array = np.asarray(episode_success)
    if episode_id < 0 or episode_id >= success_array.shape[0]:
        return True
    return bool(success_array[int(episode_id)])


def _first_descent_index(stages: np.ndarray) -> int:
    if stages.size <= 1:
        return int(stages.size)
    running_max = int(stages[0])
    for idx in range(1, stages.size):
        stage = int(stages[idx])
        if stage < running_max:
            return idx
        if stage > running_max:
            running_max = stage
    return int(stages.size)


def _window_chunk_type(stages: np.ndarray) -> int:
    if stages.size == 0:
        return CHUNK_TYPE_INVALID
    diffs = np.diff(stages)
    if not np.any(diffs != 0):
        return CHUNK_TYPE_INTRA_STAGE
    if np.any(diffs < 0):
        return CHUNK_TYPE_REGRESSION
    return CHUNK_TYPE_FORWARD_TRANSITION


def mine_stage_chunks(
    *,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    l_max_by_task: dict[int, float],
    episode_success: dict[int, bool] | np.ndarray | None = None,
    num_stages: int = 5,
    chunk_size: int = 50,
    stage_top_ratio: float = 0.3,
    stage_top_k: int = 0,
    min_stage_candidates: int = 1,
    boundary_top_k: int = 1,
    boundary_nms_iou: float = 0.5,
    boundary_mode: str = "unique_stage_boundary",
    value_smoothing_window: int = 1,
    value_normalization: str = "episode_minmax",
    failure_max_stage: int = 2,
    include_intra_stage: bool = True,
    include_boundary: bool = True,
) -> StageChunkMiningResult:
    values = np.asarray(values, dtype=np.float32)
    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    task_indices = np.asarray(task_indices, dtype=np.int64)
    _validate_inputs(values, episode_indices, frame_indices, task_indices)

    total = values.shape[0]
    normalized_value = np.zeros(total, dtype=np.float32)
    completion = np.zeros(total, dtype=np.float32)
    stage = np.full(total, -1, dtype=np.int64)
    chunk_advantage = np.full(total, np.nan, dtype=np.float32)
    chunk_type = np.zeros(total, dtype=np.int64)
    chunk_stage = np.full(total, -1, dtype=np.int64)
    boundary_id = np.full(total, -1, dtype=np.int64)
    indicator = np.zeros(total, dtype=np.int64)
    weight = np.zeros(total, dtype=np.float32)
    selection_role = np.zeros(total, dtype=np.int64)

    intra_candidates: dict[tuple[int, int, int], list[tuple[int, float]]] = defaultdict(list)
    boundary_count = 0
    boundary_candidate_count = 0
    success_episode_count = 0
    failure_episode_count = 0

    if boundary_mode not in {"unique_stage_boundary", "forward_crossing"}:
        raise ValueError(
            "'boundary_mode' must be one of ['forward_crossing', 'unique_stage_boundary'], "
            f"got {boundary_mode!r}."
        )
    if failure_max_stage < 0 or failure_max_stage >= num_stages:
        raise ValueError("'failure_max_stage' must be within [0, num_stages).")

    for episode_id, ep_slice in iter_episode_slices(episode_indices):
        _validate_episode_slice(frame_indices=frame_indices, task_indices=task_indices, ep_slice=ep_slice)
        ep_success = _episode_success_value(episode_success, episode_id)
        if ep_success:
            success_episode_count += 1
        else:
            failure_episode_count += 1

        positions = np.arange(total, dtype=np.int64)[ep_slice]
        ep_values = normalize_values(values[positions], value_normalization)
        ep_values = smooth_values(ep_values, value_smoothing_window)
        ep_completion, ep_stage = values_to_stages(ep_values, num_stages)

        # Successful demonstrations represent monotonic task progress. We keep
        # only the first arrival at each higher stage so threshold jitter cannot
        # create repeated semantic boundaries. Failed trajectories keep their
        # raw stage shape, e.g. 0->1->2->1->0, so the first descent can cut off
        # later non-advantage behavior.
        if ep_success:
            ep_stage = np.maximum.accumulate(ep_stage)

        normalized_value[positions] = ep_values
        completion[positions] = ep_completion
        stage[positions] = ep_stage

        episode_length = positions.size
        num_starts = valid_chunk_start_count(episode_length, chunk_size)
        if num_starts == 0:
            continue

        local_advantages = np.full(episode_length, np.nan, dtype=np.float32)
        local_chunk_type = np.zeros(episode_length, dtype=np.int64)
        failure_prefix_end = episode_length if ep_success else _first_descent_index(ep_stage)

        for local_t in range(num_starts):
            start_pos = int(positions[local_t])
            end_pos = int(positions[local_t + chunk_size])
            task_idx = int(task_indices[start_pos])
            l_max = safe_l_max(l_max_by_task.get(task_idx, episode_length), chunk_size)

            adv = compute_chunk_advantage(
                start_value=float(ep_values[local_t]),
                bootstrap_value=float(ep_values[local_t + chunk_size]),
                chunk_size=chunk_size,
                l_max=l_max,
            )
            stage_window = ep_stage[local_t : local_t + chunk_size + 1]
            ctype = _window_chunk_type(stage_window)

            allowed_failure_window = True
            if not ep_success:
                allowed_failure_window = (
                    local_t + chunk_size < failure_prefix_end
                    and int(np.max(stage_window)) <= failure_max_stage
                )
                if not allowed_failure_window:
                    ctype = CHUNK_TYPE_INVALID

            if ctype == CHUNK_TYPE_INTRA_STAGE:
                ctype = CHUNK_TYPE_INTRA_STAGE
                intra_candidates[(int(episode_id), task_idx, int(ep_stage[local_t]))].append((start_pos, adv))

            chunk_advantage[start_pos] = np.float32(adv)
            chunk_type[start_pos] = ctype
            chunk_stage[start_pos] = int(ep_stage[local_t])
            local_advantages[local_t] = np.float32(adv)
            local_chunk_type[local_t] = ctype

        if include_boundary and boundary_top_k > 0:
            boundary_events: list[tuple[int, int]]
            if ep_success:
                if boundary_mode == "forward_crossing":
                    boundary_events = [(int(ep_stage[idx]), int(idx)) for idx in find_forward_boundaries(ep_stage)]
                else:
                    boundary_events = find_unique_stage_boundary_events(ep_stage)
            else:
                ascent_stages = ep_stage[:failure_prefix_end]
                if boundary_mode == "forward_crossing":
                    boundary_events = [
                        (int(ascent_stages[idx]), int(idx))
                        for idx in find_forward_boundaries(ascent_stages)
                        if int(ascent_stages[idx]) <= failure_max_stage
                    ]
                else:
                    boundary_events = find_unique_stage_boundary_events(ascent_stages, max_stage=failure_max_stage)

            for _target_stage, local_boundary in boundary_events:
                boundary_count += 1
                starts = boundary_candidate_starts(
                    boundary_index=int(local_boundary),
                    episode_length=episode_length,
                    chunk_size=chunk_size,
                )
                starts = np.asarray(
                    [
                        start
                        for start in starts
                        if local_chunk_type[start] == CHUNK_TYPE_FORWARD_TRANSITION
                        and np.isfinite(local_advantages[start])
                        and (ep_success or start + chunk_size < failure_prefix_end)
                    ],
                    dtype=np.int64,
                )
                if starts.size == 0:
                    continue

                scores = local_advantages[starts]
                boundary_candidate_count += int(starts.size)
                kept_local_indices = temporal_nms(
                    starts=starts,
                    scores=scores,
                    length=chunk_size,
                    iou_threshold=boundary_nms_iou,
                    top_k=boundary_top_k,
                )
                for kept_idx in kept_local_indices:
                    local_start = int(starts[kept_idx])
                    start_pos = int(positions[local_start])
                    indicator[start_pos] = 1
                    weight[start_pos] = 1.0
                    selection_role[start_pos] = SELECTION_BOUNDARY_TRANSITION
                    boundary_id[start_pos] = boundary_count - 1

    intra_selected = 0
    if include_intra_stage:
        for (_episode_id, _task_idx, _stage_idx), candidates in intra_candidates.items():
            keep_count = _stage_keep_count(
                num_candidates=len(candidates),
                stage_top_ratio=stage_top_ratio,
                stage_top_k=stage_top_k,
                min_candidates=min_stage_candidates,
            )
            if keep_count <= 0:
                continue
            ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
            for start_pos, _score in ordered[:keep_count]:
                indicator[start_pos] = 1
                weight[start_pos] = 1.0
                selection_role[start_pos] = SELECTION_INTRA_STAGE_TOP
                intra_selected += 1

    selected_count = int(np.sum(indicator))
    report = {
        "total_frames": int(total),
        "valid_chunk_starts": int(np.sum(chunk_type != CHUNK_TYPE_INVALID)),
        "selected_chunks": selected_count,
        "selected_ratio": float(selected_count / max(int(np.sum(chunk_type != CHUNK_TYPE_INVALID)), 1)),
        "intra_candidate_groups": int(len(intra_candidates)),
        "intra_candidates": int(sum(len(items) for items in intra_candidates.values())),
        "intra_selected": int(intra_selected),
        "boundary_count": int(boundary_count),
        "boundary_candidates": int(boundary_candidate_count),
        "boundary_selected": int(np.sum(selection_role == SELECTION_BOUNDARY_TRANSITION)),
        "boundary_mode": boundary_mode,
        "success_episodes": int(success_episode_count),
        "failure_episodes": int(failure_episode_count),
        "failure_max_stage": int(failure_max_stage),
        "intra_selection_scope": "episode_stage",
    }
    return StageChunkMiningResult(
        normalized_value=normalized_value,
        completion=completion,
        stage=stage,
        chunk_advantage=chunk_advantage,
        chunk_type=chunk_type,
        chunk_stage=chunk_stage,
        boundary_id=boundary_id,
        indicator=indicator,
        weight=weight,
        selection_role=selection_role,
        report=report,
    )

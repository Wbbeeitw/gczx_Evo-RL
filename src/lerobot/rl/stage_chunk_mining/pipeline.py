#!/usr/bin/env python

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.stage_chunk_mining.advantage import percentile_or_max, safe_l_max
from lerobot.rl.stage_chunk_mining.annotation_io import (
    column_to_1d_array,
    feature_infos_for_columns,
    write_columns_in_place,
)
from lerobot.rl.stage_chunk_mining.config import StageChunkMinePipelineConfig
from lerobot.rl.stage_chunk_mining.report import build_selection_report, format_report_summary, save_report
from lerobot.rl.stage_chunk_mining.selection import mine_stage_chunks
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label


def _episode_lengths_by_task(
    episode_indices: np.ndarray,
    task_indices: np.ndarray,
) -> dict[int, list[int]]:
    lengths_by_task: dict[int, list[int]] = defaultdict(list)
    if episode_indices.size == 0:
        return lengths_by_task

    start = 0
    current = int(episode_indices[0])
    for idx in range(1, episode_indices.size + 1):
        if idx == episode_indices.size or int(episode_indices[idx]) != current:
            task_idx = int(task_indices[start])
            lengths_by_task[task_idx].append(idx - start)
            if idx < episode_indices.size:
                start = idx
                current = int(episode_indices[idx])
    return lengths_by_task


def _episode_success_by_index(
    dataset: LeRobotDataset,
    *,
    success_field: str,
    default_success: str,
) -> dict[int, bool]:
    episodes = dataset.meta.episodes.with_format(None)
    payload = episodes[:]
    has_success = success_field in episodes.column_names

    success_by_index: dict[int, bool] = {}
    for i in range(len(episodes)):
        ep_idx = int(payload["episode_index"][i])
        explicit_success = payload[success_field][i] if has_success else None
        resolved_success = resolve_episode_success_label(
            explicit_success,
            default_label=default_success,
            require_label=True,
        )
        success_by_index[ep_idx] = resolved_success == EPISODE_SUCCESS
    return success_by_index


def _success_episode_indices(episode_success: dict[int, bool]) -> set[int]:
    return {int(ep_idx) for ep_idx, success in episode_success.items() if success}


def _compute_l_max_by_task(
    *,
    episode_indices: np.ndarray,
    task_indices: np.ndarray,
    mode: str,
    chunk_size: int,
    episode_success: dict[int, bool],
) -> dict[int, float]:
    lengths_by_task = _episode_lengths_by_task(episode_indices, task_indices)
    if mode == "task_success_max":
        success_eps = _success_episode_indices(episode_success)
        if success_eps:
            success_lengths_by_task: dict[int, list[int]] = defaultdict(list)
            start = 0
            current = int(episode_indices[0])
            for idx in range(1, episode_indices.size + 1):
                if idx == episode_indices.size or int(episode_indices[idx]) != current:
                    if current in success_eps:
                        success_lengths_by_task[int(task_indices[start])].append(idx - start)
                    if idx < episode_indices.size:
                        start = idx
                        current = int(episode_indices[idx])
            lengths_by_task = {
                task_idx: success_lengths_by_task.get(task_idx) or lengths
                for task_idx, lengths in lengths_by_task.items()
            }
        else:
            logging.warning(
                "No successful episodes resolved for l_max_mode=task_success_max; falling back to task_p95."
            )
            mode = "task_p95"

    all_lengths = [length for lengths in lengths_by_task.values() for length in lengths]
    if not all_lengths:
        raise ValueError("Cannot compute L_max: dataset has no episode lengths.")

    if mode == "global_max":
        global_value = percentile_or_max(all_lengths, 100.0)
        return {task_idx: safe_l_max(global_value, chunk_size) for task_idx in lengths_by_task}
    if mode == "global_p95":
        global_value = percentile_or_max(all_lengths, 95.0)
        return {task_idx: safe_l_max(global_value, chunk_size) for task_idx in lengths_by_task}

    percentile = 100.0 if mode in {"task_max", "task_success_max"} else 95.0
    return {
        task_idx: safe_l_max(percentile_or_max(lengths, percentile), chunk_size)
        for task_idx, lengths in lengths_by_task.items()
    }


def run_stage_chunk_mining(cfg: StageChunkMinePipelineConfig) -> dict[str, Any]:
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        revision=cfg.dataset.revision,
        download_videos=cfg.dataset.download_videos,
    )
    raw_frames = dataset.hf_dataset.with_format(None)
    if cfg.mining.value_field not in raw_frames.column_names:
        raise KeyError(f"Missing value field '{cfg.mining.value_field}' in dataset.")

    absolute_indices = column_to_1d_array(raw_frames["index"], np.int64)
    episode_indices = column_to_1d_array(raw_frames["episode_index"], np.int64)
    frame_indices = column_to_1d_array(raw_frames["frame_index"], np.int64)
    task_indices = column_to_1d_array(raw_frames["task_index"], np.int64)
    values = column_to_1d_array(raw_frames[cfg.mining.value_field], np.float32)

    episode_success = _episode_success_by_index(
        dataset,
        success_field=cfg.dataset.success_field,
        default_success=cfg.dataset.default_success,
    )

    l_max_by_task = _compute_l_max_by_task(
        episode_indices=episode_indices,
        task_indices=task_indices,
        mode=cfg.mining.l_max_mode,
        chunk_size=cfg.mining.chunk_size,
        episode_success=episode_success,
    )
    logging.info("Computed L_max by task: %s", l_max_by_task)

    result = mine_stage_chunks(
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        task_indices=task_indices,
        l_max_by_task=l_max_by_task,
        episode_success=episode_success,
        num_stages=cfg.mining.num_stages,
        chunk_size=cfg.mining.chunk_size,
        stage_aware=cfg.mining.stage_aware,
        stage_top_ratio=cfg.mining.stage_top_ratio,
        stage_top_k=cfg.mining.stage_top_k,
        min_stage_candidates=cfg.mining.min_stage_candidates,
        global_top_ratio=cfg.mining.global_top_ratio,
        global_top_k=cfg.mining.global_top_k,
        global_min_candidates=cfg.mining.global_min_candidates,
        global_nms_overlap_ratio=cfg.mining.global_nms_overlap_ratio,
        boundary_top_k=cfg.mining.boundary_top_k,
        boundary_nms_iou=cfg.mining.boundary_nms_iou,
        boundary_mode=cfg.mining.boundary_mode,
        value_smoothing_window=cfg.mining.value_smoothing_window,
        value_normalization=cfg.mining.value_normalization,
        failure_max_stage=cfg.mining.failure_max_stage,
        include_intra_stage=cfg.mining.include_intra_stage,
        include_boundary=cfg.mining.include_boundary,
    )
    columns = result.as_columns(cfg.mining.output_prefix)
    feature_infos = feature_infos_for_columns(columns)
    write_columns_in_place(
        dataset_root=Path(dataset.root),
        absolute_indices=absolute_indices,
        columns=columns,
        feature_infos=feature_infos,
    )

    report = build_selection_report(
        base_report=result.report,
        task_indices=task_indices,
        stage=result.stage,
        chunk_type=result.chunk_type,
        chunk_start_indicator=result.chunk_start_indicator,
        chunk_start_role=result.chunk_start_role,
        indicator=result.indicator,
        selection_role=result.selection_role,
    )
    report["l_max_by_task"] = {str(k): float(v) for k, v in l_max_by_task.items()}
    report["value_field"] = cfg.mining.value_field
    report["value_normalization"] = cfg.mining.value_normalization
    report["boundary_mode"] = cfg.mining.boundary_mode
    report["success_field"] = cfg.dataset.success_field
    report["default_success"] = cfg.dataset.default_success
    report["failure_max_stage"] = cfg.mining.failure_max_stage
    report["output_prefix"] = cfg.mining.output_prefix
    report["stage_aware"] = bool(cfg.mining.stage_aware)
    report["global_top_ratio"] = float(cfg.mining.global_top_ratio)
    report["global_top_k"] = int(cfg.mining.global_top_k)
    report["global_nms_metric"] = "overlap_ratio"
    report["global_nms_overlap_ratio"] = float(cfg.mining.global_nms_overlap_ratio)
    report["indicator_field"] = f"{cfg.mining.output_prefix}.indicator"
    report["chunk_start_indicator_field"] = f"{cfg.mining.output_prefix}.chunk_start_indicator"

    report_path = save_report(report, Path(cfg.output_dir))
    logging.info("Wrote stage chunk annotations to dataset root: %s", dataset.root)
    logging.info("Wrote stage chunk mining report to: %s", report_path)
    logging.info("\n%s", format_report_summary(report))
    return report

#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_selection_report(
    *,
    base_report: dict[str, Any],
    task_indices: np.ndarray,
    stage: np.ndarray,
    chunk_type: np.ndarray,
    chunk_start_indicator: np.ndarray,
    chunk_start_role: np.ndarray,
    indicator: np.ndarray,
    selection_role: np.ndarray,
) -> dict[str, Any]:
    report = dict(base_report)
    per_task_stage: list[dict[str, Any]] = []
    task_indices = np.asarray(task_indices, dtype=np.int64)
    chunk_start_indicator = np.asarray(chunk_start_indicator, dtype=np.int64)
    chunk_start_role = np.asarray(chunk_start_role, dtype=np.int64)
    indicator = np.asarray(indicator, dtype=np.int64)
    selection_role = np.asarray(selection_role, dtype=np.int64)

    for task_idx in np.unique(task_indices):
        task_mask = task_indices == task_idx
        for stage_idx in sorted(int(v) for v in np.unique(stage[task_mask]) if int(v) >= 0):
            stage_mask = task_mask & (stage == stage_idx)
            chunk_start_mask = stage_mask & (chunk_type > 0)
            valid = int(np.sum(chunk_start_mask))
            selected = int(np.sum(chunk_start_mask & (chunk_start_indicator > 0)))
            stage_frames = int(np.sum(stage_mask))
            positive_frames = int(np.sum(stage_mask & (indicator > 0)))
            per_task_stage.append(
                {
                    "task_index": int(task_idx),
                    "stage": int(stage_idx),
                    "valid_chunk_starts": valid,
                    "selected_chunks": selected,
                    "selected_ratio": float(selected / valid) if valid > 0 else 0.0,
                    "stage_frames": stage_frames,
                    "positive_frames": positive_frames,
                    "positive_frame_ratio": float(positive_frames / stage_frames) if stage_frames > 0 else 0.0,
                }
            )

    report["per_task_stage"] = per_task_stage
    report["selection_role_counts"] = {
        str(int(role)): int(np.sum(selection_role == role)) for role in np.unique(selection_role)
    }
    report["chunk_start_role_counts"] = {
        str(int(role)): int(np.sum(chunk_start_role == role)) for role in np.unique(chunk_start_role)
    }
    return report


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_chunk_mining_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report_path


def format_report_summary(report: dict[str, Any]) -> str:
    lines = [
        "Stage chunk mining report:",
        f"- total_frames: {report.get('total_frames', 0)}",
        f"- valid_chunk_starts: {report.get('valid_chunk_starts', 0)}",
        f"- selected_chunks: {report.get('selected_chunks', 0)}",
        f"- selected_ratio: {float(report.get('selected_ratio', 0.0)):.6f}",
        f"- positive_frames: {report.get('positive_frames', 0)}",
        f"- positive_frame_ratio: {float(report.get('positive_frame_ratio', 0.0)):.6f}",
        f"- intra_candidates: {report.get('intra_candidates', 0)}",
        f"- intra_selected: {report.get('intra_selected', 0)}",
        f"- intra_selection_scope: {report.get('intra_selection_scope', 'unknown')}",
        f"- stage_assignment: {report.get('stage_assignment', 'unknown')}",
        f"- value_smoothing_applied: {report.get('value_smoothing_applied', 'unknown')}",
        f"- success_episodes: {report.get('success_episodes', 0)}",
        f"- failure_episodes: {report.get('failure_episodes', 0)}",
        f"- failure_max_stage: {report.get('failure_max_stage', 'n/a')}",
        f"- boundary_mode: {report.get('boundary_mode', 'unknown')}",
        f"- boundary_count: {report.get('boundary_count', 0)}",
        f"- boundary_candidates: {report.get('boundary_candidates', 0)}",
        f"- boundary_selected: {report.get('boundary_selected', 0)}",
    ]
    return "\n".join(lines)

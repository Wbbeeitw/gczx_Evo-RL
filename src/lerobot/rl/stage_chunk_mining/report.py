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
    indicator: np.ndarray,
    selection_role: np.ndarray,
) -> dict[str, Any]:
    report = dict(base_report)
    per_task_stage: list[dict[str, Any]] = []
    task_indices = np.asarray(task_indices, dtype=np.int64)

    for task_idx in np.unique(task_indices):
        task_mask = task_indices == task_idx
        for stage_idx in sorted(int(v) for v in np.unique(stage[task_mask]) if int(v) >= 0):
            mask = task_mask & (stage == stage_idx) & (chunk_type > 0)
            valid = int(np.sum(mask))
            selected = int(np.sum(mask & (indicator > 0)))
            per_task_stage.append(
                {
                    "task_index": int(task_idx),
                    "stage": int(stage_idx),
                    "valid_chunks": valid,
                    "selected_chunks": selected,
                    "selected_ratio": float(selected / valid) if valid > 0 else 0.0,
                }
            )

    report["per_task_stage"] = per_task_stage
    report["selection_role_counts"] = {
        str(int(role)): int(np.sum(selection_role == role)) for role in np.unique(selection_role)
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
        f"- intra_candidates: {report.get('intra_candidates', 0)}",
        f"- intra_selected: {report.get('intra_selected', 0)}",
        f"- boundary_count: {report.get('boundary_count', 0)}",
        f"- boundary_candidates: {report.get('boundary_candidates', 0)}",
        f"- boundary_selected: {report.get('boundary_selected', 0)}",
    ]
    return "\n".join(lines)


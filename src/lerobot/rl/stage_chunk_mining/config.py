#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus

from lerobot.utils.recording_annotations import normalize_episode_success_label


@dataclass
class StageChunkDatasetConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    download_videos: bool = False
    success_field: str = "episode_success"
    default_success: str = "success"

    def validate(self) -> None:
        if not self.repo_id:
            raise ValueError("'dataset.repo_id' must be non-empty.")
        if not self.success_field:
            raise ValueError("'dataset.success_field' must be non-empty.")
        normalized = normalize_episode_success_label(self.default_success)
        if normalized is None:
            raise ValueError("'dataset.default_success' must be either 'success' or 'failure'.")
        self.default_success = normalized


@dataclass
class StageChunkMiningConfig:
    value_field: str = "complementary_info.value"
    output_prefix: str = "complementary_info.vgsacm"
    value_normalization: str = "episode_minmax"

    num_stages: int = 5
    chunk_size: int = 50
    stage_top_ratio: float = 0.3
    stage_top_k: int = 0
    min_stage_candidates: int = 1

    boundary_top_k: int = 1
    boundary_nms_iou: float = 0.5
    boundary_mode: str = "unique_stage_boundary"
    value_smoothing_window: int = 1
    failure_max_stage: int = 2

    l_max_mode: str = "task_p95"
    include_intra_stage: bool = True
    include_boundary: bool = True

    def validate(self) -> None:
        if not self.value_field:
            raise ValueError("'mining.value_field' must be non-empty.")
        if not self.output_prefix:
            raise ValueError("'mining.output_prefix' must be non-empty.")
        valid_normalization_modes = {"clip", "episode_minmax", "none"}
        if self.value_normalization not in valid_normalization_modes:
            raise ValueError(
                "'mining.value_normalization' must be one of "
                f"{sorted(valid_normalization_modes)}, got {self.value_normalization!r}."
            )
        if self.num_stages <= 0:
            raise ValueError("'mining.num_stages' must be > 0.")
        if self.chunk_size <= 0:
            raise ValueError("'mining.chunk_size' must be > 0.")
        if not 0.0 <= self.stage_top_ratio <= 1.0:
            raise ValueError("'mining.stage_top_ratio' must be within [0, 1].")
        if self.stage_top_k < 0:
            raise ValueError("'mining.stage_top_k' must be >= 0.")
        if self.min_stage_candidates < 0:
            raise ValueError("'mining.min_stage_candidates' must be >= 0.")
        if self.boundary_top_k < 0:
            raise ValueError("'mining.boundary_top_k' must be >= 0.")
        if not 0.0 <= self.boundary_nms_iou <= 1.0:
            raise ValueError("'mining.boundary_nms_iou' must be within [0, 1].")
        valid_boundary_modes = {"unique_stage_boundary", "forward_crossing"}
        if self.boundary_mode not in valid_boundary_modes:
            raise ValueError(
                f"'mining.boundary_mode' must be one of {sorted(valid_boundary_modes)}, "
                f"got {self.boundary_mode!r}."
            )
        if self.value_smoothing_window <= 0:
            raise ValueError("'mining.value_smoothing_window' must be > 0.")
        if self.failure_max_stage < 0:
            raise ValueError("'mining.failure_max_stage' must be >= 0.")
        if self.failure_max_stage >= self.num_stages:
            raise ValueError("'mining.failure_max_stage' must be smaller than 'mining.num_stages'.")
        valid_l_max_modes = {"task_p95", "task_max", "global_p95", "global_max", "task_success_max"}
        if self.l_max_mode not in valid_l_max_modes:
            raise ValueError(
                f"'mining.l_max_mode' must be one of {sorted(valid_l_max_modes)}, got {self.l_max_mode!r}."
            )
        if not self.include_intra_stage and not self.include_boundary:
            raise ValueError("At least one of 'include_intra_stage' or 'include_boundary' must be true.")


@dataclass
class StageChunkMinePipelineConfig:
    dataset: StageChunkDatasetConfig
    mining: StageChunkMiningConfig = field(default_factory=StageChunkMiningConfig)
    output_dir: Path | None = None

    def validate(self) -> None:
        self.dataset.validate()
        self.mining.validate()
        if self.output_dir is None:
            self.output_dir = Path("outputs/stage_chunk_mining")

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]

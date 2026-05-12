#!/usr/bin/env python

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus
import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.stage_chunk_mining.annotation_io import column_to_1d_array
from lerobot.rl.stage_chunk_mining.config import StageChunkDatasetConfig
from lerobot.rl.stage_chunk_mining.visualize import (
    generate_stage_chunk_episode_figures,
    output_tag_from_prefix,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class StageChunkGalleryConfig:
    dataset: StageChunkDatasetConfig
    value_field: str = "complementary_info.value"
    output_prefix: str = "complementary_info.vgsacm"
    chunk_size: int = 50
    output_dir: Path | None = None
    camera_keys: str | None = None
    max_cameras: int = 3
    overwrite: bool = True
    strict_verification: bool = True

    def validate(self) -> None:
        self.dataset.validate()
        if not self.value_field:
            raise ValueError("'value_field' must be non-empty.")
        if not self.output_prefix:
            raise ValueError("'output_prefix' must be non-empty.")
        if self.chunk_size <= 0:
            raise ValueError("'chunk_size' must be > 0.")
        if self.max_cameras < 0:
            raise ValueError("'max_cameras' must be >= 0.")
        if self.output_dir is None:
            tag = output_tag_from_prefix(self.output_prefix)
            self.output_dir = Path("outputs/stage_chunk_gallery") / tag

    def to_dict(self) -> dict[str, Any]:
        return draccus.encode(self)  # type: ignore[no-any-return]


def run_stage_chunk_gallery(cfg: StageChunkGalleryConfig) -> dict[str, Any]:
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

    chunk_advantage_field = f"{cfg.output_prefix}.chunk_advantage"
    chunk_start_field = f"{cfg.output_prefix}.chunk_start_indicator"
    indicator_field = f"{cfg.output_prefix}.indicator"
    required_fields = [
        "episode_index",
        "frame_index",
        cfg.value_field,
        chunk_advantage_field,
        chunk_start_field,
        indicator_field,
    ]
    missing_fields = [field for field in required_fields if field not in raw_frames.column_names]
    if missing_fields:
        raise KeyError(f"Missing fields for stage chunk gallery: {missing_fields}")

    episode_indices = column_to_1d_array(raw_frames["episode_index"], np.int64)
    frame_indices = column_to_1d_array(raw_frames["frame_index"], np.int64)
    values = column_to_1d_array(raw_frames[cfg.value_field], np.float32)
    chunk_advantage = column_to_1d_array(raw_frames[chunk_advantage_field], np.float32)
    chunk_start_indicator = column_to_1d_array(raw_frames[chunk_start_field], np.int64)
    indicator = column_to_1d_array(raw_frames[indicator_field], np.int64)

    gallery_summary = generate_stage_chunk_episode_figures(
        dataset=dataset,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        values=values,
        chunk_advantage=chunk_advantage,
        chunk_start_indicator=chunk_start_indicator,
        indicator=indicator,
        chunk_size=cfg.chunk_size,
        output_dir=Path(cfg.output_dir),
        output_prefix=cfg.output_prefix,
        requested_camera_keys=cfg.camera_keys,
        max_cameras=cfg.max_cameras,
        overwrite=cfg.overwrite,
    )

    report = {
        "dataset_root": str(dataset.root),
        "repo_id": dataset.repo_id,
        "value_field": cfg.value_field,
        "output_prefix": cfg.output_prefix,
        "chunk_size": int(cfg.chunk_size),
        "camera_keys": list(gallery_summary["camera_keys"]),
        "figure_dir": gallery_summary["figure_dir"],
        "figure_count": int(gallery_summary["figure_count"]),
        "figure_paths": gallery_summary["figure_paths"],
        "verification": gallery_summary["verification"],
    }

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_chunk_gallery_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    logging.info("Wrote stage chunk gallery report to: %s", report_path)
    logging.info("Generated %d episode figures in: %s", report["figure_count"], report["figure_dir"])
    logging.info("Indicator verification pass: %s", report["verification"]["pass"])

    if cfg.strict_verification and not bool(report["verification"]["pass"]):
        failed = report["verification"].get("failed_episodes", [])
        raise RuntimeError(f"Indicator verification failed for episodes: {failed}")
    return report


@parser.wrap()
def stage_chunk_gallery(cfg: StageChunkGalleryConfig):
    init_logging()
    return run_stage_chunk_gallery(cfg)


def main():
    register_third_party_plugins()
    stage_chunk_gallery()


if __name__ == "__main__":
    main()

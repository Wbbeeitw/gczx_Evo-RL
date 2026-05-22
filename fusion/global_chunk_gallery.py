#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.stage_chunk_mining.annotation_io import column_to_1d_array
from lerobot.rl.stage_chunk_mining.visualize import (
    _compose_episode_figure,
    output_tag_from_prefix,
    select_camera_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate episode gallery figures for globally mined chunks while allowing "
            "human intervention frames to remain positive in the final indicator."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="",
        help="Repo id passed to LeRobotDataset. Defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--value-field",
        type=str,
        default="complementary_info.value",
        help="Frame-level value field.",
    )
    parser.add_argument(
        "--chunk-advantage-field",
        type=str,
        default="complementary_info.global_chunk_advantage",
        help="Chunk-start advantage field.",
    )
    parser.add_argument(
        "--chunk-start-indicator-field",
        type=str,
        default="complementary_info.global_chunk_start_indicator",
        help="Chunk-start indicator field.",
    )
    parser.add_argument(
        "--indicator-field",
        type=str,
        default="complementary_info.acp_indicator",
        help="Final frame-level indicator field.",
    )
    parser.add_argument(
        "--intervention-field",
        type=str,
        default="complementary_info.is_intervention",
        help="Human intervention field. Frames > 0.5 are treated as forced positive.",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size K.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="global_chunk",
        help="Tag used in figure titles and file names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNGs and report JSON will be written.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=None,
        help="Comma-separated camera keys. Defaults to the first available cameras.",
    )
    parser.add_argument("--max-cameras", type=int, default=3, help="Maximum cameras to render when auto-selecting.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing PNGs.")
    parser.add_argument(
        "--strict-verification",
        action="store_true",
        default=True,
        help="Raise if any episode fails the chunk/intervention consistency check.",
    )
    return parser.parse_args()


def verify_episode_chunk_mask_with_intervention(
    *,
    ep_chunk_start_indicator: np.ndarray,
    ep_indicator: np.ndarray,
    ep_intervention: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    ep_chunk_start_indicator = np.asarray(ep_chunk_start_indicator, dtype=np.int64).reshape(-1)
    ep_indicator = np.asarray(ep_indicator, dtype=np.int64).reshape(-1)
    ep_intervention = (np.asarray(ep_intervention, dtype=np.float32).reshape(-1) > 0.5).astype(np.int64)

    starts = np.flatnonzero(ep_chunk_start_indicator > 0)
    expected_chunk_mask = np.zeros_like(ep_indicator, dtype=np.int64)
    valid_start_limit = max(0, ep_indicator.shape[0] - int(chunk_size) + 1)
    invalid_start_positions = [int(start) for start in starts if int(start) >= valid_start_limit]

    for start in starts:
        start = int(start)
        if start >= valid_start_limit:
            continue
        end = min(start + int(chunk_size), ep_indicator.shape[0])
        expected_chunk_mask[start:end] = 1

    expected_indicator = np.maximum(expected_chunk_mask, ep_intervention)
    mismatch_positions = np.flatnonzero(expected_indicator != ep_indicator)
    invalid_indicator_values = [int(v) for v in np.unique(ep_indicator) if int(v) not in {0, 1}]
    invalid_start_values = [int(v) for v in np.unique(ep_chunk_start_indicator) if int(v) not in {0, 1}]

    passed = (
        len(invalid_indicator_values) == 0
        and len(invalid_start_values) == 0
        and len(invalid_start_positions) == 0
        and mismatch_positions.size == 0
    )
    return {
        "pass": bool(passed),
        "selected_chunk_count": int(starts.size),
        "expected_chunk_positive_frames": int(np.sum(expected_chunk_mask)),
        "expected_intervention_positive_frames": int(np.sum(ep_intervention)),
        "expected_final_positive_frames": int(np.sum(expected_indicator)),
        "actual_positive_frames": int(np.sum(ep_indicator)),
        "indicator_mismatch_count": int(mismatch_positions.size),
        "indicator_mismatch_positions": [int(pos) for pos in mismatch_positions[:64]],
        "invalid_indicator_values": invalid_indicator_values,
        "invalid_chunk_start_values": invalid_start_values,
        "invalid_start_positions": invalid_start_positions,
    }


def verify_chunk_mask_consistency_with_intervention(
    *,
    episode_indices: np.ndarray,
    chunk_start_indicator: np.ndarray,
    indicator: np.ndarray,
    intervention: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    chunk_start_indicator = np.asarray(chunk_start_indicator, dtype=np.int64).reshape(-1)
    indicator = np.asarray(indicator, dtype=np.int64).reshape(-1)
    intervention = np.asarray(intervention, dtype=np.float32).reshape(-1)

    per_episode: dict[str, Any] = {}
    failed_episodes: list[int] = []

    for episode_index in sorted(int(v) for v in np.unique(episode_indices)):
        positions = np.flatnonzero(episode_indices == episode_index)
        episode_report = verify_episode_chunk_mask_with_intervention(
            ep_chunk_start_indicator=chunk_start_indicator[positions],
            ep_indicator=indicator[positions],
            ep_intervention=intervention[positions],
            chunk_size=chunk_size,
        )
        per_episode[str(episode_index)] = episode_report
        if not bool(episode_report["pass"]):
            failed_episodes.append(int(episode_index))

    return {
        "pass": len(failed_episodes) == 0,
        "episodes_checked": len(per_episode),
        "failed_episodes": failed_episodes,
        "per_episode": per_episode,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    repo_id = args.repo_id if args.repo_id else dataset_root.name
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs/global_chunk_gallery") / output_tag_from_prefix(args.output_prefix)
    output_dir = output_dir.resolve()

    if args.chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    if args.max_cameras < 0:
        raise ValueError("'max_cameras' must be >= 0.")

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, download_videos=True)
    raw_frames = dataset.hf_dataset.with_format(None)

    required_fields = [
        "episode_index",
        "frame_index",
        args.value_field,
        args.chunk_advantage_field,
        args.chunk_start_indicator_field,
        args.indicator_field,
    ]
    missing_fields = [field for field in required_fields if field not in raw_frames.column_names]
    if missing_fields:
        raise KeyError(f"Missing fields for global chunk gallery: {missing_fields}")

    episode_indices = column_to_1d_array(raw_frames["episode_index"], np.int64)
    frame_indices = column_to_1d_array(raw_frames["frame_index"], np.int64)
    values = column_to_1d_array(raw_frames[args.value_field], np.float32)
    chunk_advantage = column_to_1d_array(raw_frames[args.chunk_advantage_field], np.float32)
    chunk_start_indicator = column_to_1d_array(raw_frames[args.chunk_start_indicator_field], np.int64)
    indicator = column_to_1d_array(raw_frames[args.indicator_field], np.int64)

    if args.intervention_field in raw_frames.column_names:
        intervention = column_to_1d_array(raw_frames[args.intervention_field], np.float32)
    else:
        intervention = np.zeros_like(indicator, dtype=np.float32)

    camera_keys = select_camera_keys(
        available_camera_keys=list(dataset.meta.camera_keys),
        requested_camera_keys=args.camera_keys,
        max_cameras=args.max_cameras,
    )
    verification = verify_chunk_mask_consistency_with_intervention(
        episode_indices=episode_indices,
        chunk_start_indicator=chunk_start_indicator,
        indicator=indicator,
        intervention=intervention,
        chunk_size=args.chunk_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_episodes = sorted(int(v) for v in np.unique(episode_indices))
    pad_width = max(3, len(str(max(unique_episodes) if unique_episodes else 0)))
    figure_tag = output_tag_from_prefix(args.output_prefix)
    figure_paths: list[str] = []

    for episode_index in unique_episodes:
        positions = np.flatnonzero(episode_indices == episode_index)
        if positions.size == 0:
            continue
        sort_order = np.argsort(frame_indices[positions], kind="stable")
        positions = positions[sort_order]
        output_path = output_dir / f"episode_{episode_index:0{pad_width}d}_{figure_tag}_chunk_gallery.png"
        if not output_path.exists() or args.overwrite:
            _compose_episode_figure(
                dataset=dataset,
                episode_index=episode_index,
                output_prefix=args.output_prefix,
                episode_positions=positions,
                ep_frame_indices=frame_indices[positions],
                ep_values=values[positions],
                ep_chunk_advantage=chunk_advantage[positions],
                ep_chunk_start_indicator=chunk_start_indicator[positions],
                ep_indicator=indicator[positions],
                chunk_size=args.chunk_size,
                camera_keys=camera_keys,
                output_path=output_path,
                episode_verification=verification["per_episode"][str(episode_index)],
            )
        figure_paths.append(str(output_path))

    report = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "value_field": args.value_field,
        "chunk_advantage_field": args.chunk_advantage_field,
        "chunk_start_indicator_field": args.chunk_start_indicator_field,
        "indicator_field": args.indicator_field,
        "intervention_field": args.intervention_field,
        "chunk_size": int(args.chunk_size),
        "camera_keys": camera_keys,
        "figure_dir": str(output_dir),
        "figure_count": len(figure_paths),
        "figure_paths": figure_paths,
        "verification": verification,
    }
    report_path = output_dir / "global_chunk_gallery_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, ensure_ascii=False))

    if args.strict_verification and not bool(verification["pass"]):
        failed = verification.get("failed_episodes", [])
        raise RuntimeError(f"Global chunk verification failed for episodes: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

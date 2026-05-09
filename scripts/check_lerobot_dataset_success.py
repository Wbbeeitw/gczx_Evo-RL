#!/usr/bin/env python
"""Check a local LeRobot v3 dataset and optionally label all episodes success."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUCCESS_LABEL = "success"


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def _shape(feature: dict[str, Any]) -> tuple[int, ...]:
    raw = feature.get("shape", [])
    return tuple(int(v) for v in raw)


def _stack_vector_column(df: pd.DataFrame, column: str, expected_dim: int, errors: list[str]) -> np.ndarray | None:
    if column not in df.columns:
        _fail(errors, f"missing data column: {column}")
        return None

    bad_rows = []
    arrays = []
    for i, value in enumerate(df[column].tolist()):
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (expected_dim,):
            bad_rows.append((i, arr.shape))
            if len(bad_rows) >= 5:
                break
        arrays.append(arr)

    if bad_rows:
        _fail(errors, f"{column} has rows with wrong shape; expected ({expected_dim},), examples={bad_rows}")
        return None

    stacked = np.stack(arrays, axis=0) if arrays else np.empty((0, expected_dim), dtype=np.float64)
    if not np.isfinite(stacked).all():
        _fail(errors, f"{column} contains NaN or inf values")
    return stacked


def _load_dataframes(dataset_root: Path, errors: list[str]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))

    if not data_files:
        _fail(errors, "no data parquet files found under data/chunk-*/file-*.parquet")
        data_df = None
    else:
        data_df = pd.concat([pd.read_parquet(path) for path in data_files], ignore_index=True)

    if not episode_files:
        _fail(errors, "no episode parquet files found under meta/episodes/chunk-*/file-*.parquet")
        episodes_df = None
    elif len(episode_files) != 1:
        _fail(errors, f"expected one episode metadata parquet for this script, found {len(episode_files)}")
        episodes_df = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    else:
        episodes_df = pd.read_parquet(episode_files[0])

    return data_df, episodes_df


def _check_episode_table(
    info: dict[str, Any],
    data_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    errors: list[str],
) -> None:
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))

    if len(episodes_df) != total_episodes:
        _fail(errors, f"episode table row count {len(episodes_df)} != info.total_episodes {total_episodes}")
    if len(data_df) != total_frames:
        _fail(errors, f"data row count {len(data_df)} != info.total_frames {total_frames}")

    required_episode_cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks"]
    for col in required_episode_cols:
        if col not in episodes_df.columns:
            _fail(errors, f"missing episode metadata column: {col}")
    required_data_cols = ["episode_index", "frame_index", "index", "timestamp", "task_index"]
    for col in required_data_cols:
        if col not in data_df.columns:
            _fail(errors, f"missing data column: {col}")
    if errors:
        return

    episode_indices = episodes_df["episode_index"].astype(int).tolist()
    expected_indices = list(range(total_episodes))
    if episode_indices != expected_indices:
        _fail(errors, f"episode_index is not contiguous 0..N-1: {episode_indices}")

    data_episode_indices = sorted(int(v) for v in data_df["episode_index"].unique().tolist())
    if data_episode_indices != expected_indices:
        _fail(errors, f"data episode_index values are not contiguous 0..N-1: {data_episode_indices}")

    if data_df["index"].astype(int).tolist() != list(range(len(data_df))):
        _fail(errors, "global data index column is not contiguous 0..total_frames-1")

    for _, ep in episodes_df.sort_values("episode_index").iterrows():
        ep_idx = int(ep["episode_index"])
        start = int(ep["dataset_from_index"])
        stop = int(ep["dataset_to_index"])
        length = int(ep["length"])
        if stop - start != length:
            _fail(errors, f"episode {ep_idx}: dataset_to_index - dataset_from_index != length")
            continue
        if start < 0 or stop > len(data_df) or start >= stop:
            _fail(errors, f"episode {ep_idx}: invalid dataset range [{start}, {stop})")
            continue

        part = data_df.iloc[start:stop]
        if not (part["episode_index"].astype(int).to_numpy() == ep_idx).all():
            _fail(errors, f"episode {ep_idx}: data range contains another episode_index")
        frame_index = part["frame_index"].astype(int).to_numpy()
        if not np.array_equal(frame_index, np.arange(length)):
            _fail(errors, f"episode {ep_idx}: frame_index is not 0..length-1")
        timestamps = part["timestamp"].astype(float).to_numpy()
        if len(timestamps) and not math.isclose(float(timestamps[0]), 0.0, abs_tol=1e-4):
            _fail(errors, f"episode {ep_idx}: first timestamp is not 0")
        if np.any(np.diff(timestamps) < -1e-5):
            _fail(errors, f"episode {ep_idx}: timestamp is not monotonic")


def _check_tasks(dataset_root: Path, data_df: pd.DataFrame, errors: list[str]) -> None:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        _fail(errors, "missing meta/tasks.parquet")
        return

    tasks_df = pd.read_parquet(tasks_path)
    if len(tasks_df) == 0:
        _fail(errors, "meta/tasks.parquet is empty")
        return

    valid_task_indices = set(int(v) for v in tasks_df["task_index"].tolist())
    data_task_indices = set(int(v) for v in data_df["task_index"].unique().tolist())
    missing = data_task_indices - valid_task_indices
    if missing:
        _fail(errors, f"data task_index contains values not present in meta/tasks.parquet: {sorted(missing)}")


def _ffprobe_video(path: Path) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    return streams[0] if streams else {"error": "ffprobe returned no video streams"}


def _parse_rate(rate: str | None) -> float | None:
    if not rate:
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else None
    return float(rate)


def _check_videos(
    dataset_root: Path,
    info: dict[str, Any],
    expected_fps: int | None,
    warnings: list[str],
    errors: list[str],
) -> None:
    features = info.get("features", {})
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    if not video_keys:
        _warn(warnings, "no video features found")
        return

    for key in video_keys:
        feature = features[key]
        expected_shape = _shape(feature)
        files = sorted((dataset_root / "videos" / key).glob("chunk-*/file-*.mp4"))
        if not files:
            _fail(errors, f"missing video file for {key}")
            continue
        for path in files:
            if path.stat().st_size <= 0:
                _fail(errors, f"empty video file: {path.relative_to(dataset_root)}")
                continue
            probe = _ffprobe_video(path)
            if probe is None:
                _warn(warnings, "ffprobe not found; skipped video stream probing")
                return
            if "error" in probe:
                _fail(errors, f"ffprobe failed for {path.relative_to(dataset_root)}: {probe['error']}")
                continue
            width = int(probe.get("width", -1))
            height = int(probe.get("height", -1))
            if len(expected_shape) >= 2 and (height, width) != (expected_shape[0], expected_shape[1]):
                _fail(
                    errors,
                    f"{path.relative_to(dataset_root)} resolution {(height, width)} "
                    f"!= expected {(expected_shape[0], expected_shape[1])}",
                )
            fps = _parse_rate(probe.get("avg_frame_rate"))
            if expected_fps and fps is not None and not math.isclose(fps, float(expected_fps), rel_tol=0.03):
                _warn(warnings, f"{path.relative_to(dataset_root)} fps {fps:.3f} differs from expected {expected_fps}")


def check_dataset(
    dataset_root: Path,
    expected_episodes: int | None,
    expected_action_dim: int,
    expected_fps: int | None,
    probe_videos: bool,
) -> tuple[list[str], list[str], dict[str, Any], pd.DataFrame | None]:
    errors: list[str] = []
    warnings: list[str] = []

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return [f"missing {info_path}"], warnings, {}, None
    info = json.loads(info_path.read_text())

    if expected_episodes is not None and int(info.get("total_episodes", -1)) != expected_episodes:
        _fail(errors, f"info.total_episodes {info.get('total_episodes')} != expected {expected_episodes}")
    if expected_fps is not None and int(info.get("fps", -1)) != expected_fps:
        _fail(errors, f"info.fps {info.get('fps')} != expected {expected_fps}")

    features = info.get("features", {})
    for key in ("action", "observation.state"):
        if key not in features:
            _fail(errors, f"missing feature: {key}")
        elif _shape(features[key]) != (expected_action_dim,):
            _fail(errors, f"feature {key} shape {_shape(features[key])} != ({expected_action_dim},)")

    data_df, episodes_df = _load_dataframes(dataset_root, errors)
    if data_df is None or episodes_df is None:
        return errors, warnings, info, episodes_df

    _check_episode_table(info, data_df, episodes_df, errors)
    _check_tasks(dataset_root, data_df, errors)
    _stack_vector_column(data_df, "action", expected_action_dim, errors)
    _stack_vector_column(data_df, "observation.state", expected_action_dim, errors)

    if "episode_success" not in episodes_df.columns:
        _warn(warnings, "episode_success column is missing")
    else:
        labels = sorted(set(str(v) for v in episodes_df["episode_success"].tolist()))
        if labels != [SUCCESS_LABEL]:
            _warn(warnings, f"episode_success labels are not all success: {labels}")

    if probe_videos:
        _check_videos(dataset_root, info, expected_fps, warnings, errors)

    return errors, warnings, info, episodes_df


def mark_all_success(dataset_root: Path, episodes_df: pd.DataFrame, backup: bool) -> Path:
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if len(episode_files) != 1:
        raise RuntimeError(f"marking success supports one episode parquet, found {len(episode_files)}")

    episode_path = episode_files[0]
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = episode_path.with_suffix(episode_path.suffix + f".bak_{stamp}")
        shutil.copy2(episode_path, backup_path)

    out = episodes_df.copy()
    out["episode_success"] = SUCCESS_LABEL
    out.to_parquet(episode_path, index=False)
    return episode_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=14)
    parser.add_argument("--expected-fps", type=int, default=30)
    parser.add_argument("--mark-success", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--skip-video-probe", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    errors, warnings, info, episodes_df = check_dataset(
        dataset_root=root,
        expected_episodes=args.expected_episodes,
        expected_action_dim=args.expected_action_dim,
        expected_fps=args.expected_fps,
        probe_videos=not args.skip_video_probe,
    )

    if errors:
        print("DATASET CHECK FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        raise SystemExit(1)

    if args.mark_success:
        if episodes_df is None:
            raise RuntimeError("cannot mark success because episodes metadata was not loaded")
        episode_path = mark_all_success(root, episodes_df, backup=not args.no_backup)
        errors, warnings, info, episodes_df = check_dataset(
            dataset_root=root,
            expected_episodes=args.expected_episodes,
            expected_action_dim=args.expected_action_dim,
            expected_fps=args.expected_fps,
            probe_videos=not args.skip_video_probe,
        )
        if errors:
            print("DATASET CHECK FAILED AFTER MARKING SUCCESS")
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
        print(f"marked_success_file={episode_path}")

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    fps = int(info.get("fps", 0))
    success_count = 0
    if episodes_df is not None and "episode_success" in episodes_df.columns:
        success_count = sum(str(v) == SUCCESS_LABEL for v in episodes_df["episode_success"].tolist())

    print("DATASET CHECK OK")
    print(f"dataset_root={root}")
    print(f"episodes={total_episodes}")
    print(f"frames={total_frames}")
    print(f"fps={fps}")
    print(f"success_labels={success_count}/{total_episodes}")
    if episodes_df is not None:
        print("episode_lengths=" + ",".join(str(int(v)) for v in episodes_df["length"].tolist()))
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
Create a fake tactile LeRobot dataset by copying an existing dataset and adding
four tactile video streams.

This script is intentionally isolated under ``new/`` and does not modify the
original project code. It is designed for the user's current bring-up need:
reuse an existing RGB-only dataset, synthesize four tactile streams with the
same placeholder frame repeated for every timestep, and patch the dataset
metadata so the training loop can load the tactile keys.

Default workflow:
1. Copy ``/home/enine/SACM/lerobot_dataset/5_9`` to
   ``/home/enine/SACM/lerobot_dataset/tac``.
2. Create these new video keys under ``videos/``:
   - observation.images.tactile_left_outer
   - observation.images.tactile_left_inner
   - observation.images.tactile_right_outer
   - observation.images.tactile_right_inner
3. Each tactile stream becomes one full-length ``chunk-000/file-000.mp4``
   matching the source dataset's ``total_frames`` and ``fps``.
4. Patch ``meta/info.json``, ``meta/stats.json`` and
   ``meta/episodes/chunk-000/file-000.parquet`` to register the new streams.

Notes:
- The tactile videos are fake bring-up data only.
- To minimize schema surprises, image stats are copied from the existing
  ``observation.images.left_top`` stream rather than recomputed.
- The script supports a custom placeholder image via ``--placeholder-image``.
  If omitted, it generates a synthetic tactile-like heatmap.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TACTILE_KEYS = [
    "observation.images.tactile_left_outer",
    "observation.images.tactile_left_inner",
    "observation.images.tactile_right_outer",
    "observation.images.tactile_right_inner",
]

REFERENCE_VIDEO_KEY = "observation.images.left_top"
VIDEO_INSERT_ANCHOR_KEY = "observation.images.right_wrist"
VIDEO_STATS_FIELDS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a LeRobot dataset and add four fake tactile video streams.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/enine/SACM/lerobot_dataset/5_9"),
        help="Source LeRobot dataset path.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/enine/SACM/lerobot_dataset/tac"),
        help="Destination LeRobot dataset path to create.",
    )
    parser.add_argument(
        "--placeholder-image",
        type=Path,
        default=None,
        help="Optional image file used as the repeated tactile placeholder frame.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination dataset if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def validate_paths(src: Path, dst: Path, force: bool) -> tuple[Path, Path]:
    src = src.resolve()
    dst = dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src}")
    if src == dst:
        raise ValueError("--src and --dst must be different directories.")
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot dataset root: {src}")

    if dst.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {dst}. Use --force to overwrite it.")
        if not dst.is_dir():
            raise ValueError(f"Destination exists but is not a directory: {dst}")
        # Guard against deleting an arbitrary high-level directory by mistake.
        if len(dst.parts) < 4:
            raise ValueError(f"Refusing to remove an unexpectedly shallow path: {dst}")
        print(f"[1/6] Removing existing destination: {dst}")
        shutil.rmtree(dst)

    return src, dst


def copy_dataset_tree(src: Path, dst: Path) -> None:
    print(f"[2/6] Copying dataset tree:\n  src={src}\n  dst={dst}")
    shutil.copytree(src, dst)


def ensure_tactile_keys_absent(info: dict[str, Any]) -> None:
    features = info.get("features", {})
    present = [key for key in TACTILE_KEYS if key in features]
    if present:
        raise ValueError(f"Source dataset already contains tactile keys: {present}")


def build_synthetic_placeholder(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0

    left_blob = np.exp(-(((x + 0.28) ** 2) / 0.05 + ((y - 0.05) ** 2) / 0.08))
    right_blob = np.exp(-(((x - 0.24) ** 2) / 0.06 + ((y + 0.12) ** 2) / 0.06))
    center_band = np.exp(-((y + 0.02) ** 2) / 0.22) * np.exp(-(x**2) / 0.75)
    background = 0.12 + 0.10 * (1.0 - np.sqrt(np.clip(x**2 + y**2, 0.0, 1.0)))

    red = np.clip(background + 0.80 * left_blob + 0.25 * center_band, 0.0, 1.0)
    green = np.clip(background + 0.55 * right_blob + 0.35 * center_band, 0.0, 1.0)
    blue = np.clip(background + 0.18 * left_blob + 0.18 * right_blob + 0.60 * center_band, 0.0, 1.0)

    image = np.stack([red, green, blue], axis=-1)
    return np.round(image * 255.0).astype(np.uint8)


def load_placeholder_image(image_path: Path | None, height: int, width: int) -> np.ndarray:
    if image_path is None:
        print("[3/6] Using built-in synthetic tactile placeholder image.")
        return build_synthetic_placeholder(height, width)

    if not image_path.exists():
        raise FileNotFoundError(f"Placeholder image does not exist: {image_path}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required when --placeholder-image is used, but it is not installed."
        ) from exc

    print(f"[3/6] Loading placeholder image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def try_write_video_with_cv2(path: Path, frame_rgb: np.ndarray, fps: int, num_frames: int) -> dict[str, Any] | None:
    try:
        import cv2
    except ImportError:
        return None

    height, width = frame_rgb.shape[:2]
    frame_bgr = frame_rgb[:, :, ::-1]

    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            continue

        for _ in range(num_frames):
            writer.write(frame_bgr)
        writer.release()

        if path.exists() and path.stat().st_size > 0:
            return {
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.height": height,
                "video.width": width,
                "video.fps": fps,
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            }

    return None


def try_write_video_with_imageio(path: Path, frame_rgb: np.ndarray, fps: int, num_frames: int) -> dict[str, Any] | None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    height, width = frame_rgb.shape[:2]
    writer = None
    try:
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        for _ in range(num_frames):
            writer.append_data(frame_rgb)
    except Exception:
        if writer is not None:
            writer.close()
        if path.exists():
            path.unlink()
        return None
    else:
        writer.close()

    if path.exists() and path.stat().st_size > 0:
        return {
            "video.codec": "libx264",
            "video.pix_fmt": "yuv420p",
            "video.height": height,
            "video.width": width,
            "video.fps": fps,
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        }
    return None


def try_write_video_with_ffmpeg(path: Path, frame_rgb: np.ndarray, fps: int, num_frames: int) -> dict[str, Any] | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    height, width = frame_rgb.shape[:2]
    frame_bytes = frame_rgb.tobytes()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for _ in range(num_frames):
            proc.stdin.write(frame_bytes)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr is not None else ""
        return_code = proc.wait()
        if return_code != 0:
            if path.exists():
                path.unlink()
            raise RuntimeError(stderr.strip() or f"ffmpeg exited with code {return_code}")
    except Exception:
        return None

    if path.exists() and path.stat().st_size > 0:
        return {
            "video.codec": "libx264",
            "video.pix_fmt": "yuv420p",
            "video.height": height,
            "video.width": width,
            "video.fps": fps,
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        }
    return None


def write_repeated_frame_video(path: Path, frame_rgb: np.ndarray, fps: int, num_frames: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)

    for writer in (try_write_video_with_cv2, try_write_video_with_imageio, try_write_video_with_ffmpeg):
        metadata = writer(path, frame_rgb, fps, num_frames)
        if metadata is not None:
            return metadata

    raise RuntimeError(
        "Failed to create tactile mp4. None of the supported backends worked: cv2, imageio, ffmpeg."
    )


def compute_total_video_size_mb(videos_root: Path) -> int:
    total_bytes = sum(path.stat().st_size for path in videos_root.rglob("*.mp4"))
    if total_bytes <= 0:
        return 0
    return int(math.ceil(total_bytes / (1024 * 1024)))


def insert_mapping_after(
    mapping: dict[str, Any],
    after_key: str,
    additions: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    inserted = False

    for key, value in mapping.items():
        result[key] = value
        if key == after_key:
            for add_key, add_value in additions.items():
                result[add_key] = add_value
            inserted = True

    if not inserted:
        for add_key, add_value in additions.items():
            result[add_key] = add_value

    return result


def build_tactile_feature_entry(reference_entry: dict[str, Any], video_info: dict[str, Any]) -> dict[str, Any]:
    entry = copy.deepcopy(reference_entry)
    entry["dtype"] = "video"
    entry["shape"] = [video_info["video.height"], video_info["video.width"], video_info["video.channels"]]
    entry["names"] = ["height", "width", "channels"]
    entry["info"] = {
        "video.height": video_info["video.height"],
        "video.width": video_info["video.width"],
        "video.codec": video_info["video.codec"],
        "video.pix_fmt": video_info["video.pix_fmt"],
        "video.is_depth_map": False,
        "video.fps": video_info["video.fps"],
        "video.channels": video_info["video.channels"],
        "has_audio": False,
    }
    return entry


def patch_info_json(info_path: Path, tactile_video_info: dict[str, Any]) -> None:
    info = load_json(info_path)
    ensure_tactile_keys_absent(info)

    features = info["features"]
    reference_feature = features[REFERENCE_VIDEO_KEY]
    tactile_features = {
        tactile_key: build_tactile_feature_entry(reference_feature, tactile_video_info)
        for tactile_key in TACTILE_KEYS
    }
    info["features"] = insert_mapping_after(features, VIDEO_INSERT_ANCHOR_KEY, tactile_features)
    info["video_files_size_in_mb"] = compute_total_video_size_mb(info_path.parent.parent / "videos")

    dump_json(info_path, info)


def patch_stats_json(stats_path: Path) -> None:
    stats = load_json(stats_path)
    if any(key in stats for key in TACTILE_KEYS):
        raise ValueError("Tactile stats already exist in stats.json.")

    reference_stats = stats[REFERENCE_VIDEO_KEY]
    tactile_stats = {tactile_key: copy.deepcopy(reference_stats) for tactile_key in TACTILE_KEYS}
    stats = insert_mapping_after(stats, VIDEO_INSERT_ANCHOR_KEY, tactile_stats)
    dump_json(stats_path, stats)


def video_columns(video_key: str) -> list[str]:
    return [
        f"videos/{video_key}/chunk_index",
        f"videos/{video_key}/file_index",
        f"videos/{video_key}/from_timestamp",
        f"videos/{video_key}/to_timestamp",
    ]


def stats_columns(feature_key: str) -> list[str]:
    return [f"stats/{feature_key}/{field}" for field in VIDEO_STATS_FIELDS]


def deep_copy_series(series: pd.Series) -> pd.Series:
    return series.apply(copy.deepcopy)


def patch_episodes_parquet(episodes_path: Path) -> None:
    df = pd.read_parquet(episodes_path)
    original_columns = list(df.columns)

    reference_video_cols = video_columns(REFERENCE_VIDEO_KEY)
    for col in reference_video_cols:
        if col not in df.columns:
            raise KeyError(f"Missing reference video column in episodes parquet: {col}")

    reference_stats_cols = stats_columns(REFERENCE_VIDEO_KEY)
    for col in reference_stats_cols:
        if col not in df.columns:
            raise KeyError(f"Missing reference image stats column in episodes parquet: {col}")

    for tactile_key in TACTILE_KEYS:
        for src_col, dst_col in zip(reference_video_cols, video_columns(tactile_key), strict=True):
            df[dst_col] = df[src_col].copy()

        for src_col, dst_col in zip(reference_stats_cols, stats_columns(tactile_key), strict=True):
            df[dst_col] = deep_copy_series(df[src_col])

    tactile_video_cols: list[str] = []
    for tactile_key in TACTILE_KEYS:
        tactile_video_cols.extend(video_columns(tactile_key))

    tactile_stats_cols: list[str] = []
    for tactile_key in TACTILE_KEYS:
        tactile_stats_cols.extend(stats_columns(tactile_key))

    reordered: list[str] = []
    video_anchor = f"videos/{VIDEO_INSERT_ANCHOR_KEY}/to_timestamp"
    stats_anchor = f"stats/{VIDEO_INSERT_ANCHOR_KEY}/q99"

    for col in original_columns:
        reordered.append(col)
        if col == video_anchor:
            reordered.extend(tactile_video_cols)
        if col == stats_anchor:
            reordered.extend(tactile_stats_cols)

    for col in df.columns:
        if col not in reordered:
            reordered.append(col)

    df = df[reordered]
    df.to_parquet(episodes_path, index=False)


def create_tactile_videos(
    dst_root: Path,
    total_frames: int,
    fps: int,
    frame_rgb: np.ndarray,
) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None

    print("[4/6] Writing fake tactile mp4 streams.")
    for tactile_key in TACTILE_KEYS:
        video_path = dst_root / "videos" / tactile_key / "chunk-000" / "file-000.mp4"
        print(f"  - {tactile_key} -> {video_path}")
        current_metadata = write_repeated_frame_video(video_path, frame_rgb, fps=fps, num_frames=total_frames)
        if metadata is None:
            metadata = current_metadata

    assert metadata is not None
    return metadata


def main() -> int:
    args = parse_args()
    src, dst = validate_paths(args.src, args.dst, args.force)

    info = load_json(src / "meta" / "info.json")
    ensure_tactile_keys_absent(info)

    reference_feature = info["features"].get(REFERENCE_VIDEO_KEY)
    if reference_feature is None:
        raise KeyError(f"Reference feature not found in source dataset: {REFERENCE_VIDEO_KEY}")

    height, width, channels = reference_feature["shape"]
    if channels != 3:
        raise ValueError(f"Expected 3-channel RGB reference stream, got shape={reference_feature['shape']}")

    total_frames = int(info["total_frames"])
    fps = int(info["fps"])

    print(f"[1/6] Source dataset looks valid. total_frames={total_frames}, fps={fps}, shape={height}x{width}x{channels}")
    copy_dataset_tree(src, dst)

    frame_rgb = load_placeholder_image(args.placeholder_image, height=height, width=width)
    tactile_video_info = create_tactile_videos(dst, total_frames=total_frames, fps=fps, frame_rgb=frame_rgb)

    print("[5/6] Patching metadata files.")
    patch_info_json(dst / "meta" / "info.json", tactile_video_info)
    patch_stats_json(dst / "meta" / "stats.json")
    patch_episodes_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    print("[6/6] Done.")
    print(f"Created tactile dataset at: {dst}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise

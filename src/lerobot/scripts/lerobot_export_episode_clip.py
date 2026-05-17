#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a multiview MP4 clip from a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo_id.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to export.")
    parser.add_argument("--frame-start", type=int, required=True, help="Episode-relative inclusive start frame.")
    parser.add_argument("--frame-end", type=int, required=True, help="Episode-relative inclusive end frame.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--camera-keys",
        type=str,
        default="observation.images.left_top,observation.images.left_wrist,observation.images.right_wrist",
        help="Comma-separated camera keys to include. Empty means dataset camera order.",
    )
    parser.add_argument("--video-backend", type=str, default="torchcodec", help="Video decoding backend.")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS. Defaults to dataset FPS.")
    return parser.parse_args()


def to_bgr_image(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 image tensor, got shape={arr.shape}.")

    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def overlay_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (8, 8), (340, 40), (18, 18, 18), thickness=-1)
    cv2.putText(out, label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )

    if args.episode_index < 0 or args.episode_index >= dataset.meta.total_episodes:
        raise IndexError(
            f"episode_index={args.episode_index} out of range for total_episodes={dataset.meta.total_episodes}"
        )

    ep = dataset.meta.episodes[args.episode_index]
    ep_length = int(ep["length"])
    if args.frame_start < 0 or args.frame_end < args.frame_start or args.frame_end >= ep_length:
        raise ValueError(
            f"Requested frame range [{args.frame_start}, {args.frame_end}] is invalid for episode length={ep_length}."
        )

    camera_keys = [key.strip() for key in args.camera_keys.split(",") if key.strip()]
    if not camera_keys:
        camera_keys = list(dataset.meta.camera_keys)

    missing_camera_keys = [key for key in camera_keys if key not in dataset.meta.camera_keys]
    if missing_camera_keys:
        raise KeyError(f"Missing camera keys in dataset: {missing_camera_keys}")

    abs_start = int(ep["dataset_from_index"]) + args.frame_start
    abs_end = int(ep["dataset_from_index"]) + args.frame_end
    fps = float(args.fps if args.fps is not None else dataset.fps)

    first_item = dataset[abs_start]
    first_views = [overlay_label(to_bgr_image(first_item[key]), key) for key in camera_keys]
    frame_height = max(view.shape[0] for view in first_views)
    resized_first = [
        cv2.resize(view, (int(round(view.shape[1] * frame_height / view.shape[0])), frame_height), interpolation=cv2.INTER_AREA)
        if view.shape[0] != frame_height
        else view
        for view in first_views
    ]
    first_canvas = np.concatenate(resized_first, axis=1)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first_canvas.shape[1], first_canvas.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {args.output_path}")

    try:
        for absolute_idx in range(abs_start, abs_end + 1):
            rel_frame = absolute_idx - int(ep["dataset_from_index"])
            item = dataset[absolute_idx]
            views = []
            for key in camera_keys:
                view = overlay_label(to_bgr_image(item[key]), f"{key} | ep={args.episode_index} frame={rel_frame}")
                if view.shape[0] != frame_height:
                    view = cv2.resize(
                        view,
                        (int(round(view.shape[1] * frame_height / view.shape[0])), frame_height),
                        interpolation=cv2.INTER_AREA,
                    )
                views.append(view)

            canvas = np.concatenate(views, axis=1)
            writer.write(canvas)
    finally:
        writer.release()

    print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()

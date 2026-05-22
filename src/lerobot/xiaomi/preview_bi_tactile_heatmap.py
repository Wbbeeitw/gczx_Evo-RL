#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _ensure_repo_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


_ensure_repo_src_on_path()

from lerobot.xiaomi.record_bi_piper_tactile import (
    TactileRuntimeBundle,
    TactileSideCamera,
    _load_tactile_classes,
    parse_bool,
    parse_optional_path,
)
from lerobot.xiaomi.tactile_defaults import (
    TACTILE_DEFAULT_CALIBRATE,
    TACTILE_DEFAULT_CALIBRATION_INTERVAL,
    TACTILE_DEFAULT_CALIBRATION_REDUCER,
    TACTILE_DEFAULT_CALIBRATION_SAMPLES,
    TACTILE_DEFAULT_CALIBRATION_WARMUP_FRAMES,
    TACTILE_DEFAULT_HEATMAP_COLORMAP,
    TACTILE_DEFAULT_HEATMAP_GAMMA,
    TACTILE_DEFAULT_HEATMAP_VMAX,
    TACTILE_DEFAULT_HEATMAP_VMIN,
    TACTILE_DEFAULT_MODE,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview left/right Xiaomi tactile heatmaps in real time without recording a dataset."
    )
    parser.add_argument("--xiaomi-xr0-root", type=parse_optional_path, default=None)
    parser.add_argument("--left-tactile-port", required=True, help="Left tactile controller serial port.")
    parser.add_argument("--right-tactile-port", required=True, help="Right tactile controller serial port.")
    parser.add_argument("--tactile.baudrate", "--tactile-baudrate", dest="tactile_baudrate", type=int, default=921600)
    parser.add_argument("--tactile.timeout", "--tactile-timeout", dest="tactile_timeout", type=float, default=1.0)
    parser.add_argument(
        "--tactile.mode",
        "--tactile-mode",
        dest="tactile_mode",
        choices=("auto_push", "distributed_poll"),
        default=TACTILE_DEFAULT_MODE,
    )
    parser.add_argument(
        "--tactile.distributed_scale",
        "--tactile-distributed-scale",
        dest="tactile_distributed_scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tactile.output_size",
        "--tactile-output-size",
        dest="tactile_output_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--tactile.calibrate",
        "--tactile-calibrate",
        dest="tactile_calibrate",
        type=parse_bool,
        default=TACTILE_DEFAULT_CALIBRATE,
    )
    parser.add_argument(
        "--tactile.calibration_samples",
        "--tactile-calibration-samples",
        dest="tactile_calibration_samples",
        type=int,
        default=TACTILE_DEFAULT_CALIBRATION_SAMPLES,
    )
    parser.add_argument(
        "--tactile.calibration_interval",
        "--tactile-calibration-interval",
        dest="tactile_calibration_interval",
        type=float,
        default=TACTILE_DEFAULT_CALIBRATION_INTERVAL,
    )
    parser.add_argument(
        "--tactile.calibration_warmup_frames",
        "--tactile-calibration-warmup-frames",
        dest="tactile_calibration_warmup_frames",
        type=int,
        default=TACTILE_DEFAULT_CALIBRATION_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--tactile.calibration_reducer",
        "--tactile-calibration-reducer",
        dest="tactile_calibration_reducer",
        choices=("median", "mean"),
        default=TACTILE_DEFAULT_CALIBRATION_REDUCER,
    )
    parser.add_argument(
        "--tactile.heatmap_vmin",
        "--tactile-heatmap-vmin",
        dest="tactile_heatmap_vmin",
        type=float,
        default=TACTILE_DEFAULT_HEATMAP_VMIN,
    )
    parser.add_argument(
        "--tactile.heatmap_vmax",
        "--tactile-heatmap-vmax",
        dest="tactile_heatmap_vmax",
        type=float,
        default=TACTILE_DEFAULT_HEATMAP_VMAX,
    )
    parser.add_argument(
        "--tactile.heatmap_colormap",
        "--tactile-heatmap-colormap",
        dest="tactile_heatmap_colormap",
        default=TACTILE_DEFAULT_HEATMAP_COLORMAP,
    )
    parser.add_argument(
        "--tactile.heatmap_gamma",
        "--tactile-heatmap-gamma",
        dest="tactile_heatmap_gamma",
        type=float,
        default=TACTILE_DEFAULT_HEATMAP_GAMMA,
    )
    parser.add_argument(
        "--tactile.poll_interval",
        "--tactile-poll-interval",
        dest="tactile_poll_interval",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tactile.log_level",
        "--tactile-log-level",
        dest="tactile_log_level",
        default="INFO",
    )
    parser.add_argument("--window-name", default="BiTactile Heatmap Preview")
    parser.add_argument("--show-fps", type=parse_bool, default=True)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N rendered frames. 0 means run forever.")
    return parser


def _make_tactile_camera(side_name: str, port: str, bundle: TactileRuntimeBundle, args: argparse.Namespace) -> TactileSideCamera:
    return TactileSideCamera(
        side_name=side_name,
        port=port,
        runtime_bundle=bundle,
        baudrate=args.tactile_baudrate,
        timeout=args.tactile_timeout,
        mode=args.tactile_mode,
        distributed_scale=args.tactile_distributed_scale,
        output_size=args.tactile_output_size,
        heatmap_vmin=args.tactile_heatmap_vmin,
        heatmap_vmax=args.tactile_heatmap_vmax,
        heatmap_colormap=args.tactile_heatmap_colormap,
        heatmap_gamma=args.tactile_heatmap_gamma,
        poll_interval=args.tactile_poll_interval,
        calibrate=args.tactile_calibrate,
        calibration_samples=args.tactile_calibration_samples,
        calibration_interval=args.tactile_calibration_interval,
        calibration_warmup_frames=args.tactile_calibration_warmup_frames,
        calibration_reducer=args.tactile_calibration_reducer,
        log_level=args.tactile_log_level,
    )


def _stack_preview(left_rgb: np.ndarray, right_rgb: np.ndarray, *, title: str, fps_text: str | None) -> np.ndarray:
    header_h = 48
    width = max(left_rgb.shape[1], right_rgb.shape[1])
    canvas = np.zeros((header_h + left_rgb.shape[0] + right_rgb.shape[0], width, 3), dtype=np.uint8)
    canvas[header_h : header_h + left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    canvas[header_h + left_rgb.shape[0] :, : right_rgb.shape[1]] = right_rgb

    cv2.putText(canvas, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, "Top: left tactile   Bottom: right tactile", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    if fps_text is not None:
        cv2.putText(canvas, fps_text, (width - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return canvas


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.tactile_log_level.upper(), logging.INFO))
    logger = logging.getLogger("bi-tactile-preview")

    tactile_runtime_cls, tactile_driver_cls, tactile_visualizer_cls = _load_tactile_classes(args.xiaomi_xr0_root)
    bundle = TactileRuntimeBundle(
        runtime_cls=tactile_runtime_cls,
        driver_cls=tactile_driver_cls,
        visualizer_cls=tactile_visualizer_cls,
    )
    left_camera = _make_tactile_camera("left", args.left_tactile_port, bundle, args)
    right_camera = _make_tactile_camera("right", args.right_tactile_port, bundle, args)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        args.window_name,
        int(args.tactile_output_size * 2 * 0.95),
        int((args.tactile_output_size * 2 + 48) * 0.95),
    )

    last_t = time.perf_counter()
    smoothed_fps = 0.0
    frame_count = 0
    try:
        left_camera.start()
        right_camera.start()
        logger.info("Press 'q' or ESC to exit tactile preview.")

        while args.max_frames <= 0 or frame_count < args.max_frames:
            left_rgb = left_camera.read_image()
            right_rgb = right_camera.read_image()

            now_t = time.perf_counter()
            dt = max(now_t - last_t, 1e-6)
            last_t = now_t
            fps = 1.0 / dt
            smoothed_fps = fps if smoothed_fps <= 0.0 else (0.9 * smoothed_fps + 0.1 * fps)
            fps_text = f"{smoothed_fps:5.1f} FPS" if args.show_fps else None

            canvas = _stack_preview(left_rgb, right_rgb, title=args.window_name, fps_text=fps_text)
            cv2.imshow(args.window_name, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        try:
            left_camera.stop()
        finally:
            right_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

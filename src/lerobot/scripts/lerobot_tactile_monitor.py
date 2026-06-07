#!/usr/bin/env python3
"""CLI tool for testing and monitoring tactile sensors.

Usage::

    lerobot-tactile-monitor --port /dev/ttyACM0 --calibrate
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from lerobot.cameras.tactile import TactileVisualizer
from lerobot.cameras.tactile.driver import TactileSensorDriver
from lerobot.cameras.tactile.runtime import TactileRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor tactile sensor readings in real time."
    )
    parser.add_argument(
        "--port", type=str, default="/dev/ttyACM0", help="Serial port of the tactile controller."
    )
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--mode", choices=("auto_push", "distributed_poll"), default="auto_push")
    parser.add_argument("--distributed-scale", type=float, default=0.1)
    parser.add_argument(
        "--output-size", type=int, default=256, help="Rendered panel size for each tactile view."
    )
    parser.add_argument(
        "--heatmap-vmin", type=float, default=0.0, help="Lower bound of the Fz heatmap range."
    )
    parser.add_argument(
        "--heatmap-vmax", type=float, default=25.5, help="Upper bound of the Fz heatmap range."
    )
    parser.add_argument(
        "--heatmap-colormap",
        choices=TactileVisualizer.available_colormaps(),
        default="turbo",
        help="Colormap used for the Fz heatmap.",
    )
    parser.add_argument(
        "--heatmap-gamma",
        type=float,
        default=0.75,
        help="Gamma applied after range normalisation.",
    )
    parser.add_argument(
        "--rgb-vmax-fz", type=float, default=25.5, help="Upper bound for RGB Fz channel."
    )
    parser.add_argument(
        "--rgb-vmax-shear", type=float, default=12.8, help="Upper bound for RGB Fx/Fy channels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save PNG/NPZ snapshots.",
    )
    parser.add_argument(
        "--save-every", type=int, default=0, help="Save every N frames. 0 disables."
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames. 0 = forever."
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Run zero-point calibration at startup."
    )
    parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--calibration-interval", type=float, default=0.05)
    parser.add_argument("--calibration-warmup-frames", type=int, default=20)
    parser.add_argument(
        "--calibration-reducer", choices=("median", "mean"), default="median"
    )
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def format_force(values) -> str:
    return f"({values[0]:7.3f}, {values[1]:7.3f}, {values[2]:7.3f})"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        force=True,
    )
    logger = logging.getLogger("tactile-monitor")
    print("Tactile monitor starting on %s ..." % args.port, flush=True)

    driver = TactileSensorDriver(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        enable_distributed=args.mode == "auto_push",
        distributed_scale=args.distributed_scale,
        logger=logger,
    )
    runtime = TactileRuntime(
        driver=driver,
        read_mode=args.mode,
        read_timeout=args.timeout,
        poll_interval=args.poll_interval,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        calibration_warmup_frames=args.calibration_warmup_frames,
        calibration_reducer=args.calibration_reducer,
        logger=logger,
    )
    visualizer = TactileVisualizer(
        output_size=args.output_size,
        heatmap_vmin=args.heatmap_vmin,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_colormap=args.heatmap_colormap,
        heatmap_gamma=args.heatmap_gamma,
        rgb_vmax_fz=args.rgb_vmax_fz,
        rgb_vmax_shear=args.rgb_vmax_shear,
    )

    frames_seen = 0
    last_timestamp = None

    try:
        runtime.start()
        if runtime.wait_for_frame(timeout=3.0) is None:
            raise RuntimeError("Timed out waiting for tactile frames")

        if args.calibrate:
            runtime.calibrate()

        while args.max_frames <= 0 or frames_seen < args.max_frames:
            snapshot = runtime.get_snapshot(copy_snapshot=True)
            if snapshot is None:
                time.sleep(0.02)
                continue

            if snapshot.frame.timestamp == last_timestamp:
                time.sleep(0.02)
                continue

            last_timestamp = snapshot.frame.timestamp
            frames_seen += 1

            index_force = snapshot.calibrated_force["index_middle"]
            middle_force = snapshot.calibrated_force["middle_middle"]
            logger.info(
                "[%05d] index=%s middle=%s",
                frames_seen,
                format_force(index_force),
                format_force(middle_force),
            )
            print(
                "[%05d] index=%s middle=%s"
                % (frames_seen, format_force(index_force), format_force(middle_force)),
                flush=True,
            )

            if args.output_dir is not None and args.save_every > 0 and frames_seen % args.save_every == 0:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                image = visualizer.render_snapshot(snapshot, calibrated=True)
                image_path = args.output_dir / f"tactile_{frames_seen:05d}.png"
                npz_path = args.output_dir / f"tactile_{frames_seen:05d}.npz"
                cv2.imwrite(str(image_path), image)
                runtime.save_snapshot_npz(str(npz_path), snapshot=snapshot)
                logger.info("Saved %s and %s", image_path, npz_path)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()

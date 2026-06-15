#!/usr/bin/env python3
"""Record bimanual Piper teleoperation episodes with tactile sensing into LeRobot dataset format.

Keyboard-driven episode control (like ``bi_piper_xr0_collect.py``):

- **ENTER** — start a new episode (prompts before each one).
- During recording:
  - **ENTER** — end the episode and prompt for a label (success/failure/ongoing/discard).
  - **s** — mark success and end immediately.
  - **f** — mark failure/invalid and end immediately.
  - **o** — mark ongoing and end immediately.
  - **d** — discard the episode.
  - **q** — quit the session (current episode is discarded).
- **Ctrl-C** — emergency stop.

The output is a standard LeRobot dataset, directly usable with ``lerobot-train``.

Usage::

    lerobot-record-piper-tactile \\
        --task "Pick up the red cube" \\
        --left-follower-can can0 \\
        --right-follower-can can1 \\
        --left-leader-can can2 \\
        --right-leader-can can3 \\
        --top-camera 12345678 \\
        --left-wrist-camera 12345679 \\
        --right-wrist-camera 12345680 \\
        --left-tactile-port /dev/ttyACM0 \\
        --right-tactile-port /dev/ttyACM1 \\
        --dataset.repo_id my_org/my_dataset \\
        --dataset.num_episodes 50 \\
        --fps 30
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.tactile.configuration_tactile import TactileCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import (
    make_robot_from_config,
)
from lerobot.robots.bi_piper_follower import (
    BiPiperFollower,
    BiPiperFollowerConfig,
    BiPiperXFollower,
    BiPiperXFollowerConfig,
)
from lerobot.robots.piper_follower import PiperFollowerConfigBase
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.bi_piper_leader import BiPiperLeaderConfig
from lerobot.teleoperators.bi_piper_leader import BiPiperXLeaderConfig
from lerobot.teleoperators.piper_leader import PiperLeaderConfigBase
from lerobot.utils.control_utils import (
    sanity_check_bimanual_piper_pair,
    sanity_check_dataset_name,
)
import threading
import numpy as np
from queue import Queue

from lerobot.utils.utils import init_logging


# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------

@contextmanager
def cbreak_stdin(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key_nonblocking() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    data = os.read(sys.stdin.fileno(), 1)
    if not data:
        return None
    return data.decode("utf-8", errors="ignore")


def prompt_outcome(default_value: str) -> str:
    prompt = (
        "Save as [s]uccess/[f]ailure/[o]ngoing/[d]iscard/[q]uit "
        f"(ENTER={default_value}): "
    )
    mapping = {
        "": default_value,
        "s": "success",
        "f": "failure",
        "o": "ongoing",
        "d": "discard",
        "q": "quit",
    }
    while True:
        resp = input(prompt).strip().lower()
        if resp in mapping:
            return mapping[resp]
        print("  Enter one of: s, f, o, d, q, or press ENTER.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record bimanual Piper teleop episodes + tactile into LeRobot format."
    )
    # Task
    parser.add_argument("--task", type=str, required=True, help="Task description.")
    # Robot / teleop
    parser.add_argument("--left-follower-can", type=str, required=True)
    parser.add_argument("--right-follower-can", type=str, required=True)
    parser.add_argument("--left-leader-can", type=str, required=True)
    parser.add_argument("--right-leader-can", type=str, required=True)
    parser.add_argument("--robot-id", type=str, default="bi_piper_collect")
    parser.add_argument("--teleop-id", type=str, default="bi_piper_leader_collect")
    parser.add_argument("--robot-type", choices=("bi_piper_follower", "bi_piperx_follower"),
                        default="bi_piperx_follower")
    # Cameras
    parser.add_argument("--top-camera", type=str, required=True, help="RealSense serial for ego camera.")
    parser.add_argument("--left-wrist-camera", type=str, default=None,
                        help="RealSense serial for left wrist (optional).")
    parser.add_argument("--right-wrist-camera", type=str, required=True)
    parser.add_argument("--ego-camera-side", choices=("left", "right"), default="left")
    parser.add_argument("--missing-left-wrist-fill", choices=("black", "copy-ego", "copy-right-wrist"),
                        default="black")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-warmup-s", type=int, default=2)
    # Tactile
    parser.add_argument("--left-tactile-port", type=str, default=None,
                        help="Serial port for left tactile controller.")
    parser.add_argument("--right-tactile-port", type=str, default=None,
                        help="Serial port for right tactile controller.")
    parser.add_argument("--tactile-output-size", type=int, default=256)
    parser.add_argument("--tactile-heatmap-vmin", type=float, default=0.0)
    parser.add_argument("--tactile-heatmap-vmax", type=float, default=25.5)
    parser.add_argument("--tactile-colormap", type=str, default="turbo")
    parser.add_argument("--tactile-gamma", type=float, default=0.75)
    parser.add_argument("--tactile-rgb-vmax-fz", type=float, default=25.5)
    parser.add_argument("--tactile-rgb-vmax-shear", type=float, default=12.8)
    parser.add_argument("--tactile-calibrate-on-connect", action="store_true", default=True)
    # Piper params
    parser.add_argument("--follower-startup-sleep-s", type=float, default=0.5)
    parser.add_argument("--leader-startup-sleep-s", type=float, default=0.1)
    parser.add_argument("--follower-speed-ratio", type=int, default=100)
    parser.add_argument("--follower-high-follow", action="store_true", default=True)
    parser.add_argument("--leader-command-speed-ratio", type=int, default=100)
    parser.add_argument("--leader-command-high-follow", action="store_true", default=True)
    parser.add_argument("--leader-process-isolation", action="store_true", default=True)
    parser.add_argument("--require-calibration", action="store_true", default=False)
    # Recording
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-seconds", type=float, default=0,
                        help="Max seconds per episode. 0 = unlimited.")
    parser.add_argument("--default-trajectory-type", choices=("success", "failure", "ongoing"),
                        default="success")
    # Dataset
    parser.add_argument("--dataset.repo_id", type=str, required=True, dest="dataset_repo_id",
                        help="HuggingFace-style dataset name, e.g. my_org/my_dataset")
    parser.add_argument("--dataset.root", type=str, default=None, dest="dataset_root")
    parser.add_argument("--dataset.push_to_hub", action="store_true", default=False, dest="push_to_hub")
    parser.add_argument("--dataset.private", action="store_true", default=False, dest="private_ds")
    parser.add_argument("--dataset.vcodec", type=str, default="h264", dest="vcodec")
    parser.add_argument(
        "--save-mode",
        choices=("serial", "parallel"),
        default="serial",
        help=(
            "How aggressively to save image/video data. "
            "'serial' uses one image writer thread and encodes camera videos one by one for stability; "
            "'parallel' restores the faster multi-thread/multi-process behavior."
        ),
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=None,
        help=(
            "Override image writer thread count. Defaults to 1 in serial mode, "
            "or 4 threads per camera in parallel mode."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    init_logging()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("piper-tactile-record")

    # ---- Build robot config --------------------------------------------------
    left_cams = {}
    if args.left_wrist_camera:
        left_cams["wrist"] = RealSenseCameraConfig(
            serial_number_or_name=args.left_wrist_camera,
            width=args.width, height=args.height, fps=args.fps,
            warmup_s=args.camera_warmup_s,
        )
    right_cams = {
        "wrist": RealSenseCameraConfig(
            serial_number_or_name=args.right_wrist_camera,
            width=args.width, height=args.height, fps=args.fps,
            warmup_s=args.camera_warmup_s,
        ),
    }
    ego_cfg = RealSenseCameraConfig(
        serial_number_or_name=args.top_camera,
        width=args.width, height=args.height, fps=args.fps,
        warmup_s=args.camera_warmup_s,
    )
    if args.ego_camera_side == "left":
        left_cams["ego"] = ego_cfg
    else:
        right_cams["ego"] = ego_cfg

    # Tactile cameras
    if args.left_tactile_port:
        left_cams["tactile"] = TactileCameraConfig(
            port=args.left_tactile_port,
            output_size=args.tactile_output_size,
            heatmap_vmin=args.tactile_heatmap_vmin,
            heatmap_vmax=args.tactile_heatmap_vmax,
            rgb_vmax_fz=args.tactile_rgb_vmax_fz,
            rgb_vmax_shear=args.tactile_rgb_vmax_shear,
            heatmap_colormap=args.tactile_colormap,
            heatmap_gamma=args.tactile_gamma,
            calibrate_on_connect=args.tactile_calibrate_on_connect,
        )
    if args.right_tactile_port:
        right_cams["tactile"] = TactileCameraConfig(
            port=args.right_tactile_port,
            output_size=args.tactile_output_size,
            heatmap_vmin=args.tactile_heatmap_vmin,
            heatmap_vmax=args.tactile_heatmap_vmax,
            rgb_vmax_fz=args.tactile_rgb_vmax_fz,
            rgb_vmax_shear=args.tactile_rgb_vmax_shear,
            heatmap_colormap=args.tactile_colormap,
            heatmap_gamma=args.tactile_gamma,
            calibrate_on_connect=args.tactile_calibrate_on_connect,
        )

    robot_cfg = BiPiperXFollowerConfig(
        id=args.robot_id,
        left_arm_config=PiperFollowerConfigBase(
            port=args.left_follower_can,
            startup_sleep_s=args.follower_startup_sleep_s,
            speed_ratio=args.follower_speed_ratio,
            high_follow=args.follower_high_follow,
            require_calibration=args.require_calibration,
            cameras=left_cams,
        ),
        right_arm_config=PiperFollowerConfigBase(
            port=args.right_follower_can,
            startup_sleep_s=args.follower_startup_sleep_s,
            speed_ratio=args.follower_speed_ratio,
            high_follow=args.follower_high_follow,
            require_calibration=args.require_calibration,
            cameras=right_cams,
        ),
    ) if args.robot_type == "bi_piperx_follower" else BiPiperFollowerConfig(
        id=args.robot_id,
        left_arm_config=PiperFollowerConfigBase(
            port=args.left_follower_can,
            startup_sleep_s=args.follower_startup_sleep_s,
            speed_ratio=args.follower_speed_ratio,
            high_follow=args.follower_high_follow,
            require_calibration=args.require_calibration,
            cameras=left_cams,
        ),
        right_arm_config=PiperFollowerConfigBase(
            port=args.right_follower_can,
            startup_sleep_s=args.follower_startup_sleep_s,
            speed_ratio=args.follower_speed_ratio,
            high_follow=args.follower_high_follow,
            require_calibration=args.require_calibration,
            cameras=right_cams,
        ),
    )

    teleop_cfg_cls = BiPiperXLeaderConfig if args.robot_type == "bi_piperx_follower" else BiPiperLeaderConfig
    teleop_cfg = teleop_cfg_cls(
        id=args.teleop_id,
        left_arm_config=PiperLeaderConfigBase(
            port=args.left_leader_can,
            startup_sleep_s=args.leader_startup_sleep_s,
            command_speed_ratio=args.leader_command_speed_ratio,
            command_high_follow=args.leader_command_high_follow,
            require_calibration=args.require_calibration,
        ),
        right_arm_config=PiperLeaderConfigBase(
            port=args.right_leader_can,
            startup_sleep_s=args.leader_startup_sleep_s,
            command_speed_ratio=args.leader_command_speed_ratio,
            command_high_follow=args.leader_command_high_follow,
            require_calibration=args.require_calibration,
        ),
        process_isolation=args.leader_process_isolation,
    )
    sanity_check_bimanual_piper_pair(robot_cfg, teleop_cfg)

    # ---- Connect -------------------------------------------------------------
    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    robot.connect()
    try:
        teleop.connect()
    except Exception:
        robot.disconnect()
        raise

    logger.info(
        "Connected robot=%s follower=(%s,%s) leader=(%s,%s) "
        "top=%s l_wrist=%s r_wrist=%s l_tactile=%s r_tactile=%s",
        args.robot_id, args.left_follower_can, args.right_follower_can,
        args.left_leader_can, args.right_leader_can,
        args.top_camera, args.left_wrist_camera or "masked", args.right_wrist_camera,
        args.left_tactile_port or "off", args.right_tactile_port or "off",
    )

    # ---- Create LeRobot dataset ----------------------------------------------
    sanity_check_dataset_name(args.dataset_repo_id, None)

    # Build feature schema from robot (same as lerobot-record)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 1
    if args.image_writer_threads is None:
        image_writer_threads = 1 if args.save_mode == "serial" else 4 * num_cameras
    else:
        image_writer_threads = args.image_writer_threads
    if image_writer_threads < 1:
        raise ValueError("--image-writer-threads must be >= 1.")

    parallel_video_encoding = args.save_mode == "parallel"
    logger.info(
        "Dataset save mode=%s image_writer_threads=%d parallel_video_encoding=%s",
        args.save_mode,
        image_writer_threads,
        parallel_video_encoding,
    )

    dataset = LeRobotDataset.create(
        args.dataset_repo_id,
        args.fps,
        root=args.dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
        vcodec=args.vcodec,
    )

    # ---- Episode loop --------------------------------------------------------
    saved_count = 0
    episode_index = 1
    try:
        with VideoEncodingManager(dataset):
            while True:
                input(f"\nPress ENTER to start episode {episode_index:03d} ...")

                # Episode is started implicitly by the first add_frame()
                frame_count = 0
                start_t = time.perf_counter()
                frame_period = 1.0 / float(args.fps)
                outcome = None
                quit_session = False
                interactive = sys.stdin.isatty()

                print(
                    f"  Recording episode {episode_index:03d}. "
                    "Keys: ENTER=end+prompt, s=success, f=failure, o=ongoing, d=discard, q=quit."
                )
                # Offload dataset writes to background thread for smooth teleop
                state_keys = list(robot._motors_ft.keys())
                cam_keys = list(robot._cameras_ft.keys())
                frame_queue: Queue = Queue()
                write_done = threading.Event()

                def _writer():
                    while not write_done.is_set():
                        item = frame_queue.get()
                        if item is None:
                            break
                        dataset.add_frame(item)

                writer_thread = threading.Thread(target=_writer, name="frame-writer", daemon=True)
                writer_thread.start()

                try:
                    with cbreak_stdin(interactive):
                        while True:
                            loop_t = time.perf_counter()

                            obs = robot.get_observation()
                            action = teleop.get_action()
                            robot.send_action(action)

                            # Build frame (fast path)
                            state_vals = np.array([float(obs[k]) for k in state_keys], dtype=np.float32)
                            action_vals = np.array([float(action[k]) for k in state_keys], dtype=np.float32)
                            frame_data = {
                                "observation.state": state_vals,
                                "action": action_vals,
                                "task": args.task,
                            }
                            for ck in cam_keys:
                                frame_data[f"observation.images.{ck}"] = obs[ck]
                            frame_queue.put(frame_data)
                            frame_count += 1

                            elapsed = time.perf_counter() - start_t
                            print(
                                f"\r  frames={frame_count:05d} elapsed={elapsed:.1f}s",
                                end="", flush=True,
                            )

                            # Check keyboard
                            key = read_key_nonblocking() if interactive else None
                            if key in ("\r", "\n"):
                                break
                            if key:
                                k = key.lower()
                                if k == "s":
                                    outcome = "success"; break
                                if k == "f":
                                    outcome = "failure"; break
                                if k == "o":
                                    outcome = "ongoing"; break
                                if k == "d":
                                    outcome = "discard"; break
                                if k == "q":
                                    outcome = "quit"; quit_session = True; break

                            if args.episode_seconds > 0 and elapsed >= args.episode_seconds:
                                break

                            # Frame pacing
                            next_t = start_t + (frame_count + 1) * frame_period
                            sleep_s = next_t - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            elif time.perf_counter() - loop_t > frame_period * 2:
                                # Resync if we're falling behind
                                start_t = time.perf_counter() - frame_count * frame_period

                finally:
                    write_done.set()
                    frame_queue.put(None)
                    writer_thread.join(timeout=5.0)

                print()  # newline after \r

                if frame_count == 0:
                    logger.warning("  No frames — discarding empty episode.")
                    dataset.clear_episode_buffer()
                    if quit_session:
                        break
                    episode_index += 1
                    continue

                # Resolve outcome
                if outcome in ("success", "failure", "ongoing"):
                    pass  # already set
                elif outcome == "discard":
                    logger.info("  Episode discarded.")
                    dataset.clear_episode_buffer()
                    if quit_session:
                        break
                    episode_index += 1
                    continue
                elif outcome == "quit":
                    logger.info("  Quit — discarding current episode.")
                    dataset.clear_episode_buffer()
                    break
                else:
                    outcome = prompt_outcome(args.default_trajectory_type)
                    if outcome == "discard":
                        logger.info("  Episode discarded.")
                        dataset.clear_episode_buffer()
                        episode_index += 1
                        continue
                    if outcome == "quit":
                        logger.info("  Quit — discarding current episode.")
                        dataset.clear_episode_buffer()
                        break

                # Save
                ep_success = True if outcome == "success" else False
                extra_meta = {"episode_success": ep_success} if outcome in ("success", "failure") else {}
                dataset.save_episode(
                    parallel_encoding=parallel_video_encoding,
                    extra_episode_metadata=extra_meta if extra_meta else None,
                )
                saved_count += 1
                logger.info(
                    "  Saved episode %03d as '%s' (%d frames).",
                    episode_index, outcome, frame_count,
                )

                if quit_session:
                    break
                episode_index += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()

        dataset.finalize()
        logger.info(
            "Dataset finalized: %d episodes saved to %s.\n"
            "To inspect: lerobot-dataset-report --dataset %s",
            saved_count, dataset.root, args.dataset_repo_id,
        )


if __name__ == "__main__":
    main()

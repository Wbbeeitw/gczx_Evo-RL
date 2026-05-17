#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.robots.bi_piper_follower.bi_piper_follower import BiPiperFollower
from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfigBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move both PiPER follower arms back to their calibrated center pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-id", default="piper_follower")
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--left-port", default="can0")
    parser.add_argument("--right-port", default="can1")
    parser.add_argument("--speed-ratio", type=int, default=20)
    parser.add_argument("--high-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-on-connect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-on-disconnect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--can-auto-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-flag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--startup-sleep-s", type=float, default=0.1)
    parser.add_argument("--enable-timeout-s", type=float, default=3.0)
    parser.add_argument("--mode-refresh-interval-s", type=float, default=1.0)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--arm-duration-s",
        "--duration-s",
        dest="arm_duration_s",
        type=float,
        default=8.0,
        help="Arm-joint return duration.",
    )
    parser.add_argument("--control-hz", type=float, default=20.0, help="Command rate while returning.")
    parser.add_argument(
        "--interp",
        choices=["linear", "smoothstep"],
        default="smoothstep",
        help="Interpolation profile used while returning to center.",
    )
    parser.add_argument(
        "--include-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After arm joints are centered, also center both grippers.",
    )
    parser.add_argument("--gripper-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--calibrate-on-connect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass calibrate=True to robot.connect().",
    )
    return parser.parse_args()


def build_robot(args: argparse.Namespace) -> BiPiperFollower:
    side_kwargs = dict(
        judge_flag=args.judge_flag,
        can_auto_init=args.can_auto_init,
        log_level=args.log_level,
        startup_sleep_s=args.startup_sleep_s,
        speed_ratio=args.speed_ratio,
        high_follow=args.high_follow,
        mode_refresh_interval_s=args.mode_refresh_interval_s,
        enable_on_connect=args.enable_on_connect,
        enable_timeout_s=args.enable_timeout_s,
        require_calibration=args.require_calibration,
        sync_gripper=args.sync_gripper,
        cameras={},
        disable_on_disconnect=args.disable_on_disconnect,
    )
    calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None
    cfg = BiPiperFollowerConfig(
        id=args.robot_id,
        calibration_dir=calibration_dir,
        left_arm_config=PiperFollowerConfigBase(port=args.left_port, **side_kwargs),
        right_arm_config=PiperFollowerConfigBase(port=args.right_port, **side_kwargs),
    )
    return BiPiperFollower(cfg)


def build_action_space_center_command(
    robot: BiPiperFollower,
    *,
    include_gripper: bool = False,
    gripper_only: bool = False,
) -> dict[str, float]:
    center_action: dict[str, float] = {}
    for key in robot.action_features:
        if not key.endswith(".pos"):
            continue
        is_gripper = "gripper" in key
        if gripper_only and not is_gripper:
            continue
        if not gripper_only and is_gripper and not include_gripper:
            continue
        center_action[key] = 0.0
    return center_action


def split_prefixed_piper_action_key(robot: BiPiperFollower, key: str):
    if key.startswith("left_") and hasattr(robot, "left_arm"):
        return robot.left_arm, key.removeprefix("left_")
    if key.startswith("right_") and hasattr(robot, "right_arm"):
        return robot.right_arm, key.removeprefix("right_")
    return robot, key


def calibration_home_deg(arm, key: str) -> float | None:
    calibration = getattr(arm, "calibration", None)
    if not calibration or key not in calibration:
        return None

    cal = calibration[key]
    from_units = getattr(arm, "_from_calibration_units", None)
    if callable(from_units):
        return float(from_units(cal.homing_offset))

    config = getattr(arm, "config", None)
    calibration_scale = float(getattr(config, "calibration_scale", 1000))
    return float(cal.homing_offset) / calibration_scale


def estimate_current_action_offsets(robot: BiPiperFollower, action_keys: list[str]) -> dict[str, float]:
    observation = robot.get_observation()
    offsets: dict[str, float] = {}
    for action_key in action_keys:
        if action_key not in observation:
            continue
        arm, arm_key = split_prefixed_piper_action_key(robot, action_key)
        home_deg = calibration_home_deg(arm, arm_key)
        if home_deg is None:
            continue
        current_deg = float(observation[action_key])
        cal = getattr(arm, "calibration", {}).get(arm_key)
        offset_deg = current_deg - home_deg
        if getattr(cal, "drive_mode", 0):
            offset_deg = -offset_deg
        offsets[action_key] = offset_deg
    return offsets


def interpolation_alpha(step_idx: int, steps: int, mode: str) -> float:
    alpha = float(step_idx + 1) / float(max(steps, 1))
    alpha = min(max(alpha, 0.0), 1.0)
    if mode == "linear":
        return alpha
    if mode == "smoothstep":
        return alpha * alpha * (3.0 - 2.0 * alpha)
    raise ValueError(f"Unsupported interpolation mode: {mode}")


def send_center_command_for_duration(
    robot: BiPiperFollower,
    center_action: dict[str, float],
    *,
    duration_s: float,
    control_hz: float,
    description: str,
    interp: str,
) -> None:
    if not center_action:
        raise RuntimeError(f"No '.pos' action keys available for {description} center command.")

    action_keys = list(center_action)
    start_offsets = estimate_current_action_offsets(robot, action_keys) if duration_s > 0 else {}
    start_action = {key: float(start_offsets.get(key, center_action[key])) for key in action_keys}
    steps = max(1, int(round(duration_s * control_hz)))
    sleep_s = 1.0 / control_hz if duration_s > 0 else 0.0

    logging.info(
        "Returning %s to calibrated center for %.2fs at %.1fHz (%d keys, interp=%s, ramped_keys=%d).",
        description,
        duration_s,
        control_hz,
        len(center_action),
        interp,
        len(start_offsets),
    )

    for step_idx in range(steps):
        alpha = interpolation_alpha(step_idx, steps, interp)
        action = {
            key: start_action[key] * (1.0 - alpha) + float(center_action[key]) * alpha for key in action_keys
        }
        robot.send_action(action)
        if sleep_s > 0 and step_idx < steps - 1:
            time.sleep(sleep_s)

    # Hold a final exact center command once to avoid any residual rounding error.
    robot.send_action(center_action)


def return_to_action_space_center(
    robot: BiPiperFollower,
    *,
    arm_duration_s: float,
    control_hz: float,
    include_gripper: bool,
    gripper_duration_s: float,
    interp: str,
) -> None:
    arm_center_action = build_action_space_center_command(robot, include_gripper=False)
    send_center_command_for_duration(
        robot,
        arm_center_action,
        duration_s=arm_duration_s,
        control_hz=control_hz,
        description="arm joints",
        interp=interp,
    )

    if include_gripper:
        gripper_center_action = build_action_space_center_command(
            robot,
            include_gripper=True,
            gripper_only=True,
        )
        send_center_command_for_duration(
            robot,
            gripper_center_action,
            duration_s=gripper_duration_s,
            control_hz=control_hz,
            description="grippers",
            interp=interp,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.arm_duration_s < 0:
        raise ValueError("--arm-duration-s must be >= 0.")
    if args.gripper_duration_s < 0:
        raise ValueError("--gripper-duration-s must be >= 0.")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0.")

    robot = build_robot(args)
    try:
        logging.info("Connecting bi_piper_follower | left=%s right=%s", args.left_port, args.right_port)
        robot.connect(calibrate=args.calibrate_on_connect)
        logging.info("Connected. action_features=%s", list(robot.action_features))

        return_to_action_space_center(
            robot,
            arm_duration_s=args.arm_duration_s,
            control_hz=args.control_hz,
            include_gripper=args.include_gripper,
            gripper_duration_s=args.gripper_duration_s,
            interp=args.interp,
        )
        logging.info("Finished returning both arms to calibrated center.")
    finally:
        if robot.is_connected:
            robot.disconnect()
            logging.info("Disconnected robot.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


_ensure_repo_src_on_path()

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import RobotAction, RobotObservation, make_default_processors
from lerobot.robots.bi_piper_follower.bi_piper_follower import BiPiperFollower
from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfigBase
from lerobot.robots.robot import Robot
from lerobot.scripts.recording_loop import record_loop
from lerobot.teleoperators.bi_piper_leader.bi_piper_leader import BiPiperLeader
from lerobot.teleoperators.bi_piper_leader.config_bi_piper_leader import BiPiperLeaderConfig
from lerobot.teleoperators.piper_leader.config_piper_leader import PiperLeaderConfigBase
from lerobot.utils.constants import ACTION
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.recording_annotations import normalize_episode_success_label, resolve_episode_success_label
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
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


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = value.strip()
    return None if text == "" else Path(text)


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to parse camera configuration strings. Install it with `pip install pyyaml`."
        ) from exc

    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a camera mapping dictionary, got: {type(payload)!r}")
    return payload


def _build_realsense_camera_configs(raw_text: str, arg_name: str) -> dict[str, RealSenseCameraConfig]:
    mapping = _load_yaml_mapping(raw_text)
    camera_configs: dict[str, RealSenseCameraConfig] = {}
    for camera_key, raw_config in mapping.items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"{arg_name} -> {camera_key!r} must be a dictionary.")
        camera_type = str(raw_config.get("type", "")).strip().lower()
        if camera_type != "intelrealsense":
            raise ValueError(
                f"{arg_name} -> {camera_key!r} uses unsupported camera type {camera_type!r}. "
                "This script currently supports only Intel RealSense cameras."
            )
        serial = raw_config.get("serial_number_or_name")
        if serial is None:
            raise ValueError(f"{arg_name} -> {camera_key!r} must define `serial_number_or_name`.")
        width = raw_config.get("width")
        height = raw_config.get("height")
        fps = raw_config.get("fps")
        if width is None or height is None or fps is None:
            raise ValueError(f"{arg_name} -> {camera_key!r} must define width/height/fps.")
        camera_configs[str(camera_key)] = RealSenseCameraConfig(
            serial_number_or_name=str(serial),
            width=int(width),
            height=int(height),
            fps=int(fps),
            warmup_s=int(raw_config.get("warmup_s", 1)),
            use_depth=parse_bool(raw_config.get("use_depth", False)),
            color_mode=str(raw_config.get("color_mode", "rgb")),
            rotation=_parse_realsense_rotation(raw_config.get("rotation")),
        )
    return camera_configs


def _parse_realsense_rotation(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    normalized = str(value).strip().lower()
    aliases = {
        "0": 0,
        "none": 0,
        "no_rotation": 0,
        "rotate_90": 90,
        "90": 90,
        "rotate_180": 180,
        "180": 180,
        "rotate_270": -90,
        "-90": -90,
        "270": -90,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported RealSense rotation value: {value!r}")
    return aliases[normalized]


def _resolve_xiaomi_python_root(xiaomi_root: Path | None) -> Path | None:
    if xiaomi_root is None:
        return None
    candidates = [xiaomi_root, xiaomi_root / "xr0"]
    for candidate in candidates:
        if (candidate / "mibot" / "tactile" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to locate `mibot.tactile`. Pass `--xiaomi-xr0-root` as the Xiaomi-Robotics-0 repo root "
        "or directly as its `xr0` subdirectory."
    )


def _load_tactile_classes(xiaomi_root: Path | None) -> tuple[type[Any], type[Any], type[Any]]:
    resolved_root = _resolve_xiaomi_python_root(xiaomi_root)
    if resolved_root is not None:
        resolved_root_str = str(resolved_root)
        if resolved_root_str not in sys.path:
            sys.path.insert(0, resolved_root_str)
    try:
        from mibot.tactile import TactileRuntime, TactileSensorDriver, TactileVisualizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import `mibot.tactile`. Either install the Xiaomi XR0 environment dependencies, "
            "or pass `--xiaomi-xr0-root` to point at the local Xiaomi-Robotics-0 checkout."
        ) from exc
    return TactileRuntime, TactileSensorDriver, TactileVisualizer


@dataclass
class TactileRuntimeBundle:
    runtime_cls: type[Any]
    driver_cls: type[Any]
    visualizer_cls: type[Any]


class TactileSideCamera:
    SENSOR_NAMES = ("index_middle", "middle_middle")

    def __init__(
        self,
        *,
        side_name: str,
        port: str,
        runtime_bundle: TactileRuntimeBundle,
        baudrate: int,
        timeout: float,
        mode: str,
        distributed_scale: float,
        output_size: int,
        heatmap_vmin: float,
        heatmap_vmax: float,
        heatmap_colormap: str,
        heatmap_gamma: float,
        poll_interval: float,
        calibrate: bool,
        calibration_samples: int,
        calibration_interval: float,
        calibration_warmup_frames: int,
        calibration_reducer: str,
        log_level: str,
    ) -> None:
        self.side_name = side_name
        self.port = port
        self.mode = mode
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.calibrate_on_start = calibrate
        self.logger = logging.getLogger(f"tactile-{side_name}")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        driver = runtime_bundle.driver_cls(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            enable_distributed=mode == "auto_push",
            distributed_scale=distributed_scale,
            logger=self.logger,
        )
        self.runtime = runtime_bundle.runtime_cls(
            driver=driver,
            read_mode=mode,
            read_timeout=timeout,
            poll_interval=poll_interval,
            calibration_samples=calibration_samples,
            calibration_interval=calibration_interval,
            calibration_warmup_frames=calibration_warmup_frames,
            calibration_reducer=calibration_reducer,
            logger=self.logger,
        )
        self.visualizer = runtime_bundle.visualizer_cls(
            output_size=output_size,
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            heatmap_colormap=heatmap_colormap,
            heatmap_gamma=heatmap_gamma,
        )
        self.image_shape = (output_size, output_size * len(self.SENSOR_NAMES), 3)
        self._last_image = np.zeros(self.image_shape, dtype=np.uint8)
        self._last_timestamp: float | None = None
        self._started = False

    @property
    def is_connected(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return

        self.logger.info("Starting tactile stream on %s (%s).", self.port, self.side_name)
        self.runtime.start()
        frame = self.runtime.wait_for_frame(timeout=max(self.timeout * 10.0, 3.0))
        if frame is None:
            self.runtime.stop()
            raise RuntimeError(f"Timed out waiting for tactile frames on {self.side_name} ({self.port}).")

        if self.calibrate_on_start:
            self.logger.info("Calibrating tactile stream on %s (%s).", self.port, self.side_name)
            self.runtime.calibrate()

        snapshot = self.runtime.get_snapshot(copy_snapshot=True)
        if snapshot is not None:
            self._last_timestamp = snapshot.frame.timestamp
            self._last_image = self._render_heatmap_rgb(snapshot)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.runtime.stop()
        finally:
            self._started = False

    def read_image(self) -> np.ndarray:
        snapshot = self.runtime.get_snapshot(copy_snapshot=True)
        if snapshot is None:
            return self._last_image.copy()
        if self._last_timestamp is not None and snapshot.frame.timestamp <= self._last_timestamp:
            return self._last_image.copy()

        self._last_timestamp = snapshot.frame.timestamp
        self._last_image = self._render_heatmap_rgb(snapshot)
        return self._last_image.copy()

    def _render_heatmap_rgb(self, snapshot: Any) -> np.ndarray:
        calibrated_distributed = getattr(snapshot, "calibrated_distributed", {}) or {}
        rendered_views = []
        for sensor_name in self.SENSOR_NAMES:
            distributed = calibrated_distributed.get(sensor_name)
            coords = None
            if distributed is not None:
                coords = self.visualizer.gen_coords(int(distributed.shape[0]))
            heatmap_bgr = self.visualizer.make_fz_heatmap(coords, distributed)
            rendered_views.append(np.ascontiguousarray(heatmap_bgr[..., ::-1]))
        return np.ascontiguousarray(np.hstack(rendered_views))


class BiPiperFollowerWithTactile(Robot):
    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower_tactile"

    def __init__(
        self,
        robot: BiPiperFollower,
        left_tactile: TactileSideCamera,
        right_tactile: TactileSideCamera,
        *,
        left_tactile_key: str = "left_tactile",
        right_tactile_key: str = "right_tactile",
    ) -> None:
        super().__init__(robot.config)
        self._robot = robot
        self._left_tactile = left_tactile
        self._right_tactile = right_tactile
        self._left_tactile_key = left_tactile_key
        self._right_tactile_key = right_tactile_key
        self.cameras = {
            **getattr(robot, "cameras", {}),
            left_tactile_key: left_tactile,
            right_tactile_key: right_tactile,
        }

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features = dict(self._robot.observation_features)
        features[self._left_tactile_key] = self._left_tactile.image_shape
        features[self._right_tactile_key] = self._right_tactile.image_shape
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return self._robot.action_features

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected and self._left_tactile.is_connected and self._right_tactile.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self._robot.connect(calibrate)
        try:
            self._left_tactile.start()
            self._right_tactile.start()
        except Exception:
            try:
                self._left_tactile.stop()
            finally:
                try:
                    self._right_tactile.stop()
                finally:
                    self._robot.disconnect()
            raise

    @property
    def is_calibrated(self) -> bool:
        return self._robot.is_calibrated

    def calibrate(self) -> None:
        self._robot.calibrate()

    def configure(self) -> None:
        self._robot.configure()

    def set_teleop_send_only_mode(self, enabled: bool) -> None:
        self._robot.set_teleop_send_only_mode(enabled)

    def get_observation(self) -> RobotObservation:
        observation = dict(self._robot.get_observation())
        observation[self._left_tactile_key] = self._left_tactile.read_image()
        observation[self._right_tactile_key] = self._right_tactile.read_image()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        return self._robot.send_action(action)

    def disconnect(self) -> None:
        try:
            self._left_tactile.stop()
        finally:
            try:
                self._right_tactile.stop()
            finally:
                self._robot.disconnect()


def _ensure_episode_annotation_features(dataset_features: dict[str, dict], *, action_feature_names: list[str]) -> None:
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_feature_names),),
        "names": action_feature_names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a bimanual PiPER teleoperation dataset with the original three RealSense views plus "
            "two Xiaomi tactile heatmap video streams."
        )
    )

    parser.add_argument("--xiaomi-xr0-root", type=parse_optional_path, default=None)

    parser.add_argument("--robot.id", "--robot-id", dest="robot_id", default="piper_follower")
    parser.add_argument("--teleop.id", "--teleop-id", dest="teleop_id", default="piper_leader")
    parser.add_argument("--calibration-dir", type=parse_optional_path, default=None)

    parser.add_argument(
        "--robot.left_arm_config.port",
        "--left-follower-can",
        dest="left_follower_can",
        required=True,
    )
    parser.add_argument(
        "--robot.right_arm_config.port",
        "--right-follower-can",
        dest="right_follower_can",
        required=True,
    )
    parser.add_argument(
        "--teleop.left_arm_config.port",
        "--left-leader-can",
        dest="left_leader_can",
        required=True,
    )
    parser.add_argument(
        "--teleop.right_arm_config.port",
        "--right-leader-can",
        dest="right_leader_can",
        required=True,
    )

    parser.add_argument(
        "--robot.left_arm_config.cameras",
        "--robot-left-arm-cameras",
        dest="left_arm_cameras",
        required=True,
        help="YAML-like mapping for the left follower cameras, e.g. '{top: {...}, wrist: {...}}'.",
    )
    parser.add_argument(
        "--robot.right_arm_config.cameras",
        "--robot-right-arm-cameras",
        dest="right_arm_cameras",
        required=True,
        help="YAML-like mapping for the right follower cameras, e.g. '{wrist: {...}}'.",
    )

    parser.add_argument(
        "--robot.left_arm_config.require_calibration",
        dest="left_robot_require_calibration",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--robot.right_arm_config.require_calibration",
        dest="right_robot_require_calibration",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--teleop.left_arm_config.require_calibration",
        dest="left_teleop_require_calibration",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--teleop.right_arm_config.require_calibration",
        dest="right_teleop_require_calibration",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--robot.left_arm_config.enable_on_connect",
        dest="left_robot_enable_on_connect",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--robot.right_arm_config.enable_on_connect",
        dest="right_robot_enable_on_connect",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--robot.left_arm_config.high_follow",
        dest="left_robot_high_follow",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--robot.right_arm_config.high_follow",
        dest="right_robot_high_follow",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--robot.left_arm_config.speed_ratio",
        dest="left_robot_speed_ratio",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--robot.right_arm_config.speed_ratio",
        dest="right_robot_speed_ratio",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--teleop.process_isolation",
        "--teleop-process-isolation",
        dest="teleop_process_isolation",
        type=parse_bool,
        default=True,
    )

    parser.add_argument("--left-tactile-port", required=True, help="Serial port for the left tactile controller.")
    parser.add_argument("--right-tactile-port", required=True, help="Serial port for the right tactile controller.")
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
        help="Per-sensor heatmap size. Final tactile video size per side is <output_size> x <2*output_size>.",
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

    parser.add_argument("--dataset.repo_id", "--dataset-repo-id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", "--dataset-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.single_task", "--task", dest="task", required=True)
    parser.add_argument("--dataset.num_episodes", "--num-episodes", dest="num_episodes", type=int, default=10)
    parser.add_argument(
        "--dataset.episode_time_s",
        "--episode-time-s",
        dest="episode_time_s",
        type=float,
        default=240.0,
    )
    parser.add_argument("--dataset.reset_time_s", "--reset-time-s", dest="reset_time_s", type=float, default=5.0)
    parser.add_argument("--dataset.fps", "--fps", dest="fps", type=int, default=30)
    parser.add_argument("--dataset.video", "--dataset-video", dest="dataset_video", type=parse_bool, default=True)
    parser.add_argument(
        "--dataset.push_to_hub",
        "--push-to-hub",
        dest="dataset_push_to_hub",
        type=parse_bool,
        default=False,
    )
    parser.add_argument("--dataset.private", "--dataset-private", dest="dataset_private", type=parse_bool, default=False)
    parser.add_argument("--dataset.vcodec", "--dataset-vcodec", dest="dataset_vcodec", default="h264")
    parser.add_argument(
        "--dataset.num_image_writer_processes",
        dest="num_image_writer_processes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataset.num_image_writer_threads_per_camera",
        dest="num_image_writer_threads_per_camera",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dataset.video_encoding_batch_size",
        dest="video_encoding_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument("--resume", dest="resume", type=parse_bool, default=False)

    parser.add_argument("--display_data", "--display-data", dest="display_data", type=parse_bool, default=False)
    parser.add_argument("--display_ip", "--display-ip", dest="display_ip", default=None)
    parser.add_argument("--display_port", "--display-port", dest="display_port", type=int, default=None)
    parser.add_argument(
        "--display_compressed_images",
        "--display-compressed-images",
        dest="display_compressed_images",
        type=parse_bool,
        default=False,
    )
    parser.add_argument("--play_sounds", "--play-sounds", dest="play_sounds", type=parse_bool, default=False)

    parser.add_argument(
        "--enable_episode_outcome_labeling",
        "--enable-episode-outcome-labeling",
        dest="enable_episode_outcome_labeling",
        type=parse_bool,
        default=True,
    )
    parser.add_argument("--episode_success_key", "--episode-success-key", dest="episode_success_key", default="s")
    parser.add_argument("--episode_failure_key", "--episode-failure-key", dest="episode_failure_key", default="f")
    parser.add_argument(
        "--require_episode_success_label",
        "--require-episode-success-label",
        dest="require_episode_success_label",
        type=parse_bool,
        default=True,
    )
    parser.add_argument("--default_episode_success", "--default-episode-success", dest="default_episode_success", default=None)

    return parser


def _make_bi_piper_follower(args: argparse.Namespace) -> BiPiperFollower:
    left_cameras = _build_realsense_camera_configs(args.left_arm_cameras, "--robot.left_arm_config.cameras")
    right_cameras = _build_realsense_camera_configs(args.right_arm_cameras, "--robot.right_arm_config.cameras")
    follower_config = BiPiperFollowerConfig(
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        left_arm_config=PiperFollowerConfigBase(
            port=args.left_follower_can,
            require_calibration=args.left_robot_require_calibration,
            enable_on_connect=args.left_robot_enable_on_connect,
            high_follow=args.left_robot_high_follow,
            speed_ratio=args.left_robot_speed_ratio,
            cameras=left_cameras,
        ),
        right_arm_config=PiperFollowerConfigBase(
            port=args.right_follower_can,
            require_calibration=args.right_robot_require_calibration,
            enable_on_connect=args.right_robot_enable_on_connect,
            high_follow=args.right_robot_high_follow,
            speed_ratio=args.right_robot_speed_ratio,
            cameras=right_cameras,
        ),
    )
    return BiPiperFollower(follower_config)


def _make_bi_piper_leader(args: argparse.Namespace) -> BiPiperLeader:
    leader_config = BiPiperLeaderConfig(
        id=args.teleop_id,
        calibration_dir=args.calibration_dir,
        process_isolation=args.teleop_process_isolation,
        left_arm_config=PiperLeaderConfigBase(
            port=args.left_leader_can,
            require_calibration=args.left_teleop_require_calibration,
        ),
        right_arm_config=PiperLeaderConfigBase(
            port=args.right_leader_can,
            require_calibration=args.right_teleop_require_calibration,
        ),
    )
    return BiPiperLeader(leader_config)


def _make_tactile_side_camera(
    *,
    side_name: str,
    port: str,
    runtime_bundle: TactileRuntimeBundle,
    args: argparse.Namespace,
) -> TactileSideCamera:
    return TactileSideCamera(
        side_name=side_name,
        port=port,
        runtime_bundle=runtime_bundle,
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


def _build_dataset_features(robot: Robot, use_videos: bool) -> dict[str, dict]:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )
    action_names = dataset_features[ACTION]["names"]
    action_names = list(robot.action_features) if action_names is None else list(action_names)
    _ensure_episode_annotation_features(dataset_features, action_feature_names=action_names)
    return dataset_features


def _create_or_resume_dataset(
    *,
    args: argparse.Namespace,
    robot: Robot,
    dataset_features: dict[str, dict],
) -> LeRobotDataset:
    if args.resume:
        dataset = LeRobotDataset(
            args.dataset_repo_id,
            root=args.dataset_root,
            batch_encoding_size=args.video_encoding_batch_size,
            vcodec=args.dataset_vcodec,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=args.num_image_writer_processes,
                num_threads=args.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
        return dataset

    sanity_check_dataset_name(args.dataset_repo_id, None)
    return LeRobotDataset.create(
        args.dataset_repo_id,
        args.fps,
        root=args.dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=args.dataset_video,
        image_writer_processes=args.num_image_writer_processes,
        image_writer_threads=args.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=args.video_encoding_batch_size,
        vcodec=args.dataset_vcodec,
    )


def _record(args: argparse.Namespace) -> None:
    init_logging()

    if not args.dataset_video:
        raise ValueError(
            "This tactile recorder requires `--dataset.video true`. "
            "RGB views and tactile heatmaps are expected to be encoded as MP4 videos, not stored as images."
        )

    if args.require_episode_success_label and not args.enable_episode_outcome_labeling:
        raise ValueError(
            "`require_episode_success_label=true` requires `enable_episode_outcome_labeling=true`."
        )

    if args.default_episode_success is not None:
        args.default_episode_success = normalize_episode_success_label(args.default_episode_success)

    if args.display_data:
        init_rerun(session_name="xiaomi_bi_piper_tactile_recording", ip=args.display_ip, port=args.display_port)

    display_compressed_images = (
        True
        if (args.display_data and args.display_ip is not None and args.display_port is not None)
        else args.display_compressed_images
    )

    tactile_runtime_cls, tactile_driver_cls, tactile_visualizer_cls = _load_tactile_classes(args.xiaomi_xr0_root)
    runtime_bundle = TactileRuntimeBundle(
        runtime_cls=tactile_runtime_cls,
        driver_cls=tactile_driver_cls,
        visualizer_cls=tactile_visualizer_cls,
    )

    base_robot = _make_bi_piper_follower(args)
    teleop = _make_bi_piper_leader(args)
    wrapped_robot = BiPiperFollowerWithTactile(
        base_robot,
        _make_tactile_side_camera(side_name="left", port=args.left_tactile_port, runtime_bundle=runtime_bundle, args=args),
        _make_tactile_side_camera(
            side_name="right",
            port=args.right_tactile_port,
            runtime_bundle=runtime_bundle,
            args=args,
        ),
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = _build_dataset_features(wrapped_robot, use_videos=args.dataset_video)

    dataset: LeRobotDataset | None = None
    listener = None

    try:
        dataset = _create_or_resume_dataset(args=args, robot=wrapped_robot, dataset_features=dataset_features)

        wrapped_robot.connect()
        teleop.connect()

        listener, events = init_keyboard_listener(
            intervention_toggle_key="i",
            episode_success_key=args.episode_success_key if args.enable_episode_outcome_labeling else None,
            episode_failure_key=args.episode_failure_key if args.enable_episode_outcome_labeling else None,
        )

        logging.info(
            "Recording with video-backed image keys: %s",
            ", ".join(
                sorted(k for k in wrapped_robot.observation_features if isinstance(wrapped_robot.observation_features[k], tuple))
            ),
        )
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < args.num_episodes and not events["stop_recording"]:
                events["episode_outcome"] = None
                log_say(f"Recording episode {dataset.num_episodes}", args.play_sounds)
                record_loop(
                    robot=wrapped_robot,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    teleop=teleop,
                    control_time_s=args.episode_time_s,
                    single_task=args.task,
                    display_data=args.display_data,
                    display_compressed_images=display_compressed_images,
                )

                episode_success = None
                if args.enable_episode_outcome_labeling:
                    episode_success = resolve_episode_success_label(
                        explicit_label=events.get("episode_outcome"),
                        default_label=args.default_episode_success,
                        require_label=args.require_episode_success_label,
                    )
                    if events.get("episode_outcome") is None and episode_success is not None:
                        logging.warning(
                            "Episode %s has no explicit success/failure label, defaulting to '%s'.",
                            dataset.num_episodes,
                            episode_success,
                        )

                if not events["stop_recording"] and (
                    (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", args.play_sounds)
                    record_loop(
                        robot=wrapped_robot,
                        events=events,
                        fps=args.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=args.reset_time_s,
                        single_task=args.task,
                        display_data=args.display_data,
                        display_compressed_images=display_compressed_images,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", args.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    events["episode_outcome"] = None
                    dataset.clear_episode_buffer()
                    continue

                extra_episode_metadata = (
                    {"episode_success": episode_success} if args.enable_episode_outcome_labeling else None
                )
                dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
                recorded_episodes += 1
                logging.info("Saved episode %d/%d to %s", recorded_episodes, args.num_episodes, args.dataset_root)
    finally:
        log_say("Stop recording", args.play_sounds, blocking=True)

        if dataset is not None:
            dataset.finalize()
            logging.info(
                "Dataset finalized under %s. To inspect it, run:\n  lerobot-dataset-report --dataset %s",
                args.dataset_root,
                dataset.repo_id,
            )

        try:
            if wrapped_robot.is_connected:
                wrapped_robot.disconnect()
        except Exception:
            logging.exception("Failed while disconnecting the wrapped robot.")

        try:
            if teleop.is_connected:
                teleop.disconnect()
        except Exception:
            logging.exception("Failed while disconnecting the teleoperator.")

        if listener and hasattr(listener, "stop"):
            listener.stop()

        if args.dataset_push_to_hub:
            if dataset is not None:
                dataset.push_to_hub(private=args.dataset_private)
            else:
                logging.warning(
                    "`dataset.push_to_hub=true` was requested, but the dataset was not initialized successfully."
                )

        log_say("Exiting", args.play_sounds)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _record(args)


if __name__ == "__main__":
    main()

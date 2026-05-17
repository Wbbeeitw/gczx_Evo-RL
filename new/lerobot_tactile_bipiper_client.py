#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec B403: internal trusted transport
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import grpc
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.bi_piper_follower.bi_piper_follower import BiPiperFollower
from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfigBase
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE

from tactile_client_server_common import (
    DEFAULT_IMAGE_TOPICS,
    DEFAULT_ORDERED_IMAGE_KEYS,
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX,
    TactilePolicyRequestConfig,
    TactileTimedAction,
    build_tactile_prompt,
    encode_observation_packet,
    parse_csv_items,
    resolve_dataset_root,
    to_rgb_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROS1 tactile client that directly reads/sends actions to a bi_piper_follower robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-address", required=True, help="gRPC server address like 127.0.0.1:8080")
    parser.add_argument("--dataset-repo-id", required=True, help="Dataset repo id used for training.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root or parent root.")
    parser.add_argument("--checkpoint", default="", help="Optional note-only arg for parity with deployment docs.")
    parser.add_argument("--task", required=True, help="High-level instruction.")
    parser.add_argument("--robot-type", default="bi_piper_follower")
    parser.add_argument("--robot-id", default="piper_follower")
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--left-port", default="can0")
    parser.add_argument("--right-port", default="can1")
    parser.add_argument("--speed-ratio", type=int, default=100)
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
    parser.add_argument("--policy-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-staleness-s", type=float, default=0.5)
    parser.add_argument("--refill-threshold", type=int, default=2)
    parser.add_argument("--actions-per-chunk", type=int, default=30)
    parser.add_argument(
        "--queue-merge-mode",
        choices=["replace", "append"],
        default="replace",
        help="How to handle a newly returned action chunk.",
    )
    parser.add_argument("--request-retry-s", type=float, default=0.2)
    parser.add_argument("--rpc-timeout-s", type=float, default=5.0)
    parser.add_argument("--rgb-msg-type", choices=["image", "compressed"], default="image")
    parser.add_argument("--tactile-msg-type", choices=["image", "compressed"], default="image")
    parser.add_argument("--rgb-transport-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--tactile-transport-format", choices=["jpeg", "png"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--left-top-topic", default=DEFAULT_IMAGE_TOPICS["observation.images.left_top"])
    parser.add_argument("--left-wrist-topic", default=DEFAULT_IMAGE_TOPICS["observation.images.left_wrist"])
    parser.add_argument("--right-wrist-topic", default=DEFAULT_IMAGE_TOPICS["observation.images.right_wrist"])
    parser.add_argument(
        "--tactile-left-outer-topic",
        default=DEFAULT_IMAGE_TOPICS["observation.images.tactile_left_outer"],
    )
    parser.add_argument(
        "--tactile-left-inner-topic",
        default=DEFAULT_IMAGE_TOPICS["observation.images.tactile_left_inner"],
    )
    parser.add_argument(
        "--tactile-right-outer-topic",
        default=DEFAULT_IMAGE_TOPICS["observation.images.tactile_right_outer"],
    )
    parser.add_argument(
        "--tactile-right-inner-topic",
        default=DEFAULT_IMAGE_TOPICS["observation.images.tactile_right_inner"],
    )
    parser.add_argument(
        "--ordered-image-keys",
        default=",".join(DEFAULT_ORDERED_IMAGE_KEYS),
        help="Comma-separated image key order expected by the server-side policy.",
    )
    parser.add_argument("--action-topic", default="/policy/action")
    parser.add_argument("--action-json-topic", default="/policy/action_json")
    parser.add_argument("--reset-topic", default="/policy/reset")
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--print-prompt-once", action="store_true")
    return parser.parse_args()


@dataclass
class LatestValue:
    value: np.ndarray | None = None
    receipt_time: float = 0.0


@dataclass
class ImageSnapshot:
    images: dict[str, np.ndarray]
    version: int


@dataclass
class ImageBuffer:
    image_keys: list[str]
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _version: int = field(default=0, init=False)
    images: dict[str, LatestValue] = field(default_factory=dict, init=False)

    def update_image(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self.images[key] = LatestValue(value=value, receipt_time=time.monotonic())
            self._version += 1

    def reset(self) -> None:
        with self._lock:
            self.images.clear()
            self._version += 1

    def get_snapshot(self, max_staleness_s: float) -> tuple[ImageSnapshot | None, str | None]:
        now = time.monotonic()
        with self._lock:
            images: dict[str, np.ndarray] = {}
            for key in self.image_keys:
                latest = self.images.get(key)
                if latest is None or latest.value is None:
                    return None, f"missing image stream: {key}"
                if now - latest.receipt_time > max_staleness_s:
                    return None, f"stale image stream: {key}"
                images[key] = latest.value.copy()
            return ImageSnapshot(images=images, version=self._version), None


class ActionQueue:
    def __init__(self):
        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def pop_left(self) -> np.ndarray | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def update(self, actions: list[np.ndarray], mode: str) -> None:
        with self._lock:
            if mode == "replace":
                self._queue.clear()
            self._queue.extend(actions)


class TactileBiPiperClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ordered_image_keys = parse_csv_items(args.ordered_image_keys)
        self.prompt = build_tactile_prompt(
            task=args.task,
            image_keys=self.ordered_image_keys,
            prompt_prefix=args.prompt_prefix,
            prompt_suffix=args.prompt_suffix,
        )
        if args.print_prompt_once:
            logging.info("Injected prompt: %s", self.prompt)

        (
            self.rospy,
            self.cv_bridge,
            self.cv2,
            self.sensor_msgs,
            self.std_msgs,
        ) = self._import_ros_modules()

        dataset_root = resolve_dataset_root(args.dataset_root, args.dataset_repo_id)
        self.dataset_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
        self.state_names = self._load_state_names()
        self.action_names = self._load_action_names()

        self.image_buffer = ImageBuffer(image_keys=self.ordered_image_keys)
        self.action_queue = ActionQueue()
        self.bridge = self.cv_bridge.CvBridge()
        self.reset_requested = False
        self.running = True
        self.request_counter = 0
        self.last_requested_version = -1
        self.last_request_time = 0.0
        self._last_wait_log_t = 0.0
        self._last_no_action_log_t = 0.0

        self.robot = self._make_robot()
        self.robot.connect(calibrate=True)
        logging.info("Connected bi_piper_follower | left=%s right=%s", args.left_port, args.right_port)

        self.channel = grpc.insecure_channel(args.server_address, options=grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.image_topics = {
            "observation.images.left_top": args.left_top_topic,
            "observation.images.left_wrist": args.left_wrist_topic,
            "observation.images.right_wrist": args.right_wrist_topic,
            "observation.images.tactile_left_outer": args.tactile_left_outer_topic,
            "observation.images.tactile_left_inner": args.tactile_left_inner_topic,
            "observation.images.tactile_right_outer": args.tactile_right_outer_topic,
            "observation.images.tactile_right_inner": args.tactile_right_inner_topic,
        }

        self.rospy.init_node("lerobot_tactile_bipiper_client", anonymous=False)
        self.action_pub = self.rospy.Publisher(args.action_topic, self.std_msgs.Float32MultiArray, queue_size=1)
        self.action_json_pub = None
        if args.action_json_topic.strip():
            self.action_json_pub = self.rospy.Publisher(args.action_json_topic, self.std_msgs.String, queue_size=1)

        self._handshake()
        self._init_subscribers()
        self.fetch_thread = threading.Thread(target=self._fetch_loop, name="tactile-fetch-loop", daemon=True)

    def _import_ros_modules(self):
        try:
            import cv2
            import rospy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import CompressedImage, Image
            from std_msgs.msg import Bool, Float32MultiArray, String
        except ImportError as exc:
            raise RuntimeError(
                "ROS1 client requires `rospy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, and `opencv-python`."
            ) from exc

        sensor_msgs = type(
            "SensorMsgs",
            (),
            {
                "Image": Image,
                "CompressedImage": CompressedImage,
            },
        )
        std_msgs = type(
            "StdMsgs",
            (),
            {
                "Bool": Bool,
                "Float32MultiArray": Float32MultiArray,
                "String": String,
            },
        )
        return rospy, type("CvBridgeModule", (), {"CvBridge": CvBridge}), cv2, sensor_msgs, std_msgs

    def _make_robot(self) -> BiPiperFollower:
        side_kwargs = dict(
            judge_flag=self.args.judge_flag,
            can_auto_init=self.args.can_auto_init,
            log_level=self.args.log_level,
            startup_sleep_s=self.args.startup_sleep_s,
            speed_ratio=self.args.speed_ratio,
            high_follow=self.args.high_follow,
            mode_refresh_interval_s=self.args.mode_refresh_interval_s,
            enable_on_connect=self.args.enable_on_connect,
            enable_timeout_s=self.args.enable_timeout_s,
            require_calibration=self.args.require_calibration,
            sync_gripper=self.args.sync_gripper,
            cameras={},
            disable_on_disconnect=self.args.disable_on_disconnect,
        )
        cfg = BiPiperFollowerConfig(
            type="bi_piper_follower",
            id=self.args.robot_id,
            calibration_dir=self.args.calibration_dir if self.args.calibration_dir else None,
            left_arm_config=PiperFollowerConfigBase(port=self.args.left_port, **side_kwargs),
            right_arm_config=PiperFollowerConfigBase(port=self.args.right_port, **side_kwargs),
        )
        return BiPiperFollower(cfg)

    def _load_state_names(self) -> list[str]:
        feature_info = self.dataset_meta.info.get("features", {}).get("observation.state", {})
        names = feature_info.get("names")
        if not isinstance(names, list) or not names:
            raise ValueError("Dataset metadata does not contain observation.state names.")
        return [str(name) for name in names]

    def _load_action_names(self) -> list[str]:
        feature_info = self.dataset_meta.info.get("features", {}).get("action", {})
        names = feature_info.get("names")
        if not isinstance(names, list) or not names:
            raise ValueError("Dataset metadata does not contain action names.")
        return [str(name) for name in names]

    def _handshake(self) -> None:
        self.stub.Ready(services_pb2.Empty(), timeout=self.args.rpc_timeout_s)
        setup = TactilePolicyRequestConfig(actions_per_chunk=self.args.actions_per_chunk)
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(setup)),  # nosec B301: same workspace controlled scripts
            timeout=self.args.rpc_timeout_s,
        )
        logging.info("Connected to tactile policy server at %s", self.args.server_address)

    def _init_subscribers(self) -> None:
        for key, topic in self.image_topics.items():
            if key not in self.ordered_image_keys:
                continue
            if "tactile_" in key:
                msg_type = (
                    self.sensor_msgs.CompressedImage if self.args.tactile_msg_type == "compressed" else self.sensor_msgs.Image
                )
            else:
                msg_type = self.sensor_msgs.CompressedImage if self.args.rgb_msg_type == "compressed" else self.sensor_msgs.Image
            self.rospy.Subscriber(topic, msg_type, self._make_image_callback(key), queue_size=1, buff_size=2**24)
            logging.info("Subscribed image stream %s <= %s", key, topic)

        if self.args.reset_topic.strip():
            self.rospy.Subscriber(self.args.reset_topic, self.std_msgs.Bool, self._reset_callback, queue_size=1)
            logging.info("Subscribed reset <= %s", self.args.reset_topic)

    def _make_image_callback(self, feature_key: str):
        def callback(msg: Any) -> None:
            try:
                image, encoding = self._decode_image_message(msg)
                image = to_rgb_uint8(image, encoding)
                self.image_buffer.update_image(feature_key, image)
            except Exception as exc:
                logging.warning("Failed to decode image for %s: %s", feature_key, exc)

        return callback

    def _decode_image_message(self, msg: Any) -> tuple[np.ndarray, str | None]:
        if isinstance(msg, self.sensor_msgs.CompressedImage):
            array = np.frombuffer(msg.data, dtype=np.uint8)
            image = self.cv2.imdecode(array, self.cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("cv2.imdecode returned None for compressed image.")
            return image, "bgr8"
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), getattr(msg, "encoding", None)

    def _reset_callback(self, msg: Any) -> None:
        if hasattr(msg, "data") and not bool(msg.data):
            return
        self.reset_requested = True

    def _build_state_vector(self, robot_obs: RobotObservation) -> np.ndarray:
        missing = [name for name in self.state_names if name not in robot_obs]
        if missing:
            raise KeyError(
                f"Robot observation is missing state keys required by dataset metadata: {missing}. "
                f"Available keys sample: {list(robot_obs.keys())[:20]}"
            )
        return np.asarray([robot_obs[name] for name in self.state_names], dtype=np.float32)

    def _build_action_dict(self, action_vector: np.ndarray) -> RobotAction:
        vector = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != len(self.action_names):
            raise ValueError(
                f"Action dim mismatch. Policy returned {vector.shape[0]} dims, dataset metadata expects {len(self.action_names)}."
            )
        return {name: float(value) for name, value in zip(self.action_names, vector, strict=True)}

    def _publish_action_debug(self, action_dict: RobotAction) -> None:
        vector = np.asarray([action_dict[name] for name in self.action_names], dtype=np.float32)
        self.action_pub.publish(self.std_msgs.Float32MultiArray(data=vector.tolist()))
        if self.action_json_pub is not None:
            self.action_json_pub.publish(self.std_msgs.String(data=json.dumps(action_dict, ensure_ascii=True)))

    def _send_observation_and_get_actions(self, images_snapshot: ImageSnapshot) -> list[np.ndarray]:
        robot_obs = self.robot.get_observation()
        state_vector = self._build_state_vector(robot_obs)

        observation = {OBS_STATE: state_vector}
        observation.update(images_snapshot.images)

        self.request_counter += 1
        packet = encode_observation_packet(
            observation=observation,
            ordered_image_keys=self.ordered_image_keys,
            task_prompt=self.prompt,
            robot_type=self.args.robot_type,
            timestep=self.request_counter,
            rgb_transport_format=self.args.rgb_transport_format,
            tactile_transport_format=self.args.tactile_transport_format,
            jpeg_quality=self.args.jpeg_quality,
            png_compression=self.args.png_compression,
        )
        payload = pickle.dumps(packet)  # nosec B301: same workspace controlled scripts
        request_iterator = send_bytes_in_chunks(
            payload,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        self.stub.SendObservations(request_iterator, timeout=self.args.rpc_timeout_s)
        response = self.stub.GetActions(services_pb2.Empty(), timeout=self.args.rpc_timeout_s)
        if not response.data:
            return []
        actions = pickle.loads(response.data)  # nosec B301: same workspace controlled scripts
        decoded_actions: list[np.ndarray] = []
        for action in actions:
            if isinstance(action, TactileTimedAction):
                decoded_actions.append(np.asarray(action.get_action(), dtype=np.float32))
        return decoded_actions

    def _fetch_loop(self) -> None:
        while self.running and not self.rospy.is_shutdown():
            if self.reset_requested:
                self.action_queue.clear()
                self.last_requested_version = -1
                self.reset_requested = False

            if self.action_queue.size() > self.args.refill_threshold:
                time.sleep(0.01)
                continue

            snapshot, reason = self.image_buffer.get_snapshot(self.args.max_staleness_s)
            if snapshot is None:
                now = time.monotonic()
                if now - self._last_wait_log_t > 2.0:
                    logging.info("Waiting for complete live image observation: %s", reason)
                    self._last_wait_log_t = now
                time.sleep(0.05)
                continue

            now = time.monotonic()
            should_resend_same_snapshot = (
                self.action_queue.size() == 0 and (now - self.last_request_time) >= self.args.request_retry_s
            )
            if snapshot.version == self.last_requested_version and not should_resend_same_snapshot:
                time.sleep(0.01)
                continue

            try:
                actions = self._send_observation_and_get_actions(snapshot)
                self.last_requested_version = snapshot.version
                self.last_request_time = now
                if actions:
                    self.action_queue.update(actions, mode=self.args.queue_merge_mode)
                    logging.info(
                        "Fetched action chunk from server | count=%d queue_size=%d",
                        len(actions),
                        self.action_queue.size(),
                    )
                else:
                    logging.warning("Server returned an empty action chunk.")
                    time.sleep(0.05)
            except grpc.RpcError as exc:
                logging.warning("RPC error while fetching action chunk: %s", exc)
                time.sleep(0.2)
            except Exception as exc:
                logging.warning("Failed to fetch action chunk: %s", exc)
                time.sleep(0.2)

    def run(self) -> None:
        self.fetch_thread.start()
        rate = self.rospy.Rate(self.args.policy_rate_hz)
        logging.info(
            "BiPiper tactile client started | rate=%.2fHz left=%s right=%s",
            self.args.policy_rate_hz,
            self.args.left_port,
            self.args.right_port,
        )

        try:
            while not self.rospy.is_shutdown():
                if self.reset_requested:
                    self.action_queue.clear()
                    self.last_requested_version = -1
                    self.reset_requested = False

                action = self.action_queue.pop_left()
                if action is not None:
                    action_dict = self._build_action_dict(action)
                    sent_action = self.robot.send_action(action_dict)
                    self._publish_action_debug(sent_action if sent_action else action_dict)
                else:
                    now = time.monotonic()
                    if now - self._last_no_action_log_t > 2.0:
                        logging.info("No action available yet. Waiting for server response.")
                        self._last_no_action_log_t = now
                rate.sleep()
        finally:
            self.running = False
            self.channel.close()
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            except Exception as exc:
                logging.warning("Failed to disconnect robot cleanly: %s", exc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    client = TactileBiPiperClient(args)
    client.run()


if __name__ == "__main__":
    main()

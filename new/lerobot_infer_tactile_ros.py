#!/usr/bin/env python

"""
Standalone ROS1 tactile inference entrypoint for the simplified tactile-prompt PI0.5 setup.

This file does not modify any existing source under `src/lerobot`.

Assumptions:
1. The checkpoint is a `pi05` policy trained on a dataset that contains:
   - observation.images.left_top
   - observation.images.left_wrist
   - observation.images.right_wrist
   - observation.images.tactile_left_outer
   - observation.images.tactile_left_inner
   - observation.images.tactile_right_outer
   - observation.images.tactile_right_inner
2. Live inference uses the latest frame from each topic. We do not enforce strict cross-topic synchronization.
3. The tactile heatmaps are published as ROS image topics by some node such as `tac_map`.
   The subscriber only needs the topic names, not the node name itself.
4. The policy still requires `observation.state`, so a robot state topic must also be provided.

Default behavior:
- Subscribe to 3 RGB topics + 4 tactile topics + 1 state topic.
- Rebuild the same sensor-aware prompt used by the standalone training script.
- Run PI0.5 inference and publish a single action vector as `std_msgs/Float32MultiArray`.
- Optionally publish a JSON action dictionary for downstream bridges.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.control_utils import predict_action

DEFAULT_ORDERED_IMAGE_KEYS = [
    "observation.images.left_top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.tactile_left_outer",
    "observation.images.tactile_left_inner",
    "observation.images.tactile_right_outer",
    "observation.images.tactile_right_inner",
]

DEFAULT_SENSOR_TEXT = {
    "observation.images.left_top": "Global RGB camera: observe the whole scene and object layout.",
    "observation.images.left_wrist": "Left wrist RGB camera: observe the local left gripper view.",
    "observation.images.right_wrist": "Right wrist RGB camera: observe the local right gripper view.",
    "observation.images.tactile_left_outer": (
        "Left gripper outer tactile heatmap: describe contact on the outer surface of the left gripper."
    ),
    "observation.images.tactile_left_inner": (
        "Left gripper inner tactile heatmap: describe contact on the inner grasping surface of the left gripper."
    ),
    "observation.images.tactile_right_outer": (
        "Right gripper outer tactile heatmap: describe contact on the outer surface of the right gripper."
    ),
    "observation.images.tactile_right_inner": (
        "Right gripper inner tactile heatmap: describe contact on the inner grasping surface of the right gripper."
    ),
}

DEFAULT_IMAGE_TOPICS = {
    "observation.images.left_top": "/camera/left_top/image_raw",
    "observation.images.left_wrist": "/camera/left_wrist/image_raw",
    "observation.images.right_wrist": "/camera/right_wrist/image_raw",
    "observation.images.tactile_left_outer": "/tac_map/tactile_left_left",
    "observation.images.tactile_left_inner": "/tac_map/tactile_left_right",
    "observation.images.tactile_right_outer": "/tac_map/tactile_right_left",
    "observation.images.tactile_right_inner": "/tac_map/tactile_right_right",
}

DEFAULT_PROMPT_PREFIX = (
    "You control a bimanual robot from multi-view RGB observations, tactile heatmaps, and robot state."
)
DEFAULT_PROMPT_SUFFIX = "Generate the next action chunk that follows the instruction."

POLICY_PREPROCESSOR_FILENAME = "policy_preprocessor.json"
POLICY_POSTPROCESSOR_FILENAME = "policy_postprocessor.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ROS1 tactile inference bridge for the simplified PI0.5 tactile setup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory containing config + safetensors.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id used for training.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root. Can be the dataset directory itself, or its parent directory.",
    )
    parser.add_argument("--task", required=True, help="High-level task instruction for prompt injection.")
    parser.add_argument("--robot-type", default="bi_piper_follower", help="Optional robot type string.")
    parser.add_argument("--device", default="auto", help="Inference device: auto/cpu/cuda/mps.")
    parser.add_argument("--policy-rate-hz", type=float, default=30.0, help="Action publication rate.")
    parser.add_argument(
        "--max-staleness-s",
        type=float,
        default=0.5,
        help="Reject observations if any required stream is older than this.",
    )
    parser.add_argument(
        "--state-topic",
        default="/robot/state",
        help="State topic used to build observation.state.",
    )
    parser.add_argument(
        "--state-msg-type",
        choices=["float32multiarray", "jointstate"],
        default="float32multiarray",
        help="ROS message type for the state topic.",
    )
    parser.add_argument(
        "--state-indices",
        default="",
        help="Optional comma-separated indices used to reorder/slice the incoming state vector.",
    )
    parser.add_argument("--action-topic", default="/policy/action", help="Output topic for raw action vector.")
    parser.add_argument(
        "--action-json-topic",
        default="/policy/action_json",
        help="Optional output topic for JSON action dictionary. Set empty string to disable.",
    )
    parser.add_argument(
        "--reset-topic",
        default="/policy/reset",
        help="Optional topic that resets the internal policy action queue.",
    )
    parser.add_argument(
        "--rgb-msg-type",
        choices=["image", "compressed"],
        default="image",
        help="ROS message type used by all RGB topics.",
    )
    parser.add_argument(
        "--tactile-msg-type",
        choices=["image", "compressed"],
        default="image",
        help="ROS message type used by all tactile topics.",
    )
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
        help="Comma-separated image key order expected by the policy.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default=DEFAULT_PROMPT_PREFIX,
        help="Text inserted before the sensor descriptions.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default=DEFAULT_PROMPT_SUFFIX,
        help="Text inserted after the task instruction.",
    )
    parser.add_argument(
        "--rebuild-processors",
        action="store_true",
        help="Ignore saved processor JSON files in the checkpoint and rebuild processors from config + dataset stats.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable torch autocast when running on CUDA.",
    )
    parser.add_argument(
        "--print-prompt-once",
        action="store_true",
        help="Log the final injected prompt once at startup.",
    )
    return parser.parse_args()


def parse_csv_items(csv_text: str) -> list[str]:
    return [item.strip() for item in csv_text.split(",") if item.strip()]


def parse_indices(csv_text: str) -> list[int] | None:
    values = parse_csv_items(csv_text)
    if not values:
        return None
    return [int(value) for value in values]


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _label_for_image_key(image_key: str) -> str:
    if image_key in DEFAULT_SENSOR_TEXT:
        return DEFAULT_SENSOR_TEXT[image_key]

    suffix = image_key.split(".")[-1].replace("_", " ")
    return f"{suffix.title()}: auxiliary observation stream available for action prediction."


def build_tactile_prompt(
    task: str,
    image_keys: list[str],
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
) -> str:
    parts: list[str] = []
    if prompt_prefix.strip():
        parts.append(prompt_prefix.strip())
    for image_key in image_keys:
        parts.append(_label_for_image_key(image_key))
    parts.append(f"Instruction: {task.strip()}")
    if prompt_suffix.strip():
        parts.append(prompt_suffix.strip())
    return " ".join(part for part in parts if part)


def resolve_dataset_root(root_arg: str, repo_id: str) -> Path:
    root_path = Path(root_arg).expanduser()
    repo_path = Path(*repo_id.split("/"))
    candidates = [
        root_path,
        root_path / repo_path,
    ]
    for candidate in candidates:
        if (candidate / "meta").exists():
            return candidate
    return root_path


def configure_policy_features_for_tactile_metadata(
    policy_cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata,
    ordered_image_keys: list[str],
) -> None:
    all_features = dataset_to_policy_features(ds_meta.features)
    output_features = {key: ft for key, ft in all_features.items() if ft.type is FeatureType.ACTION}
    ordered_input_features: dict[str, Any] = {}

    missing_image_keys: list[str] = []
    for image_key in ordered_image_keys:
        feature = all_features.get(image_key)
        if feature is None:
            missing_image_keys.append(image_key)
            continue
        if feature.type is not FeatureType.VISUAL:
            raise ValueError(f"Expected visual feature for '{image_key}', got {feature.type}.")
        ordered_input_features[image_key] = feature

    if missing_image_keys:
        raise ValueError(
            "Dataset metadata is missing required tactile/image keys: "
            f"{missing_image_keys}. Available keys: {sorted(ds_meta.features.keys())}"
        )

    for key, feature in all_features.items():
        if key in output_features or key in ordered_input_features:
            continue
        ordered_input_features[key] = feature

    policy_cfg.input_features = ordered_input_features
    policy_cfg.output_features = output_features


def maybe_get_action_names(ds_meta: LeRobotDatasetMetadata, action_dim: int) -> list[str] | None:
    feature_info = ds_meta.info.get("features", {}).get(ACTION, {})
    names = feature_info.get("names")
    if isinstance(names, list) and len(names) == action_dim:
        return [str(name) for name in names]
    return None


def to_rgb_uint8(image: np.ndarray, encoding: str | None = None) -> np.ndarray:
    import cv2

    img = np.asarray(image)

    if img.ndim == 2:
        img = _normalize_to_uint8(img)
        img = np.repeat(img[:, :, None], 3, axis=2)
        return np.ascontiguousarray(img)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image ndim={img.ndim}. Expected 2 or 3.")

    if img.shape[2] == 1:
        img = _normalize_to_uint8(img[:, :, 0])
        img = np.repeat(img[:, :, None], 3, axis=2)
        return np.ascontiguousarray(img)

    if img.shape[2] >= 3:
        img = img[:, :, :3]
        img = _normalize_to_uint8(img)
        enc = (encoding or "").lower()
        if enc.startswith("bgr"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img)

    raise ValueError(f"Unsupported image shape {img.shape}.")


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        dtype_max = np.iinfo(arr.dtype).max
        if dtype_max <= 0:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = arr.astype(np.float32) / float(dtype_max)
        return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        min_value = float(np.nanmin(arr)) if arr.size else 0.0
        if max_value <= 1.0 and min_value >= 0.0:
            return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


@dataclass
class LatestValue:
    value: np.ndarray | None = None
    stamp: float = 0.0
    receipt_time: float = 0.0


@dataclass
class ObservationBuffer:
    image_keys: list[str]
    expected_state_dim: int
    state_indices: list[int] | None = None
    _lock: Lock = field(default_factory=Lock, init=False)
    images: dict[str, LatestValue] = field(default_factory=dict, init=False)
    state: LatestValue = field(default_factory=LatestValue, init=False)

    def update_image(self, key: str, value: np.ndarray, stamp: float) -> None:
        with self._lock:
            self.images[key] = LatestValue(value=value, stamp=stamp, receipt_time=time.monotonic())

    def update_state(self, value: np.ndarray, stamp: float) -> None:
        vector = np.asarray(value, dtype=np.float32)
        if self.state_indices is not None:
            if max(self.state_indices) >= vector.shape[0]:
                raise ValueError(
                    f"State indices {self.state_indices} exceed incoming state length {vector.shape[0]}."
                )
            vector = vector[self.state_indices]
        if vector.shape[0] != self.expected_state_dim:
            raise ValueError(
                f"Incoming state dim {vector.shape[0]} does not match expected dim {self.expected_state_dim}."
            )
        with self._lock:
            self.state = LatestValue(value=vector, stamp=stamp, receipt_time=time.monotonic())

    def reset(self) -> None:
        with self._lock:
            self.images.clear()
            self.state = LatestValue()

    def get_observation(self, max_staleness_s: float) -> tuple[dict[str, np.ndarray] | None, str | None]:
        now = time.monotonic()
        with self._lock:
            if self.state.value is None:
                return None, "state is missing"
            if now - self.state.receipt_time > max_staleness_s:
                return None, "state is stale"

            observation: dict[str, np.ndarray] = {OBS_STATE: self.state.value.copy()}
            for key in self.image_keys:
                latest = self.images.get(key)
                if latest is None or latest.value is None:
                    return None, f"missing image stream: {key}"
                if now - latest.receipt_time > max_staleness_s:
                    return None, f"stale image stream: {key}"
                observation[key] = latest.value.copy()
        return observation, None


class TactileRosInferenceNode:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ordered_image_keys = parse_csv_items(args.ordered_image_keys)
        self.prompt = build_tactile_prompt(
            task=args.task,
            image_keys=self.ordered_image_keys,
            prompt_prefix=args.prompt_prefix,
            prompt_suffix=args.prompt_suffix,
        )
        self.device = resolve_device(args.device)
        self.use_amp = self.device == "cuda" and not args.disable_amp
        self.reset_requested = False
        self._last_wait_log_t = 0.0
        self._last_infer_log_t = 0.0
        self._was_ready = False

        (
            self.rospy,
            self.cv_bridge,
            self.cv2,
            self.sensor_msgs,
            self.std_msgs,
        ) = self._import_ros_modules()

        logging.info("Resolved device: %s", self.device)
        logging.info("Ordered image keys: %s", self.ordered_image_keys)
        if args.print_prompt_once:
            logging.info("Injected prompt: %s", self.prompt)

        dataset_root = resolve_dataset_root(args.dataset_root, args.dataset_repo_id)
        logging.info("Resolved dataset root: %s", dataset_root)
        self.ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)

        policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        if policy_cfg.type != "pi05":
            raise ValueError(f"This script currently targets pi05 only, got '{policy_cfg.type}'.")
        policy_cfg.pretrained_path = args.checkpoint
        policy_cfg.device = self.device
        configure_policy_features_for_tactile_metadata(policy_cfg, self.ds_meta, self.ordered_image_keys)

        self.policy = make_policy(policy_cfg, ds_meta=self.ds_meta)
        self.preprocessor, self.postprocessor = self._make_processors(policy_cfg)
        self.policy.reset()

        self.expected_state_dim = int(policy_cfg.input_features[OBS_STATE].shape[0])
        self.action_dim = int(policy_cfg.output_features[ACTION].shape[0])
        self.action_names = maybe_get_action_names(self.ds_meta, self.action_dim)
        self.buffer = ObservationBuffer(
            image_keys=self.ordered_image_keys,
            expected_state_dim=self.expected_state_dim,
            state_indices=parse_indices(args.state_indices),
        )
        self.bridge = self.cv_bridge.CvBridge()

        self.image_topics = {
            "observation.images.left_top": args.left_top_topic,
            "observation.images.left_wrist": args.left_wrist_topic,
            "observation.images.right_wrist": args.right_wrist_topic,
            "observation.images.tactile_left_outer": args.tactile_left_outer_topic,
            "observation.images.tactile_left_inner": args.tactile_left_inner_topic,
            "observation.images.tactile_right_outer": args.tactile_right_outer_topic,
            "observation.images.tactile_right_inner": args.tactile_right_inner_topic,
        }

        self.rospy.init_node("lerobot_tactile_pi05_infer", anonymous=False)
        self.action_pub = self.rospy.Publisher(
            args.action_topic, self.std_msgs.Float32MultiArray, queue_size=1
        )
        self.action_json_pub = None
        if args.action_json_topic.strip():
            self.action_json_pub = self.rospy.Publisher(args.action_json_topic, self.std_msgs.String, queue_size=1)

        self._init_subscribers()

    def _import_ros_modules(self):
        try:
            import cv2
            import rospy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import CompressedImage, Image, JointState
            from std_msgs.msg import Bool, Float32MultiArray, String
        except ImportError as exc:
            raise RuntimeError(
                "ROS1 inference requires `rospy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, and `opencv-python` "
                "in the current environment."
            ) from exc

        sensor_msgs = type(
            "SensorMsgs",
            (),
            {
                "Image": Image,
                "CompressedImage": CompressedImage,
                "JointState": JointState,
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

    def _make_processors(self, policy_cfg: PreTrainedConfig):
        checkpoint_dir = Path(self.args.checkpoint)
        has_saved_processors = (
            (checkpoint_dir / POLICY_PREPROCESSOR_FILENAME).exists()
            and (checkpoint_dir / POLICY_POSTPROCESSOR_FILENAME).exists()
        )
        if has_saved_processors and not self.args.rebuild_processors:
            logging.info("Loading saved pre/post processors from checkpoint.")
            return make_pre_post_processors(
                policy_cfg,
                pretrained_path=str(checkpoint_dir),
                dataset_stats=self.ds_meta.stats,
            )

        logging.info("Rebuilding pre/post processors from policy config and dataset stats.")
        return make_pre_post_processors(
            policy_cfg,
            dataset_stats=self.ds_meta.stats,
        )

    def _init_subscribers(self) -> None:
        for key, topic in self.image_topics.items():
            if key not in self.ordered_image_keys:
                continue
            msg_type = (
                self.sensor_msgs.CompressedImage
                if self._is_tactile_key(key) and self.args.tactile_msg_type == "compressed"
                else self.sensor_msgs.Image
                if self._is_tactile_key(key)
                else self.sensor_msgs.CompressedImage
                if self.args.rgb_msg_type == "compressed"
                else self.sensor_msgs.Image
            )
            self.rospy.Subscriber(topic, msg_type, self._make_image_callback(key), queue_size=1, buff_size=2**24)
            logging.info("Subscribed image stream %s <= %s", key, topic)

        if self.args.state_msg_type == "jointstate":
            state_msg_type = self.sensor_msgs.JointState
        else:
            state_msg_type = self.std_msgs.Float32MultiArray
        self.rospy.Subscriber(self.args.state_topic, state_msg_type, self._state_callback, queue_size=1)
        logging.info("Subscribed state <= %s (%s)", self.args.state_topic, self.args.state_msg_type)

        if self.args.reset_topic.strip():
            self.rospy.Subscriber(self.args.reset_topic, self.std_msgs.Bool, self._reset_callback, queue_size=1)
            logging.info("Subscribed reset <= %s", self.args.reset_topic)

    def _make_image_callback(self, feature_key: str):
        def callback(msg: Any) -> None:
            try:
                stamp = self._extract_stamp(msg)
                image, encoding = self._decode_image_message(msg)
                image = to_rgb_uint8(image, encoding)
                self.buffer.update_image(feature_key, image, stamp)
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

    def _state_callback(self, msg: Any) -> None:
        try:
            if self.args.state_msg_type == "jointstate":
                vector = np.asarray(msg.position, dtype=np.float32)
            else:
                vector = np.asarray(msg.data, dtype=np.float32)
            self.buffer.update_state(vector, self._extract_stamp(msg))
        except Exception as exc:
            logging.warning("Failed to decode state message: %s", exc)

    def _reset_callback(self, msg: Any) -> None:
        if hasattr(msg, "data") and not bool(msg.data):
            return
        self.reset_requested = True

    @staticmethod
    def _is_tactile_key(key: str) -> bool:
        return "tactile_" in key

    def _extract_stamp(self, msg: Any) -> float:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return time.time()
        try:
            value = float(stamp.to_sec())
        except Exception:
            return time.time()
        return value if value > 0 else time.time()

    def _publish_action(self, action: np.ndarray) -> None:
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        action_msg = self.std_msgs.Float32MultiArray(data=vector.tolist())
        self.action_pub.publish(action_msg)

        if self.action_json_pub is not None:
            if self.action_names is not None:
                payload = {name: float(value) for name, value in zip(self.action_names, vector, strict=True)}
            else:
                payload = {str(index): float(value) for index, value in enumerate(vector)}
            self.action_json_pub.publish(self.std_msgs.String(data=json.dumps(payload, ensure_ascii=True)))

    def _maybe_reset_policy(self) -> None:
        if not self.reset_requested:
            return
        logging.info("Reset signal received. Clearing policy action queue.")
        self.policy.reset()
        self.reset_requested = False

    def run(self) -> None:
        rate = self.rospy.Rate(self.args.policy_rate_hz)
        logging.info(
            "Inference loop started | rate=%.2fHz state_dim=%d action_dim=%d",
            self.args.policy_rate_hz,
            self.expected_state_dim,
            self.action_dim,
        )

        while not self.rospy.is_shutdown():
            self._maybe_reset_policy()

            observation, reason = self.buffer.get_observation(max_staleness_s=self.args.max_staleness_s)
            if observation is None:
                now = time.monotonic()
                if self._was_ready:
                    logging.warning("Observation stream became invalid: %s. Resetting policy queue.", reason)
                    self.policy.reset()
                    self._was_ready = False
                elif now - self._last_wait_log_t > 2.0:
                    logging.info("Waiting for complete live observation: %s", reason)
                    self._last_wait_log_t = now
                rate.sleep()
                continue

            self._was_ready = True
            start_t = time.perf_counter()
            action_tensor = predict_action(
                observation=observation,
                policy=self.policy,
                device=torch.device(self.device),
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.use_amp,
                task=self.prompt,
                robot_type=self.args.robot_type,
            )
            action_np = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            self._publish_action(action_np)

            infer_dt_ms = (time.perf_counter() - start_t) * 1000.0
            now = time.monotonic()
            if now - self._last_infer_log_t > 2.0:
                logging.info("Published action | latency=%.2fms", infer_dt_ms)
                self._last_infer_log_t = now

            rate.sleep()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    node = TactileRosInferenceNode(args)
    node.run()


if __name__ == "__main__":
    main()

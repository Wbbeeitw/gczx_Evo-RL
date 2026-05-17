#!/usr/bin/env python

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION

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


@dataclass
class TactilePolicyRequestConfig:
    actions_per_chunk: int | None = None


@dataclass
class EncodedTactileObservation:
    timestamp: float
    timestep: int
    state: np.ndarray
    encoded_images: dict[str, bytes]
    image_encodings: dict[str, str]
    task_prompt: str
    robot_type: str = ""

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep


@dataclass
class TactileTimedAction:
    timestamp: float
    timestep: int
    action: np.ndarray

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep

    def get_action(self) -> np.ndarray:
        return self.action


def parse_csv_items(csv_text: str) -> list[str]:
    return [item.strip() for item in csv_text.split(",") if item.strip()]


def parse_indices(csv_text: str) -> list[int] | None:
    values = parse_csv_items(csv_text)
    if not values:
        return None
    return [int(value) for value in values]


def resolve_device(requested: str) -> str:
    import torch

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


def to_rgb_uint8(image: np.ndarray, encoding: str | None = None) -> np.ndarray:
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


def is_tactile_key(image_key: str) -> bool:
    return "tactile_" in image_key


def encode_image_for_transport(
    image: np.ndarray,
    transport_format: str,
    jpeg_quality: int,
    png_compression: int,
) -> bytes:
    image_uint8 = to_rgb_uint8(image, encoding=None)
    if transport_format == "jpeg":
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    elif transport_format == "png":
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    else:
        raise ValueError(f"Unsupported transport format '{transport_format}'.")
    if not ok:
        raise ValueError(f"Failed to encode image as {transport_format}.")
    return bytes(buffer)


def decode_image_from_transport(buffer: bytes, transport_format: str) -> np.ndarray:
    if transport_format not in {"jpeg", "png"}:
        raise ValueError(f"Unsupported transport format '{transport_format}'.")
    image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode {transport_format} buffer.")
    return to_rgb_uint8(image, encoding="bgr8")


def encode_observation_packet(
    observation: dict[str, np.ndarray],
    ordered_image_keys: list[str],
    task_prompt: str,
    robot_type: str,
    timestep: int,
    rgb_transport_format: str,
    tactile_transport_format: str,
    jpeg_quality: int,
    png_compression: int,
) -> EncodedTactileObservation:
    encoded_images: dict[str, bytes] = {}
    image_encodings: dict[str, str] = {}

    for image_key in ordered_image_keys:
        transport_format = tactile_transport_format if is_tactile_key(image_key) else rgb_transport_format
        encoded_images[image_key] = encode_image_for_transport(
            observation[image_key],
            transport_format=transport_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )
        image_encodings[image_key] = transport_format

    return EncodedTactileObservation(
        timestamp=time.time(),
        timestep=timestep,
        state=np.asarray(observation["observation.state"], dtype=np.float32),
        encoded_images=encoded_images,
        image_encodings=image_encodings,
        task_prompt=task_prompt,
        robot_type=robot_type,
    )


def decode_observation_packet(packet: EncodedTactileObservation) -> dict[str, np.ndarray]:
    observation = {"observation.state": np.asarray(packet.state, dtype=np.float32)}
    for image_key, buffer in packet.encoded_images.items():
        transport_format = packet.image_encodings[image_key]
        observation[image_key] = decode_image_from_transport(buffer, transport_format)
    return observation

#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec B403: internal trusted transport
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import grpc
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE

from tactile_client_server_common import (
    DEFAULT_ORDERED_IMAGE_KEYS,
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX,
    EncodedTactileObservation,
    TactilePolicyRequestConfig,
    TactileTimedAction,
    build_tactile_prompt,
    encode_observation_packet,
    parse_csv_items,
    resolve_dataset_root,
)

TACTILE_IMAGE_KEYS = [
    "observation.images.tactile_left_outer",
    "observation.images.tactile_left_inner",
    "observation.images.tactile_right_outer",
    "observation.images.tactile_right_inner",
]

RGB_IMAGE_KEYS = [
    "observation.images.left_top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock local loopback client for tactile pi05 policy demos on a single server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Local gRPC policy server address.")
    parser.add_argument("--dataset-repo-id", required=True, help="Dataset repo id used for training.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root or dataset parent root.")
    parser.add_argument("--task", required=True, help="High-level task instruction for prompt injection.")
    parser.add_argument("--robot-type", default="bi_piper_follower")
    parser.add_argument(
        "--ordered-image-keys",
        default=",".join(DEFAULT_ORDERED_IMAGE_KEYS),
        help="Comma-separated image key order expected by the server-side policy.",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Which dataset episode to replay for RGB bases.")
    parser.add_argument("--steps", type=int, default=30, help="Number of demo loop iterations.")
    parser.add_argument("--frame-stride", type=int, default=3, help="Frame stride inside the selected episode.")
    parser.add_argument("--policy-rate-hz", type=float, default=2.0, help="Loop rate for the demo.")
    parser.add_argument("--actions-per-chunk", type=int, default=30, help="Requested action chunk size.")
    parser.add_argument("--rpc-timeout-s", type=float, default=10.0)
    parser.add_argument("--rgb-transport-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--tactile-transport-format", choices=["jpeg", "png"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--state-action-gain", type=float, default=0.20, help="How strongly the last action drives the next state.")
    parser.add_argument("--state-base-blend", type=float, default=0.10, help="How much the current dataset frame anchors the simulated state.")
    parser.add_argument("--state-noise-std", type=float, default=0.0, help="Gaussian noise added to the simulated state.")
    parser.add_argument("--tactile-intensity", type=float, default=0.75, help="Blend weight for synthetic tactile overlays.")
    parser.add_argument("--save-dir", default="", help="Optional directory to save per-step composite PNGs + summary JSONL.")
    parser.add_argument("--print-prompt-once", action="store_true")
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    return parser.parse_args()


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_image_to_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy(image)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported image shape {arr.shape}.")
    arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        min_value = float(np.nanmin(arr)) if arr.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _normalize_feature_value(value: float) -> float:
    return 0.5 * (np.tanh(float(value)) + 1.0)


def _pick_dim(vector: np.ndarray, index: int) -> float:
    if vector.size == 0:
        return 0.0
    return float(vector[index % vector.size])


def _load_named_feature_list(ds_meta: LeRobotDatasetMetadata, feature_key: str, fallback_prefix: str) -> list[str]:
    feature_info = ds_meta.info.get("features", {}).get(feature_key, {})
    names = feature_info.get("names")
    shape = feature_info.get("shape", [])
    dim = int(shape[0]) if shape else 0
    if isinstance(names, list) and len(names) == dim:
        return [str(name) for name in names]
    return [f"{fallback_prefix}_{index}" for index in range(dim)]


def _episode_range(ds_meta: LeRobotDatasetMetadata, episode_index: int) -> tuple[int, int]:
    episodes = ds_meta.episodes
    if episodes is None or len(episodes) == 0:
        raise ValueError("Dataset metadata does not contain episode index ranges.")
    if not (0 <= episode_index < len(episodes)):
        raise IndexError(f"episode_index={episode_index} is out of range for {len(episodes)} episodes.")
    row = episodes.iloc[episode_index] if hasattr(episodes, "iloc") else episodes[episode_index]
    start = int(row["dataset_from_index"])
    stop = int(row["dataset_to_index"])
    if stop <= start:
        raise ValueError(f"Episode {episode_index} has invalid range [{start}, {stop}).")
    return start, stop


def _overlay_banner(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 255, 180), 1, cv2.LINE_AA)
    return canvas


def _make_tactile_overlay(
    base_image: np.ndarray,
    state: np.ndarray,
    action: np.ndarray | None,
    key_index: int,
    step_index: int,
    blend: float,
) -> np.ndarray:
    base_rgb = base_image.copy()
    height, width = base_rgb.shape[:2]

    overlay_gray = np.zeros((height, width), dtype=np.uint8)
    x_value = _pick_dim(state, key_index * 2)
    y_value = _pick_dim(state, key_index * 2 + 1)
    a_value = 0.0 if action is None else _pick_dim(action, key_index * 3)
    b_value = 0.0 if action is None else _pick_dim(action, key_index * 3 + 1)

    center_x = int(np.clip(_normalize_feature_value(x_value + 0.5 * a_value) * (width - 1), 0, width - 1))
    center_y = int(np.clip(_normalize_feature_value(y_value + 0.5 * b_value) * (height - 1), 0, height - 1))
    radius = max(18, int(min(height, width) * (0.08 + 0.06 * _normalize_feature_value(a_value - b_value))))
    intensity = int(80 + 175 * _normalize_feature_value(x_value + y_value + a_value))

    cv2.circle(overlay_gray, (center_x, center_y), radius, intensity, thickness=-1)
    cv2.circle(
        overlay_gray,
        ((center_x + 37 * (step_index + 1)) % width, (center_y + 23 * (step_index + 1)) % height),
        max(12, radius // 2),
        max(50, intensity // 2),
        thickness=-1,
    )
    overlay_gray = cv2.GaussianBlur(overlay_gray, (0, 0), sigmaX=20.0, sigmaY=20.0)
    overlay_bgr = cv2.applyColorMap(overlay_gray, cv2.COLORMAP_TURBO)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    mixed = cv2.addWeighted(base_rgb, 1.0 - blend, overlay_rgb, blend, 0.0)
    return mixed


def _compose_panel(images: list[np.ndarray], footer_lines: list[str]) -> np.ndarray:
    resized = [cv2.resize(image, (360, 240), interpolation=cv2.INTER_AREA) for image in images]
    blank = np.full_like(resized[0], 16)
    top_row = cv2.hconcat([resized[0], resized[1], resized[2], resized[3]])
    bottom_row = cv2.hconcat([resized[4], resized[5], resized[6], blank])
    body = cv2.vconcat([top_row, bottom_row])
    footer_h = 110
    footer = np.full((footer_h, body.shape[1], 3), 18, dtype=np.uint8)
    for line_index, line in enumerate(footer_lines[:4]):
        cv2.putText(
            footer,
            line,
            (16, 28 + line_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return cv2.vconcat([body, footer])


class MockLoopbackClient:
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

        dataset_root = resolve_dataset_root(args.dataset_root, args.dataset_repo_id)
        logging.info("Resolved dataset root: %s", dataset_root)
        self.ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)
        self.dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root)
        self.episode_start, self.episode_stop = _episode_range(self.ds_meta, args.episode_index)
        self.episode_length = self.episode_stop - self.episode_start
        self.state_names = _load_named_feature_list(self.ds_meta, OBS_STATE, "state")
        self.action_names = _load_named_feature_list(self.ds_meta, "action", "action")

        first_item = self.dataset[self.episode_start]
        self.current_state = _to_numpy(first_item[OBS_STATE]).astype(np.float32).reshape(-1)
        self.last_action: np.ndarray | None = None
        self.request_counter = 0

        self.channel = grpc.insecure_channel(args.server_address, options=grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.save_dir = Path(args.save_dir).expanduser() if args.save_dir.strip() else None
        self.summary_path = None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.summary_path = self.save_dir / "summary.jsonl"

    def _handshake(self) -> None:
        self.stub.Ready(services_pb2.Empty(), timeout=self.args.rpc_timeout_s)
        setup = TactilePolicyRequestConfig(actions_per_chunk=self.args.actions_per_chunk)
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(setup)),  # nosec B301: same workspace controlled scripts
            timeout=self.args.rpc_timeout_s,
        )
        logging.info("Connected to tactile policy server at %s", self.args.server_address)

    def _dataset_item_for_step(self, step_index: int) -> dict:
        offset = (step_index * self.args.frame_stride) % self.episode_length
        dataset_index = self.episode_start + offset
        return self.dataset[dataset_index]

    def _build_rgb_images(self, item: dict, step_index: int) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for image_key in RGB_IMAGE_KEYS:
            image = _tensor_image_to_uint8_hwc(item[image_key])
            title = image_key.split(".")[-1]
            subtitle = f"step={step_index:03d}"
            images[image_key] = _overlay_banner(image, title=title, subtitle=subtitle)
        return images

    def _build_tactile_images(self, item: dict, step_index: int) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for key_index, image_key in enumerate(TACTILE_IMAGE_KEYS):
            base_image = _tensor_image_to_uint8_hwc(item[image_key])
            tactile_image = _make_tactile_overlay(
                base_image=base_image,
                state=self.current_state,
                action=self.last_action,
                key_index=key_index,
                step_index=step_index,
                blend=self.args.tactile_intensity,
            )
            title = image_key.split(".")[-1]
            subtitle = f"loopback tactile {step_index:03d}"
            images[image_key] = _overlay_banner(tactile_image, title=title, subtitle=subtitle)
        return images

    def _build_observation(self, item: dict, step_index: int) -> tuple[dict[str, np.ndarray], int]:
        item_state = _to_numpy(item[OBS_STATE]).astype(np.float32).reshape(-1)
        if item_state.shape != self.current_state.shape:
            raise ValueError(
                f"State shape mismatch: dataset frame has {item_state.shape}, simulated state has {self.current_state.shape}."
            )

        rgb_images = self._build_rgb_images(item, step_index)
        tactile_images = self._build_tactile_images(item, step_index)
        observation = {OBS_STATE: self.current_state.astype(np.float32)}
        observation.update(rgb_images)
        observation.update(tactile_images)

        dataset_offset = (step_index * self.args.frame_stride) % self.episode_length
        dataset_index = self.episode_start + dataset_offset
        return observation, dataset_index

    def _request_action_chunk(self, observation: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.request_counter += 1
        packet: EncodedTactileObservation = encode_observation_packet(
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
            log_prefix="[MOCK CLIENT] Observation",
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

    def _update_state(self, item: dict) -> None:
        item_state = _to_numpy(item[OBS_STATE]).astype(np.float32).reshape(-1)
        next_state = self.current_state.copy()
        if self.last_action is not None:
            action_drive = self.last_action[: next_state.shape[0]]
            next_state = next_state + self.args.state_action_gain * action_drive
        next_state = (1.0 - self.args.state_base_blend) * next_state + self.args.state_base_blend * item_state
        if self.args.state_noise_std > 0.0:
            next_state = next_state + np.random.normal(0.0, self.args.state_noise_std, size=next_state.shape).astype(
                np.float32
            )
        self.current_state = next_state.astype(np.float32)

    def _action_summary(self, action: np.ndarray | None) -> str:
        if action is None:
            return "action=None"
        pairs = []
        for index, value in enumerate(action[: min(6, action.shape[0])]):
            name = self.action_names[index] if index < len(self.action_names) else f"a{index}"
            pairs.append(f"{name}={float(value):.3f}")
        return "action_head: " + ", ".join(pairs)

    def _state_summary(self) -> str:
        pairs = []
        for index, value in enumerate(self.current_state[: min(6, self.current_state.shape[0])]):
            name = self.state_names[index] if index < len(self.state_names) else f"s{index}"
            pairs.append(f"{name}={float(value):.3f}")
        return "state_head: " + ", ".join(pairs)

    def _save_step_artifacts(
        self,
        step_index: int,
        dataset_index: int,
        observation: dict[str, np.ndarray],
        action: np.ndarray | None,
    ) -> None:
        if self.save_dir is None or self.summary_path is None:
            return

        panel_images = [observation[key] for key in RGB_IMAGE_KEYS + TACTILE_IMAGE_KEYS]
        footer_lines = [
            f"step={step_index:03d} dataset_index={dataset_index} server={self.args.server_address}",
            self._state_summary(),
            self._action_summary(action),
            f"task={self.args.task}",
        ]
        panel = _compose_panel(panel_images, footer_lines)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.save_dir / f"frame_{step_index:04d}.png"), panel_bgr)

        record = {
            "step": step_index,
            "dataset_index": dataset_index,
            "state": self.current_state.tolist(),
            "action": None if action is None else action.tolist(),
        }
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def run(self) -> None:
        self._handshake()
        logging.info(
            "Mock loopback started | episode=%d frames=%d steps=%d rate=%.2fHz",
            self.args.episode_index,
            self.episode_length,
            self.args.steps,
            self.args.policy_rate_hz,
        )
        sleep_s = 0.0 if self.args.policy_rate_hz <= 0 else 1.0 / self.args.policy_rate_hz
        start_t = time.perf_counter()

        try:
            for step_index in range(self.args.steps):
                step_t0 = time.perf_counter()
                item = self._dataset_item_for_step(step_index)
                observation, dataset_index = self._build_observation(item, step_index)
                action_chunk = self._request_action_chunk(observation)
                action = action_chunk[0] if action_chunk else None
                self.last_action = action

                logging.info(
                    "step=%03d dataset_index=%d action_chunk=%d | %s | %s",
                    step_index,
                    dataset_index,
                    len(action_chunk),
                    self._state_summary(),
                    self._action_summary(action),
                )

                self._save_step_artifacts(step_index, dataset_index, observation, action)
                self._update_state(item)

                elapsed = time.perf_counter() - step_t0
                if sleep_s > elapsed:
                    time.sleep(sleep_s - elapsed)
        finally:
            total_s = time.perf_counter() - start_t
            logging.info("Mock loopback finished in %.2fs", total_s)
            self.channel.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    client = MockLoopbackClient(args)
    client.run()


if __name__ == "__main__":
    main()

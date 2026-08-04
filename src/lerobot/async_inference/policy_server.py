# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import copy
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RTCInferenceMetadata,
    RemoteActionChunk,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    decode_observation_images,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


@dataclass(frozen=True)
class _TorchRNGSnapshot:
    device: torch.device
    state: torch.Tensor


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        self._policy_inference_lock = threading.Lock()
        self._diagnostic_state_lock = threading.Lock()
        self._diagnostic_thread: threading.Thread | None = None
        self._diagnostic_running = False
        self._last_qwen_text_debug_at = float("-inf")
        self._last_tactile_action_debug_at = float("-inf")
        self._tactile_image_keys: tuple[str, ...] = ()
        self._tactile_baseline_samples: dict[str, list[torch.Tensor]] = {}
        self._tactile_baselines: dict[str, torch.Tensor] = {}
        self._debug_qwen_text_enabled = False
        self._debug_tactile_counterfactual_enabled = False

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.rtc_config = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[
            dict[str, Any], dict[str, Any]
        ] | None = None
        self.postprocessor: PolicyProcessorPipeline[
            PolicyAction, PolicyAction
        ] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def _wait_for_diagnostic_thread(self) -> None:
        with self._diagnostic_state_lock:
            thread = self._diagnostic_thread
        if thread is not None and thread.is_alive():
            self.logger.info("Waiting for the active XR0 diagnostic task to finish.")
            thread.join()

    def _reset_diagnostic_state(self) -> None:
        self._last_qwen_text_debug_at = float("-inf")
        self._last_tactile_action_debug_at = float("-inf")
        self._tactile_baseline_samples = {key: [] for key in self._tactile_image_keys}
        self._tactile_baselines = {}
        with self._diagnostic_state_lock:
            self._diagnostic_thread = None
            self._diagnostic_running = False

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(
                f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}"
            )

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.rtc_config = policy_specs.rtc_config

        policy_class = get_policy_class(self.policy_type)
        self._wait_for_diagnostic_thread()

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        policy_chunk_size = getattr(
            self.policy.config, "chunk_size", self.actions_per_chunk
        )
        if self.actions_per_chunk > policy_chunk_size:
            raise ValueError(
                f"actions_per_chunk ({self.actions_per_chunk}) exceeds policy chunk_size "
                f"({policy_chunk_size})"
            )
        if (
            self.rtc_config is not None
            and self.rtc_config.enabled
            and self.rtc_config.execution_horizon > self.actions_per_chunk
        ):
            raise ValueError(
                f"RTC execution_horizon ({self.rtc_config.execution_horizon}) exceeds "
                f"actions_per_chunk ({self.actions_per_chunk})"
            )

        expected_image_keys = set(self.policy.config.image_features)
        client_image_keys = {
            key for key in self.lerobot_features if key.startswith(OBS_IMAGES)
        }
        if client_image_keys != expected_image_keys:
            raise ValueError(
                "Robot image features do not match policy checkpoint. "
                f"Missing: {sorted(expected_image_keys - client_image_keys)}; "
                f"unexpected: {sorted(client_image_keys - expected_image_keys)}"
            )

        if self.rtc_config is not None and self.rtc_config.enabled:
            if not (
                getattr(self.policy, "supports_rtc", False)
                or hasattr(self.policy, "init_rtc_processor")
            ):
                raise ValueError(
                    f"Policy type '{self.policy_type}' does not support RTC inference."
                )
            self.policy.config.rtc_config = self.rtc_config
            if hasattr(self.policy, "init_rtc_processor"):
                self.policy.init_rtc_processor()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.policy.to(self.device)

        self._tactile_image_keys = tuple(
            sorted(key for key in expected_image_keys if "tactile" in key.lower())
        )
        self._debug_qwen_text_enabled = self.config.debug_qwen_text and hasattr(
            self.policy, "generate_debug_text"
        )
        self._debug_tactile_counterfactual_enabled = bool(
            self.config.debug_tactile_counterfactual and self._tactile_image_keys
        )
        self._reset_diagnostic_state()

        if self.config.debug_qwen_text and not self._debug_qwen_text_enabled:
            self.logger.warning(
                "Qwen text diagnostics requested but policy type '%s' does not support them; skipping.",
                self.policy_type,
            )
        elif self._debug_qwen_text_enabled:
            self.logger.warning(
                "XR0 Qwen text diagnostics are enabled. The VLM is frozen during standard XR0 training, "
                "so decoded text is auxiliary evidence and not proof of action-head causality."
            )

        if self.config.debug_tactile_counterfactual and not self._tactile_image_keys:
            self.logger.warning(
                "XR0 tactile counterfactual requested but checkpoint has no tactile image features; skipping."
            )
        elif self._debug_tactile_counterfactual_enabled:
            self.logger.info(
                "XR0 tactile counterfactual enabled keys=%s baseline_frames=%d",
                list(self._tactile_image_keys),
                self.config.debug_tactile_baseline_frames,
            )

        if self._tactile_image_keys and (
            self._debug_qwen_text_enabled or self._debug_tactile_counterfactual_enabled
        ):
            self.logger.warning(
                "Keep all tactile-equipped grippers unloaded for the first %d action requests "
                "so XR0 diagnostics can build unloaded tactile baselines.",
                self.config.debug_tactile_baseline_frames,
            )

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {
                    "rename_map": policy_specs.rename_map
                },
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(
            f"Time taken to put policy on {self.device}: {end - start:.4f} seconds"
        )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize
        decoded_observation, codec_stats = decode_observation_images(
            timed_observation.get_observation()
        )
        timed_observation.observation = decoded_observation
        if codec_stats.image_count > 0:
            compression_ratio = codec_stats.raw_bytes / max(
                codec_stats.encoded_bytes, 1
            )
            self.logger.info(
                "Observation images decoded count=%d encoded=%.1fKiB raw=%.1fKiB "
                "compression=%.2fx decode=%.1fms",
                codec_stats.image_count,
                codec_stats.encoded_bytes / 1024.0,
                codec_stats.raw_bytes / 1024.0,
                compression_ratio,
                codec_stats.elapsed_s * 1000.0,
            )

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(
                    0,
                    self.config.inference_latency
                    - max(0, time.perf_counter() - getactions_starts),
                )
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def _obs_sanity_checks(
        self, obs: TimedObservation, previous_obs: TimedObservation
    ) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!"
            )
            return False

        elif observations_similar(
            obs, previous_obs, lerobot_features=self.lerobot_features
        ):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = (
                self.last_processed_obs.get_timestep()
                if self.last_processed_obs
                else "None"
            )
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug(
                    "Observation queue was full, removed oldest observation"
                )

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(
        self, t_0: float, action_chunk: list[torch.Tensor], i_0: int
    ) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action,
            )
            for i, action in enumerate(action_chunk)
        ]

    def _capture_policy_rng_state(self) -> _TorchRNGSnapshot:
        device = torch.device(self.device)
        if device.type == "cuda":
            state = torch.cuda.get_rng_state(device)
        else:
            state = torch.random.get_rng_state()
        return _TorchRNGSnapshot(device=device, state=state.clone())

    @staticmethod
    def _restore_policy_rng_state(snapshot: _TorchRNGSnapshot) -> None:
        if snapshot.device.type == "cuda":
            torch.cuda.set_rng_state(snapshot.state, snapshot.device)
        else:
            torch.random.set_rng_state(snapshot.state)

    def _get_policy_predict_kwargs(
        self,
        rtc: RTCInferenceMetadata | None,
    ) -> dict[str, Any]:
        if rtc is None:
            return {}

        prev_chunk_left_over = rtc.prev_chunk_left_over
        if prev_chunk_left_over is not None:
            prev_chunk_left_over = prev_chunk_left_over.to(self.device)
        return {
            "prev_chunk_left_over": prev_chunk_left_over,
            "inference_delay": rtc.inference_delay,
            "execution_horizon": rtc.execution_horizon,
        }

    @staticmethod
    def _clone_debug_observation(observation: Observation) -> Observation:
        return {
            key: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for key, value in observation.items()
        }

    def _update_tactile_baselines(self, observation: Observation) -> None:
        if not self._tactile_image_keys:
            return
        if not (
            self._debug_qwen_text_enabled or self._debug_tactile_counterfactual_enabled
        ):
            return

        target_count = self.config.debug_tactile_baseline_frames
        for key in self._tactile_image_keys:
            if key in self._tactile_baselines:
                continue
            image = observation.get(key)
            if not isinstance(image, torch.Tensor):
                continue
            samples = self._tactile_baseline_samples.setdefault(key, [])
            if len(samples) < target_count:
                samples.append(image.detach().cpu().clone())
            if len(samples) == target_count:
                self._tactile_baselines[key] = torch.median(
                    torch.stack(samples, dim=0), dim=0
                ).values
                samples.clear()
                self.logger.info(
                    "XR0 unloaded tactile baseline ready key=%s frames=%d",
                    key,
                    target_count,
                )

    def _make_tactile_masked_observation(
        self, observation: Observation
    ) -> Observation | None:
        if not self._tactile_image_keys:
            return None
        if any(key not in self._tactile_baselines for key in self._tactile_image_keys):
            return None

        masked = self._clone_debug_observation(observation)
        for key in self._tactile_image_keys:
            image = masked.get(key)
            if not isinstance(image, torch.Tensor):
                return None
            masked[key] = (
                self._tactile_baselines[key]
                .to(device=image.device, dtype=image.dtype)
                .clone()
            )
        return masked

    @staticmethod
    def _action_difference_metrics(
        primary_action: torch.Tensor,
        counterfactual_action: torch.Tensor,
    ) -> dict[str, float]:
        primary = primary_action.detach().float().cpu()
        counterfactual = counterfactual_action.detach().float().cpu()
        if primary.ndim == 3:
            primary = primary.squeeze(0)
        if counterfactual.ndim == 3:
            counterfactual = counterfactual.squeeze(0)

        horizon = min(primary.shape[-2], counterfactual.shape[-2])
        action_dim = min(primary.shape[-1], counterfactual.shape[-1])
        delta = (
            primary[:horizon, :action_dim] - counterfactual[:horizon, :action_dim]
        ).abs()
        metrics = {
            "mean_abs_delta": float(delta.mean().item()),
            "max_abs_delta": float(delta.max().item()),
        }

        if action_dim >= 27:
            metrics.update(
                {
                    "left_gripper_delta": float(delta[:, 6].mean().item()),
                    "left_arm_joint_delta": float(delta[:, 7:13].mean().item()),
                    "right_gripper_delta": float(delta[:, 20].mean().item()),
                    "right_arm_joint_delta": float(delta[:, 21:27].mean().item()),
                }
            )
        elif action_dim >= 14:
            metrics.update(
                {
                    "left_gripper_delta": float(delta[:, 6].mean().item()),
                    "left_arm_joint_delta": float(delta[:, 0:6].mean().item()),
                    "right_gripper_delta": float(delta[:, 13].mean().item()),
                    "right_arm_joint_delta": float(delta[:, 7:13].mean().item()),
                }
            )
        return metrics

    def _run_xr0_diagnostics(
        self,
        *,
        policy,
        observation: Observation,
        predict_kwargs: dict[str, Any],
        primary_action: torch.Tensor,
        primary_rng_state: _TorchRNGSnapshot | None,
        timestep: int,
        run_text: bool,
        run_counterfactual: bool,
    ) -> None:
        started_at = time.perf_counter()
        try:
            if self.shutdown_event.is_set() or not self.observation_queue.empty():
                self.logger.debug(
                    "Skipping XR0 diagnostics because an action observation is pending."
                )
                return
            if not self._policy_inference_lock.acquire(blocking=False):
                self.logger.debug(
                    "Skipping XR0 diagnostics because policy inference is busy."
                )
                return
            try:
                if self.shutdown_event.is_set() or not self.observation_queue.empty():
                    self.logger.debug(
                        "Skipping XR0 diagnostics because an action observation arrived."
                    )
                    return

                masked_observation = self._make_tactile_masked_observation(observation)
                full_text = None
                masked_text = None
                metrics = None

                if run_text:
                    full_outputs = policy.generate_debug_text(
                        observation,
                        max_new_tokens=self.config.debug_qwen_text_max_new_tokens,
                    )
                    full_text = full_outputs[0] if full_outputs else ""
                    if masked_observation is not None:
                        masked_outputs = policy.generate_debug_text(
                            masked_observation,
                            max_new_tokens=self.config.debug_qwen_text_max_new_tokens,
                        )
                        masked_text = masked_outputs[0] if masked_outputs else ""

                if run_counterfactual and masked_observation is not None:
                    if primary_rng_state is None:
                        raise RuntimeError(
                            "XR0 tactile counterfactual requires the primary inference RNG state."
                        )
                    current_rng_state = self._capture_policy_rng_state()
                    try:
                        self._restore_policy_rng_state(primary_rng_state)
                        counterfactual_action = policy.predict_action_chunk(
                            masked_observation,
                            **predict_kwargs,
                        )[:, : self.actions_per_chunk, :]
                    finally:
                        self._restore_policy_rng_state(current_rng_state)
                    metrics = self._action_difference_metrics(
                        primary_action, counterfactual_action
                    )
            finally:
                self._policy_inference_lock.release()

            lines = ["[XR0 DEBUG]", f"timestep={timestep}"]
            if full_text is not None:
                lines.append(f"qwen_full={full_text}")
                if self._tactile_image_keys and masked_text is None:
                    lines.append("qwen_tactile_masked=SKIPPED_BASELINE_NOT_READY")
                elif masked_text is not None:
                    lines.append(f"qwen_tactile_masked={masked_text}")
            if run_counterfactual and metrics is None:
                lines.append("tactile_action_counterfactual=SKIPPED_BASELINE_NOT_READY")
            elif metrics is not None:
                lines.extend(f"{key}={value:.6f}" for key, value in metrics.items())
            lines.append(f"diagnostic_time_s={time.perf_counter() - started_at:.3f}")
            self.logger.info("\n%s", "\n".join(lines))
        except Exception:
            self.logger.exception(
                "XR0 diagnostic task failed; normal policy inference remains active."
            )
        finally:
            with self._diagnostic_state_lock:
                self._diagnostic_running = False

    def _maybe_schedule_xr0_diagnostics(
        self,
        *,
        observation: Observation,
        rtc: RTCInferenceMetadata | None,
        primary_action: torch.Tensor,
        primary_rng_state: _TorchRNGSnapshot | None,
        timestep: int,
    ) -> None:
        now = time.monotonic()
        run_text = self._debug_qwen_text_enabled and (
            now - self._last_qwen_text_debug_at
            >= self.config.debug_qwen_text_interval_s
        )
        run_counterfactual = self._debug_tactile_counterfactual_enabled and (
            now - self._last_tactile_action_debug_at
            >= self.config.debug_tactile_action_interval_s
        )
        if not (run_text or run_counterfactual):
            return

        with self._diagnostic_state_lock:
            if self._diagnostic_running:
                return
            self._diagnostic_running = True
            if run_text:
                self._last_qwen_text_debug_at = now
            if run_counterfactual:
                self._last_tactile_action_debug_at = now

        debug_observation = self._clone_debug_observation(observation)
        predict_kwargs = {
            key: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
            for key, value in self._get_policy_predict_kwargs(rtc).items()
        }
        thread = threading.Thread(
            target=self._run_xr0_diagnostics,
            kwargs={
                "policy": self.policy,
                "observation": debug_observation,
                "predict_kwargs": predict_kwargs,
                "primary_action": primary_action.detach().clone(),
                "primary_rng_state": primary_rng_state,
                "timestep": timestep,
                "run_text": run_text,
                "run_counterfactual": run_counterfactual,
            },
            name="xr0-diagnostics",
            daemon=True,
        )
        with self._diagnostic_state_lock:
            self._diagnostic_thread = thread
        thread.start()

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        rtc: RTCInferenceMetadata | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(
            observation,
            **self._get_policy_predict_kwargs(rtc),
        )
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(
                0
            )  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(
        self, observation_t: TimedObservation
    ) -> list[TimedAction] | RemoteActionChunk:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess
        diagnostics_enabled = (
            self._debug_qwen_text_enabled or self._debug_tactile_counterfactual_enabled
        )
        if diagnostics_enabled:
            self._update_tactile_baselines(observation)

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        primary_rng_state = None
        if diagnostics_enabled:
            with self._policy_inference_lock:
                if self._debug_tactile_counterfactual_enabled:
                    primary_rng_state = self._capture_policy_rng_state()
                action_tensor = self._get_action_chunk(observation, observation_t.rtc)
        else:
            action_tensor = self._get_action_chunk(observation, observation_t.rtc)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        original_action_tensor = action_tensor.squeeze(0).detach().cpu()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()
        if diagnostics_enabled:
            self._maybe_schedule_xr0_diagnostics(
                observation=observation,
                rtc=observation_t.rtc,
                primary_action=original_action_tensor,
                primary_rng_state=primary_rng_state,
                timestep=observation_t.get_timestep(),
            )

        if observation_t.rtc is not None:
            return RemoteActionChunk(
                request_id=observation_t.rtc.request_id,
                original_actions=original_action_tensor,
                processed_actions=action_tensor,
                observation_timestamp=observation_t.get_timestamp(),
                observation_timestep=observation_t.get_timestep(),
                action_count_before_inference=observation_t.rtc.action_count_before_inference,
                starvation_count_before_inference=observation_t.rtc.starvation_count_before_inference,
                inference_time_s=inference_time,
            )

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self._wait_for_diagnostic_thread()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()

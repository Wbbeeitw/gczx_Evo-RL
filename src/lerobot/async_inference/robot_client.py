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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import math
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.tactile.configuration_tactile import TactileCameraConfig  # noqa: F401
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_piper_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RTCInferenceMetadata,
    RawObservation,
    RemoteActionChunk,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    encode_observation_images,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


def _start_live_display(enabled: bool):
    if not enabled:
        return None

    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    init_rerun(session_name="async_inference_robot_client")
    return log_rerun_data


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        try:
            self.display_logger = _start_live_display(config.display_data)
        except Exception:
            self.logger.exception("Could not start Rerun live display; continuing without it.")
            self.display_logger = None
        self.display_action_lock = threading.Lock()
        self.latest_display_action = None
        self.gripper_diagnostics_lock = threading.Lock()
        self.latest_right_gripper_target = None
        self.latest_right_gripper_sent = None
        self.latest_right_gripper_feedback = None
        self.latest_right_gripper_feedback_at = None
        self.right_gripper_target_min = None
        self.right_gripper_target_max = None
        self.right_gripper_sent_min = None
        self.right_gripper_sent_max = None
        self.right_gripper_feedback_min = None
        self.right_gripper_feedback_max = None
        self.startup_left_gripper_hold_started_at = None
        self.startup_left_gripper_hold_finished = False
        self.startup_right_gripper_hold_started_at = None
        self.startup_right_gripper_hold_finished = False
        self.robot = make_robot_from_config(config.robot)
        for side in ("left", "right"):
            position = getattr(config, f"startup_{side}_gripper_position")
            if position is not None and f"{side}_gripper.pos" not in self.robot.action_features:
                raise ValueError(
                    f"startup_{side}_gripper_position requires robot action feature "
                    f"'{side}_gripper.pos'"
                )
        self.robot.connect()
        try:
            self._command_startup_gripper_positions()
        except BaseException:
            self.robot.disconnect()
            raise

        if config.controlled_arms != "both":
            suppressed_prefix = "left_" if config.controlled_arms == "right" else "right_"
            suppressed_keys = [
                key for key in self.robot.action_features if key.startswith(suppressed_prefix)
            ]
            if suppressed_keys:
                self.logger.info(
                    "Action arm filter enabled controlled_arms=%s suppressed_keys=%s",
                    config.controlled_arms,
                    suppressed_keys,
                )
            else:
                self.logger.warning(
                    "Action arm filter controlled_arms=%s found no %s-prefixed action keys",
                    config.controlled_arms,
                    suppressed_prefix.rstrip("_"),
                )

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            rtc_config=config.rtc,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.executed_action_count = 0
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.rtc_action_queue = ActionQueue(config.rtc) if config.rtc.enabled else None
        self.rtc_latency_tracker = LatencyTracker()
        self.rtc_request_lock = threading.Lock()
        self.rtc_request_in_flight = threading.Event()
        self.rtc_request_id = 0
        self.active_rtc_request_id = -1
        self.rtc_warmup_complete = threading.Event()
        self.rtc_warmup_request_ids: set[int] = set()
        self.rtc_warmup_completed_requests = 0
        if self.rtc_action_queue is None or config.rtc.warmup_requests == 0:
            self.rtc_warmup_complete.set()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop
        self.latest_captured_observation = None
        self.latest_performed_action = None
        self.last_action_send_duration_s = 0.0
        self.max_action_send_duration_s = 0.0

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        if self.display_logger is not None:
            self.logger.info(
                "Rerun live display enabled compressed_images=%s",
                self.config.display_compressed_images,
            )

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()
        try:
            self.robot.disconnect()
            self.logger.debug("Robot disconnected")
        finally:
            self.channel.close()
            self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            rpc_started_at = time.perf_counter()
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            rpc_time = time.perf_counter() - rpc_started_at
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")
            if self.config.observation_image_codec != "raw":
                self.logger.info(
                    "Observation transport step=%d payload=%.1fKiB serialize=%.1fms rpc=%.1fms",
                    obs_timestep,
                    len(observation_bytes) / 1024.0,
                    serialize_time * 1000.0,
                    rpc_time * 1000.0,
                )

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        replace_existing: bool = False,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        with self.latest_action_lock:
            latest_action = self.latest_action

        with self.action_queue_lock:
            queued_actions = (
                {}
                if replace_existing
                else {
                    action.get_timestep(): action
                    for action in self.action_queue.queue
                    if action.get_timestep() > latest_action
                }
            )

            for new_action in incoming_actions:
                timestep = new_action.get_timestep()
                if timestep <= latest_action:
                    continue

                old_action = queued_actions.get(timestep)
                if old_action is None:
                    queued_actions[timestep] = new_action
                    continue

                queued_actions[timestep] = TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=timestep,
                    action=aggregate_fn(old_action.get_action(), new_action.get_action()),
                )

            future_action_queue = Queue()
            for timestep in sorted(queued_actions):
                future_action_queue.put(queued_actions[timestep])
            self.action_queue = future_action_queue

    def _begin_rtc_request(self) -> int:
        with self.rtc_request_lock:
            if self.rtc_request_in_flight.is_set():
                raise RuntimeError("An RTC inference request is already in flight.")
            request_id = self.rtc_request_id
            self.rtc_request_id += 1
            self.active_rtc_request_id = request_id
            self.rtc_request_in_flight.set()
            if not self.rtc_warmup_complete.is_set():
                self.rtc_warmup_request_ids.add(request_id)
            return request_id

    def _finish_rtc_request(self, request_id: int) -> None:
        with self.rtc_request_lock:
            if request_id != self.active_rtc_request_id:
                return
            self.active_rtc_request_id = -1
            self.rtc_request_in_flight.clear()

    def _cancel_active_rtc_request(self) -> None:
        with self.rtc_request_lock:
            self.rtc_warmup_request_ids.discard(self.active_rtc_request_id)
            self.active_rtc_request_id = -1
            self.rtc_request_in_flight.clear()

    def _is_rtc_warmup_request(self, request_id: int) -> bool:
        with self.rtc_request_lock:
            return request_id in self.rtc_warmup_request_ids

    def _complete_rtc_warmup_request(self, request_id: int) -> None:
        with self.rtc_request_lock:
            self.rtc_warmup_request_ids.discard(request_id)
            self.rtc_warmup_completed_requests += 1
            completed_requests = self.rtc_warmup_completed_requests

        if completed_requests < self.config.rtc.warmup_requests:
            self.logger.info(
                "RTC warm-up request %d/%d complete",
                completed_requests,
                self.config.rtc.warmup_requests,
            )
            return

        self.rtc_action_queue.reset_for_reanchor()
        self.rtc_latency_tracker.reset()
        self.must_go.set()
        self.rtc_warmup_complete.set()
        self.logger.info("RTC warm-up complete; queue and latency statistics reset")

    def _wait_for_rtc_warmup(self) -> bool:
        if self.rtc_warmup_complete.is_set():
            return True

        timeout_s = self.config.rtc.warmup_timeout_s
        deadline = time.perf_counter() + timeout_s
        self.logger.info(
            "Waiting for %d RTC warm-up requests before enabling action execution",
            self.config.rtc.warmup_requests,
        )
        while self.running:
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0:
                self.logger.error("RTC warm-up timed out after %.1fs; no actions were executed", timeout_s)
                self._cancel_active_rtc_request()
                self.shutdown_event.set()
                return False
            if self.rtc_warmup_complete.wait(min(0.1, remaining_s)):
                return True
        return False

    def _prepare_rtc_metadata(self, request_id: int) -> RTCInferenceMetadata:
        prev_actions, action_count, starvation_count = self.rtc_action_queue.get_rtc_snapshot()
        execution_horizon = self.config.rtc.execution_horizon
        if prev_actions is not None:
            leftover_len = len(prev_actions)
            if leftover_len <= 0:
                prev_actions = None
                execution_horizon = 0
            else:
                execution_horizon = min(execution_horizon, leftover_len)
                padded = prev_actions.new_zeros(
                    (self.config.actions_per_chunk, prev_actions.shape[-1])
                )
                copy_steps = min(leftover_len, self.config.actions_per_chunk)
                padded[:copy_steps] = prev_actions[:copy_steps]
                prev_actions = padded.cpu()

        median_latency = self.rtc_latency_tracker.percentile(0.5) or 0.0
        estimated_delay_steps = (
            self.config.rtc.inference_delay_multiplier
            * median_latency
            / self.config.environment_dt
        )
        inference_delay = math.ceil(max(0.0, estimated_delay_steps - 1e-6))
        inference_delay = min(inference_delay, execution_horizon)
        return RTCInferenceMetadata(
            request_id=request_id,
            prev_chunk_left_over=prev_actions,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            action_count_before_inference=action_count,
            starvation_count_before_inference=starvation_count,
        )

    def _merge_rtc_action_chunk(self, chunk: RemoteActionChunk, receive_time: float) -> None:
        request_latency = max(0.0, receive_time - chunk.observation_timestamp)
        real_delay = math.ceil(request_latency / self.config.environment_dt)
        self.rtc_latency_tracker.add(request_latency)
        self.action_chunk_size = max(self.action_chunk_size, len(chunk.processed_actions))
        is_warmup = self._is_rtc_warmup_request(chunk.request_id)

        merged = self.rtc_action_queue.merge(
            chunk.original_actions,
            chunk.processed_actions,
            0 if is_warmup else real_delay,
            action_index_before_inference=None,
            action_count_before_inference=chunk.action_count_before_inference,
            starvation_count_before_inference=chunk.starvation_count_before_inference,
        )
        if not merged:
            self.rtc_action_queue.reset_for_reanchor()
            self.rtc_latency_tracker.reset()
            self.must_go.set()
            self.logger.warning(
                "RTC chunk request=%d had no future actions; requesting a fresh anchor",
                chunk.request_id,
            )
        else:
            self.must_go.set()
            self.logger.info(
                "RTC chunk request=%d inference=%.1fms round_trip=%.1fms consumed=%d queue=%d",
                chunk.request_id,
                chunk.inference_time_s * 1000.0,
                request_latency * 1000.0,
                self.rtc_action_queue.last_replace_real_delay,
                self.rtc_action_queue.qsize(),
            )

        if is_warmup:
            if not merged:
                self.logger.error("RTC warm-up request %d returned no usable actions", chunk.request_id)
                self._finish_rtc_request(chunk.request_id)
                self.shutdown_event.set()
                return
            self._complete_rtc_warmup_request(chunk.request_id)
        self._finish_rtc_request(chunk.request_id)

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                if isinstance(timed_actions, RemoteActionChunk):
                    self._merge_rtc_action_chunk(timed_actions, receive_time)
                    continue

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")
                if self.rtc_action_queue is not None:
                    self._cancel_active_rtc_request()
                    self.must_go.set()

    def actions_available(self):
        """Check if there are actions available in the queue"""
        if self.rtc_action_queue is not None:
            return not self.rtc_action_queue.empty()
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def get_executed_action_count(self) -> int:
        if self.rtc_action_queue is not None:
            return self.rtc_action_queue.get_action_count()
        with self.latest_action_lock:
            return self.executed_action_count

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        if action_tensor.numel() != len(self.robot.action_features):
            raise ValueError(
                f"Received action with {action_tensor.numel()} values, but robot expects "
                f"{len(self.robot.action_features)}"
            )
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        if self.config.controlled_arms == "right":
            action = {key: value for key, value in action.items() if not key.startswith("left_")}
        elif self.config.controlled_arms == "left":
            action = {key: value for key, value in action.items() if not key.startswith("right_")}
        return action

    def _command_startup_gripper_positions(self) -> None:
        startup_action = {
            f"{side}_gripper.pos": float(getattr(self.config, f"startup_{side}_gripper_position"))
            for side in ("left", "right")
            if getattr(self.config, f"startup_{side}_gripper_position") is not None
        }
        if not startup_action:
            return

        sent_action = (
            startup_action
            if self.config.dry_run_actions
            else self.robot.send_action(startup_action)
        )
        self._remember_display_action(sent_action)
        for side in ("left", "right"):
            key = f"{side}_gripper.pos"
            if key not in startup_action:
                continue
            if side == "right":
                self._remember_gripper_action(
                    {key: startup_action[key]},
                    None if sent_action is None else {key: sent_action.get(key)},
                )
            sent_position = None if sent_action is None else sent_action.get(key)
            self.logger.info(
                "Startup %s gripper position commanded target=%.2f sent=%.2f dry_run=%s hold_s=%.2f",
                side,
                startup_action[key],
                float("nan") if sent_position is None else sent_position,
                self.config.dry_run_actions,
                getattr(self.config, f"startup_{side}_gripper_hold_s"),
            )

    def _command_startup_right_gripper_position(self) -> None:
        """Backward-compatible wrapper for callers using the old helper name."""
        self._command_startup_gripper_positions()

    def _apply_startup_gripper_holds(
        self,
        action: dict[str, float],
        now: float | None = None,
    ) -> dict[str, float]:
        now = time.perf_counter() if now is None else now
        held_action = dict(action)
        for side in ("left", "right"):
            position = getattr(self.config, f"startup_{side}_gripper_position")
            hold_s = getattr(self.config, f"startup_{side}_gripper_hold_s")
            started_at_attr = f"startup_{side}_gripper_hold_started_at"
            finished_attr = f"startup_{side}_gripper_hold_finished"
            if position is None or hold_s <= 0 or getattr(self, finished_attr):
                continue

            started_at = getattr(self, started_at_attr)
            if started_at is None:
                setattr(self, started_at_attr, now)
                self.logger.info(
                    "Startup %s gripper hold started position=%.2f duration=%.2fs",
                    side,
                    position,
                    hold_s,
                )
                started_at = now

            if now - started_at < hold_s:
                held_action[f"{side}_gripper.pos"] = float(position)
                continue

            setattr(self, finished_attr, True)
            self.logger.info(
                "Startup %s gripper hold complete; policy now controls the gripper",
                side,
            )

        return held_action

    def _apply_startup_right_gripper_hold(
        self,
        action: dict[str, float],
        now: float | None = None,
    ) -> dict[str, float]:
        """Backward-compatible wrapper for callers using the old helper name."""
        return self._apply_startup_gripper_holds(action, now=now)

    def _remember_display_action(self, action: dict[str, Any] | None) -> None:
        if action is None:
            return
        with self.display_action_lock:
            self.latest_display_action = dict(action)

    def _remember_gripper_action(
        self,
        target_action: dict[str, Any],
        sent_action: dict[str, Any] | None,
    ) -> None:
        target = target_action.get("right_gripper.pos")
        sent = None if sent_action is None else sent_action.get("right_gripper.pos")
        if sent is None:
            sent = target
        with self.gripper_diagnostics_lock:
            self.latest_right_gripper_target = None if target is None else float(target)
            self.latest_right_gripper_sent = None if sent is None else float(sent)
            if target is not None:
                target = float(target)
                self.right_gripper_target_min = (
                    target
                    if self.right_gripper_target_min is None
                    else min(self.right_gripper_target_min, target)
                )
                self.right_gripper_target_max = (
                    target
                    if self.right_gripper_target_max is None
                    else max(self.right_gripper_target_max, target)
                )
            if sent is not None:
                sent = float(sent)
                self.right_gripper_sent_min = (
                    sent
                    if self.right_gripper_sent_min is None
                    else min(self.right_gripper_sent_min, sent)
                )
                self.right_gripper_sent_max = (
                    sent
                    if self.right_gripper_sent_max is None
                    else max(self.right_gripper_sent_max, sent)
                )

    def _remember_gripper_feedback(self, observation: RawObservation) -> None:
        feedback = observation.get("right_gripper.pos")
        if feedback is None:
            return
        with self.gripper_diagnostics_lock:
            feedback = float(feedback)
            self.latest_right_gripper_feedback = feedback
            self.latest_right_gripper_feedback_at = time.perf_counter()
            self.right_gripper_feedback_min = (
                feedback
                if self.right_gripper_feedback_min is None
                else min(self.right_gripper_feedback_min, feedback)
            )
            self.right_gripper_feedback_max = (
                feedback
                if self.right_gripper_feedback_max is None
                else max(self.right_gripper_feedback_max, feedback)
            )

    def _get_gripper_diagnostics(
        self, now: float | None = None, reset_window: bool = False
    ) -> dict[str, float]:
        with self.gripper_diagnostics_lock:
            target = self.latest_right_gripper_target
            sent = self.latest_right_gripper_sent
            feedback = self.latest_right_gripper_feedback
            feedback_at = self.latest_right_gripper_feedback_at
            target_min = self.right_gripper_target_min
            target_max = self.right_gripper_target_max
            sent_min = self.right_gripper_sent_min
            sent_max = self.right_gripper_sent_max
            feedback_min = self.right_gripper_feedback_min
            feedback_max = self.right_gripper_feedback_max
            if reset_window:
                self.right_gripper_target_min = None
                self.right_gripper_target_max = None
                self.right_gripper_sent_min = None
                self.right_gripper_sent_max = None
                self.right_gripper_feedback_min = None
                self.right_gripper_feedback_max = None

        nan = float("nan")
        feedback_age_ms = (
            nan
            if feedback_at is None
            else max(0.0, (time.perf_counter() if now is None else now) - feedback_at) * 1000.0
        )
        return {
            "target": nan if target is None else target,
            "sent": nan if sent is None else sent,
            "feedback": nan if feedback is None else feedback,
            "error": nan if feedback is None or sent is None else feedback - sent,
            "feedback_age_ms": feedback_age_ms,
            "target_min": nan if target_min is None else target_min,
            "target_max": nan if target_max is None else target_max,
            "sent_min": nan if sent_min is None else sent_min,
            "sent_max": nan if sent_max is None else sent_max,
            "feedback_min": nan if feedback_min is None else feedback_min,
            "feedback_max": nan if feedback_max is None else feedback_max,
        }

    def _log_live_observation(self, observation: RawObservation) -> None:
        if self.display_logger is None:
            return

        with self.display_action_lock:
            action = None if self.latest_display_action is None else dict(self.latest_display_action)

        try:
            self.display_logger(
                observation=observation,
                action=action,
                compress_images=self.config.display_compressed_images,
            )
        except Exception:
            self.logger.exception("Rerun live display failed; disabling it for this client run.")
            self.display_logger = None

    def _capture_raw_observation(self, task: str) -> RawObservation:
        raw_observation: RawObservation = self.robot.get_observation()
        self._remember_gripper_feedback(raw_observation)
        raw_observation["task"] = task
        self._log_live_observation(raw_observation)
        return raw_observation

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Reading and performing actions in local queue"""

        if self.rtc_action_queue is not None:
            action_tensor = self.rtc_action_queue.get_for_execution()
            if action_tensor is None:
                return None

            policy_action_dict = self._action_tensor_to_action_dict(action_tensor)
            action_dict = self._apply_startup_gripper_holds(policy_action_dict)
            send_started_at = time.perf_counter()
            try:
                performed_action = (
                    action_dict if self.config.dry_run_actions else self.robot.send_action(action_dict)
                )
            except Exception:
                self.rtc_action_queue.mark_action_failed()
                raise
            self.last_action_send_duration_s = time.perf_counter() - send_started_at
            self.max_action_send_duration_s = max(
                self.max_action_send_duration_s, self.last_action_send_duration_s
            )

            self.rtc_action_queue.mark_action_sent(action_tensor)
            with self.latest_action_lock:
                self.latest_action = self.rtc_action_queue.get_action_count() - 1
                self.executed_action_count = self.rtc_action_queue.get_action_count()
            display_action = performed_action if performed_action is not None else action_dict
            self._remember_gripper_action(policy_action_dict, performed_action)
            self._remember_display_action(display_action)
            return performed_action

        get_start = time.perf_counter()
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = None
            while not self.action_queue.empty():
                candidate = self.action_queue.get_nowait()
                if candidate.get_timestep() > latest_action:
                    timed_action = candidate
                    break
        get_end = time.perf_counter() - get_start

        if timed_action is None:
            return None

        policy_action_dict = self._action_tensor_to_action_dict(timed_action.get_action())
        action_dict = self._apply_startup_gripper_holds(policy_action_dict)
        send_started_at = time.perf_counter()
        performed_action = (
            action_dict if self.config.dry_run_actions else self.robot.send_action(action_dict)
        )
        self.last_action_send_duration_s = time.perf_counter() - send_started_at
        self.max_action_send_duration_s = max(
            self.max_action_send_duration_s, self.last_action_send_duration_s
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.executed_action_count += 1
        display_action = performed_action if performed_action is not None else action_dict
        self._remember_gripper_action(policy_action_dict, performed_action)
        self._remember_display_action(display_action)

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        if self.rtc_action_queue is not None:
            if self.rtc_request_in_flight.is_set():
                return False
            if not self.rtc_warmup_complete.is_set():
                return True
            if self.action_chunk_size <= 0:
                return True

            configured_threshold = math.floor(self.action_chunk_size * self._chunk_size_threshold)
            horizon_threshold = max(0, self.action_chunk_size - self.config.rtc.execution_horizon)
            p95_latency = self.rtc_latency_tracker.p95() or 0.0
            latency_threshold = math.ceil(p95_latency / self.config.environment_dt) + 2
            refill_threshold = min(
                self.action_chunk_size - 1,
                max(configured_threshold, horizon_threshold, latency_threshold),
            )
            return self.rtc_action_queue.qsize() <= refill_threshold

        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation = self._capture_raw_observation(task)
            observation_timestamp = time.time()

            wire_observation, codec_stats = encode_observation_images(
                raw_observation,
                codec=self.config.observation_image_codec,
                jpeg_quality=self.config.observation_jpeg_quality,
            )
            if codec_stats.image_count > 0:
                compression_ratio = codec_stats.raw_bytes / max(codec_stats.encoded_bytes, 1)
                self.logger.info(
                    "Observation images codec=%s count=%d raw=%.1fKiB encoded=%.1fKiB "
                    "compression=%.2fx encode=%.1fms",
                    self.config.observation_image_codec,
                    codec_stats.image_count,
                    codec_stats.raw_bytes / 1024.0,
                    codec_stats.encoded_bytes / 1024.0,
                    compression_ratio,
                    codec_stats.elapsed_s * 1000.0,
                )

            if self.rtc_action_queue is not None:
                request_id = self._begin_rtc_request()
                try:
                    rtc_metadata = self._prepare_rtc_metadata(request_id)
                    is_warmup = self._is_rtc_warmup_request(request_id)
                    if is_warmup:
                        self.logger.info(
                            "RTC warm-up request %d/%d prefix=%s horizon=%d",
                            self.rtc_warmup_completed_requests + 1,
                            self.config.rtc.warmup_requests,
                            rtc_metadata.prev_chunk_left_over is not None,
                            rtc_metadata.execution_horizon,
                        )
                    observation = TimedObservation(
                        timestamp=observation_timestamp,
                        observation=wire_observation,
                        timestep=rtc_metadata.action_count_before_inference,
                        must_go=True,
                        rtc=rtc_metadata,
                    )
                    sent = self.send_observation(observation)
                    if not sent:
                        self._finish_rtc_request(request_id)
                    self.logger.info(
                        "RTC observation request=%d step=%d queue=%d delay=%d horizon=%d",
                        request_id,
                        observation.get_timestep(),
                        self.rtc_action_queue.qsize(),
                        rtc_metadata.inference_delay,
                        rtc_metadata.execution_horizon,
                    )
                    return raw_observation
                except Exception:
                    self._finish_rtc_request(request_id)
                    raise

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=observation_timestamp,
                observation=wire_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def action_control_loop(self, verbose: bool = False) -> Action | None:
        """Execute queued actions independently from camera and network I/O."""
        self.start_barrier.wait()
        self.logger.info("Action control loop starting")
        if not self._wait_for_rtc_warmup():
            return self.latest_performed_action
        started_at = time.perf_counter()
        next_action_at = started_at
        first_action_at = None
        first_action_count = 0
        missed_deadlines = 0
        max_deadline_lag_s = 0.0
        log_interval = max(1, round(self.config.fps))

        while self.running:
            now = time.perf_counter()
            if self.config.duration > 0 and now - started_at >= self.config.duration:
                self.logger.info("Stopping because duration reached %.1fs", self.config.duration)
                self.shutdown_event.set()
                break

            wait_s = next_action_at - now
            if wait_s > 0 and self.shutdown_event.wait(wait_s):
                break

            control_loop_start = time.perf_counter()
            action_count_before = self.get_executed_action_count()
            if self.actions_available():
                performed_action = self.control_loop_action(verbose)
                if self.get_executed_action_count() > action_count_before:
                    self.latest_performed_action = performed_action

            action_count_after = self.get_executed_action_count()
            if action_count_after > action_count_before:
                if first_action_at is None:
                    first_action_at = control_loop_start
                    first_action_count = action_count_before
                if action_count_after % log_interval == 0:
                    elapsed_s = time.perf_counter() - first_action_at
                    queue_size = (
                        self.rtc_action_queue.qsize()
                        if self.rtc_action_queue is not None
                        else self.action_queue.qsize()
                    )
                    gripper = self._get_gripper_diagnostics(reset_window=True)
                    self.logger.info(
                        "Action #%d queue=%d send=%.1fms max_send=%.1fms "
                        "effective_hz=%.2f missed_deadlines=%d max_lag=%.1fms dry_run=%s "
                        "right_gripper_target=%.2f right_gripper_sent=%.2f "
                        "right_gripper_feedback=%.2f right_gripper_error=%.2f feedback_age_ms=%.1f "
                        "right_gripper_target_range=[%.2f,%.2f] "
                        "right_gripper_sent_range=[%.2f,%.2f] "
                        "right_gripper_feedback_range=[%.2f,%.2f]",
                        action_count_after,
                        queue_size,
                        self.last_action_send_duration_s * 1000.0,
                        self.max_action_send_duration_s * 1000.0,
                        (action_count_after - first_action_count) / max(elapsed_s, 1e-6),
                        missed_deadlines,
                        max_deadline_lag_s * 1000.0,
                        self.config.dry_run_actions,
                        gripper["target"],
                        gripper["sent"],
                        gripper["feedback"],
                        gripper["error"],
                        gripper["feedback_age_ms"],
                        gripper["target_min"],
                        gripper["target_max"],
                        gripper["sent_min"],
                        gripper["sent_max"],
                        gripper["feedback_min"],
                        gripper["feedback_max"],
                    )

            self.logger.debug(
                f"Action control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}"
            )
            next_action_at += self.config.environment_dt
            deadline_lag_s = time.perf_counter() - next_action_at
            if deadline_lag_s > 0:
                missed_deadlines += 1
                max_deadline_lag_s = max(max_deadline_lag_s, deadline_lag_s)
                if deadline_lag_s > self.config.environment_dt:
                    next_action_at = time.perf_counter() + self.config.environment_dt

        return self.latest_performed_action

    def observation_loop(self, task: str, verbose: bool = False) -> Observation | None:
        """Capture and send observations independently from action execution."""
        self.logger.info("Observation loop starting")

        while self.running:
            observation_loop_start = time.perf_counter()
            if self._ready_to_send_observation():
                captured_observation = self.control_loop_observation(task, verbose)
                if captured_observation is not None:
                    self.latest_captured_observation = captured_observation
            elif self.display_logger is not None:
                try:
                    self.latest_captured_observation = self._capture_raw_observation(task)
                except Exception as error:
                    self.logger.error("Error capturing live display observation: %s", error)

            self.logger.debug(
                f"Observation loop (ms): {(time.perf_counter() - observation_loop_start) * 1000:.2f}"
            )
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - observation_loop_start)))

        return self.latest_captured_observation

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Run independent action and observation loops."""
        observation_thread = threading.Thread(
            target=self.observation_loop,
            args=(task, verbose),
            daemon=True,
            name="robot-observation-loop",
        )
        observation_thread.start()

        try:
            self.action_control_loop(verbose)
        finally:
            self.shutdown_event.set()
            observation_thread.join(timeout=max(1.0, 2 * self.config.environment_dt))

        return self.latest_captured_observation, self.latest_performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)
    action_receiver_thread = None
    try:
        if not client.start():
            raise RuntimeError("Failed to start async inference client.")
        client.logger.info("Starting action receiver thread...")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        client.control_loop(task=cfg.task)
    finally:
        client.stop()
        if action_receiver_thread is not None:
            action_receiver_thread.join(timeout=5.0)
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client

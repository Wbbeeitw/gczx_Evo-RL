#!/usr/bin/env python

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
Demo script showing how to use Real-Time Chunking (RTC) with action chunking policies on real robots.

This script demonstrates:
1. Creating a robot and policy (SmolVLA, Pi0, etc.) with RTC
2. Consuming actions from the policy while the robot executes
3. Periodically requesting new action chunks in the background using threads
4. Managing action buffers and timing for real-time operation

For simulation environments, see eval_with_simulation.py

Usage:
    # Run RTC with Real robot with RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot without RTC
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=helper2424/smolvla_check_rtc_last3 \
        --policy.device=mps \
        --rtc.enabled=false \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120

    # Run RTC with Real robot with pi0.5 policy
    uv run examples/rtc/eval_with_real_robot.py \
        --policy.path=helper2424/pi05_check_rtc \
        --policy.device=mps \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem58FA0834591 \
        --robot.id=so100_follower \
        --robot.cameras="{ gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --task="Move green small object into the purple platform" \
        --duration=120
"""

import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    bi_piper_follower,
    koch_follower,
    piper_follower,
    so_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 30

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def _prepare_rtc_leftover(
    prev_actions: Tensor | None,
    chunk_size: int,
    action_dim: int | None,
    execution_horizon: int,
) -> tuple[Tensor | None, int, int]:
    """Pad RTC leftover to a fixed shape to avoid shape-churn in inference."""
    if prev_actions is None:
        return None, 0, execution_horizon

    leftover_len = int(prev_actions.shape[0])
    leftover_action_dim = int(prev_actions.shape[1]) if prev_actions.ndim > 1 else 1
    target_action_dim = action_dim or leftover_action_dim

    runtime_execution_horizon = min(execution_horizon, leftover_len)
    if runtime_execution_horizon <= 0:
        return None, leftover_len, 0

    padded = prev_actions.new_zeros((chunk_size, target_action_dim))
    copy_steps = min(leftover_len, chunk_size)
    copy_dims = min(leftover_action_dim, target_action_dim)
    padded[:copy_steps, :copy_dims] = prev_actions[:copy_steps, :copy_dims]
    return padded, leftover_len, runtime_execution_horizon


def _wait_for_first_action_chunk(
    action_queue: ActionQueue,
    get_actions_thread: Thread,
    shutdown_event: Event,
    poll_interval_s: float = 0.5,
    log_interval_s: float = 5.0,
):
    """Wait until the first action chunk is ready before starting the actor loop."""
    start_time = time.perf_counter()
    last_log_time = start_time

    while not shutdown_event.is_set():
        queue_size = action_queue.qsize()
        if queue_size > 0:
            logger.info(
                "[MAIN] First action chunk ready after %.3fs. Initial queue size: %d",
                time.perf_counter() - start_time,
                queue_size,
            )
            return

        if not get_actions_thread.is_alive():
            raise RuntimeError("Get actions thread exited before producing the first action chunk")

        now = time.perf_counter()
        if now - last_log_time >= log_interval_s:
            logger.info(
                "[MAIN] Waiting for first action chunk... elapsed=%.3fs queue_size=%d",
                now - start_time,
                queue_size,
            )
            last_log_time = now

        time.sleep(poll_interval_s)


def get_actions(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device
        chunk_size = int(getattr(policy.config, "chunk_size", 0) or 0)
        action_dim = None
        try:
            action_dim = len(robot.action_features())
        except Exception:
            action_dim = None

        # Load preprocessor and postprocessor from pretrained files
        # The stats are embedded in the processor .safetensors files
        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )

        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                prev_actions, leftover_len, runtime_execution_horizon = _prepare_rtc_leftover(
                    prev_actions=prev_actions,
                    chunk_size=chunk_size,
                    action_dim=action_dim,
                    execution_horizon=cfg.rtc.execution_horizon,
                )

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)
                runtime_inference_delay = min(inference_delay, runtime_execution_horizon)

                obs_t0 = time.perf_counter()
                obs = robot.get_observation()
                obs_t1 = time.perf_counter()

                # Apply robot observation processor
                obs_processed = robot_observation_processor(obs)
                obs_t2 = time.perf_counter()

                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
                )
                obs_t3 = time.perf_counter()

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

                obs_with_policy_features["task"] = [cfg.task]  # Task should be a list, not a string!
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )

                preproceseded_obs = preprocessor(obs_with_policy_features)
                obs_t4 = time.perf_counter()

                # Generate actions WITH RTC
                logger.info(
                    "[GET_ACTIONS] Requesting chunk queue_before=%d inference_delay=%d "
                    "runtime_exec_horizon=%d leftover_len=%d",
                    action_queue.qsize(),
                    runtime_inference_delay,
                    runtime_execution_horizon,
                    leftover_len,
                )
                if policy_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_t0 = time.perf_counter()
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=runtime_inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=runtime_execution_horizon,
                )
                if policy_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_t1 = time.perf_counter()

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()
                chunk_len = int(original_actions.shape[0])

                postprocessed_actions = postprocessor(actions)
                obs_t5 = time.perf_counter()

                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)
                latency_max = latency_tracker.max() or 0.0
                latency_p95 = latency_tracker.p95() or 0.0
                delay_p95 = math.ceil(latency_p95 / time_per_chunk)
                delay_max = math.ceil(latency_max / time_per_chunk)
                required_threshold = cfg.rtc.execution_horizon + new_delay
                required_threshold_p95 = cfg.rtc.execution_horizon + delay_p95
                required_threshold_max = cfg.rtc.execution_horizon + delay_max
                safe_max_fps = max((chunk_len - max(cfg.rtc.execution_horizon, 1)) / max(new_latency, 1e-6), 0.1)
                queue_before_merge = action_queue.qsize()

                if new_delay >= chunk_len:
                    logger.warning(
                        "[GET_ACTIONS] Inference delay consumed the whole chunk: delay=%d >= chunk_len=%d. "
                        "Current fps=%.1f is too high for latency=%.3fs. Recommended fps <= %.2f.",
                        new_delay,
                        chunk_len,
                        cfg.fps,
                        new_latency,
                        safe_max_fps,
                    )

                if cfg.action_queue_size_to_get_new_actions < required_threshold:
                    logger.warning(
                        "[GET_ACTIONS] action_queue_size_to_get_new_actions=%d too small. Need > %d "
                        "(delay=%d + execution_horizon=%d).",
                        cfg.action_queue_size_to_get_new_actions,
                        required_threshold,
                        new_delay,
                        cfg.rtc.execution_horizon,
                    )

                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )
                queue_after_merge = action_queue.qsize()

                logger.info(
                    "[GET_ACTIONS] chunk_len=%d latency=%.3fs delay=%d max_latency=%.3fs p95_latency=%.3fs "
                    "queue_before=%d queue_after=%d threshold=%d required_now=%d required_p95=%d "
                    "required_max=%d exec_horizon=%d runtime_exec_horizon=%d leftover_len=%d blend=%d "
                    "t_obs=%.3fs t_obs_proc=%.3fs t_frame=%.3fs t_pre=%.3fs t_infer=%.3fs t_post=%.3fs",
                    chunk_len,
                    new_latency,
                    new_delay,
                    latency_max,
                    latency_p95,
                    queue_before_merge,
                    queue_after_merge,
                    cfg.action_queue_size_to_get_new_actions,
                    required_threshold,
                    required_threshold_p95,
                    required_threshold_max,
                    cfg.rtc.execution_horizon,
                    runtime_execution_horizon,
                    leftover_len,
                    action_queue.last_blend_steps,
                    obs_t1 - obs_t0,
                    obs_t2 - obs_t1,
                    obs_t3 - obs_t2,
                    obs_t4 - obs_t3,
                    infer_t1 - infer_t0,
                    obs_t5 - infer_t1,
                )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / cfg.fps

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
            action = action_queue.get()

            if action is not None:
                action = action.cpu()
                action_dict = {key: action[i].item() for i, key in enumerate(robot.action_features())}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        logger.info(f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    policy_type = (
        getattr(policy, "type", None)
        or getattr(getattr(policy, "config", None), "type", None)
        or getattr(policy, "name", None)
    )

    # PI models handle their own compilation inside their model constructors.
    if policy_type in {"pi05", "pi0"}:
        logger.info("Skipping outer torch.compile for %s; using policy-native compilation path.", policy_type)
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {"backend": cfg.torch_compile_backend}

        # torch.compile currently rejects specifying both mode and options together.
        # Prefer explicit options when CUDA graphs must be disabled.
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}
        else:
            compile_kwargs["mode"] = cfg.torch_compile_mode

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


@parser.wrap()
def demo_cli(cfg: RTCDemoConfig):
    """Main entry point for RTC demo with draccus configuration."""

    # Initialize logging
    init_logging()

    logger.info(f"Using device: {cfg.device}")

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy = None
    robot = None
    get_actions_thread = None
    actor_thread = None

    policy_class = get_policy_class(cfg.policy.type)

    # Use the CLI-resolved policy config so runtime overrides are honored in RTC inference.
    config = cfg.policy

    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile
        if hasattr(config, "compile_mode"):
            config.compile_mode = cfg.torch_compile_mode

    logger.info(
        "Effective policy config: type=%s dtype=%s gradient_checkpointing=%s compile_model=%s "
        "compile_mode=%s device=%s",
        getattr(config, "type", None),
        getattr(config, "dtype", None),
        getattr(config, "gradient_checkpointing", None),
        getattr(config, "compile_model", None),
        getattr(config, "compile_mode", None),
        getattr(config, "device", None),
    )

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_pretrained_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    # Turn on RTC
    policy.config.rtc_config = cfg.rtc

    # Init RTC processort, as by default if RTC disabled in the config
    # The processor won't be created
    policy.init_rtc_processor()

    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()

    # Apply torch.compile to predict_action_chunk method if enabled
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    # Create robot observation processor
    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()

    # Create action queue for communication between threads
    action_queue = ActionQueue(cfg.rtc)

    # Start chunk requester thread
    get_actions_thread = Thread(
        target=get_actions,
        args=(policy, robot_wrapper, robot_observation_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()
    logger.info("Started get actions thread")

    _wait_for_first_action_chunk(action_queue, get_actions_thread, shutdown_event)

    # Start action executor thread
    actor_thread = Thread(
        target=actor_control,
        args=(robot_wrapper, robot_action_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()
    logger.info("Started actor thread")

    logger.info("Started stop by duration thread")

    # Main thread monitors for duration or shutdown
    logger.info(f"Running demo for {cfg.duration} seconds...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(10)

        # Log queue status periodically
        if int(time.time() - start_time) % 5 == 0:
            logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

        if time.time() - start_time > cfg.duration:
            break

    logger.info("Demo duration reached or shutdown requested")

    # Signal shutdown
    shutdown_event.set()

    # Wait for threads to finish
    if get_actions_thread and get_actions_thread.is_alive():
        logger.info("Waiting for chunk requester thread to finish...")
        get_actions_thread.join()

    if actor_thread and actor_thread.is_alive():
        logger.info("Waiting for action executor thread to finish...")
        actor_thread.join()

    # Cleanup robot
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected")

    logger.info("Cleanup completed")


if __name__ == "__main__":
    demo_cli()
    logging.info("RTC demo finished")

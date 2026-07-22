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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Condition, Lock

import torch
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self._execution_idle = Condition(self.lock)
        self.last_index = 0
        self.cfg = cfg
        self.last_blend_steps = 0
        self.last_replace_old_remaining = 0
        self.last_replace_new_length = 0
        self.last_replace_real_delay = 0
        self.action_count = 0
        self.starvation_count = 0
        self._starved = False
        self._has_actions = False
        self._action_in_flight = False
        self.last_executed_action = None

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self._action_in_flight:
                raise RuntimeError("Cannot consume another action while one awaits acknowledgement.")
            if self.queue is None or self.last_index >= len(self.queue):
                self._mark_starved()
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            self.action_count += 1
            self.last_executed_action = action.detach().clone()
            if self.last_index >= len(self.queue):
                self._mark_starved()
            return action.detach().clone()

    def get_for_execution(self) -> Tensor | None:
        """Reserve the next action until the robot confirms it was sent."""
        with self.lock:
            if self._action_in_flight:
                raise RuntimeError("An action is already awaiting execution acknowledgement.")
            if self.queue is None or self.last_index >= len(self.queue):
                self._mark_starved()
                return None

            self._action_in_flight = True
            return self.queue[self.last_index].detach().clone()

    def mark_action_sent(self, action: Tensor) -> None:
        """Acknowledge a reserved action after it was sent successfully."""
        with self._execution_idle:
            if not self._action_in_flight:
                raise RuntimeError("No action is awaiting execution acknowledgement.")

            self.last_index += 1
            self.action_count += 1
            self.last_executed_action = action.detach().clone()
            self._action_in_flight = False
            if self.queue is not None and self.last_index >= len(self.queue):
                self._mark_starved()
            self._execution_idle.notify_all()

    def mark_action_failed(self) -> None:
        """Release a reserved action without counting it as executed."""
        with self._execution_idle:
            self._action_in_flight = False
            self._execution_idle.notify_all()

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        with self.lock:
            if self.queue is None:
                return 0
            return max(0, len(self.queue) - self.last_index)

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        with self.lock:
            return self.queue is None or self.last_index >= len(self.queue)

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        with self.lock:
            return self.last_index

    def get_action_count(self) -> int:
        """Return the monotonic number of executed actions."""
        with self.lock:
            return self.action_count

    def get_starvation_count(self) -> int:
        """Return the monotonic number of starvation transitions."""
        with self.lock:
            return self.starvation_count

    def needs_reanchor(self) -> bool:
        """Return whether the queue has emptied since its last merge."""
        with self.lock:
            return self._starved

    def reset_for_reanchor(self) -> None:
        """Clear queued history while preserving monotonic counters."""
        with self.lock:
            self._wait_until_execution_idle()
            self.queue = None
            self.original_queue = None
            self.last_index = 0
            self._starved = False
            self._has_actions = False
            self.last_blend_steps = 0
            self.last_replace_old_remaining = 0
            self.last_replace_new_length = 0
            self.last_replace_real_delay = 0

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_rtc_snapshot(self) -> tuple[Tensor | None, int, int]:
        """Return leftover actions and monotonic counters atomically."""
        with self.lock:
            left_over = None
            if self.original_queue is not None:
                left_over = self.original_queue[self.last_index :].clone()
            return left_over, self.action_count, self.starvation_count

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
        action_count_before_inference: int | None = None,
        starvation_count_before_inference: int | None = None,
    ) -> bool:
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            self._wait_until_execution_idle()
            starved_during_inference = starvation_count_before_inference is not None and (
                self._starved or self.starvation_count != starvation_count_before_inference
            )

            resolved_delay = real_delay
            if action_count_before_inference is not None:
                resolved_delay = max(0, self.action_count - action_count_before_inference)
                if resolved_delay != real_delay:
                    log_delay_mismatch = (
                        logger.warning if abs(resolved_delay - real_delay) > 1 else logger.debug
                    )
                    log_delay_mismatch(
                        "[ACTION_QUEUE] Wall-clock delay differs from sent actions. "
                        f"Sent actions: {resolved_delay}, wall-clock delay: {real_delay}"
                    )
            else:
                self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                if resolved_delay >= len(processed_actions):
                    logger.warning(
                        "[ACTION_QUEUE] No future actions remain after inference; "
                        "discarding generated chunk for a fresh reanchor."
                    )
                    return False
                if starved_during_inference:
                    logger.warning(
                        "[ACTION_QUEUE] Queue starved during inference; keeping the valid future tail "
                        "and blending it from the last sent action."
                    )
                self._replace_actions_queue(original_actions, processed_actions, resolved_delay)
                return True

            self._append_actions_queue(original_actions, processed_actions)
            return True

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        old_processed_remaining = None
        if self.queue is not None and self.last_index < len(self.queue):
            old_processed_remaining = self.queue[self.last_index :].clone()

        new_original_queue = original_actions[real_delay:].detach().clone()
        new_processed_queue = processed_actions[real_delay:].detach().clone()

        self.last_replace_old_remaining = 0 if old_processed_remaining is None else len(old_processed_remaining)
        self.last_replace_new_length = len(new_processed_queue)
        self.last_replace_real_delay = real_delay
        self.last_blend_steps = 0

        blend_steps = self._get_blend_steps(new_processed_queue)
        if blend_steps > 0:
            new_processed_queue = self._blend_prefix_from_last_action(new_processed_queue, blend_steps)
            self.last_blend_steps = blend_steps

        self.original_queue = new_original_queue
        self.queue = new_processed_queue

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}")

        self.last_index = 0
        self._set_merged_queue_state()

    def _get_blend_steps(self, new_processed_queue: Tensor) -> int:
        if self.cfg.queue_blend_steps <= 0 or self.last_executed_action is None:
            return 0
        return min(self.cfg.queue_blend_steps, len(new_processed_queue))

    def _blend_prefix_from_last_action(self, new_actions: Tensor, blend_steps: int) -> Tensor:
        if blend_steps <= 0 or self.last_executed_action is None:
            return new_actions

        blended_actions = new_actions.clone()
        progress = torch.arange(
            1,
            blend_steps + 1,
            device=new_actions.device,
            dtype=new_actions.dtype,
        ) / float(blend_steps + 1)
        weights = progress**3 * (progress * (progress * 6 - 15) + 10)
        weights = weights.view(-1, *([1] * (new_actions.ndim - 1)))
        last_action = self.last_executed_action.to(device=new_actions.device, dtype=new_actions.dtype)
        initial_offset = last_action - new_actions[0]
        blended_actions[:blend_steps] += initial_offset * (1 - weights)
        return blended_actions

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
        """
        if self.queue is None:
            self.original_queue = original_actions.detach().clone()
            self.queue = processed_actions.detach().clone()
            self._set_merged_queue_state()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.detach().clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.detach().clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0
        self._set_merged_queue_state()

    def _set_merged_queue_state(self) -> None:
        self._starved = False
        self._has_actions = self.queue is not None and len(self.queue) > 0
        if not self._has_actions:
            self._mark_starved()

    def _mark_starved(self) -> None:
        if self._has_actions and not self._starved:
            self._starved = True
            self.starvation_count += 1

    def _wait_until_execution_idle(self) -> None:
        while self._action_in_flight:
            self._execution_idle.wait()

    def _check_delays(self, real_delay: int, action_index_before_inference: int | None = None):
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.
        """
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            # Let's check that action index difference (real delay calculated based on action queue)
            # is the same as delay calculated based on inference latency
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )

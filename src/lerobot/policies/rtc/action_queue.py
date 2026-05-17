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
from threading import Lock

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
        self.last_index = 0
        self.cfg = cfg
        self.last_blend_steps = 0
        self.last_replace_old_remaining = 0
        self.last_replace_new_length = 0
        self.last_replace_real_delay = 0

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

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
            return self.original_queue[self.last_index :]

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ):
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
            self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, real_delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        old_original_remaining = None
        old_processed_remaining = None
        if self.original_queue is not None and self.last_index < len(self.original_queue):
            old_original_remaining = self.original_queue[self.last_index :].clone()
        if self.queue is not None and self.last_index < len(self.queue):
            old_processed_remaining = self.queue[self.last_index :].clone()

        new_original_queue = original_actions[real_delay:].clone()
        new_processed_queue = processed_actions[real_delay:].clone()

        self.last_replace_old_remaining = 0 if old_processed_remaining is None else len(old_processed_remaining)
        self.last_replace_new_length = len(new_processed_queue)
        self.last_replace_real_delay = real_delay
        self.last_blend_steps = 0

        blend_steps = self._get_blend_steps(old_processed_remaining, new_processed_queue)
        if blend_steps > 0:
            new_original_queue = self._blend_prefix(
                old_original_remaining,
                new_original_queue,
                blend_steps,
            )
            new_processed_queue = self._blend_prefix(
                old_processed_remaining,
                new_processed_queue,
                blend_steps,
            )
            self.last_blend_steps = blend_steps

        self.original_queue = new_original_queue
        self.queue = new_processed_queue

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}")

        self.last_index = 0

    def _get_blend_steps(self, old_processed_remaining: Tensor | None, new_processed_queue: Tensor) -> int:
        """Compute how many prefix actions should be blended during replacement."""
        if self.cfg.queue_blend_steps <= 0:
            return 0

        old_len = 0 if old_processed_remaining is None else len(old_processed_remaining)
        new_len = len(new_processed_queue)
        return min(self.cfg.queue_blend_steps, old_len, new_len)

    def _blend_prefix(self, old_actions: Tensor | None, new_actions: Tensor, blend_steps: int) -> Tensor:
        """Blend the first few actions to soften chunk boundary discontinuities."""
        if old_actions is None or blend_steps <= 0:
            return new_actions

        blended_actions = new_actions.clone()
        weights = torch.arange(
            1,
            blend_steps + 1,
            device=new_actions.device,
            dtype=new_actions.dtype,
        ) / float(blend_steps + 1)
        weights = weights.view(-1, *([1] * (new_actions.ndim - 1)))
        old_prefix = old_actions[:blend_steps].to(device=new_actions.device, dtype=new_actions.dtype)

        blended_actions[:blend_steps] = old_prefix * (1 - weights) + blended_actions[:blend_steps] * weights
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
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

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

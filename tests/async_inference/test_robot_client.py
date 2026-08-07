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
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import threading
import time
from queue import Queue

import numpy as np
import pytest
import torch

# Skip entire module if grpc is not available
pytest.importorskip("grpc")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_rtc_dry_run_does_not_send_robot_action(monkeypatch, robot_client):
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    robot_client.config.dry_run_actions = True
    robot_client.config.rtc = RTCConfig(enabled=True, execution_horizon=5)
    robot_client.rtc_action_queue = ActionQueue(robot_client.config.rtc)
    actions = torch.ones(3, len(robot_client.robot.action_features))
    assert robot_client.rtc_action_queue.merge(actions, actions, real_delay=0)

    monkeypatch.setattr(
        robot_client.robot,
        "send_action",
        lambda _: pytest.fail("dry-run must not call robot.send_action"),
    )

    performed_action = robot_client.control_loop_action()

    assert performed_action == {
        key: 1.0 for key in robot_client.robot.action_features
    }
    assert robot_client.rtc_action_queue.get_action_count() == 1


@pytest.mark.parametrize("rtc_enabled", [False, True])
def test_right_arm_filter_suppresses_left_actions_for_all_queue_modes(
    monkeypatch, robot_client, rtc_enabled: bool
):
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    action_features = {
        "left_joint_1.pos": float,
        "left_gripper.pos": float,
        "right_joint_1.pos": float,
        "right_gripper.pos": float,
    }
    robot_client.robot.action_features = action_features
    robot_client.config.controlled_arms = "right"
    action_tensor = torch.tensor([10.0, 20.0, 30.0, 40.0])
    sent_actions = []

    def record_action(action):
        sent_actions.append(action)
        return action

    monkeypatch.setattr(robot_client.robot, "send_action", record_action)

    if rtc_enabled:
        robot_client.config.rtc = RTCConfig(enabled=True, execution_horizon=1)
        robot_client.rtc_action_queue = ActionQueue(robot_client.config.rtc)
        actions = action_tensor.unsqueeze(0)
        assert robot_client.rtc_action_queue.merge(actions, actions, real_delay=0)
    else:
        robot_client.rtc_action_queue = None
        robot_client.action_queue.put(
            TimedAction(action=action_tensor, timestep=0, timestamp=time.time())
        )

    performed_action = robot_client.control_loop_action()

    expected = {"right_joint_1.pos": 30.0, "right_gripper.pos": 40.0}
    assert sent_actions == [expected]
    assert performed_action == expected
    assert robot_client.latest_display_action == expected
    diagnostics = robot_client._get_gripper_diagnostics()
    assert diagnostics["target"] == 40.0
    assert diagnostics["sent"] == 40.0


@pytest.mark.parametrize(
    "controlled_arms, expected",
    [
        ("both", {"left_joint.pos": 1.0, "right_joint.pos": 2.0}),
        ("left", {"left_joint.pos": 1.0}),
        ("right", {"right_joint.pos": 2.0}),
    ],
)
def test_action_arm_filter_mapping(robot_client, controlled_arms: str, expected: dict[str, float]):
    robot_client.robot.action_features = {
        "left_joint.pos": float,
        "right_joint.pos": float,
    }
    robot_client.config.controlled_arms = controlled_arms

    action = robot_client._action_tensor_to_action_dict(torch.tensor([1.0, 2.0]))

    assert action == expected


def test_live_display_logs_observation_and_latest_executed_action(robot_client):
    calls = []
    robot_client.config.display_compressed_images = True
    robot_client.display_logger = lambda **kwargs: calls.append(kwargs)
    robot_client._remember_display_action({"right_joint.pos": 1.25})
    observation = {
        "right_tactile": np.zeros((8, 8, 3), dtype=np.uint8),
        "right_joint.pos": 0.5,
    }

    robot_client._log_live_observation(observation)

    assert calls == [
        {
            "observation": observation,
            "action": {"right_joint.pos": 1.25},
            "compress_images": True,
        }
    ]


def test_live_display_failure_disables_display_without_raising(robot_client):
    def fail_display(**kwargs):
        del kwargs
        raise RuntimeError("viewer closed")

    robot_client.display_logger = fail_display

    robot_client._log_live_observation({"right_joint.pos": 0.5})

    assert robot_client.display_logger is None


def test_live_display_startup_failure_does_not_block_client(monkeypatch):
    from lerobot.async_inference import robot_client as robot_client_module
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    def fail_startup(enabled):
        del enabled
        raise RuntimeError("viewer unavailable")

    monkeypatch.setattr(robot_client_module, "_start_live_display", fail_startup)
    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        display_data=True,
    )

    client = robot_client_module.RobotClient(config)
    try:
        assert client.display_logger is None
        assert client.robot.is_connected
    finally:
        client.stop()


def test_capture_raw_observation_adds_task_and_logs_display(monkeypatch, robot_client):
    observation = {
        "right_tactile": np.zeros((8, 8, 3), dtype=np.uint8),
        "right_gripper.pos": np.float32(64.5),
    }
    displayed = []
    monkeypatch.setattr(robot_client.robot, "get_observation", lambda: dict(observation))
    monkeypatch.setattr(robot_client, "_log_live_observation", displayed.append)

    captured = robot_client._capture_raw_observation("move the cup")

    assert captured == {**observation, "task": "move the cup"}
    assert displayed == [captured]
    diagnostics = robot_client._get_gripper_diagnostics()
    assert diagnostics["feedback"] == pytest.approx(64.5)
    assert diagnostics["feedback_age_ms"] >= 0.0


def test_gripper_diagnostics_track_target_sent_feedback_and_age(robot_client):
    robot_client._remember_gripper_feedback({"right_gripper.pos": np.float32(65.0)})
    robot_client._remember_gripper_action(
        {"right_gripper.pos": 60.0},
        {"right_gripper.pos": 61.5},
    )
    feedback_at = robot_client.latest_right_gripper_feedback_at
    assert feedback_at is not None

    diagnostics = robot_client._get_gripper_diagnostics(now=feedback_at + 0.25)

    assert diagnostics == {
        "target": 60.0,
        "sent": 61.5,
        "feedback": 65.0,
        "error": 3.5,
        "feedback_age_ms": 250.0,
        "target_min": 60.0,
        "target_max": 60.0,
        "sent_min": 61.5,
        "sent_max": 61.5,
        "feedback_min": 65.0,
        "feedback_max": 65.0,
    }


def test_gripper_diagnostics_use_target_as_dry_run_sent_value(robot_client):
    robot_client._remember_gripper_action({"right_gripper.pos": 72.0}, None)

    diagnostics = robot_client._get_gripper_diagnostics()

    assert diagnostics["target"] == 72.0
    assert diagnostics["sent"] == 72.0
    assert np.isnan(diagnostics["feedback"])
    assert np.isnan(diagnostics["error"])
    assert np.isnan(diagnostics["feedback_age_ms"])
    assert diagnostics["target_min"] == 72.0
    assert diagnostics["target_max"] == 72.0
    assert diagnostics["sent_min"] == 72.0
    assert diagnostics["sent_max"] == 72.0
    assert np.isnan(diagnostics["feedback_min"])
    assert np.isnan(diagnostics["feedback_max"])


def test_gripper_diagnostics_report_and_reset_window_ranges(robot_client):
    robot_client._remember_gripper_action(
        {"right_gripper.pos": 90.0},
        {"right_gripper.pos": 89.0},
    )
    robot_client._remember_gripper_action(
        {"right_gripper.pos": 60.0},
        {"right_gripper.pos": 61.0},
    )
    robot_client._remember_gripper_feedback({"right_gripper.pos": 88.0})
    robot_client._remember_gripper_feedback({"right_gripper.pos": 64.0})

    diagnostics = robot_client._get_gripper_diagnostics(reset_window=True)
    after_reset = robot_client._get_gripper_diagnostics()

    assert diagnostics["target_min"] == 60.0
    assert diagnostics["target_max"] == 90.0
    assert diagnostics["sent_min"] == 61.0
    assert diagnostics["sent_max"] == 89.0
    assert diagnostics["feedback_min"] == 64.0
    assert diagnostics["feedback_max"] == 88.0
    assert after_reset["target"] == 60.0
    assert after_reset["sent"] == 61.0
    assert after_reset["feedback"] == 64.0
    assert np.isnan(after_reset["target_min"])
    assert np.isnan(after_reset["sent_min"])
    assert np.isnan(after_reset["feedback_min"])


def test_gripper_diagnostics_are_thread_safe(robot_client):
    errors = []

    def write_diagnostics():
        try:
            for value in range(500):
                robot_client._remember_gripper_action(
                    {"right_gripper.pos": value},
                    {"right_gripper.pos": value + 1},
                )
                robot_client._remember_gripper_feedback({"right_gripper.pos": value + 2})
        except Exception as error:
            errors.append(error)

    writer = threading.Thread(target=write_diagnostics)
    writer.start()
    while writer.is_alive():
        snapshot = robot_client._get_gripper_diagnostics()
        assert set(snapshot) == {
            "target",
            "sent",
            "feedback",
            "error",
            "feedback_age_ms",
            "target_min",
            "target_max",
            "sent_min",
            "sent_max",
            "feedback_min",
            "feedback_max",
        }
    writer.join()

    assert errors == []
    assert robot_client._get_gripper_diagnostics()["feedback"] == 501.0


def test_robot_client_config_exposes_live_display_settings():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        display_data=True,
        display_compressed_images=False,
    )

    config_dict = config.to_dict()
    assert config_dict["display_data"] is True
    assert config_dict["display_compressed_images"] is False


def test_robot_client_config_exposes_startup_right_gripper_settings():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        controlled_arms="right",
        startup_right_gripper_position=100.0,
        startup_right_gripper_hold_s=1.5,
    )

    config_dict = config.to_dict()
    assert config_dict["startup_right_gripper_position"] == 100.0
    assert config_dict["startup_right_gripper_hold_s"] == 1.5


def test_robot_client_config_exposes_both_startup_gripper_settings():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        controlled_arms="both",
        startup_left_gripper_position=100.0,
        startup_left_gripper_hold_s=1.0,
        startup_right_gripper_position=101.0,
        startup_right_gripper_hold_s=1.5,
    )

    config_dict = config.to_dict()
    assert config_dict["startup_left_gripper_position"] == 100.0
    assert config_dict["startup_left_gripper_hold_s"] == 1.0
    assert config_dict["startup_right_gripper_position"] == 101.0
    assert config_dict["startup_right_gripper_hold_s"] == 1.5


@pytest.mark.parametrize(
    "position, hold_s, controlled_arms, expected_message",
    [
        (float("nan"), 0.0, "right", "must be finite"),
        (0.0, -1.0, "right", "finite and non-negative"),
        (None, 1.0, "right", "requires startup_right_gripper_position"),
        (0.0, 0.0, "left", "controlled_arms='left'"),
    ],
)
def test_robot_client_config_rejects_invalid_startup_right_gripper_settings(
    position, hold_s, controlled_arms, expected_message
):
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match=expected_message):
        RobotClientConfig(
            robot=MockRobotConfig(),
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            controlled_arms=controlled_arms,
            startup_right_gripper_position=position,
            startup_right_gripper_hold_s=hold_s,
        )


@pytest.mark.parametrize(
    "position, hold_s, controlled_arms, expected_message",
    [
        (float("nan"), 0.0, "both", "must be finite"),
        (0.0, -1.0, "both", "finite and non-negative"),
        (None, 1.0, "both", "requires startup_left_gripper_position"),
        (0.0, 0.0, "right", "controlled_arms='right'"),
    ],
)
def test_robot_client_config_rejects_invalid_startup_left_gripper_settings(
    position, hold_s, controlled_arms, expected_message
):
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match=expected_message):
        RobotClientConfig(
            robot=MockRobotConfig(),
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            controlled_arms=controlled_arms,
            startup_left_gripper_position=position,
            startup_left_gripper_hold_s=hold_s,
        )


def test_startup_right_gripper_command_and_hold(monkeypatch, robot_client):
    sent_actions = []
    robot_client.robot.action_features = {
        "right_joint.pos": float,
        "right_gripper.pos": float,
    }
    robot_client.config.startup_right_gripper_position = 0.0
    robot_client.config.startup_right_gripper_hold_s = 1.5
    monkeypatch.setattr(
        robot_client.robot,
        "send_action",
        lambda action: sent_actions.append(action) or dict(action),
    )

    robot_client._command_startup_right_gripper_position()
    held = robot_client._apply_startup_right_gripper_hold(
        {"right_joint.pos": 12.0, "right_gripper.pos": -25.0},
        now=100.0,
    )
    still_held = robot_client._apply_startup_right_gripper_hold(
        {"right_joint.pos": 13.0, "right_gripper.pos": -30.0},
        now=101.49,
    )
    released = robot_client._apply_startup_right_gripper_hold(
        {"right_joint.pos": 14.0, "right_gripper.pos": -35.0},
        now=101.5,
    )

    assert sent_actions == [{"right_gripper.pos": 0.0}]
    assert held == {"right_joint.pos": 12.0, "right_gripper.pos": 0.0}
    assert still_held == {"right_joint.pos": 13.0, "right_gripper.pos": 0.0}
    assert released == {"right_joint.pos": 14.0, "right_gripper.pos": -35.0}
    assert robot_client.startup_right_gripper_hold_finished is True


def test_startup_both_gripper_command_and_hold(monkeypatch, robot_client):
    sent_actions = []
    robot_client.robot.action_features = {
        "left_gripper.pos": float,
        "right_gripper.pos": float,
    }
    robot_client.config.startup_left_gripper_position = 100.0
    robot_client.config.startup_left_gripper_hold_s = 1.0
    robot_client.config.startup_right_gripper_position = 101.0
    robot_client.config.startup_right_gripper_hold_s = 1.5
    monkeypatch.setattr(
        robot_client.robot,
        "send_action",
        lambda action: sent_actions.append(dict(action)) or dict(action),
    )

    robot_client._command_startup_gripper_positions()
    held = robot_client._apply_startup_gripper_holds(
        {"left_gripper.pos": -25.0, "right_gripper.pos": -30.0},
        now=100.0,
    )
    released = robot_client._apply_startup_gripper_holds(
        {"left_gripper.pos": -35.0, "right_gripper.pos": -40.0},
        now=101.5,
    )

    assert sent_actions == [
        {"left_gripper.pos": 100.0, "right_gripper.pos": 101.0}
    ]
    assert held == {"left_gripper.pos": 100.0, "right_gripper.pos": 101.0}
    assert released == {"left_gripper.pos": -35.0, "right_gripper.pos": -40.0}
    assert robot_client.startup_left_gripper_hold_finished is True
    assert robot_client.startup_right_gripper_hold_finished is True


def test_startup_right_gripper_hold_preserves_policy_target_in_diagnostics(
    monkeypatch, robot_client
):
    from lerobot.async_inference.helpers import TimedAction

    robot_client.robot.action_features = {
        "right_joint.pos": float,
        "right_gripper.pos": float,
    }
    robot_client.config.controlled_arms = "right"
    robot_client.config.startup_right_gripper_position = 0.0
    robot_client.config.startup_right_gripper_hold_s = 10.0
    robot_client.action_queue.put(
        TimedAction(
            action=torch.tensor([12.0, -25.0]),
            timestep=0,
            timestamp=time.time(),
        )
    )
    sent_actions = []
    monkeypatch.setattr(
        robot_client.robot,
        "send_action",
        lambda action: sent_actions.append(dict(action)) or dict(action),
    )

    robot_client.control_loop_action()

    assert sent_actions == [{"right_joint.pos": 12.0, "right_gripper.pos": 0.0}]
    diagnostics = robot_client._get_gripper_diagnostics()
    assert diagnostics["target"] == -25.0
    assert diagnostics["sent"] == 0.0


def test_robot_client_config_rejects_invalid_controlled_arms():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="controlled_arms"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            controlled_arms="invalid",
        )


def test_rtc_metadata_contains_padded_leftover_actions(robot_client):
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    robot_client.config.rtc = RTCConfig(enabled=True, execution_horizon=2)
    robot_client.rtc_action_queue = ActionQueue(robot_client.config.rtc)
    actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert robot_client.rtc_action_queue.merge(actions, actions, real_delay=0)
    action = robot_client.rtc_action_queue.get_for_execution()
    robot_client.rtc_action_queue.mark_action_sent(action)

    metadata = robot_client._prepare_rtc_metadata(request_id=7)

    assert metadata.request_id == 7
    assert metadata.execution_horizon == 2
    assert metadata.action_count_before_inference == 1
    assert metadata.prev_chunk_left_over.shape == (robot_client.config.actions_per_chunk, 3)
    assert torch.equal(metadata.prev_chunk_left_over[:3], actions[1:])


def test_rtc_warmup_uses_prefix_then_resets_runtime_state(robot_client):
    from lerobot.async_inference.helpers import RemoteActionChunk
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    robot_client.config.rtc = RTCConfig(enabled=True, execution_horizon=6, warmup_requests=3)
    robot_client.rtc_action_queue = ActionQueue(robot_client.config.rtc)
    robot_client.rtc_warmup_complete.clear()
    actions = torch.ones(20, len(robot_client.robot.action_features))

    for warmup_index in range(3):
        assert robot_client._ready_to_send_observation()
        request_id = robot_client._begin_rtc_request()
        metadata = robot_client._prepare_rtc_metadata(request_id)
        assert (metadata.prev_chunk_left_over is None) is (warmup_index == 0)
        if warmup_index > 0:
            assert metadata.execution_horizon == 6

        robot_client._merge_rtc_action_chunk(
            RemoteActionChunk(
                request_id=request_id,
                original_actions=actions,
                processed_actions=actions,
                observation_timestamp=time.time(),
                observation_timestep=0,
                action_count_before_inference=0,
                starvation_count_before_inference=0,
            ),
            receive_time=time.time(),
        )

    assert robot_client.rtc_warmup_complete.is_set()
    assert robot_client.rtc_action_queue.empty()
    assert len(robot_client.rtc_latency_tracker) == 0
    assert robot_client.get_executed_action_count() == 0


def test_rtc_warmup_blocks_actions_and_duration_starts_after_completion(monkeypatch, robot_client):
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    robot_client.config.rtc = RTCConfig(enabled=True, warmup_requests=1, warmup_timeout_s=1.0)
    robot_client.config.duration = 0.05
    robot_client.rtc_action_queue = ActionQueue(robot_client.config.rtc)
    robot_client.rtc_warmup_complete.clear()
    actions = torch.ones(3, len(robot_client.robot.action_features))
    assert robot_client.rtc_action_queue.merge(actions, actions, real_delay=0)
    robot_client.start_barrier = type("ImmediateBarrier", (), {"wait": lambda self: None})()
    sent_actions = []

    def record_action(action):
        sent_actions.append((time.perf_counter(), action))
        return action

    monkeypatch.setattr(robot_client.robot, "send_action", record_action)

    warmup_delay_s = 0.08
    timer = threading.Timer(warmup_delay_s, robot_client.rtc_warmup_complete.set)
    started_at = time.perf_counter()
    timer.start()
    try:
        robot_client.action_control_loop()
    finally:
        timer.cancel()

    assert time.perf_counter() - started_at >= warmup_delay_s + robot_client.config.duration
    assert sent_actions
    assert sent_actions[0][0] - started_at >= warmup_delay_s


def test_rtc_warmup_timeout_stops_without_sending_actions(monkeypatch, robot_client):
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    robot_client.config.rtc = RTCConfig(enabled=True, warmup_requests=1, warmup_timeout_s=0.02)
    robot_client.rtc_action_queue = ActionQueue(robot_client.config.rtc)
    robot_client.rtc_warmup_complete.clear()
    actions = torch.ones(3, len(robot_client.robot.action_features))
    assert robot_client.rtc_action_queue.merge(actions, actions, real_delay=0)
    robot_client.start_barrier = type("ImmediateBarrier", (), {"wait": lambda self: None})()
    monkeypatch.setattr(
        robot_client.robot,
        "send_action",
        lambda _: pytest.fail("warm-up timeout must fail closed before sending actions"),
    )

    assert robot_client.action_control_loop() is None
    assert robot_client.shutdown_event.is_set()
    assert robot_client.get_executed_action_count() == 0

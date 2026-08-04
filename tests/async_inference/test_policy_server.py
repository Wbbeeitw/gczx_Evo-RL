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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import require_package

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@require_package("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(
        PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True
    )

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(
        range(5, 5 + actions_per_chunk)
    )

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_predict_action_chunk_does_not_capture_rng_when_debug_is_disabled(
    monkeypatch, policy_server
):
    from lerobot.async_inference.policy_server import PolicyServer

    policy_server.policy_type = "act"
    policy_server.preprocessor = lambda observation: observation
    policy_server.postprocessor = lambda action: action

    def fail_if_called(_self):
        raise AssertionError(
            "RNG state should not be captured when diagnostics are disabled"
        )

    class FailIfLocked:
        def __enter__(self):
            raise AssertionError(
                "Policy lock should not be used when diagnostics are disabled"
            )

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        PolicyServer, "_capture_policy_rng_state", fail_if_called, raising=True
    )
    monkeypatch.setattr(
        policy_server,
        "_update_tactile_baselines",
        lambda observation: pytest.fail(
            "Tactile baselines should not update when diagnostics are disabled"
        ),
    )
    monkeypatch.setattr(
        policy_server,
        "_maybe_schedule_xr0_diagnostics",
        lambda **kwargs: pytest.fail(
            "Diagnostics should not be scheduled when diagnostics are disabled"
        ),
    )
    policy_server._policy_inference_lock = FailIfLocked()

    timed_actions = policy_server._predict_action_chunk(
        _make_obs(torch.zeros(6), timestep=5)
    )

    assert len(timed_actions) == policy_server.actions_per_chunk


def test_get_action_chunk_forwards_rtc_metadata(policy_server):
    from lerobot.async_inference.helpers import RTCInferenceMetadata

    captured_kwargs = {}

    def predict_action_chunk(observation, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.zeros(1, 20, 6)

    policy_server.policy.predict_action_chunk = predict_action_chunk
    prefix = torch.ones(20, 6)
    rtc = RTCInferenceMetadata(
        request_id=3,
        prev_chunk_left_over=prefix,
        inference_delay=2,
        execution_horizon=5,
    )

    chunk = policy_server._get_action_chunk({OBS_STATE: torch.zeros(1, 6)}, rtc)

    assert chunk.shape == (1, 20, 6)
    assert torch.equal(captured_kwargs["prev_chunk_left_over"], prefix)
    assert captured_kwargs["inference_delay"] == 2
    assert captured_kwargs["execution_horizon"] == 5


def test_policy_server_debug_options_default_to_disabled():
    from lerobot.async_inference.configs import PolicyServerConfig

    config = PolicyServerConfig()

    assert config.debug_qwen_text is False
    assert config.debug_tactile_counterfactual is False
    assert config.debug_action_comparison_dir is None
    assert config.debug_action_comparison_save_images is False


def test_tactile_baseline_uses_initial_unloaded_frames(policy_server):
    policy_server.config.debug_qwen_text = True
    policy_server.config.debug_tactile_baseline_frames = 3
    policy_server._debug_qwen_text_enabled = True
    tactile_key = "observation.images.right_tactile"
    policy_server._tactile_image_keys = (tactile_key,)
    policy_server._reset_diagnostic_state()

    for value in (1.0, 3.0, 2.0):
        policy_server._update_tactile_baselines(
            {
                OBS_STATE: torch.zeros(1, 6),
                tactile_key: torch.full((1, 3, 2, 2), value),
            }
        )

    baseline = policy_server._tactile_baselines[tactile_key]
    assert torch.equal(baseline, torch.full((1, 3, 2, 2), 2.0))


def test_tactile_counterfactual_metrics_support_xr0_32d_layout(policy_server):
    primary = torch.zeros(30, 32)
    counterfactual = primary.clone()
    counterfactual[:, 6] = 2.0
    counterfactual[:, 20] = 4.0
    counterfactual[:, 7:13] = 1.0
    counterfactual[:, 21:27] = 3.0

    metrics = policy_server._action_difference_metrics(primary, counterfactual)

    assert metrics["left_gripper_delta"] == pytest.approx(2.0)
    assert metrics["right_gripper_delta"] == pytest.approx(4.0)
    assert metrics["left_arm_joint_delta"] == pytest.approx(1.0)
    assert metrics["right_arm_joint_delta"] == pytest.approx(3.0)


def test_tactile_counterfactual_suffix_metrics_exclude_rtc_prefix(policy_server):
    primary = torch.zeros(30, 32)
    counterfactual = primary.clone()
    right_dimensions = [20, 21, 22, 23, 24, 25, 26]
    counterfactual[:6, right_dimensions] = 100.0
    counterfactual[6:, 20] = 4.0
    for joint_index, dimension in enumerate(range(21, 27), start=1):
        counterfactual[6:, dimension] = float(joint_index)

    metrics = policy_server._action_difference_metrics(
        primary,
        counterfactual,
        prefix_length=6,
        action_std=torch.ones(30, 32),
        controlled_arms="right",
    )

    assert metrics["rtc_prefix_length"] == 6
    assert metrics["max_abs_delta"] == pytest.approx(100.0)
    assert metrics["suffix_max_abs_delta"] == pytest.approx(6.0)
    assert metrics["suffix_mean_abs_delta"] == pytest.approx(25.0 / 7.0)
    assert metrics["next_generated_action_delta"] == pytest.approx(25.0 / 7.0)
    assert metrics["right_joint_3_delta"] == pytest.approx(3.0)
    assert metrics["right_gripper_suffix_delta"] == pytest.approx(4.0)
    assert metrics["most_affected_dimension"] == "right_joint_6"
    assert metrics["most_affected_score_std"] == pytest.approx(6.0)


def test_tactile_difference_metrics_compare_current_to_unloaded_baseline(
    policy_server,
):
    tactile_key = "observation.images.right_tactile"
    policy_server._tactile_image_keys = (tactile_key,)
    policy_server._tactile_baselines = {
        tactile_key: torch.zeros(1, 3, 2, 2),
    }
    observation = {tactile_key: torch.full((1, 3, 2, 2), 0.25)}

    metrics, frames = policy_server._tactile_difference_metrics(observation)

    assert metrics["tactile_pixel_mean_delta"] == pytest.approx(0.25)
    assert metrics["tactile_pixel_max_delta"] == pytest.approx(0.25)
    assert metrics["tactile_active_pixel_ratio"] == pytest.approx(1.0)
    assert tactile_key in frames


def test_action_comparison_record_saves_csv_npz_and_tactile_images(
    tmp_path, policy_server
):
    output_dir = tmp_path / "session"
    (output_dir / "action_chunks").mkdir(parents=True)
    (output_dir / "tactile_frames").mkdir()
    (output_dir / "reports").mkdir()
    policy_server._diagnostic_output_dir = output_dir
    policy_server.config.debug_action_comparison_save_images = True
    tactile_key = "observation.images.right_tactile"
    tactile_frames = {
        tactile_key: (
            torch.ones(1, 3, 2, 2),
            torch.zeros(1, 3, 2, 2),
            torch.ones(1, 3, 2, 2),
        )
    }

    policy_server._save_action_comparison_record(
        timestep=12,
        prefix_length=6,
        primary_action=torch.zeros(30, 32),
        counterfactual_action=torch.ones(1, 30, 32),
        metrics={
            "rtc_prefix_length": 6,
            "suffix_mean_abs_delta": 1.0,
            "most_affected_dimension": "right_joint_1",
        },
        tactile_frames=tactile_frames,
        diagnostic_time_s=0.2,
    )

    assert (output_dir / "summary.csv").is_file()
    assert len(list((output_dir / "action_chunks").glob("*.npz"))) == 1
    assert len(list((output_dir / "tactile_frames").glob("*.jpg"))) == 3

    pytest.importorskip("matplotlib")
    from lerobot.scripts.lerobot_xr0_tactile_report import generate_report

    report_dir = generate_report(output_dir)
    assert (report_dir / "action_delta_timeline.png").is_file()
    assert (report_dir / "right_gripper_comparison.png").is_file()
    assert (report_dir / "right_joint_deltas.png").is_file()
    assert (report_dir / "tactile_vs_action_delta.png").is_file()


def test_tactile_counterfactual_reuses_primary_rng_state(monkeypatch, policy_server):
    tactile_key = "observation.images.right_tactile"
    observation = {
        OBS_STATE: torch.zeros(1, 6),
        tactile_key: torch.ones(1, 3, 2, 2),
    }
    policy_server._tactile_image_keys = (tactile_key,)
    policy_server._tactile_baselines = {
        tactile_key: torch.zeros(1, 3, 2, 2),
    }

    class StochasticTactilePolicy:
        @staticmethod
        def predict_action_chunk(observation, **kwargs):
            tactile_offset = observation[tactile_key].mean()
            return torch.rand(1, 20, 6) + tactile_offset

    policy = StochasticTactilePolicy()
    primary_rng_state = policy_server._capture_policy_rng_state()
    primary_action = policy.predict_action_chunk(observation)
    rng_state_before_diagnostic = torch.random.get_rng_state().clone()
    captured = {}

    def capture_metrics(primary, counterfactual, **kwargs):
        captured["primary"] = primary
        captured["counterfactual"] = counterfactual
        return {"mean_abs_delta": float((primary - counterfactual).abs().mean())}

    monkeypatch.setattr(policy_server, "_action_difference_metrics", capture_metrics)

    policy_server._run_xr0_diagnostics(
        policy=policy,
        observation=observation,
        predict_kwargs={},
        primary_action=primary_action,
        primary_rng_state=primary_rng_state,
        timestep=0,
        run_text=False,
        run_counterfactual=True,
    )

    assert torch.allclose(
        captured["primary"] - captured["counterfactual"],
        torch.ones_like(primary_action),
    )
    assert torch.equal(torch.random.get_rng_state(), rng_state_before_diagnostic)


def test_tactile_masking_is_skipped_for_rgb_only_checkpoint(policy_server):
    policy_server._tactile_image_keys = ()
    policy_server._tactile_baselines = {}

    observation = {OBS_STATE: torch.zeros(1, 6)}

    assert policy_server._make_tactile_masked_observation(observation) is None

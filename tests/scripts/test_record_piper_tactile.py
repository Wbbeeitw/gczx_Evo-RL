import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.scripts.lerobot_record_piper_tactile import (
    _disconnect_hardware,
    _episode_metadata,
    _FrameWriter,
    _log_live_frame,
    _missing_left_wrist_frame,
    parse_args,
)


class _RecordingDataset:
    def __init__(self, fail_at: int | None = None) -> None:
        self.frames = []
        self.fail_at = fail_at

    def add_frame(self, frame) -> None:
        if self.fail_at is not None and frame["index"] == self.fail_at:
            raise ValueError("write failed")
        self.frames.append(frame)


class _BlockingDataset(_RecordingDataset):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def add_frame(self, frame) -> None:
        if not self.frames:
            self.started.set()
            self.release.wait()
        super().add_frame(frame)


def test_frame_writer_drains_frames_in_fifo_order() -> None:
    dataset = _RecordingDataset()
    writer = _FrameWriter(dataset)

    for index in range(100):
        writer.submit({"index": index})
    writer.close()

    assert [frame["index"] for frame in dataset.frames] == list(range(100))


def test_frame_writer_close_is_idempotent_and_rejects_new_frames() -> None:
    writer = _FrameWriter(_RecordingDataset())

    writer.close()
    writer.close()

    with pytest.raises(RuntimeError, match="already closed"):
        writer.submit({"index": 0})


def test_frame_writer_surfaces_dataset_errors() -> None:
    writer = _FrameWriter(_RecordingDataset(fail_at=3))
    for index in range(10):
        try:
            writer.submit({"index": index})
        except RuntimeError:
            break

    with pytest.raises(RuntimeError, match="failed while adding") as exc_info:
        writer.close()

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_frame_writer_abort_does_not_drain_backlog() -> None:
    dataset = _BlockingDataset()
    writer = _FrameWriter(dataset)
    writer.submit({"index": 0})
    assert dataset.started.wait(timeout=1.0)
    for index in range(1, 100):
        writer.submit({"index": index})

    abort_thread = threading.Thread(target=writer.abort)
    abort_thread.start()
    dataset.release.set()
    abort_thread.join(timeout=1.0)

    assert not abort_thread.is_alive()
    assert [frame["index"] for frame in dataset.frames] == [0]


def test_frame_writer_reports_sustained_queue_overload() -> None:
    dataset = _BlockingDataset()
    writer = _FrameWriter(dataset, max_queue_size=1)
    writer.submit({"index": 0})
    assert dataset.started.wait(timeout=1.0)
    writer.submit({"index": 1})

    with pytest.raises(RuntimeError, match="queue stayed full"):
        writer.submit({"index": 2})

    dataset.release.set()
    writer.abort()


def test_disconnect_hardware_cleans_partially_connected_arms() -> None:
    class FakeArm:
        def __init__(self, *, is_connected: bool = False, process_running: bool = False) -> None:
            self.is_connected = is_connected
            self._process = object() if process_running else None
            self.disconnect_calls = 0

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.is_connected = False
            self._process = None

    teleop = SimpleNamespace(
        left_arm=FakeArm(is_connected=True),
        right_arm=FakeArm(process_running=True),
    )
    robot = SimpleNamespace(
        left_arm=FakeArm(),
        right_arm=FakeArm(is_connected=True),
    )

    _disconnect_hardware(robot, teleop, logging.getLogger(__name__))

    assert teleop.left_arm.disconnect_calls == 1
    assert teleop.right_arm.disconnect_calls == 1
    assert robot.left_arm.disconnect_calls == 0
    assert robot.right_arm.disconnect_calls == 1


def test_disconnect_hardware_retries_partial_tactile_runtime_cleanup() -> None:
    class FakeCamera:
        is_connected = False

        def __init__(self) -> None:
            self._runtime = object()

    class FakeArm:
        is_connected = False

        def __init__(self) -> None:
            self.camera = FakeCamera()
            self.cameras = {"tactile": self.camera}
            self.disconnect_calls = 0

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            if self.disconnect_calls == 1:
                raise RuntimeError("stop failed")
            self.camera._runtime = None

    arm = FakeArm()
    robot = SimpleNamespace(left_arm=arm, right_arm=SimpleNamespace(is_connected=False, cameras={}))
    teleop = SimpleNamespace(is_connected=False)
    logger = logging.getLogger(__name__)

    _disconnect_hardware(robot, teleop, logger)
    assert arm.disconnect_calls == 2
    assert arm.camera._runtime is None


@pytest.mark.parametrize("outcome", ["success", "failure", "ongoing"])
def test_episode_metadata_uses_a_stable_string_schema(outcome: str) -> None:
    assert _episode_metadata(outcome) == {"trajectory_type": outcome}


def test_episode_metadata_rejects_unsaved_outcomes() -> None:
    with pytest.raises(ValueError, match="unsupported episode outcome"):
        _episode_metadata("discard")


def test_missing_left_wrist_black_fill() -> None:
    frame = _missing_left_wrist_frame(
        {},
        fill_mode="black",
        ego_camera_side="left",
        height=4,
        width=6,
    )

    assert frame.shape == (4, 6, 3)
    assert frame.dtype == np.uint8
    assert not frame.any()


@pytest.mark.parametrize(
    ("fill_mode", "ego_camera_side", "source_key"),
    [
        ("copy-ego", "left", "left_ego"),
        ("copy-ego", "right", "right_ego"),
        ("copy-right-wrist", "left", "right_wrist"),
    ],
)
def test_missing_left_wrist_copies_configured_source(
    fill_mode: str,
    ego_camera_side: str,
    source_key: str,
) -> None:
    source = np.full((3, 5, 3), 7, dtype=np.uint8)

    frame = _missing_left_wrist_frame(
        {source_key: source},
        fill_mode=fill_mode,
        ego_camera_side=ego_camera_side,
        height=3,
        width=5,
    )

    np.testing.assert_array_equal(frame, source)
    assert frame is not source


def test_live_display_receives_rgb_and_tactile_observations() -> None:
    captured = {}

    def display_logger(**kwargs) -> None:
        captured.update(kwargs)

    observation = {
        "left_ego": np.zeros((4, 6, 3), dtype=np.uint8),
        "left_wrist": np.zeros((4, 6, 3), dtype=np.uint8),
        "right_wrist": np.zeros((4, 6, 3), dtype=np.uint8),
        "right_tactile": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    action = {"right_joint_1.pos": 1.0}

    _log_live_frame(
        display_logger,
        observation,
        action,
        compress_images=True,
    )

    assert captured == {
        "observation": observation,
        "action": action,
        "compress_images": True,
    }


def test_live_display_is_a_noop_when_disabled() -> None:
    _log_live_frame(
        None,
        {"right_tactile": np.zeros((8, 8, 3), dtype=np.uint8)},
        {},
        compress_images=True,
    )


def test_display_cli_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "lerobot-record-piper-tactile",
            "--task",
            "test task",
            "--left-follower-can",
            "can0",
            "--right-follower-can",
            "can1",
            "--left-leader-can",
            "can2",
            "--right-leader-can",
            "can3",
            "--top-camera",
            "top",
            "--right-wrist-camera",
            "right-wrist",
            "--dataset.repo_id",
            "test/repo",
            "--display_data",
            "--no-display-compressed-images",
        ],
    )

    args = parse_args()

    assert args.display_data is True
    assert args.display_compressed_images is False

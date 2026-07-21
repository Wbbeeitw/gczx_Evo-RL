import pytest

from lerobot.datasets.video_utils import VideoEncodingManager


def test_video_encoding_manager_waits_for_images_before_exception_cleanup(tmp_path) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.episodes_since_last_encoding = 0
            self.num_episodes = 0
            self.root = tmp_path
            self.meta = type("Meta", (), {"video_keys": []})()
            self.events = []

        def _wait_image_writer(self) -> None:
            self.events.append("wait")

        def finalize(self) -> None:
            self.events.append("finalize")

    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="stop"):
        with VideoEncodingManager(dataset):
            raise RuntimeError("stop")

    assert dataset.events == ["wait", "finalize"]


def test_video_encoding_manager_finalizes_when_image_wait_fails(tmp_path) -> None:
    class FakeDataset:
        episodes_since_last_encoding = 0
        num_episodes = 0
        root = tmp_path
        meta = type("Meta", (), {"video_keys": []})()

        def __init__(self) -> None:
            self.finalized = False

        def _wait_image_writer(self) -> None:
            raise RuntimeError("image failure")

        def finalize(self) -> None:
            self.finalized = True

    dataset = FakeDataset()

    with pytest.raises(RuntimeError, match="recording failure"):
        with VideoEncodingManager(dataset):
            raise RuntimeError("recording failure")

    assert dataset.finalized

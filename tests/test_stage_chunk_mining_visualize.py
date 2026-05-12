#!/usr/bin/env python

from pathlib import Path

import numpy as np
import torch

from lerobot.rl.stage_chunk_mining.visualize import (
    generate_stage_chunk_episode_figures,
    select_camera_keys,
    verify_chunk_mask_consistency,
    verify_episode_chunk_mask,
)


class _FakeMeta:
    camera_keys = [
        "observation.images.left_top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]


class _FakeDataset:
    meta = _FakeMeta()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        value = (idx + 1) * 20
        frame = np.full((24, 32, 3), value, dtype=np.uint8)
        return {
            "observation.images.left_top": torch.from_numpy(frame.copy()),
            "observation.images.left_wrist": torch.from_numpy(np.flip(frame, axis=1).copy()),
            "observation.images.right_wrist": torch.from_numpy(np.flip(frame, axis=0).copy()),
        }


def test_select_camera_keys_uses_requested_subset():
    selected = select_camera_keys(
        available_camera_keys=_FakeMeta.camera_keys,
        requested_camera_keys="observation.images.right_wrist, observation.images.left_top",
        max_cameras=3,
    )
    assert selected == ["observation.images.right_wrist", "observation.images.left_top"]


def test_verify_episode_chunk_mask_checks_union_of_selected_chunks():
    report = verify_episode_chunk_mask(
        ep_chunk_start_indicator=np.asarray([1, 0, 0, 1, 0, 0], dtype=np.int64),
        ep_indicator=np.asarray([1, 1, 0, 1, 1, 0], dtype=np.int64),
        chunk_size=2,
    )
    assert report["pass"] is True
    assert report["selected_chunk_count"] == 2
    assert report["actual_positive_frames"] == 4


def test_verify_chunk_mask_consistency_reports_mismatch():
    report = verify_chunk_mask_consistency(
        episode_indices=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
        chunk_start_indicator=np.asarray([1, 0, 0, 1, 0], dtype=np.int64),
        indicator=np.asarray([1, 0, 0, 1, 1], dtype=np.int64),
        chunk_size=2,
    )
    assert report["pass"] is False
    assert report["failed_episodes"] == [0]
    assert report["per_episode"]["0"]["indicator_mismatch_count"] == 1


def test_generate_stage_chunk_episode_figures_writes_gallery(tmp_path: Path):
    dataset = _FakeDataset()
    result = generate_stage_chunk_episode_figures(
        dataset=dataset,
        episode_indices=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        frame_indices=np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        values=np.asarray([-0.8, -0.7, -0.6, -0.5, -0.9, -0.8, -0.7, -0.6], dtype=np.float32),
        chunk_advantage=np.asarray([0.4, np.nan, 0.2, np.nan, 0.5, np.nan, 0.1, np.nan], dtype=np.float32),
        chunk_start_indicator=np.asarray([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64),
        indicator=np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int64),
        chunk_size=2,
        output_dir=tmp_path / "figures",
        output_prefix="complementary_info.vgsacm_global_r050_nooverlap_chunkmask",
    )
    assert result["figure_count"] == 2
    assert result["verification"]["pass"] is True
    for path in result["figure_paths"]:
        assert Path(path).exists()

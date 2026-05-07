#!/usr/bin/env python

import unittest

import numpy as np

from lerobot.rl.stage_chunk_mining.advantage import compute_chunk_advantage
from lerobot.rl.stage_chunk_mining.boundary import find_forward_boundaries, find_unique_stage_boundaries
from lerobot.rl.stage_chunk_mining.selection import (
    SELECTION_BOUNDARY_TRANSITION,
    SELECTION_INTRA_STAGE_TOP,
    mine_stage_chunks,
)
from lerobot.rl.stage_chunk_mining.stages import normalize_values, values_to_stages


class TestStageChunkMining(unittest.TestCase):
    def test_value_stage_partition(self):
        values = np.asarray([-1.0, -0.9, -0.7, -0.5, -0.1, 0.0], dtype=np.float32)
        _completion, stages = values_to_stages(values, num_stages=5)
        np.testing.assert_array_equal(stages, np.asarray([0, 0, 1, 2, 4, 4], dtype=np.int64))

    def test_episode_minmax_normalizes_values_to_negative_unit_range(self):
        values = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
        normalized = normalize_values(values, mode="episode_minmax")
        np.testing.assert_allclose(normalized, np.asarray([-1.0, -0.5, 0.0], dtype=np.float32))

    def test_chunk_advantage_formula(self):
        advantage = compute_chunk_advantage(start_value=-0.8, bootstrap_value=-0.4, chunk_size=50, l_max=200)
        self.assertAlmostEqual(advantage, 0.15, places=6)

    def test_stage_top_ratio_selects_intra_stage_chunks(self):
        values = np.asarray([-0.9, -0.89, -0.88, -0.87, -0.86, -0.85, -0.84], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            num_stages=5,
            chunk_size=1,
            value_normalization="clip",
            stage_top_ratio=0.5,
            include_boundary=False,
        )
        self.assertGreater(int(np.sum(result.selection_role == SELECTION_INTRA_STAGE_TOP)), 0)
        self.assertEqual(int(np.sum(result.indicator)), int(np.sum(result.selection_role == SELECTION_INTRA_STAGE_TOP)))

    def test_success_episode_uses_episode_minmax_monotonic_stages(self):
        values = np.asarray([100.0, 125.0, 150.0, 175.0, 200.0], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: True},
            num_stages=5,
            chunk_size=1,
            include_intra_stage=False,
            include_boundary=False,
        )
        np.testing.assert_array_equal(result.stage, np.asarray([0, 1, 2, 3, 4], dtype=np.int64))
        self.assertEqual(result.report["success_episodes"], 1)
        self.assertEqual(result.report["failure_episodes"], 0)

    def test_success_episode_suppresses_stage_jitter(self):
        values = np.asarray([0.0, 2.0, 1.0, 4.0, 3.0], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: True},
            num_stages=5,
            chunk_size=1,
            include_intra_stage=False,
            include_boundary=False,
        )
        self.assertTrue(bool(np.all(np.diff(result.stage) >= 0)))
        self.assertEqual(int(result.stage[0]), 0)
        self.assertEqual(int(result.stage[-1]), 4)

    def test_success_episode_stage_range_uses_unsmoothed_values(self):
        values = np.asarray([0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: True},
            num_stages=5,
            chunk_size=1,
            value_smoothing_window=3,
            include_intra_stage=False,
            include_boundary=False,
        )
        self.assertEqual(int(np.min(result.stage)), 0)
        self.assertEqual(int(np.max(result.stage)), 4)
        np.testing.assert_array_equal(result.stage, np.asarray([0, 0, 0, 0, 4], dtype=np.int64))

    def test_success_episode_minmax_starts_from_initial_progress(self):
        values = np.asarray([5.0, 4.0, 6.25, 7.5, 8.75, 10.0], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: True},
            num_stages=5,
            chunk_size=1,
            include_intra_stage=False,
            include_boundary=False,
        )
        np.testing.assert_array_equal(result.stage, np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64))

    def test_success_episode_waits_for_next_threshold_when_start_is_not_minimum(self):
        values = np.asarray([6.0, 4.0, 6.25, 7.5, 8.75, 10.0], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: True},
            num_stages=5,
            chunk_size=1,
            include_intra_stage=False,
            include_boundary=False,
        )
        np.testing.assert_array_equal(result.stage, np.asarray([0, 0, 0, 1, 2, 3], dtype=np.int64))

    def test_failed_episode_only_mines_before_first_stage_descent(self):
        values = np.asarray(
            [-0.95, -0.95, -0.75, -0.75, -0.55, -0.55, -0.75, -0.75, -0.95, -0.95],
            dtype=np.float32,
        )
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: False},
            num_stages=5,
            chunk_size=1,
            value_normalization="clip",
            stage_top_ratio=1.0,
            boundary_top_k=0,
            failure_max_stage=2,
            include_intra_stage=True,
            include_boundary=False,
        )
        selected_starts = set(int(v) for v in np.flatnonzero(result.selection_role == SELECTION_INTRA_STAGE_TOP))
        self.assertEqual(selected_starts, {0, 2, 4})
        self.assertEqual(result.report["failure_episodes"], 1)

    def test_failed_episode_boundary_selection_stops_before_descent(self):
        values = np.asarray([-0.95, -0.75, -0.55, -0.55, -0.75, -0.95], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            episode_success={0: False},
            num_stages=5,
            chunk_size=2,
            value_normalization="clip",
            stage_top_ratio=0.0,
            boundary_top_k=1,
            boundary_mode="unique_stage_boundary",
            failure_max_stage=2,
            include_intra_stage=False,
            include_boundary=True,
        )
        selected_starts = set(int(v) for v in np.flatnonzero(result.selection_role == SELECTION_BOUNDARY_TRANSITION))
        self.assertEqual(selected_starts, {0, 1})
        self.assertEqual(result.report["boundary_count"], 2)
        self.assertEqual(result.report["boundary_selected"], 2)

    def test_boundary_top_one_selects_best_transition_chunk(self):
        values = np.asarray([-0.7, -0.61, -0.59, -0.3], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            num_stages=5,
            chunk_size=2,
            value_normalization="clip",
            stage_top_ratio=0.0,
            boundary_top_k=1,
            include_intra_stage=False,
            include_boundary=True,
        )
        self.assertEqual(int(np.sum(result.selection_role == SELECTION_BOUNDARY_TRANSITION)), 1)
        self.assertEqual(int(np.sum(result.indicator)), 1)

    def test_unique_stage_boundaries_ignore_threshold_jitter(self):
        stages = np.asarray([3, 3, 4, 4, 3, 3, 4, 4], dtype=np.int64)
        np.testing.assert_array_equal(find_forward_boundaries(stages), np.asarray([2, 6], dtype=np.int64))
        np.testing.assert_array_equal(find_unique_stage_boundaries(stages), np.asarray([2], dtype=np.int64))

    def test_unique_boundary_mode_selects_once_for_repeated_crossing(self):
        values = np.asarray([-0.35, -0.35, -0.15, -0.15, -0.35, -0.35, -0.15, -0.15], dtype=np.float32)
        result = mine_stage_chunks(
            values=values,
            episode_indices=np.zeros(values.shape[0], dtype=np.int64),
            frame_indices=np.arange(values.shape[0], dtype=np.int64),
            task_indices=np.zeros(values.shape[0], dtype=np.int64),
            l_max_by_task={0: 100},
            num_stages=5,
            chunk_size=2,
            value_normalization="clip",
            stage_top_ratio=0.0,
            boundary_top_k=1,
            boundary_mode="unique_stage_boundary",
            include_intra_stage=False,
            include_boundary=True,
        )
        self.assertEqual(result.report["boundary_count"], 1)
        self.assertEqual(int(np.sum(result.selection_role == SELECTION_BOUNDARY_TRANSITION)), 1)


if __name__ == "__main__":
    unittest.main()

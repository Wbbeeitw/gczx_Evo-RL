#!/usr/bin/env python

import unittest

import numpy as np

from lerobot.rl.stage_chunk_mining.advantage import (
    compute_chunk_advantage,
    compute_chunk_advantages_batch,
)


class TestStageChunkAdvantage(unittest.TestCase):
    def test_lam_one_matches_endpoint_formula(self):
        values = np.asarray([-0.8, -0.72, -0.67, -0.6, -0.5, -0.3], dtype=np.float64)
        chunk_size = values.shape[0] - 1
        l_max = 200.0

        advantage = compute_chunk_advantage(values, chunk_size=chunk_size, l_max=l_max, lam=1.0)
        expected = float(values[-1] - values[0] - chunk_size / l_max)

        self.assertAlmostEqual(advantage, expected, delta=1e-9)

    def test_lam_zero_matches_single_step_td(self):
        values = np.asarray([-0.8, -0.72, -0.67, -0.6], dtype=np.float64)
        chunk_size = values.shape[0] - 1
        l_max = 200.0

        advantage = compute_chunk_advantage(values, chunk_size=chunk_size, l_max=l_max, lam=0.0)
        expected = float(values[1] - values[0] - 1.0 / l_max)

        self.assertAlmostEqual(advantage, expected, delta=1e-9)

    def test_linear_values_lam_point_nine_five_is_stable(self):
        values = np.linspace(-0.8, -0.3, 51, dtype=np.float64)
        advantage = compute_chunk_advantage(values, chunk_size=50, l_max=200.0, lam=0.95)

        self.assertGreater(advantage, 0.0)
        self.assertLess(advantage, 0.25)
        self.assertTrue(np.isfinite(advantage))

    def test_batch_matches_single_chunk_results(self):
        values = np.asarray([-0.9, -0.86, -0.83, -0.79, -0.74, -0.7, -0.66], dtype=np.float64)
        chunk_size = 3
        l_max = 120.0
        lam = 0.95

        batch_advantages = compute_chunk_advantages_batch(
            values=values,
            chunk_size=chunk_size,
            l_max=l_max,
            lam=lam,
        )
        single_advantages = np.asarray(
            [
                compute_chunk_advantage(values[i : i + chunk_size + 1], chunk_size, l_max, lam=lam)
                for i in range(values.shape[0] - chunk_size)
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(batch_advantages, single_advantages, rtol=0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

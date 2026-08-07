from __future__ import annotations

import unittest

import numpy as np

from core.selection_stats import (
    RUIN_FLOOR,
    expected_max_sharpe,
    pbo_cscv,
    periodic_returns,
    probabilistic_sharpe_ratio,
)


class SelectionStatisticsTests(unittest.TestCase):
    def test_expected_max_matches_documented_reference(self) -> None:
        self.assertAlmostEqual(expected_max_sharpe(200, 1.0), 2.765523904633547)

    def test_psr_equals_half_at_its_benchmark(self) -> None:
        self.assertAlmostEqual(
            probabilistic_sharpe_ratio(
                observed_sr=0.2,
                n_obs=100,
                skewness=0.0,
                kurt=3.0,
                benchmark_sr=0.2,
            ),
            0.5,
        )

    def test_pbo_detects_persistent_edge_better_than_noise(self) -> None:
        returns = np.random.default_rng(7).normal(0.0, 1.0, size=(600, 100))
        noise_pbo = pbo_cscv(returns, n_splits=10)["pbo"]
        returns[:, 0] += 0.5
        edge_pbo = pbo_cscv(returns, n_splits=10)["pbo"]

        self.assertLess(edge_pbo, noise_pbo)
        self.assertEqual(edge_pbo, 0.0)

    def test_periodic_returns_stay_flat_after_ruin(self) -> None:
        equity = np.array([1.0, 0.5, RUIN_FLOOR / 2.0, RUIN_FLOOR / 4.0])
        returns = periodic_returns(equity, bars_per_period=1)

        self.assertEqual(returns[-1], 0.0)

    def test_pbo_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D"):
            pbo_cscv(np.ones(10))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            pbo_cscv(np.ones((30, 1)))
        with self.assertRaisesRegex(ValueError, "even"):
            pbo_cscv(np.ones((30, 2)), n_splits=5)


if __name__ == "__main__":
    unittest.main()

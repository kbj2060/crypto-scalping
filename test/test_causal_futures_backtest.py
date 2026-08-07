from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.causal_futures_backtest import (
    fit_tail_thresholds,
    purged_decision_mask,
    simulate_single_position,
)


class CausalThresholdTests(unittest.TestCase):
    def test_thresholds_depend_only_on_calibration_scores(self) -> None:
        calibration = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        thresholds = fit_tail_thresholds(calibration, upper_quantile=0.8, lower_quantile=0.2)

        self.assertAlmostEqual(thresholds.upper, 0.42)
        self.assertAlmostEqual(thresholds.lower, 0.18)

        changed_test_scores = np.array([-100.0, 100.0])
        self.assertEqual(
            thresholds,
            fit_tail_thresholds(calibration, upper_quantile=0.8, lower_quantile=0.2),
        )
        self.assertFalse(np.isclose(thresholds.upper, np.quantile(changed_test_scores, 0.8)))


class PurgedSplitTests(unittest.TestCase):
    def test_target_must_finish_before_split_end(self) -> None:
        timestamps = pd.date_range("2025-08-31 23:40", periods=10, freq="5min")
        mask = purged_decision_mask(
            timestamps,
            start=pd.Timestamp("2025-08-31 23:40"),
            end=pd.Timestamp("2025-09-01 00:00"),
            horizon_bars=2,
        )

        self.assertEqual(mask.tolist(), [True, True, False, False, False, False, False, False, False, False])


class SinglePositionBacktestTests(unittest.TestCase):
    def _market(self) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        timestamps = pd.date_range("2026-04-01", periods=8, freq="5min")
        open_px = np.full(8, 100.0)
        high = np.array([100.0, 100.5, 102.0, 100.5, 100.5, 100.5, 100.5, 100.5])
        low = np.full(8, 99.5)
        close = np.full(8, 100.0)
        return timestamps, open_px, high, low, close

    def test_notional_applies_to_both_price_move_and_cost(self) -> None:
        timestamps, open_px, high, low, close = self._market()
        result = simulate_single_position(
            timestamps=timestamps,
            open_px=open_px,
            high=high,
            low=low,
            close=close,
            decision_indices=np.array([0]),
            scores=np.array([0.9]),
            tp_moves=np.array([0.02]),
            sl_moves=np.array([0.01]),
            upper_threshold=0.8,
            lower_threshold=0.2,
            horizon_bars=3,
            margin_fraction=0.30,
            leverage=3.0,
            roundtrip_cost_rate=0.0014,
        )

        expected = 0.02 * 0.90 - 0.0014 * 0.90
        self.assertEqual(len(result.ledger), 1)
        self.assertAlmostEqual(result.ledger.iloc[0]["trade_return"], expected)
        self.assertAlmostEqual(result.equity[-1], 1.0 + expected)

    def test_second_signal_is_ignored_while_position_is_open(self) -> None:
        timestamps, open_px, high, low, close = self._market()
        result = simulate_single_position(
            timestamps=timestamps,
            open_px=open_px,
            high=high,
            low=low,
            close=close,
            decision_indices=np.array([0, 1]),
            scores=np.array([0.9, 0.9]),
            tp_moves=np.array([0.50, 0.50]),
            sl_moves=np.array([0.50, 0.50]),
            upper_threshold=0.8,
            lower_threshold=0.2,
            horizon_bars=3,
            margin_fraction=0.30,
            leverage=3.0,
            roundtrip_cost_rate=0.0014,
        )

        self.assertEqual(len(result.ledger), 1)
        self.assertEqual(result.skipped_while_open, 1)


if __name__ == "__main__":
    unittest.main()

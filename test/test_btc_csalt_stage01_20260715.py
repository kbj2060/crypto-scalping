from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import test_btc_csalt_stage01_20260715 as csalt
import test_btc_csalt_causal_baseline_20260715 as baseline


class CsaltStage01Test(unittest.TestCase):
    def test_dollar_events_reset_to_zero_and_discard_warmup(self) -> None:
        activity = np.array([6.0, 6.0, 4.0, 6.0, 10.0, 1.0])
        events = csalt.build_dollar_events(activity, 0, len(activity) - 1, threshold=10.0)
        self.assertEqual(events.tolist(), [3, 4])

    def test_lifecycle_reward_uses_fill_fees_and_funding(self) -> None:
        funding = csalt.FundingTape(
            timestamp_ns=np.array([pd.Timestamp("2025-01-01 08:00").value], dtype=np.int64),
            rate_x_price_cumsum=np.array([0.0, 0.0001 * 105.0]),
        )
        reward = csalt.lifecycle_log_return(
            1,
            100.0,
            110.0,
            pd.Timestamp("2025-01-01 00:05").value,
            pd.Timestamp("2025-01-01 12:00").value,
            funding,
        )
        entry_fill = 100.0 * (1.0 + csalt.SLIPPAGE_RATE)
        exit_fill = 110.0 * (1.0 - csalt.SLIPPAGE_RATE)
        ratio = exit_fill / entry_fill
        account_return = (
            (ratio - 1.0) * csalt.NOTIONAL
            - csalt.FEE_RATE * csalt.NOTIONAL
            - csalt.FEE_RATE * csalt.NOTIONAL * ratio
            - csalt.NOTIONAL * (0.0001 * 105.0) / entry_fill
        )
        self.assertAlmostEqual(reward, np.log1p(account_return), places=12)

    def test_stop_is_detected_before_trailing(self) -> None:
        result = csalt.first_risk_trigger(
            side=1,
            entry_index=0,
            scan_end=1,
            entry_fill=100.0,
            entry_atr=1.0,
            high=np.array([102.0, 103.0]),
            low=np.array([97.0, 101.0]),
            close=np.array([101.0, 102.0]),
        )
        self.assertEqual(result, (0, "stop"))

    def test_smdp_continuation_changes_the_preferred_action(self) -> None:
        events = np.array([0, 10, 20], dtype=np.int64)
        rewards = np.full((3, len(csalt.ACTIONS)), np.nan)
        rewards[:, 0] = 0.0
        rewards[0, 1] = 0.05
        rewards[0, 2] = 0.04
        rewards[1, 1] = 0.20
        rewards[2, 1] = 0.01
        successors = np.full_like(rewards, -1, dtype=np.int64)
        successors[:-1, 0] = np.array([1, 2])
        successors[0, 1] = 2
        successors[0, 2] = 1
        successors[1, 1] = 2
        original = csalt.PLANNING_BARS
        try:
            csalt.PLANNING_BARS = 400
            q = csalt.finite_smdp_q(events, rewards, successors)
        finally:
            csalt.PLANNING_BARS = original
        self.assertGreater(q[0, 2], q[0, 1])

    def test_causal_features_do_not_change_when_future_prices_change(self) -> None:
        timestamp = pd.date_range("2025-01-01", periods=400, freq="5min")
        close = np.linspace(100.0, 120.0, len(timestamp))
        frame = pd.DataFrame(
            {
                "timestamp": timestamp,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.linspace(10.0, 20.0, len(timestamp)),
                "atr": np.ones(len(timestamp)),
            }
        )
        original = baseline.build_causal_features(frame)
        changed = frame.copy()
        changed.loc[350:, "close"] *= 2.0
        modified = baseline.build_causal_features(changed)
        np.testing.assert_allclose(original.iloc[:350], modified.iloc[:350], equal_nan=True)


if __name__ == "__main__":
    unittest.main()

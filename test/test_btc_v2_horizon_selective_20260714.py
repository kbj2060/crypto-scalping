from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_btc_v2_horizon_selective_20260714.py"
SPEC = importlib.util.spec_from_file_location("btc_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
btc_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(btc_v2)


def _frame(close: list[float]) -> pd.DataFrame:
    values = np.asarray(close, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(values), freq="5min"),
            "open": values,
            "high": values * 1.001,
            "low": values * 0.999,
            "close": values,
        }
    )


class BtcV2Test(unittest.TestCase):
    def test_execution_labels_enter_on_next_bar_and_time_exit(self) -> None:
        frame = _frame([100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        tp = np.full(len(frame), 0.50)
        sl = np.full(len(frame), 0.50)
        result = btc_v2._execution_labels_numba(
            *(frame[column].to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close")),
            tp,
            sl,
            2,
            btc_v2.FEE_RATE,
            btc_v2.SLIP_RATE,
            btc_v2.MAKER_FEE_MULT,
        )
        long_return, _, long_reason, _, long_hold, _ = result
        self.assertEqual(long_reason[0], 3)
        self.assertEqual(long_hold[0], 2)
        self.assertGreater(long_return[0], 0.0)


    def test_stationary_feature_contract_excludes_price_and_ou_halflife(self) -> None:
        frame = _frame([100.0] * 300)
        for column in btc_v2.PATCH_COLUMNS:
            frame[column] = np.linspace(-1.0, 1.0, len(frame))
        frame["ou_halflife"] = 999.0
        frame["rsi"] = 50.0
        features, columns = btc_v2._feature_frame(frame)
        self.assertNotIn("close", columns)
        self.assertNotIn("ou_halflife", columns)
        self.assertIn("rsi", columns)
        self.assertIn("patch_log_return_mean_288", features.columns)


    def test_fresh_forward_replay_enforces_time_exit(self) -> None:
        frame = _frame([100.0] * 10)
        side = np.zeros(len(frame), dtype=np.int8)
        side[0] = 1
        tp = np.full(len(frame), 0.50)
        sl = np.full(len(frame), 0.50)
        metrics, ledger, _ = btc_v2._fresh_forward_replay(frame, side, tp, sl, max_hold_bars=2)
        self.assertEqual(metrics["trades"], 1)
        self.assertEqual(ledger.iloc[0]["reason"], "time_exit")
        self.assertEqual(int(ledger.iloc[0]["entry_fill_i"]), 1)
        self.assertEqual(int(ledger.iloc[0]["exit_signal_i"]), 3)

    def test_causal_direction_uses_only_trailing_prices(self) -> None:
        close = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        before = btc_v2._causal_direction(close, horizon_bars=2)
        close[-1] = 1.0
        after = btc_v2._causal_direction(close, horizon_bars=2)
        np.testing.assert_array_equal(before[:-1], after[:-1])
        self.assertEqual(before[2], 1)


if __name__ == "__main__":
    unittest.main()

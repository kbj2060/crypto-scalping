from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_eval_btc_v2_regime_trendscan_20260714.py"
SPEC = importlib.util.spec_from_file_location("btc_v2_regime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
btc_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(btc_v2)


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_action": [1, 1, 2, 2],
            "parent_quality": [0.60, 0.60, 0.70, 0.70],
            f"{btc_v2.REGIME_PREFIX}bull_prob": [0.60, 0.60, 0.10, 0.10],
            f"{btc_v2.REGIME_PREFIX}bear_prob": [0.10, 0.10, 0.70, 0.70],
            "is_new_parent_signal": [True, False, True, False],
        }
    )


class BtcV2RegimeTrendScanTest(unittest.TestCase):
    def test_candidate_only_fires_on_new_parent_signal(self) -> None:
        side = btc_v2._candidate_side(_candidate_frame(), quality_threshold=0.55, regime_threshold=0.50)
        np.testing.assert_array_equal(side, np.asarray([1, 0, -1, 0], dtype=np.int8))

    def test_regime_must_agree_with_parent_side(self) -> None:
        frame = _candidate_frame()
        frame.loc[0, f"{btc_v2.REGIME_PREFIX}bull_prob"] = 0.49
        frame.loc[2, f"{btc_v2.REGIME_PREFIX}bear_prob"] = 0.49
        side = btc_v2._candidate_side(frame, quality_threshold=0.55, regime_threshold=0.50)
        np.testing.assert_array_equal(side, np.zeros(4, dtype=np.int8))

    def test_gate_off_keeps_new_parent_events(self) -> None:
        frame = _candidate_frame()
        frame[f"{btc_v2.REGIME_PREFIX}bull_prob"] = 0.0
        frame[f"{btc_v2.REGIME_PREFIX}bear_prob"] = 0.0
        side = btc_v2._candidate_side(frame, quality_threshold=0.55, regime_threshold=None)
        np.testing.assert_array_equal(side, np.asarray([1, 0, -1, 0], dtype=np.int8))

    def test_risk_contract_uses_margin_times_leverage_once(self) -> None:
        self.assertAlmostEqual(btc_v2.NOTIONAL, btc_v2.MARGIN_FRACTION * btc_v2.LEVERAGE)
        self.assertAlmostEqual(btc_v2.STOP_ATR_PRICE, btc_v2.LEGACY_STOP_ATR_ACCOUNT / btc_v2.NOTIONAL)


if __name__ == "__main__":
    unittest.main()

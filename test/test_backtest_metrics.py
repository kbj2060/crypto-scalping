"""P1-1 of docs/pipeline_integrity_and_research_redesign_20260730.md: verifies
core/backtest_metrics.bar_level_performance reproduces the 6 recorded ETH/SOL/BTC VAL/OOS cells
from tmp/research_20260728/three_asset_bar_level_mdd/summary.json using that run's own saved
equity curves and ledgers -- confirms the promoted function is a faithful port, not a
reimplementation that happens to look similar.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest_metrics import bar_level_performance

DATA_DIR = ROOT / "tmp/research_20260728/three_asset_bar_level_mdd"


class BarLevelPerformanceReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATA_DIR.exists():
            raise unittest.SkipTest(f"{DATA_DIR} not present -- cannot verify against recorded cells")
        cls.recorded = json.loads((DATA_DIR / "summary.json").read_text())["results"]

    def _check(self, asset: str, split: str) -> None:
        equity = np.load(DATA_DIR / f"equity_{asset}_{split}.npy")
        ledger = pd.read_csv(DATA_DIR / f"ledger_{asset}_{split}.csv")
        recorded = self.recorded[asset][split]

        result = bar_level_performance(equity, ledger)

        self.assertAlmostEqual(result["pnl"], recorded["pnl"], places=1)
        self.assertAlmostEqual(result["mdd_bar_level"], recorded["bar_level_mdd"], places=1)
        self.assertAlmostEqual(result["mdd_trade_ledger"], recorded["trade_ledger_mdd"], places=1)
        self.assertEqual(result["trades"], recorded["trades"])
        self.assertAlmostEqual(result["wr"], recorded["wr"], places=2)

    def test_eth_val(self) -> None:
        self._check("eth", "VAL")

    def test_eth_oos(self) -> None:
        self._check("eth", "OOS")

    def test_sol_val(self) -> None:
        self._check("sol", "VAL")

    def test_sol_oos(self) -> None:
        self._check("sol", "OOS")

    def test_btc_val(self) -> None:
        self._check("btc", "VAL")

    def test_btc_oos(self) -> None:
        self._check("btc", "OOS")


if __name__ == "__main__":
    unittest.main()

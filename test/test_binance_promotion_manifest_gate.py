from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_bot_modules.binance_execution import BinanceFuturesExecutionAdapter
from trading_bot_modules.binance_runtime_config import BinanceExecutionConfig


class _Fetcher:
    account_exchange = object()
    account_enabled = True
    account_testnet = True
    account_symbol = "ETH/USDT:USDT"


class BinancePromotionManifestGateTests(unittest.TestCase):
    def _manifest(self, *, eligible: bool) -> str:
        path = Path(tempfile.mkdtemp()) / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "current_live_manifest_v1",
                    "promotion_eligible": eligible,
                    "promotion_blockers": [] if eligible else ["test_blocker"],
                    "artifact_integrity": {"promotion_pass": eligible},
                    "fresh_forward": {
                        split: {
                            "fresh_forward_bar_by_bar": True,
                            "trade_ledgers_used_as_input": False,
                            "saved_parent_exit_timestamps_used": False,
                            "future_rows_used_for_entry": False,
                        }
                        for split in ("validation", "oos")
                    },
                    "selection_statistics": {
                        "gate_pass": eligible,
                        "deflated_sharpe_ratio": 0.97,
                        "minimum_deflated_sharpe_ratio": 0.95,
                        "probability_backtest_overfit": 0.10,
                        "maximum_probability_backtest_overfit": 0.20,
                    },
                }
            ),
            encoding="utf-8",
        )
        return str(path)

    def _adapter(self, manifest_path: str) -> BinanceFuturesExecutionAdapter:
        env = {
            "BINANCE_EXECUTION_ENABLED": "true",
            "BINANCE_EXECUTION_DRY_RUN": "true",
            "FINAL_GOVERNOR_OMEGA4_6_1_ENABLE": "true",
            "FINAL_GOVERNOR_OMEGA4_6_1_MANIFEST_PATH": manifest_path,
        }
        with patch.dict(os.environ, env, clear=False):
            return BinanceFuturesExecutionAdapter(
                _Fetcher(), config=BinanceExecutionConfig.from_env()
            )

    def test_ineligible_manifest_disables_execution(self) -> None:
        adapter = self._adapter(self._manifest(eligible=False))
        status = adapter.status()

        self.assertFalse(adapter.enabled)
        self.assertTrue(status["requested_enabled"])
        self.assertEqual(status["health"], "blocked")
        self.assertEqual(status["disabled_reason"], "promotion_manifest_failed")
        self.assertTrue(status["last_error"])
        self.assertTrue(status["last_error_at"])

    def test_eligible_manifest_allows_execution(self) -> None:
        status = self._adapter(self._manifest(eligible=True)).status()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["health"], "ready")
        self.assertEqual(status["disabled_reason"], "")

    def test_missing_manifest_disables_execution(self) -> None:
        self.assertFalse(self._adapter("/missing/promotion-manifest.json").enabled)

    def test_eligible_flag_without_evidence_disables_execution(self) -> None:
        path = Path(tempfile.mkdtemp()) / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "current_live_manifest_v1",
                    "promotion_eligible": True,
                    "promotion_blockers": [],
                }
            ),
            encoding="utf-8",
        )
        self.assertFalse(self._adapter(str(path)).enabled)

    def test_failed_statistical_threshold_disables_execution(self) -> None:
        path = Path(self._manifest(eligible=True))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selection_statistics"]["probability_backtest_overfit"] = 0.50
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(self._adapter(str(path)).enabled)

    def test_noncausal_fresh_forward_evidence_disables_execution(self) -> None:
        path = Path(self._manifest(eligible=True))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fresh_forward"]["oos"]["future_rows_used_for_entry"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(self._adapter(str(path)).enabled)


if __name__ == "__main__":
    unittest.main()

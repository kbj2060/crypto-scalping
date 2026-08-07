from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_omega461_promotion_manifest_20260729 import build_promotion_manifest
from trading_bot_modules.omega4_6_1_runtime_contract import (
    require_execution_promotion_manifest,
)


class Omega461PromotionManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())

    def _write(self, name: str, payload: dict[str, object]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _fresh() -> dict[str, object]:
        return {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
        }

    @staticmethod
    def _selection() -> dict[str, object]:
        return {
            "gate_pass": True,
            "deflated_sharpe_ratio": 0.97,
            "minimum_deflated_sharpe_ratio": 0.95,
            "probability_backtest_overfit": 0.10,
            "maximum_probability_backtest_overfit": 0.20,
        }

    def _build(self) -> dict[str, object]:
        return build_promotion_manifest(
            artifact_report_path=self._write("artifact.json", {"promotion_pass": True}),
            validation_report_path=self._write("validation.json", self._fresh()),
            oos_report_path=self._write("oos.json", self._fresh()),
            selection_report_path=self._write("selection.json", self._selection()),
            source_commit="abc123",
        )

    def test_builds_manifest_accepted_by_execution_gate(self) -> None:
        manifest = self._build()
        output = self._write("manifest.json", manifest)

        accepted = require_execution_promotion_manifest(output)

        self.assertTrue(accepted["promotion_eligible"])
        self.assertEqual(len(accepted["artifact_integrity"]["report_sha256"]), 64)

    def test_rejects_noncausal_oos_report_before_build(self) -> None:
        oos = self._fresh()
        oos["trade_ledgers_used_as_input"] = True
        with self.assertRaisesRegex(ValueError, "oos fresh-forward"):
            build_promotion_manifest(
                artifact_report_path=self._write("artifact.json", {"promotion_pass": True}),
                validation_report_path=self._write("validation.json", self._fresh()),
                oos_report_path=self._write("oos.json", oos),
                selection_report_path=self._write("selection.json", self._selection()),
                source_commit="abc123",
            )

    def test_rejects_failed_artifact_report_before_build(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact integrity"):
            build_promotion_manifest(
                artifact_report_path=self._write("artifact.json", {"promotion_pass": False}),
                validation_report_path=self._write("validation.json", self._fresh()),
                oos_report_path=self._write("oos.json", self._fresh()),
                selection_report_path=self._write("selection.json", self._selection()),
                source_commit="abc123",
            )


if __name__ == "__main__":
    unittest.main()

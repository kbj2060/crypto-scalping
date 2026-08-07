from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardExecutionAlertTests(unittest.TestCase):
    def test_dashboard_has_persistent_execution_alert_surface(self) -> None:
        html = (ROOT / "dashboard/live/index.html").read_text(encoding="utf-8")

        self.assertIn('id="executionAlertBanner"', html)
        self.assertIn('aria-live="assertive"', html)
        self.assertIn('id="executionAlertReason"', html)
        self.assertIn('id="executionAlertTime"', html)

    def test_trading_bot_publishes_alert_to_dashboard_and_telegram(self) -> None:
        source = (ROOT / "trading_bot.py").read_text(encoding="utf-8")

        self.assertIn('"execution_alert": dict(_execution_alert)', source)
        self.assertIn("telegram-execution-alert", source)
        self.assertIn("execution alert dashboard write failed", source)
        self.assertIn("[트레이딩봇 치명적 오류]", source)

    def test_dashboard_renders_execution_alert_contract(self) -> None:
        javascript = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")
        stylesheet = (ROOT / "dashboard/live/styles.css").read_text(encoding="utf-8")

        self.assertIn("function renderExecutionAlert", javascript)
        self.assertIn("state?.execution_alert", javascript)
        self.assertIn('renderExecutionAlert(state, compactState);', javascript)
        self.assertIn(".execution-alert-banner", stylesheet)
        self.assertIn(".execution-alert-banner.hidden", stylesheet)


if __name__ == "__main__":
    unittest.main()

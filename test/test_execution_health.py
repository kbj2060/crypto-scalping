from __future__ import annotations

import unittest

from trading_bot_modules.execution_health import (
    ExecutionAlertDeduper,
    build_execution_alert,
)


class ExecutionHealthTests(unittest.TestCase):
    def test_manifest_failure_is_error_with_original_time(self) -> None:
        alert = build_execution_alert(
            {
                "requested_enabled": True,
                "enabled": False,
                "disabled_reason": "promotion_manifest_failed",
                "last_error": "manifest is not eligible",
                "last_error_at": "2026-07-30T01:02:03Z",
            }
        )

        self.assertTrue(alert["active"])
        self.assertEqual(alert["severity"], "error")
        self.assertEqual(alert["reason"], "manifest is not eligible")
        self.assertEqual(alert["occurred_at"], "2026-07-30T01:02:03Z")

    def test_configured_off_is_visible_but_not_an_error(self) -> None:
        alert = build_execution_alert(
            {
                "requested_enabled": False,
                "enabled": False,
                "disabled_reason": "configured_off",
                "disabled_at": "2026-07-30T01:00:00Z",
            }
        )

        self.assertEqual(alert["severity"], "disabled")
        self.assertEqual(alert["title"], "실제 주문 실행 비활성")

    def test_model_error_reason_is_not_hidden_by_execution_status(self) -> None:
        alert = build_execution_alert(
            {"requested_enabled": True, "enabled": True, "status": "no_order_needed"},
            decision_reason="final_governor_error",
            observed_at="2026-07-30T01:03:00+09:00",
        )

        self.assertEqual(alert["severity"], "error")
        self.assertEqual(alert["reason"], "final_governor_error")
        self.assertEqual(alert["occurred_at"], "2026-07-30T01:03:00+09:00")

    def test_ready_execution_has_no_alert(self) -> None:
        alert = build_execution_alert({"requested_enabled": True, "enabled": True})
        self.assertFalse(alert["active"])

    def test_repeated_alert_is_suppressed_until_recovery(self) -> None:
        deduper = ExecutionAlertDeduper()
        alert = build_execution_alert(
            {"requested_enabled": True, "enabled": False, "disabled_reason": "account_not_ready"},
            observed_at="2026-07-30T01:00:00Z",
        )

        self.assertTrue(deduper.should_notify(alert))
        repeated = dict(alert, occurred_at="2026-07-30T02:00:00Z")
        self.assertFalse(deduper.should_notify(repeated))
        self.assertEqual(repeated["occurred_at"], alert["occurred_at"])
        self.assertFalse(deduper.should_notify(build_execution_alert({"enabled": True})))
        self.assertTrue(deduper.should_notify(alert))


if __name__ == "__main__":
    unittest.main()

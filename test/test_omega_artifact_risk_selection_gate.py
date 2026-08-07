from __future__ import annotations

import unittest

from scripts.audit_omega_artifact_integrity_20260630 import risk_selection_contract_checks


def _report() -> dict:
    return {
        "risk_model": {"selection_scope": "validation_only"},
        "selected": {
            "constraint_pass": True,
            "fallback_used": False,
            "full_replay_selection_applied": True,
            "constraints": {
                "validation_trade_floor": 10,
                "validation_mdd_floor": -8.0,
            },
            "selected_full_replay": {
                "validation": {"trades": 12, "mdd": -7.0}
            },
        },
    }


class OmegaArtifactRiskSelectionGateTests(unittest.TestCase):
    def test_valid_machine_readable_contract_passes(self) -> None:
        self.assertTrue(all(check.status == "pass" for check in risk_selection_contract_checks(_report())))

    def test_oos_selection_scope_fails(self) -> None:
        report = _report()
        report["risk_model"]["selection_scope"] = "validation_oos_guard"
        self.assertTrue(any(check.status == "fail" for check in risk_selection_contract_checks(report)))

    def test_missing_machine_readable_contract_fails(self) -> None:
        report = _report()
        report["selected"].pop("constraint_pass")
        self.assertTrue(any(check.status == "fail" for check in risk_selection_contract_checks(report)))

    def test_full_replay_mdd_violation_fails(self) -> None:
        report = _report()
        report["selected"]["selected_full_replay"]["validation"]["mdd"] = -9.0
        self.assertTrue(any(check.status == "fail" for check in risk_selection_contract_checks(report)))


if __name__ == "__main__":
    unittest.main()

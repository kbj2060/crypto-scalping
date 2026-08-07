from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_btc_v2_artifact_20260716 import audit


def test_audit_requires_both_historical_and_future_gates(tmp_path: Path) -> None:
    report = {
        "feature_contract": {
            "btc_native_stationary_only": True,
            "cross_asset_features": False,
            "legacy_aliases": False,
        },
        "execution_contract": {
            "notional": 0.30,
            "same_bar_policy": "stop_first_conservative",
            "next_bar_entry": True,
        },
        "fresh_forward_contract": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
        },
        "historical_gate_passed": True,
        "future_gate": {"passed": False},
        "promotion_eligible": False,
    }
    content = json.dumps(report).encode()
    (tmp_path / "report.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / "manifest.sha256.json").write_text(
        json.dumps({"report.json": digest}), encoding="utf-8"
    )
    result = audit(tmp_path)
    assert result["manifest_pass"]
    assert result["contract_pass"]
    assert not result["future_gate_passed"]
    assert not result["promotion_pass"]

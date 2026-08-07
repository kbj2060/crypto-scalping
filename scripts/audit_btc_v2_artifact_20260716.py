#!/usr/bin/env python3
"""Fail-fast integrity and promotion audit for a BTC v2 research artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(artifact_dir: Path) -> dict[str, Any]:
    report_path = artifact_dir / "report.json"
    manifest_path = artifact_dir / "manifest.sha256.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("report.json and manifest.sha256.json are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in manifest.items():
        path = artifact_dir / name
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    fresh = report.get("fresh_forward_contract", {})
    contract_checks = {
        "fresh_forward_bar_by_bar": fresh.get("fresh_forward_bar_by_bar") is True,
        "trade_ledgers_not_input": fresh.get("trade_ledgers_used_as_input") is False,
        "saved_parent_exits_not_input": fresh.get("saved_parent_exit_timestamps_used") is False,
        "future_rows_not_used": fresh.get("future_rows_used_for_entry") is False,
        "btc_native_features": report.get("feature_contract", {}).get("btc_native_stationary_only") is True,
        "no_cross_asset_features": report.get("feature_contract", {}).get("cross_asset_features") is False,
        "no_legacy_aliases": report.get("feature_contract", {}).get("legacy_aliases") is False,
        "notional_identity": abs(
            report.get("execution_contract", {}).get("notional", -1.0)
            - 0.15 * 2.0
        ) < 1e-12,
        "stop_first": report.get("execution_contract", {}).get("same_bar_policy")
        == "stop_first_conservative",
        "next_bar_entry": report.get("execution_contract", {}).get("next_bar_entry") is True,
    }
    result = {
        "artifact_dir": str(artifact_dir.resolve()),
        "manifest_pass": not mismatches,
        "manifest_mismatches": mismatches,
        "contract_checks": contract_checks,
        "contract_pass": all(contract_checks.values()),
        "historical_gate_passed": report.get("historical_gate_passed") is True,
        "future_gate_passed": report.get("future_gate", {}).get("passed") is True,
    }
    result["promotion_pass"] = bool(
        result["manifest_pass"]
        and result["contract_pass"]
        and result["historical_gate_passed"]
        and result["future_gate_passed"]
        and report.get("promotion_eligible") is True
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.artifact_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["promotion_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

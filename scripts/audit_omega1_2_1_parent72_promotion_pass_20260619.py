#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_parent72_price_move_contract_20260619"
SOURCE_MODEL_ID = "omega1_2_1_aggressive_compensated_scale200_cap090"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / f"{MODEL_ID}_promotion_readiness"
MANIFEST = ROOT / "data/ensemble/supervised" / MODEL_ID / "candidate_manifest.json"
REDTEAM = ROOT / "tmp/causal_regen_20260516" / MODEL_ID / "redteam_report.json"
LIVE_ADAPTER = ROOT / "trading_bot_modules/omega1_2_1_live.py"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
LIVE_SNAPSHOT = ROOT / "data/live/decision_feature_frame_snapshot.pkl.gz"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _py_module_available(module: str) -> bool:
    try:
        proc = subprocess.run(
            ["python3", "-c", f"import {module}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=3,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _contract_ledger_check() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for split in ("validation", "oos"):
        path = ROOT / "tmp/causal_regen_20260516" / MODEL_ID / f"{MODEL_ID}_{split}_contract_ledger.csv"
        rows = _read_csv(path)
        bad: list[str] = []
        for row in rows:
            margin = float(row["margin_fraction"])
            leverage = float(row["execution_leverage"])
            notional = float(row["notional_exposure"])
            tp_move = float(row["tp_price_move"])
            sl_move = float(row["sl_price_move"])
            if abs(margin * leverage - notional) > 1.0e-9:
                bad.append(str(row["trade_id"]))
            if abs(tp_move * notional - float(row["take_profit"])) > 1.0e-9:
                bad.append(str(row["trade_id"]))
            if abs(sl_move * notional - float(row["stop_loss"])) > 1.0e-9:
                bad.append(str(row["trade_id"]))
        checks[split] = {"rows": len(rows), "bad_contract_rows": bad, "path": str(path.relative_to(ROOT))}
    return checks


def _live_adapter_model_id() -> dict[str, Any]:
    text = LIVE_ADAPTER.read_text(errors="ignore")
    model_match = re.search(r'OMEGA121_MODEL_ID\s*=\s*"([^"]+)"', text)
    base_match = re.search(r"BASE_NOTIONAL\s*=\s*([0-9.]+)", text)
    return {
        "path": str(LIVE_ADAPTER.relative_to(ROOT)),
        "omega121_model_id": model_match.group(1) if model_match else None,
        "base_notional": float(base_match.group(1)) if base_match else None,
    }


def _fresh_prediction_inventory() -> dict[str, Any]:
    files = sorted(str(p.relative_to(ROOT)) for p in PARENT_DIR.glob("*predictions*true3head*.csv"))
    post_oos = [
        f
        for f in files
        if any(token in Path(f).name for token in ("202603", "2026_03", "202604", "2026_04", "202605", "2026_05", "202606", "2026_06", "fresh", "forward"))
    ]
    return {"prediction_files": files, "post_2026_02_prediction_files": post_oos}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST)
    redteam = _read_json(REDTEAM)
    ledger_contract = _contract_ledger_check()
    live_adapter = _live_adapter_model_id()
    prediction_inventory = _fresh_prediction_inventory()
    py_deps = {m: _py_module_available(m) for m in ("numpy", "pandas", "torch", "joblib", "sklearn")}

    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, evidence: list[str] | None = None) -> None:
        checks.append({"check": name, "status": status, "detail": detail, "evidence": evidence or []})

    add(
        "research_repackage_redteam",
        "PASS" if redteam.get("redteam_pass") else "FAIL",
        f"Research red-team verdict: {redteam.get('verdict')}.",
        [str(REDTEAM.relative_to(ROOT))],
    )
    bad_rows = sum(len(v["bad_contract_rows"]) for v in ledger_contract.values())
    add(
        "contract_native_ledger",
        "PASS" if bad_rows == 0 else "FAIL",
        f"Contract ledger rows checked; bad rows={bad_rows}.",
        [v["path"] for v in ledger_contract.values()],
    )
    add(
        "validation_only_selection",
        "PASS" if not manifest["selection"]["selected_uses_oos_for_decision"] else "FAIL",
        "Candidate manifest locks selected row by validation-only reconstruction.",
        [str(MANIFEST.relative_to(ROOT))],
    )
    add(
        "fresh_untouched_predictions",
        "FAIL" if not prediction_inventory["post_2026_02_prediction_files"] else "PASS",
        "No parent true3head prediction CSV after 2026-02 was found, so fresh untouched replay cannot be completed from current artifacts.",
        prediction_inventory["prediction_files"],
    )
    add(
        "parent_reinference_environment",
        "FAIL" if not py_deps.get("torch") else "PASS",
        f"python3 dependency availability: {py_deps}. Parent TabM fresh inference requires torch/joblib/numpy/pandas.",
        [],
    )
    add(
        "live_adapter_contract_match",
        "FAIL" if live_adapter["omega121_model_id"] != MODEL_ID else "PASS",
        f"Live adapter OMEGA121_MODEL_ID={live_adapter['omega121_model_id']}; target={MODEL_ID}.",
        [live_adapter["path"]],
    )
    add(
        "live_snapshot_available",
        "PASS" if LIVE_SNAPSHOT.exists() else "FAIL",
        "Live decision feature snapshot exists for future shadow/parity replay, but candidate adapter must be wired before it can be used for this model.",
        [str(LIVE_SNAPSHOT.relative_to(ROOT))],
    )

    blockers = [c for c in checks if c["status"] == "FAIL"]
    report = {
        "audit_id": f"{MODEL_ID}_promotion_readiness_20260619",
        "target_model_id": MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "verdict": "PROMOTION_PASS_BLOCKED" if blockers else "PROMOTION_PASS",
        "promotion_pass": not blockers,
        "redteam_pass": not blockers,
        "blockers": [f"{c['check']}: {c['detail']}" for c in blockers],
        "checks": checks,
        "ledger_contract": ledger_contract,
        "live_adapter": live_adapter,
        "prediction_inventory": prediction_inventory,
        "python3_dependencies": py_deps,
        "required_next_actions": [
            "Generate parent true3head predictions on a post-2026-02 untouched OHLCV feature frame, or collect forward shadow decisions from the candidate adapter.",
            "Wire a fail-fast live/shadow adapter for omega1_2_1_parent72_price_move_contract_20260619 with exact manifest model id and risk constants.",
            "Run contract-native replay from the adapter outputs, then rerun this promotion readiness audit.",
        ],
    }
    _write_json(OUT_DIR / "promotion_readiness_report.json", report)
    print(
        json.dumps(
            {
                "report": str((OUT_DIR / "promotion_readiness_report.json").relative_to(ROOT)),
                "verdict": report["verdict"],
                "promotion_pass": report["promotion_pass"],
                "blockers": len(blockers),
                "blocker_checks": [c["check"] for c in blockers],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

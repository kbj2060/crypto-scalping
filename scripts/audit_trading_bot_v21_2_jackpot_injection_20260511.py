#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
REPORT = ROOT / "data/ensemble/reports/hf_v13_jackpot_runner_v21_2_20260511_summary.json"
AUDIT = ROOT / "data/ensemble/reports/hf_v13_jackpot_runner_v21_2_20260511_audit.json"
BOT = ROOT / "trading_bot.py"
OUT = ROOT / "data/ensemble/reports/trading_bot_v21_2_jackpot_injection_20260511_redteam_audit.json"


FORBIDDEN_PATTERNS = (
    "regime_v2",
    "hdb",
    "hmm",
    "v22_sniper_ledger",
    "contaminated",
)


def main() -> int:
    blocking: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}
    for name, path in {"model": MODEL, "report": REPORT, "audit": AUDIT, "bot": BOT}.items():
        checks[f"{name}_exists"] = path.exists()
        if not path.exists():
            blocking.append(f"missing_{name}:{path}")
    if blocking:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "fail", "blocking": blocking, "checks": checks}, indent=2), encoding="utf-8")
        print(OUT)
        return 1

    payload = joblib.load(MODEL)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    source = BOT.read_text(encoding="utf-8")
    runner = dict(payload.get("cost_runner") or {})
    feature_cols = list(runner.get("feature_cols") or [])
    forbidden_cols = [c for c in feature_cols if any(p in c.lower() for p in FORBIDDEN_PATTERNS)]

    checks["audit_status"] = audit.get("status")
    checks["audit_verdict"] = audit.get("verdict")
    checks["audit_selection_uses_2026"] = audit.get("selection_uses_2026")
    checks["audit_train_eval_timestamp_overlap"] = audit.get("feature_audit", {}).get("train_eval_timestamp_overlap")
    checks["forbidden_feature_cols"] = forbidden_cols
    checks["clean_regime_feature_count"] = sum(1 for c in feature_cols if c.startswith("clean_regime_2024_unsup_v4_"))
    checks["model_id"] = payload.get("model_id")
    checks["selected_config"] = payload.get("selected_config")
    checks["base_model"] = payload.get("base_model")
    checks["report_model_matches"] = str(report.get("model")) == str(MODEL)
    checks["bot_default_enables_v21_2"] = "FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE" in source and re.search(
        r"FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE\s*=\s*_env_flag\([^\\n]+True\)",
        source,
        flags=re.S,
    ) is not None
    checks["bot_points_to_v21_2_model"] = str(MODEL.relative_to(ROOT)) in source
    checks["bot_points_to_v21_2_report"] = str(REPORT.relative_to(ROOT)) in source
    checks["bot_points_to_v21_2_audit"] = str(AUDIT.relative_to(ROOT)) in source
    checks["bot_parent_default_margin110"] = "hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl" in source
    checks["bot_5x_not_default"] = "hf_v13_jackpot_runner_5x_v21_3_20260511" not in source
    checks["bot_bypasses_runtime_risk_for_v21_2"] = "FINAL_GOVERNOR_V21_BYPASS_RUNTIME_RISK_GATES" in source and "FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE" in source
    checks["bot_has_jackpot_resize_path"] = "v21_2_jackpot_runner|add_on_resize" in source
    checks["bot_has_parent_exit_parity"] = "parent_tp_sl_max_hold" in source and "learned_take_profit" in source and "learned_max_hold" in source

    metrics = dict(report.get("metrics") or {})
    checks["oos_cost1_pnl"] = metrics.get("cost1", {}).get("pnl")
    checks["oos_cost2_pnl"] = metrics.get("cost2", {}).get("pnl")
    checks["oos_cost3_pnl"] = metrics.get("cost3", {}).get("pnl")

    if str(audit.get("status", "")).lower() != "pass":
        blocking.append("v21_2_audit_not_pass")
    if audit.get("selection_uses_2026") is not False:
        blocking.append("selection_uses_2026_not_false")
    if int(audit.get("feature_audit", {}).get("train_eval_timestamp_overlap", -1)) != 0:
        blocking.append("train_eval_timestamp_overlap_nonzero")
    if forbidden_cols:
        blocking.append("forbidden_feature_cols_present")
    for key in (
        "bot_default_enables_v21_2",
        "bot_points_to_v21_2_model",
        "bot_points_to_v21_2_report",
        "bot_points_to_v21_2_audit",
        "bot_parent_default_margin110",
        "bot_5x_not_default",
        "bot_bypasses_runtime_risk_for_v21_2",
        "bot_has_jackpot_resize_path",
        "bot_has_parent_exit_parity",
    ):
        if not checks.get(key):
            blocking.append(key)
    if float(metrics.get("cost3", {}).get("pnl", 0.0) or 0.0) <= 0.0:
        warnings.append("cost3_not_positive")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "pass" if not blocking else "fail",
        "verdict": "pass_for_shadow_injection" if not blocking else "block_injection",
        "blocking": blocking,
        "warnings": warnings,
        "checks": checks,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit native supervised portfolio ranker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_supervised_ranker_native_20260709"
REPORT_PATH = ART_DIR / "report.json"
OUT_JSON = ART_DIR / "redteam_audit.json"
OUT_MD = ROOT / "docs/audits/portfolio_supervised_ranker_native_redteam_20260709.md"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    for key, expected in (
        ("fresh_forward_bar_by_bar", True),
        ("trade_ledgers_used_as_input", False),
        ("saved_parent_exit_timestamps_used", False),
        ("future_rows_used_for_entry", False),
    ):
        if report.get(key) is not expected:
            issues.append({"severity": "P0", "check": key, "message": f"expected {expected}, got {report.get(key)}"})
    train = pd.read_csv(ART_DIR / "validation_candidate_training_set.csv")
    if len(train) < 50:
        issues.append({"severity": "P1", "check": "training_rows_thin", "message": f"training rows={len(train)}"})
    res = report["results"]
    if res["oos_extended"]["pnl"] <= 4.046534614137842 + 1e-9 and res["oos_extended"]["trades"] == 49:
        issues.append({"severity": "P2", "check": "oos_equivalent_to_rule_take_all", "message": "OOS result matches rule_take_all baseline"})
    if res["oos_extended"]["cash_decisions"] == 0:
        issues.append({"severity": "P2", "check": "oos_no_cash_filtering", "message": "ranker made no OOS cash decisions"})
    issues.append({"severity": "P1", "check": "promotion_grade", "message": "validation improvement did not transfer to OOS"})
    audit = {
        "audit_id": "portfolio_supervised_ranker_native_redteam_20260709",
        "promotion_pass": False,
        "p0_count": int(sum(i["severity"] == "P0" for i in issues)),
        "issues": issues,
        "training_rows": int(len(train)),
        "results": res,
    }
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    lines = [
        "# Portfolio Supervised Ranker Native Red-Team Audit - 2026-07-09",
        "",
        f"Promotion pass: `{audit['promotion_pass']}`",
        f"P0 count: `{audit['p0_count']}`",
        "",
        "## Findings",
        "",
    ]
    for issue in issues:
        lines.append(f"- `{issue['severity']}` `{issue['check']}`: {issue['message']}")
    lines.extend(["", "## Results", "", "| split | PnL | MDD | trades | decisions | cash |", "|---|---:|---:|---:|---:|---:|"])
    for split, m in res.items():
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['trades']} | {m.get('decisions', 0)} | {m.get('cash_decisions', 0)} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(OUT_JSON), "md": str(OUT_MD), "promotion_pass": audit["promotion_pass"], "p0_count": audit["p0_count"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

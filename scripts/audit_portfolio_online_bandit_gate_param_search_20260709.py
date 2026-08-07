#!/usr/bin/env python3
"""Audit validation-only param search for online bandit portfolio gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_online_bandit_gate_param_search_20260709"
REPORT_PATH = ART_DIR / "report.json"
OUT_JSON = ART_DIR / "redteam_audit.json"
OUT_MD = ROOT / "docs/audits/portfolio_online_bandit_gate_param_search_redteam_20260709.md"


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
        ("uses_only_past_closed_trades_for_learning", True),
    ):
        if report.get(key) is not expected:
            issues.append({"severity": "P0", "check": key, "message": f"expected {expected}, got {report.get(key)}"})
    res = report["results"]
    if res["oos_extended"]["skips"] == 0:
        issues.append({"severity": "P2", "check": "oos_no_policy_effect", "message": "selected config makes no OOS skips; equivalent to rule_take_all"})
    if res["validation"]["skips"] == 0:
        issues.append({"severity": "P2", "check": "validation_no_policy_effect", "message": "selected config makes no validation skips"})
    issues.append({"severity": "P1", "check": "promotion_grade", "message": "param search collapses to no-op rule baseline; not an RL improvement"})
    audit = {
        "audit_id": "portfolio_online_bandit_gate_param_search_redteam_20260709",
        "promotion_pass": False,
        "p0_count": int(sum(i["severity"] == "P0" for i in issues)),
        "issues": issues,
        "selected": report["selected"],
        "results": res,
    }
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    lines = [
        "# Portfolio Online Bandit Gate Param Search Red-Team Audit - 2026-07-09",
        "",
        f"Promotion pass: `{audit['promotion_pass']}`",
        f"P0 count: `{audit['p0_count']}`",
        "",
        "## Findings",
        "",
    ]
    for issue in issues:
        lines.append(f"- `{issue['severity']}` `{issue['check']}`: {issue['message']}")
    lines.extend(["", "## Results", "", "| split | PnL | MDD | trades | decisions | skips |", "|---|---:|---:|---:|---:|---:|"])
    for split, m in res.items():
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['trades']} | {m.get('decisions', 0)} | {m.get('skips', 0)} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(OUT_JSON), "md": str(OUT_MD), "promotion_pass": audit["promotion_pass"], "p0_count": audit["p0_count"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

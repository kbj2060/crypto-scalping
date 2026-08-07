#!/usr/bin/env python3
"""Audit TAKE-only causal native asset switch router."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_online_asset_switch_native_20260709"
REPORT_PATH = ART_DIR / "report.json"
OUT_JSON = ART_DIR / "redteam_audit.json"
OUT_MD = ROOT / "docs/audits/portfolio_online_asset_switch_native_redteam_20260709.md"


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


def _metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in df["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(df)), "wr": float(wins / len(df))}


def _issue(issues: list[dict[str, Any]], severity: str, check: str, message: str) -> None:
    issues.append({"severity": severity, "check": check, "message": message})


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    for key, expected in (
        ("fresh_forward_bar_by_bar", True),
        ("trade_ledgers_used_as_input", False),
        ("saved_parent_exit_timestamps_used", False),
        ("future_rows_used_for_entry", False),
        ("uses_only_past_closed_trades_for_learning", True),
        ("state_receives_all_asset_candidates", True),
        ("action_masking", True),
        ("skip_action_removed", True),
    ):
        if report.get(key) is not expected:
            _issue(issues, "P0", key, f"expected {expected}, got {report.get(key)}")
    diag: dict[str, Any] = {}
    action_counts: dict[str, dict[str, int]] = {}
    for split in ("validation", "oos_extended"):
        ledger = pd.read_csv(ART_DIR / f"{split}_ledger.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
        decisions = pd.read_csv(ART_DIR / f"{split}_decisions.csv")
        overlap = int((ledger.sort_values("entry_timestamp")["entry_timestamp"].shift(-1) <= ledger.sort_values("entry_timestamp")["exit_timestamp"]).fillna(False).sum()) if not ledger.empty else 0
        if overlap:
            _issue(issues, "P0", f"{split}_overlap", f"{overlap} overlapping positions")
        diag[split] = {"metrics": _metrics(ledger), "overlap_count": overlap, "asset_counts": {str(k): int(v) for k, v in ledger["asset"].value_counts().items()}}
        action_counts[split] = {str(k): int(v) for k, v in decisions["action"].value_counts().items()}
    res = report["results"]
    if res["oos_extended"]["pnl"] <= 4.046534614137842 + 1e-9 and res["oos_extended"]["trades"] == 49:
        _issue(issues, "P2", "equivalent_to_rule_take_all", "TAKE-only asset switch matched existing rule_take_all result")
    _issue(issues, "P1", "promotion_grade", "no demonstrated improvement over rule baseline")
    audit = {
        "audit_id": "portfolio_online_asset_switch_native_redteam_20260709",
        "promotion_pass": False,
        "p0_count": int(sum(i["severity"] == "P0" for i in issues)),
        "issues": issues,
        "ledger_diagnostics": diag,
        "action_counts": action_counts,
        "results": res,
    }
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    lines = [
        "# Portfolio Online Asset Switch Native Red-Team Audit - 2026-07-09",
        "",
        f"Promotion pass: `{audit['promotion_pass']}`",
        f"P0 count: `{audit['p0_count']}`",
        "",
        "## Findings",
        "",
    ]
    for issue in issues:
        lines.append(f"- `{issue['severity']}` `{issue['check']}`: {issue['message']}")
    lines.extend(["", "## Action Counts", "", "```json", json.dumps(action_counts, ensure_ascii=False, indent=2), "```", ""])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit": str(OUT_JSON), "md": str(OUT_MD), "promotion_pass": audit["promotion_pass"], "p0_count": audit["p0_count"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the native bar-by-bar 2-action portfolio RL gate replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_rl_gate_2action_native_20260708"
REPORT_PATH = ART_DIR / "report.json"
OUT_JSON = ART_DIR / "redteam_audit.json"
OUT_MD = ROOT / "docs/audits/portfolio_rl_gate_2action_native_redteam_20260708.md"


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


def _audit_ledger(name: str, path: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    diag: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        _issue(issues, "P0", f"{name}_presence", f"missing {path}")
        return diag
    df = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp"])
    diag["rows"] = int(len(df))
    if df.empty:
        _issue(issues, "P1", f"{name}_empty", "empty ledger")
        return diag
    df = df.sort_values("entry_timestamp").reset_index(drop=True)
    overlap = int((df["entry_timestamp"].shift(-1) <= df["exit_timestamp"]).fillna(False).sum())
    diag["overlap_count"] = overlap
    if overlap:
        _issue(issues, "P0", f"{name}_overlap", f"{overlap} overlapping positions")
    if bool((df["exit_timestamp"] < df["entry_timestamp"]).any()):
        _issue(issues, "P0", f"{name}_exit_before_entry", "exit before entry found")
    notional = pd.to_numeric(df["notional"], errors="raise")
    margin = pd.to_numeric(df["margin_fraction"], errors="raise")
    leverage = pd.to_numeric(df["leverage"], errors="raise")
    err = (notional - margin * leverage).abs()
    diag["max_notional_identity_error"] = float(err.max())
    if float(err.max()) > 1e-8:
        _issue(issues, "P1", f"{name}_notional_identity", f"max error {err.max()}")
    diag["metrics"] = _metrics(df)
    diag["asset_counts"] = {str(k): int(v) for k, v in df["asset"].value_counts().items()}
    diag["reason_counts"] = {str(k): int(v) for k, v in df["reason"].value_counts().items()}
    diag["max_leverage"] = float(leverage.max())
    diag["max_notional"] = float(notional.max())
    if float(leverage.max()) > 5.0 + 1e-9:
        _issue(issues, "P0", f"{name}_leverage_cap", f"max leverage {leverage.max()}")
    if float(notional.max()) > 1.8 + 1e-9:
        _issue(issues, "P0", f"{name}_notional_cap", f"max notional {notional.max()}")
    return diag


def main() -> int:
    issues: list[dict[str, Any]] = []
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("fresh_forward_bar_by_bar") is not True:
        _issue(issues, "P0", "fresh_forward_bar_by_bar", f"got {report.get('fresh_forward_bar_by_bar')}")
    if report.get("trade_ledgers_used_as_input") is not False:
        _issue(issues, "P0", "trade_ledgers_used_as_input", f"got {report.get('trade_ledgers_used_as_input')}")
    if report.get("saved_parent_exit_timestamps_used") is not False:
        _issue(issues, "P0", "saved_parent_exit_timestamps_used", f"got {report.get('saved_parent_exit_timestamps_used')}")
    if report.get("future_rows_used_for_entry") is not False:
        _issue(issues, "P0", "future_rows_used_for_entry", f"got {report.get('future_rows_used_for_entry')}")
    if report.get("policy_training_used_event_ledger") is True:
        _issue(issues, "P1", "policy_training_event_ledger", "frozen policy was trained from event-level ledger prototype")
    if report.get("promotion_grade") is not True:
        _issue(issues, "P1", "promotion_grade", f"got {report.get('promotion_grade')}")

    ledgers = {
        "validation_rule_take_all": ART_DIR / "validation_rule_take_all_ledger.csv",
        "validation_rl_gate": ART_DIR / "validation_rl_gate_ledger.csv",
        "oos_rule_take_all": ART_DIR / "oos_extended_rule_take_all_ledger.csv",
        "oos_rl_gate": ART_DIR / "oos_extended_rl_gate_ledger.csv",
    }
    diag = {name: _audit_ledger(name, path, issues) for name, path in ledgers.items()}

    res = report["results"]
    if res["rl_gate"]["validation"]["pnl"] < res["rule_take_all"]["validation"]["pnl"]:
        _issue(issues, "P1", "rl_validation_underperforms_rule", "RL validation PnL is below rule_take_all")
    if res["rl_gate"]["oos_extended"]["pnl"] < res["rule_take_all"]["oos_extended"]["pnl"]:
        _issue(issues, "P1", "rl_oos_underperforms_rule", "RL OOS PnL is below rule_take_all")
    if res["rl_gate"]["oos_extended"]["mdd"] < res["rule_take_all"]["oos_extended"]["mdd"]:
        _issue(issues, "P1", "rl_oos_mdd_underperforms_rule", "RL OOS MDD is worse than rule_take_all")
    if res["rl_gate"]["oos_extended"]["mdd"] < -25.0:
        _issue(issues, "P1", "rl_oos_mdd_budget", f"RL OOS MDD={res['rl_gate']['oos_extended']['mdd']:.2f}%")

    blockers = [x for x in issues if x["severity"] == "P0"]
    audit = {
        "audit_id": "portfolio_rl_gate_2action_native_redteam_20260708",
        "promotion_pass": len(blockers) == 0 and not any(x["severity"] == "P1" for x in issues),
        "p0_count": len(blockers),
        "issues": issues,
        "ledger_diagnostics": diag,
        "reported_results": res,
    }
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    lines = [
        "# Portfolio RL Gate 2-Action Native Red-Team Audit - 2026-07-08",
        "",
        f"Promotion pass: `{audit['promotion_pass']}`",
        f"P0 count: `{audit['p0_count']}`",
        "",
        "## Findings",
        "",
    ]
    for issue in issues:
        lines.append(f"- `{issue['severity']}` `{issue['check']}`: {issue['message']}")
    lines.extend(["", "## Ledger Metrics", "", "| ledger | PnL | MDD | trades | WR | overlap |", "|---|---:|---:|---:|---:|---:|"])
    for name, d in diag.items():
        m = d.get("metrics", {})
        lines.append(f"| {name} | {m.get('pnl', 0):.2f}% | {m.get('mdd', 0):.2f}% | {m.get('trades', 0)} | {m.get('wr', 0):.2%} | {d.get('overlap_count', 0)} |")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit": str(OUT_JSON), "md": str(OUT_MD), "promotion_pass": audit["promotion_pass"], "p0_count": audit["p0_count"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

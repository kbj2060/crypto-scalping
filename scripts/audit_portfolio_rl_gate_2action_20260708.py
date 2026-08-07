#!/usr/bin/env python3
"""Red-team audit for the 2-action portfolio RL gate prototype."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_rl_gate_2action_20260708"
REPORT_PATH = ART_DIR / "report.json"
OUT_JSON = ART_DIR / "redteam_audit.json"
OUT_MD = ROOT / "docs/audits/portfolio_rl_gate_2action_redteam_20260708.md"


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


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(ledger)), "wr": float(wins / len(ledger))}


def _add(issues: list[dict[str, Any]], severity: str, check: str, message: str) -> None:
    issues.append({"severity": severity, "check": check, "message": message})


def _audit_ledger(name: str, path: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    diag: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        _add(issues, "P0", f"{name}_presence", f"missing {path}")
        return diag
    df = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp"])
    diag["rows"] = int(len(df))
    required = {"asset", "entry_timestamp", "exit_timestamp", "trade_return", "notional", "margin_fraction", "leverage"}
    missing = sorted(required - set(df.columns))
    if missing:
        _add(issues, "P0", f"{name}_schema", f"missing columns {missing}")
        return diag
    if df.empty:
        _add(issues, "P1", f"{name}_empty", "ledger is empty")
        return diag
    df = df.sort_values("entry_timestamp").reset_index(drop=True)
    overlaps = int((df["entry_timestamp"].shift(-1) <= df["exit_timestamp"]).fillna(False).sum())
    diag["overlap_count"] = overlaps
    if overlaps:
        _add(issues, "P0", f"{name}_overlap", f"{overlaps} overlapping selected positions")
    if bool((df["exit_timestamp"] < df["entry_timestamp"]).any()):
        _add(issues, "P0", f"{name}_exit_before_entry", "some exits precede entries")
    if bool((pd.to_numeric(df["trade_return"], errors="raise") <= -1.0).any()):
        _add(issues, "P0", f"{name}_bankruptcy_return", "trade_return <= -100%")
    notional = pd.to_numeric(df["notional"], errors="raise")
    margin = pd.to_numeric(df["margin_fraction"], errors="raise")
    lev = pd.to_numeric(df["leverage"], errors="raise")
    identity_err = (notional - margin * lev).abs()
    diag["max_notional_identity_error"] = float(identity_err.max())
    if float(identity_err.max()) > 1e-8:
        _add(issues, "P1", f"{name}_notional_identity", f"max notional identity error {identity_err.max()}")
    diag["asset_counts"] = {str(k): int(v) for k, v in df["asset"].value_counts().items()}
    diag["max_leverage"] = float(lev.max())
    diag["max_notional"] = float(notional.max())
    diag["metrics"] = _compound_metrics(df)
    if "router_action" in df.columns:
        actions = set(df["router_action"].astype(str))
        diag["router_actions"] = sorted(actions)
        if actions - {"TAKE_TOP"}:
            _add(issues, "P0", f"{name}_router_action", f"unexpected selected actions {sorted(actions)}")
    if {"q_skip", "q_take"}.issubset(df.columns):
        qdiff = pd.to_numeric(df["q_take"], errors="raise") - pd.to_numeric(df["q_skip"], errors="raise")
        diag["qdiff_min"] = float(qdiff.min())
        diag["qdiff_max"] = float(qdiff.max())
        if bool((qdiff <= 0.0).any()):
            _add(issues, "P0", f"{name}_q_consistency", "selected TAKE_TOP row has q_take <= q_skip")
    return diag


def main() -> int:
    issues: list[dict[str, Any]] = []
    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    # Promotion-contract gates.
    if report.get("fresh_forward_bar_by_bar") is not True:
        _add(issues, "P0", "fresh_forward_bar_by_bar", f"expected true, got {report.get('fresh_forward_bar_by_bar')}")
    if report.get("trade_ledgers_used_as_input") is not False:
        _add(issues, "P0", "trade_ledgers_used_as_input", f"expected false, got {report.get('trade_ledgers_used_as_input')}")
    if report.get("saved_parent_exit_timestamps_used") is not False:
        _add(issues, "P0", "saved_parent_exit_timestamps_used", f"expected false, got {report.get('saved_parent_exit_timestamps_used')}")
    if report.get("future_rows_used_for_entry") is not False:
        _add(issues, "P0", "future_rows_used_for_entry", f"expected false, got {report.get('future_rows_used_for_entry')}")
    if report.get("promotion_grade") is not True:
        _add(issues, "P0", "promotion_grade", f"expected true, got {report.get('promotion_grade')}")
    if report.get("training_data") != "validation_only":
        _add(issues, "P1", "training_data", f"training_data={report.get('training_data')}")
    if report.get("oos_usage") != "reported_once_after_policy_training":
        _add(issues, "P1", "oos_usage", f"oos_usage={report.get('oos_usage')}")
    if set(report.get("action_space", {}).values()) != {"SKIP", "TAKE_TOP"}:
        _add(issues, "P0", "action_space", f"unexpected action_space={report.get('action_space')}")

    ledgers = {
        "validation_rl_gate": ART_DIR / "validation_rl_gate_ledger.csv",
        "oos_rl_gate": ART_DIR / "oos_rl_gate_ledger.csv",
        "validation_rule_take_all": ART_DIR / "validation_rule_take_all_ledger.csv",
        "oos_rule_take_all": ART_DIR / "oos_rule_take_all_ledger.csv",
    }
    ledger_diag = {name: _audit_ledger(name, path, issues) for name, path in ledgers.items()}

    p0 = [x for x in issues if x["severity"] == "P0"]
    audit = {
        "audit_id": "portfolio_rl_gate_2action_redteam_20260708",
        "report": str(REPORT_PATH),
        "promotion_pass": len(p0) == 0,
        "blocker_count": len(p0),
        "issues": issues,
        "ledger_diagnostics": ledger_diag,
        "reported_results": report.get("results", {}),
    }
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    lines = [
        "# Portfolio RL Gate 2-Action Red-Team Audit - 2026-07-08",
        "",
        f"Promotion pass: `{audit['promotion_pass']}`",
        f"Blocker count: `{audit['blocker_count']}`",
        "",
        "## Findings",
        "",
    ]
    for issue in issues:
        lines.append(f"- `{issue['severity']}` `{issue['check']}`: {issue['message']}")
    lines.extend(["", "## Ledger Metrics", "", "| ledger | PnL | MDD | trades | WR | overlap |", "|---|---:|---:|---:|---:|---:|"])
    for name, diag in ledger_diag.items():
        m = diag.get("metrics", {})
        lines.append(f"| {name} | {m.get('pnl', 0):.2f}% | {m.get('mdd', 0):.2f}% | {m.get('trades', 0)} | {m.get('wr', 0):.2%} | {diag.get('overlap_count', 0)} |")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit": str(OUT_JSON), "md": str(OUT_MD), "promotion_pass": audit["promotion_pass"], "blocker_count": audit["blocker_count"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

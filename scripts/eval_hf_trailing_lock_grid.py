#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_hf_entry_overlay_grid import DEFAULT_EVAL_CSV, DEFAULT_POLICY, DEFAULT_TRAIN_CSV, _audit, _decisions, _quality_scaled_decisions  # noqa: E402
from scripts.eval_hf_risk_overlay_grid import _read, backtest_hf_risk_overlay  # noqa: E402
from scripts.eval_lifecycle_ai_stress import _stress_frame  # noqa: E402


DEFAULT_SELECTION = ROOT / "data/ensemble/reports/hf_final_selection_hf_v4_2026.json"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/hf_trailing_lock_grid_hf_v4_2026.json"


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "long_entries", "short_entries", "entry_blocks", "exits")
    }


def _base_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in ("monthly_balanced_selection", "highest_return_mdd20_selection"):
        src = selection[key]
        rows.append(
            {
                "name": src["name"],
                "entry_config": src["entry_config"],
                "risk_config": dict(src["risk_config"]),
            }
        )
    return rows


def _trail_cfgs() -> list[dict[str, float]]:
    out = [{"trailing_trigger": 999.0, "trailing_gap": 999.0}]
    for trigger in (0.006, 0.010, 0.015, 0.025):
        for gap in (0.003, 0.005, 0.008, 0.012):
            out.append({"trailing_trigger": trigger, "trailing_gap": gap})
    return out


def _eval(df: pd.DataFrame, dec0: pd.DataFrame, row: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    dec = _quality_scaled_decisions(dec0, **row["entry_config"])
    return _compact(backtest_hf_risk_overlay(df, dec, fee=fee, slip=slip, **row["risk_config"]))


def _monthly(df: pd.DataFrame, dec0: pd.DataFrame, row: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    jan = df["timestamp"] < pd.Timestamp("2026-02-01")
    feb = df["timestamp"] >= pd.Timestamp("2026-02-01")
    jan_eval = _eval(df.loc[jan].reset_index(drop=True), dec0.loc[jan].reset_index(drop=True), row, fee=fee, slip=slip)
    feb_eval = _eval(df.loc[feb].reset_index(drop=True), dec0.loc[feb].reset_index(drop=True), row, fee=fee, slip=slip)
    return {"jan": jan_eval, "feb": feb_eval, "min_month_pnl": float(min(jan_eval["pnl"], feb_eval["pnl"]))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trailing lock search for selected HF risk/profit configs.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    eval_df = _read(args.eval_csv)
    dec0 = _decisions(eval_df, policy)
    rows: list[dict[str, Any]] = []
    for base in _base_rows(selection):
        for trail in _trail_cfgs():
            risk_cfg = dict(base["risk_config"])
            risk_cfg.update(trail)
            row = {
                "name": f"{base['name']}_tr{trail['trailing_trigger']}_gap{trail['trailing_gap']}",
                "base_name": base["name"],
                "entry_config": base["entry_config"],
                "risk_config": risk_cfg,
            }
            row["eval"] = _eval(eval_df, dec0, row, fee=float(args.fee), slip=float(args.slip))
            rows.append(row)

    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    goal = [r for r in ranked if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 20.0]
    mdd15 = [r for r in goal if float(r["eval"].get("mdd") or -1e18) >= -15.0]
    top = (mdd15 or goal)[:8]
    monthly = []
    for row in top:
        monthly.append({"name": row["name"], "full": row["eval"], **_monthly(eval_df, dec0, row, fee=float(args.fee), slip=float(args.slip))})
    monthly_balanced = sorted(monthly, key=lambda r: (float(r["min_month_pnl"]), float(r["full"]["pnl"])), reverse=True)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = [{"name": row["name"], "eval": _eval(eval_df, dec0, row, fee=float(args.fee) * mult, slip=float(args.slip) * mult)} for row in top[:5]]

    ai_stress: dict[str, Any] = {}
    for mode in ("normal", "all_ai_zero", "patchtst_zero", "tide_zero", "dlinear_zero"):
        df, meta = _stress_frame(eval_df, mode)
        decs = _decisions(df, policy)
        ai_stress[mode] = {"stress": meta, "results": [{"name": row["name"], "eval": _eval(df, decs, row, fee=float(args.fee), slip=float(args.slip))} for row in top[:3]]}

    report = {
        "type": "hf_trailing_lock_grid_hf_v4_2026",
        "policy": str(args.policy),
        "selection": str(args.selection),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "selection_overfit_note": "Trailing parameters are research-selected on 2026; monthly section is an additional robustness check.",
        "grid_size": len(rows),
        "grid": rows,
        "ranked_goal_mdd_lte_15": [{"name": r["name"], **r["eval"]} for r in mdd15[:30]],
        "monthly_balanced": monthly_balanced,
        "cost_stress": cost_stress,
        "ai_stress": ai_stress,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "top": report["ranked_goal_mdd_lte_15"][:8], "monthly_balanced": monthly_balanced[:5]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

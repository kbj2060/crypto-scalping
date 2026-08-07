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
from scripts.eval_hf_risk_overlay_grid import backtest_hf_risk_overlay  # noqa: E402
from scripts.eval_lifecycle_ai_stress import _stress_frame  # noqa: E402


DEFAULT_BASE_REPORT = ROOT / "data/ensemble/reports/hf_risk_overlay_grid_hf_v4_2026.json"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/hf_profit_pyramid_grid_hf_v4_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "long_entries", "short_entries", "entry_blocks", "exits")
    }


def _boosts() -> list[dict[str, float]]:
    out = [{"daily_profit_boost_start": 999.0, "daily_profit_boost_mult": 1.0, "equity_high_boost_dd": -1.0, "equity_high_boost_mult": 1.0}]
    for start in (0.005, 0.015, 0.030):
        for mult in (1.10, 1.20, 1.35):
            out.append(
                {
                    "daily_profit_boost_start": start,
                    "daily_profit_boost_mult": mult,
                    "equity_high_boost_dd": -1.0,
                    "equity_high_boost_mult": 1.0,
                }
            )
    for dd in (0.005, 0.015, 0.030):
        for mult in (1.08, 1.16, 1.28):
            out.append(
                {
                    "daily_profit_boost_start": 999.0,
                    "daily_profit_boost_mult": 1.0,
                    "equity_high_boost_dd": dd,
                    "equity_high_boost_mult": mult,
                }
            )
    for start in (0.005, 0.015):
        for pmult in (1.10, 1.20):
            for dd in (0.005, 0.015):
                for emult in (1.08, 1.16):
                    out.append(
                        {
                            "daily_profit_boost_start": start,
                            "daily_profit_boost_mult": pmult,
                            "equity_high_boost_dd": dd,
                            "equity_high_boost_mult": emult,
                        }
                    )
    return out


def _unique_base_rows(report: dict[str, Any], n: int) -> list[dict[str, Any]]:
    by_name = {r["name"]: r for r in report["grid"]}
    names: list[str] = []
    for section in ("ranked_goal_mdd_lte_30", "ranked_goal_5_to_20_trades_per_day", "ranked_by_pnl"):
        for row in report.get(section, []):
            name = row["name"]
            if name not in names and name in by_name:
                names.append(name)
            if len(names) >= n:
                return [by_name[x] for x in names]
    return [by_name[x] for x in names]


def _eval(df: pd.DataFrame, base_dec: pd.DataFrame, row: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    dec = _quality_scaled_decisions(base_dec, **row["entry_config"])
    return _compact(backtest_hf_risk_overlay(df, dec, fee=fee, slip=slip, **row["risk_config"]))


def _monthly(df: pd.DataFrame, dec: pd.DataFrame, row: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    jan = df["timestamp"] < pd.Timestamp("2026-02-01")
    feb = df["timestamp"] >= pd.Timestamp("2026-02-01")
    jan_eval = _eval(df.loc[jan].reset_index(drop=True), dec.loc[jan].reset_index(drop=True), row, fee=fee, slip=slip)
    feb_eval = _eval(df.loc[feb].reset_index(drop=True), dec.loc[feb].reset_index(drop=True), row, fee=fee, slip=slip)
    return {"jan": jan_eval, "feb": feb_eval, "min_month_pnl": float(min(jan_eval["pnl"], feb_eval["pnl"]))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profit-pyramid overlay search on top HF risk configs.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--top-base", type=int, default=12)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    base_rows = _unique_base_rows(base_report, int(args.top_base))
    eval_df = _read(args.eval_csv)
    base_dec = _decisions(eval_df, policy)
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        for boost in _boosts():
            risk_cfg = dict(base["risk_config"])
            risk_cfg.update(boost)
            row = {
                "name": (
                    f"{base['name']}"
                    f"_pb{boost['daily_profit_boost_start']}x{boost['daily_profit_boost_mult']}"
                    f"_hb{boost['equity_high_boost_dd']}x{boost['equity_high_boost_mult']}"
                ),
                "base_name": base["name"],
                "entry_config": base["entry_config"],
                "risk_config": risk_cfg,
            }
            row["eval"] = _eval(eval_df, base_dec, row, fee=float(args.fee), slip=float(args.slip))
            rows.append(row)

    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    goal = [r for r in ranked if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 20.0]
    mdd20 = [r for r in goal if float(r["eval"].get("mdd") or -1e18) >= -20.0]
    mdd15 = [r for r in goal if float(r["eval"].get("mdd") or -1e18) >= -15.0]
    top = (mdd20 or goal)[:8]

    monthly = []
    for row in top:
        m = _monthly(eval_df, base_dec, row, fee=float(args.fee), slip=float(args.slip))
        monthly.append({"name": row["name"], "full": row["eval"], **m})
    monthly_balanced = sorted(monthly, key=lambda r: (float(r["min_month_pnl"]), float(r["full"]["pnl"])), reverse=True)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = []
        for row in top[:5]:
            cost_stress[f"cost_{mult:g}x"].append({
                "name": row["name"],
                "eval": _eval(eval_df, base_dec, row, fee=float(args.fee) * mult, slip=float(args.slip) * mult),
            })

    ai_stress: dict[str, Any] = {}
    for mode in ("normal", "all_ai_zero", "patchtst_zero", "tide_zero", "dlinear_zero"):
        df, meta = _stress_frame(eval_df, mode)
        dec0 = _decisions(df, policy)
        ai_stress[mode] = {"stress": meta, "results": []}
        for row in top[:3]:
            ai_stress[mode]["results"].append({"name": row["name"], "eval": _eval(df, dec0, row, fee=float(args.fee), slip=float(args.slip))})

    report = {
        "type": "hf_profit_pyramid_grid_hf_v4_2026",
        "policy": str(args.policy),
        "base_report": str(args.base_report),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "selection_overfit_note": "Grid is research search on 2026; monthly section reports Jan/Feb robustness for top configs.",
        "grid_size": len(rows),
        "grid": rows,
        "ranked_goal_5_to_20": [{"name": r["name"], **r["eval"]} for r in goal[:30]],
        "ranked_goal_mdd_lte_20": [{"name": r["name"], **r["eval"]} for r in mdd20[:30]],
        "ranked_goal_mdd_lte_15": [{"name": r["name"], **r["eval"]} for r in mdd15[:30]],
        "monthly_top": monthly,
        "monthly_balanced": monthly_balanced,
        "cost_stress": cost_stress,
        "ai_stress": ai_stress,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(args.report_out),
        "top_mdd20": report["ranked_goal_mdd_lte_20"][:8],
        "monthly_balanced": report["monthly_balanced"][:5],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

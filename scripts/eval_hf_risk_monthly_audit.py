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


DEFAULT_GRID_REPORT = ROOT / "data/ensemble/reports/hf_risk_overlay_grid_hf_v4_2026.json"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/hf_risk_monthly_audit_hf_v4_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "long_entries", "short_entries")
    }


def _eval_row(df: pd.DataFrame, dec: pd.DataFrame, row: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    d = _quality_scaled_decisions(dec, **row["entry_config"])
    return _compact(backtest_hf_risk_overlay(df, d, fee=fee, slip=slip, **row["risk_config"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly selection audit for HF risk overlay grid.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--grid-report", type=Path, default=DEFAULT_GRID_REPORT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    grid_report = json.loads(args.grid_report.read_text(encoding="utf-8"))
    rows = list(grid_report["grid"])
    full_df = _read(args.eval_csv)
    full_dec = _decisions(full_df, policy)
    jan_mask = full_df["timestamp"] < pd.Timestamp("2026-02-01")
    feb_mask = full_df["timestamp"] >= pd.Timestamp("2026-02-01")
    jan_df = full_df.loc[jan_mask].reset_index(drop=True)
    feb_df = full_df.loc[feb_mask].reset_index(drop=True)
    jan_dec = full_dec.loc[jan_mask].reset_index(drop=True)
    feb_dec = full_dec.loc[feb_mask].reset_index(drop=True)

    jan_rows: list[dict[str, Any]] = []
    feb_rows: list[dict[str, Any]] = []
    for row in rows:
        jan_rows.append({"name": row["name"], "entry_config": row["entry_config"], "risk_config": row["risk_config"], "eval": _eval_row(jan_df, jan_dec, row, fee=float(args.fee), slip=float(args.slip))})
        feb_rows.append({"name": row["name"], "entry_config": row["entry_config"], "risk_config": row["risk_config"], "eval": _eval_row(feb_df, feb_dec, row, fee=float(args.fee), slip=float(args.slip))})

    jan_goal = [
        r for r in sorted(jan_rows, key=lambda x: float(x["eval"].get("pnl") or -1e18), reverse=True)
        if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 20.0 and float(r["eval"].get("mdd") or -1e18) >= -30.0
    ]
    feb_by_name = {r["name"]: r for r in feb_rows}
    selected = jan_goal[0]
    selected_feb = feb_by_name[selected["name"]]
    full_top_name = grid_report["ranked_goal_mdd_lte_30"][0]["name"]

    report = {
        "type": "hf_risk_monthly_audit_hf_v4_2026",
        "selection_rule": "select best Jan 2026 pnl among 5-20 trades/day and MDD >= -30%, then evaluate Feb 2026 untouched",
        "audit": {
            **_audit(args.train_csv, args.eval_csv, policy),
            "jan_rows": int(len(jan_df)),
            "feb_rows": int(len(feb_df)),
            "jan_range": [str(jan_df["timestamp"].min()), str(jan_df["timestamp"].max())],
            "feb_range": [str(feb_df["timestamp"].min()), str(feb_df["timestamp"].max())],
            "jan_feb_timestamp_overlap_rows": int(len(set(jan_df["timestamp"].astype("int64")) & set(feb_df["timestamp"].astype("int64")))),
        },
        "jan_selected": {"name": selected["name"], **selected["eval"]},
        "jan_selected_feb_eval": {"name": selected_feb["name"], **selected_feb["eval"]},
        "full_2026_top_monthly_breakdown": {
            "name": full_top_name,
            "jan": next({"name": r["name"], **r["eval"]} for r in jan_rows if r["name"] == full_top_name),
            "feb": next({"name": r["name"], **r["eval"]} for r in feb_rows if r["name"] == full_top_name),
            "full": next({"name": r["name"], **r["eval"]} for r in rows if r["name"] == full_top_name),
        },
        "ranked_jan_goal_mdd30": [{"name": r["name"], **r["eval"]} for r in jan_goal[:20]],
        "ranked_feb_goal_mdd30": [
            {"name": r["name"], **r["eval"]}
            for r in sorted(
                [
                    r for r in feb_rows
                    if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 20.0 and float(r["eval"].get("mdd") or -1e18) >= -30.0
                ],
                key=lambda x: float(x["eval"].get("pnl") or -1e18),
                reverse=True,
            )[:20]
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(args.report_out),
        "jan_selected": report["jan_selected"],
        "jan_selected_feb_eval": report["jan_selected_feb_eval"],
        "full_top_monthly": report["full_2026_top_monthly_breakdown"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

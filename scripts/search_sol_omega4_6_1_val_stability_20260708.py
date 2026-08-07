#!/usr/bin/env python3
"""Validation-only stability search for the SOL Omega4.6.1 baseline.

OOS is intentionally not used for selection. The selected parameter set is
chosen from the SOL zig075 q070 validation ledger using risk-policy gates:
- validation MDD must be inside the configured drawdown budget
- month-level validation stress must stay inside configured budgets
- trades must remain large enough after the duration gate

After selection, OOS is reported once as a sealed holdout check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/causal_regen_20260516"
RISK_DIR = BASE / "sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707"
VAL_FEATURES = ROOT / "data/splits/year_oos/sol_features_2025.csv"
OOS_FEATURES = ROOT / "data/splits/year_oos/sol_features_2026.csv"
OUT_DIR = BASE / "sol_omega4_6_1_val_stability_search_20260708"
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8


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
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)),
    }


def _load_ledger(path: Path, features_path: Path) -> pd.DataFrame:
    ledger = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp"])
    feats = pd.read_csv(features_path, usecols=["timestamp", "ou_halflife"], parse_dates=["timestamp"])
    out = ledger.merge(feats.rename(columns={"timestamp": "entry_timestamp"}), on="entry_timestamp", how="left", validate="one_to_one")
    if out["ou_halflife"].isna().any():
        raise RuntimeError(f"{path}: ou_halflife merge produced NaN")
    return out


def _apply_scale(ledger: pd.DataFrame, *, long_scale: float, short_scale: float) -> pd.DataFrame:
    out = ledger.copy()
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    margin = pd.to_numeric(out["margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    lev = pd.to_numeric(out["leverage"], errors="raise").to_numpy(dtype=np.float64)
    scale = np.where(side > 0, float(long_scale), np.where(side < 0, float(short_scale), 1.0))
    lev2 = np.minimum(lev * scale, LEVERAGE_CAP)
    notional = np.minimum(margin * lev2, NOTIONAL_CAP)
    lev2 = np.where(margin > 0.0, notional / np.maximum(margin, 1e-12), lev2)
    out["leverage"] = lev2
    out["notional"] = notional
    out["trade_return"] = pd.to_numeric(out["net_per_notional"], errors="raise").to_numpy(dtype=np.float64) * notional
    return out


def _duration_variants(ledger: pd.DataFrame, *, min_trades: int) -> list[tuple[float, float | None, pd.DataFrame]]:
    out: list[tuple[float, float | None, pd.DataFrame]] = [(0.0, None, ledger)]
    values = ledger["ou_halflife"].to_numpy(dtype=np.float64)
    for q in np.arange(0.05, 0.85, 0.05):
        threshold = float(np.quantile(values, q))
        gated = ledger.loc[ledger["ou_halflife"] > threshold].reset_index(drop=True)
        if len(gated) >= int(min_trades):
            out.append((threshold, float(q), gated))
    return out


def _monthly_metrics(ledger: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ledger.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    month = ledger["entry_timestamp"].dt.to_period("M").astype(str)
    for key, group in ledger.groupby(month, sort=True):
        out[str(key)] = _compound_metrics(group.reset_index(drop=True))
    return out


def _score(validation: dict[str, Any], monthlies: dict[str, dict[str, Any]]) -> float:
    mdd_abs = max(abs(float(validation["mdd"])), 1.0)
    worst_month_pnl = min(float(x["pnl"]) for x in monthlies.values()) if monthlies else -100.0
    return float(validation["pnl"]) / mdd_abs + 0.05 * worst_month_pnl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-grid", default="0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--min-validation-trades", type=int, default=24)
    ap.add_argument("--max-validation-mdd-abs", type=float, default=20.0)
    ap.add_argument("--max-monthly-mdd-abs", type=float, default=12.0)
    ap.add_argument("--min-worst-month-pnl", type=float, default=-8.0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(x) for x in str(args.scale_grid).split(",") if x.strip()]
    val = _load_ledger(RISK_DIR / "validation_selected_risk_replayed_trade_ledger.csv", VAL_FEATURES)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -np.inf
    for long_scale in scales:
        for short_scale in scales:
            scaled = _apply_scale(val, long_scale=long_scale, short_scale=short_scale)
            for duration_threshold, duration_quantile, gated in _duration_variants(scaled, min_trades=int(args.min_validation_trades)):
                validation = _compound_metrics(gated)
                monthlies = _monthly_metrics(gated)
                worst_month_pnl = min((float(x["pnl"]) for x in monthlies.values()), default=-100.0)
                worst_month_mdd = min((float(x["mdd"]) for x in monthlies.values()), default=-100.0)
                eligible = (
                    int(validation["trades"]) >= int(args.min_validation_trades)
                    and float(validation["mdd"]) >= -abs(float(args.max_validation_mdd_abs))
                    and worst_month_mdd >= -abs(float(args.max_monthly_mdd_abs))
                    and worst_month_pnl >= float(args.min_worst_month_pnl)
                )
                row = {
                    "asset": "sol",
                    "component": "zig075",
                    "quality_tag": "q070",
                    "quality_threshold": 0.70,
                    "long_scale": float(long_scale),
                    "short_scale": float(short_scale),
                    "duration_threshold": float(duration_threshold),
                    "duration_quantile": duration_quantile,
                    "validation": validation,
                    "validation_monthly": monthlies,
                    "worst_month_pnl": float(worst_month_pnl),
                    "worst_month_mdd": float(worst_month_mdd),
                    "score": _score(validation, monthlies),
                    "eligible": bool(eligible),
                }
                rows.append(row)
                if eligible and float(row["score"]) > best_score:
                    best = row
                    best_score = float(row["score"])
    if best is None:
        raise RuntimeError("no validation-stable candidate found")

    oos = _load_ledger(RISK_DIR / "oos_selected_risk_replayed_trade_ledger.csv", OOS_FEATURES)
    oos_scaled = _apply_scale(oos, long_scale=float(best["long_scale"]), short_scale=float(best["short_scale"]))
    oos_gated = oos_scaled.loc[oos_scaled["ou_halflife"] > float(best["duration_threshold"])].reset_index(drop=True)
    oos_q1 = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)

    report = {
        "method": "sol_omega4_6_1_val_only_stability_search",
        "promotion_grade": False,
        "selection_data": "validation_only_2025_09_01_to_2025_12_31",
        "oos_usage": "reported_once_after_selection_not_used_for_selection",
        "selection_objective": "max validation stability score after validation total/monthly drawdown and worst-month gates",
        "search_space": {
            "component": "zig075",
            "quality_tag": "q070",
            "scale_grid": scales,
            "duration_quantiles": "0.05..0.80 by 0.05 plus no-gate",
            "min_validation_trades": int(args.min_validation_trades),
            "max_validation_mdd_abs": float(args.max_validation_mdd_abs),
            "max_monthly_mdd_abs": float(args.max_monthly_mdd_abs),
            "min_worst_month_pnl": float(args.min_worst_month_pnl),
            "leverage_cap": LEVERAGE_CAP,
            "notional_cap": NOTIONAL_CAP,
        },
        "selected": best,
        "oos_one_shot": _compound_metrics(oos_gated),
        "oos_frozen_q1_2026": _compound_metrics(oos_q1),
        "candidate_count": len(rows),
        "eligible_count": int(sum(bool(x["eligible"]) for x in rows)),
    }
    pd.DataFrame(rows).to_csv(args.out_dir / "candidate_grid.csv", index=False)
    oos_gated.to_csv(args.out_dir / "selected_oos_gated_ledger.csv", index=False)
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

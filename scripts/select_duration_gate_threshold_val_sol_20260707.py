#!/usr/bin/env python3
"""SOL duration-gate (ou_halflife) threshold calibration, VAL-only.

Simplified single-component analogue of select_duration_gate_threshold_val_
20260706.py: this SOL build only has one viable component (zig075 q070, see
GATE 2), so there is no priority_route/SCALE_MAP two-component reconciliation
step to replicate. Grid-searches quantile thresholds of ou_halflife (computed
on VAL trades only) against the VAL trade ledger, selects by an
ETH-duration_priority_score-style objective (monthly-worst-PnL weighted
heavily, small weight on overall PnL and trade-count retention), then reports
the frozen threshold applied ONCE to the OOS ledger (no threshold re-selection
on OOS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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


def _load_ledger_with_halflife(ledger_path: Path, features_path: Path) -> pd.DataFrame:
    ledger = pd.read_csv(ledger_path, parse_dates=["entry_timestamp"])
    ledger = ledger.sort_values("entry_timestamp").reset_index(drop=True)
    feats = pd.read_csv(features_path, usecols=["timestamp", "ou_halflife"], parse_dates=["timestamp"])
    merged = ledger.merge(feats.rename(columns={"timestamp": "entry_timestamp"}), on="entry_timestamp", how="left", validate="one_to_one")
    if merged["ou_halflife"].isna().any():
        raise RuntimeError("ou_halflife merge produced NaN for some trades")
    return merged


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


def _monthly_min_pnl(ledger: pd.DataFrame) -> float:
    if ledger.empty:
        return 0.0
    month = ledger["entry_timestamp"].dt.to_period("M")
    pnls = []
    for _, group in ledger.groupby(month):
        pnls.append(_compound_metrics(group)["pnl"])
    return float(min(pnls)) if pnls else 0.0


def _priority_score(gated: pd.DataFrame, baseline: pd.DataFrame) -> float:
    val_m = _compound_metrics(gated)
    monthly_min = _monthly_min_pnl(gated)
    return 2.0 * monthly_min + 0.10 * float(val_m["pnl"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ledger", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707/validation_selected_risk_replayed_trade_ledger.csv")
    ap.add_argument("--oos-ledger", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707/oos_selected_risk_replayed_trade_ledger.csv")
    ap.add_argument("--val-features", type=Path, default=ROOT / "data/splits/year_oos/sol_features_2025.csv")
    ap.add_argument("--oos-features", type=Path, default=ROOT / "data/splits/year_oos/sol_features_2026.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_duration_gate_threshold_val_20260707")
    ap.add_argument("--max-mdd-abs", type=float, default=20.0)
    ap.add_argument("--min-trade-ratio", type=float, default=0.65)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val = _load_ledger_with_halflife(args.val_ledger, args.val_features)
    oos = _load_ledger_with_halflife(args.oos_ledger, args.oos_features)
    baseline_val_m = _compound_metrics(val)
    baseline_oos_m = _compound_metrics(oos)

    quantiles = np.arange(0.05, 0.85, 0.05)
    thresholds = sorted(set(float(np.quantile(val["ou_halflife"].to_numpy(dtype=np.float64), q)) for q in quantiles))
    trade_floor = int(np.floor(int(baseline_val_m["trades"]) * float(args.min_trade_ratio)))
    mdd_floor = -abs(float(args.max_mdd_abs))

    candidates: list[dict[str, Any]] = []
    for q, th in zip(quantiles, thresholds):
        gated_val = val.loc[val["ou_halflife"] > th].reset_index(drop=True)
        val_m = _compound_metrics(gated_val)
        eligible = int(val_m["trades"]) >= trade_floor and float(val_m["mdd"]) >= mdd_floor
        candidates.append(
            {
                "quantile": float(q),
                "threshold": float(th),
                "validation": val_m,
                "validation_monthly_min_pnl": _monthly_min_pnl(gated_val),
                "eligible": bool(eligible),
                "priority_score": _priority_score(gated_val, val) if eligible else float("-inf"),
            }
        )

    eligible_candidates = [c for c in candidates if c["eligible"]]
    no_gate_candidate = {
        "quantile": None,
        "threshold": 0.0,
        "validation": baseline_val_m,
        "validation_monthly_min_pnl": _monthly_min_pnl(val),
        "eligible": True,
        "priority_score": _priority_score(val, val),
    }
    pool = eligible_candidates + [no_gate_candidate]
    selected = max(pool, key=lambda c: float(c["priority_score"]))
    selected_threshold = float(selected["threshold"])

    gated_oos = oos.loc[oos["ou_halflife"] > selected_threshold].reset_index(drop=True)
    selected_oos_m = _compound_metrics(gated_oos)

    report = {
        "method": "single_component_sol_duration_gate_val_grid_search",
        "baseline_no_gate": {"validation": baseline_val_m, "oos": baseline_oos_m},
        "candidates": candidates,
        "selection_gate": {"min_trades_ratio": float(args.min_trade_ratio), "max_mdd_abs": float(args.max_mdd_abs)},
        "selected": {
            "threshold": selected_threshold,
            "quantile": selected["quantile"],
            "validation": selected["validation"],
            "oos_one_shot": selected_oos_m,
        },
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"selected_threshold": selected_threshold, "validation": selected["validation"], "oos_one_shot": selected_oos_m, "baseline_no_gate": report["baseline_no_gate"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

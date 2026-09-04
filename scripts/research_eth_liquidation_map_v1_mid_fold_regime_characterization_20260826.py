#!/usr/bin/env python3
"""2026-08-26 user follow-up to the v1_mid 4-fold walk-forward re-verification: "그럼 데이터 구간
split 해놓은 것들이 불장인지 베어장인지에 따라 바뀌는건 아니야?" -- the 4 folds so far were split
purely by calendar time (equal eval-point counts), never checked for whether they actually span
different market regimes or happen to be 4 similar (e.g. all-bullish) windows, which would make
"consistent across 4 folds" a much weaker claim than it sounds (regime-confounded, not regime-
independent).

Computes, for each of the SAME 4 fold boundaries used in
research_eth_liquidation_map_v1_mid_walkforward_multifold_20260826.py (identical eval-point split,
reusing v1dir's grid so the fold boundaries are byte-identical): net % price change from first to
last eval point in the fold, max drawdown, max rally (drawup), and a simple bull/bear/choppy label
so the fold-to-fold support consistency finding can be read against actual realized regime, not
assumed from calendar position alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_v1_mid_fold_regime_characterization_20260826.json"

N_FOLDS = 4
BULL_THRESHOLD_PCT = 10.0
BEAR_THRESHOLD_PCT = -10.0


def max_drawdown_drawup(close: np.ndarray) -> tuple[float, float]:
    running_max = np.maximum.accumulate(close)
    running_min = np.minimum.accumulate(close)
    dd = float(((close - running_max) / running_max).min() * 100)   # most negative dip from a prior peak
    du = float(((close - running_min) / running_min).max() * 100)   # most positive rally from a prior trough
    return dd, du


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)

    n = len(df)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    close_full = df["close"].to_numpy(dtype="float64")
    ts_full = df["timestamp"]

    fold_bounds = np.linspace(0, len(eval_idxs), N_FOLDS + 1).astype(int)
    print(f"\n{'fold':5s} {'period':25s} {'net_pct':9s} {'max_dd%':9s} {'max_rally%':11s} {'label':10s}", flush=True)
    rows = []
    for f in range(N_FOLDS):
        fold_eval_idxs = eval_idxs[fold_bounds[f]: fold_bounds[f + 1]]
        idx_lo, idx_hi = fold_eval_idxs[0], fold_eval_idxs[-1]
        # Full hourly closes across the fold's actual calendar span (not just the eval-point subset)
        seg = close_full[idx_lo: idx_hi + 1]
        net_pct = float((seg[-1] - seg[0]) / seg[0] * 100)
        dd, du = max_drawdown_drawup(seg)
        label = "BULL" if net_pct > BULL_THRESHOLD_PCT else ("BEAR" if net_pct < BEAR_THRESHOLD_PCT else "CHOPPY")
        row = {
            "fold": f, "ts_lo": str(ts_full.iloc[idx_lo]), "ts_hi": str(ts_full.iloc[idx_hi]),
            "n_hourly_bars": int(idx_hi - idx_lo + 1), "start_price": float(seg[0]), "end_price": float(seg[-1]),
            "net_pct_change": round(net_pct, 2), "max_drawdown_pct": round(dd, 2), "max_rally_pct": round(du, 2),
            "label": label,
        }
        rows.append(row)
        period = f"{row['ts_lo'][:10]}~{row['ts_hi'][:10]}"
        print(f"{f:<5d} {period:25s} {net_pct:+8.2f} {dd:8.2f} {du:10.2f}  {label:10s}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"n_bars": n, "n_folds": N_FOLDS, "folds": rows}, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

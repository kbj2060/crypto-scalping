#!/usr/bin/env python3
"""ATR trailing-stop cost-gate grid (SL x ARM x Trail, 96 combos) for the 2 Homer candidates at
their final confirmed (HORIZON, GAP, K), VAL+OOS only (HOLDOUT untouched). Reuses core.
causal_futures_backtest.simulate_single_position verbatim -- same standard engine, same constants
(MARGIN_FRACTION=0.30/LEVERAGE=3.0/ROUNDTRIP_COST_RATE=0.001=10bp), same SL/ARM/TRAIL grids, and the
same "trade every cluster-anchored candidate unconditionally (score=+-1, not proba-filtered)"
convention every prior economics gate in this lineage used (docs/homer/README.md: proba-based R:R
scaling was tried and rejected elsewhere, corr(proba,pnl)~=0) -- exactly
backtest_eth_dalton_rule2_balance_edge_trailing_gridsearch_20260831.py's structure, signal-specific
bits swapped in. horizon_bars for purging/trade-cap reuses each signal's own confirmed metalabel
HORIZON (8/12), matching how dalton reused its own 30 -- not an independent choice.

Runs locally (CPU only, no GPU needed).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    HOLDOUT_START,
    OOS_START,
    VAL_START,
    build_fires,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

OUT_DIR = ROOT / "tmp/eth_kalman_demarker_trailing_gridsearch_20260831"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001  # 10bp round-trip, this lineage's standard cost

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5},
}


def log(msg: str) -> None:
    print(f"[kalman_demarker_trailing_gridsearch] {msg}", flush=True)


def gate_signal(name: str, klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
                 trigger_bottom: pd.Series, extremeness: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    cfg = SIGNAL_CONFIG[name]
    horizon = cfg["horizon"]
    fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols,
                        horizon, cfg["gap"], cfg["K"])
    fires = fires.sort_values("pos").reset_index(drop=True)  # decision_indices must be sorted
    log(f"\n=== {name} (H={horizon}, GAP={cfg['gap']}, K={cfg['K']}): {len(fires)} cluster-anchored candidates ===")

    open_px = klines["open"].to_numpy()
    high = klines["high"].to_numpy()
    low = klines["low"].to_numpy()
    close = klines["close"].to_numpy()
    ts_full = klines["timestamp"]
    atr_pct = ind["atr_pct"].to_numpy()

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    scores = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
    atr = atr_pct[decision_indices]

    eligible_val = purged_decision_mask(ts_full, start=VAL_START, end=OOS_START, horizon_bars=horizon)
    eligible_oos = purged_decision_mask(ts_full, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
    val_set = set(np.flatnonzero(eligible_val).tolist())
    oos_set = set(np.flatnonzero(eligible_oos).tolist())
    val_mask = np.array([d in val_set for d in decision_indices])
    oos_mask = np.array([d in oos_set for d in decision_indices])
    log(f"  decisions eligible: val={val_mask.sum()} oos={oos_mask.sum()}")

    tp_placeholder = np.full(len(fires), 999.0)
    results = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                sl_moves = sl * atr
                arm_moves = arm * atr
                trail_moves = trail * atr
                row = {"sl": sl, "arm": arm, "trail": trail}
                ok = True
                for wname, mask in (("val", val_mask), ("oos", oos_mask)):
                    result = simulate_single_position(
                        timestamps=ts_full, open_px=open_px, high=high, low=low, close=close,
                        decision_indices=decision_indices[mask], scores=scores[mask],
                        tp_moves=tp_placeholder[mask], sl_moves=sl_moves[mask],
                        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=horizon,
                        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                        arm_moves=arm_moves[mask], trail_moves=trail_moves[mask],
                    )
                    ledger = result.ledger
                    n_trades = int(len(ledger))
                    avg_bp = float(ledger["trade_return"].mean() * 10000) if n_trades else float("nan")
                    win_rate = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
                    row[f"{wname}_n"] = n_trades
                    row[f"{wname}_avg_bp"] = round(avg_bp, 3)
                    row[f"{wname}_win_rate"] = round(win_rate, 4)
                    if not (n_trades > 0 and avg_bp > 0):
                        ok = False
                row["both_positive"] = ok
                results.append(row)

    table = pd.DataFrame(results)
    table["signal"] = name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"{name}_gridsearch.csv", index=False)

    passing = table[table["both_positive"]].copy()
    log(f"  {len(passing)}/{len(table)} combos pass VAL+OOS both-positive")
    if len(passing):
        passing["min_bp"] = passing[["val_avg_bp", "oos_avg_bp"]].min(axis=1)
        passing = passing.sort_values("min_bp", ascending=False)
        log("  top 5 by min(val_bp, oos_bp):")
        for _, r in passing.head(5).iterrows():
            log(f"    SL={r['sl']} ARM={r['arm']} Trail={r['trail']}: "
                f"VAL={r['val_avg_bp']:+.2f}bp(n={int(r['val_n'])},win={r['val_win_rate']:.1%}) "
                f"OOS={r['oos_avg_bp']:+.2f}bp(n={int(r['oos_n'])},win={r['oos_win_rate']:.1%})")
    else:
        log("  best (still-failing) combos by min(val_bp, oos_bp):")
        table["min_bp"] = table[["val_avg_bp", "oos_avg_bp"]].min(axis=1)
        for _, r in table.sort_values("min_bp", ascending=False).head(5).iterrows():
            log(f"    SL={r['sl']} ARM={r['arm']} Trail={r['trail']}: "
                f"VAL={r['val_avg_bp']:+.2f}bp(n={int(r['val_n'])}) OOS={r['oos_avg_bp']:+.2f}bp(n={int(r['oos_n'])})")
    return table


def main() -> int:
    log("loading klines + Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    gate_signal("demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
               dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"])

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    gate_signal("kalman_deviation_meanrev", klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
               kalman_dev_z.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"])

    log(f"\nfull grids saved -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

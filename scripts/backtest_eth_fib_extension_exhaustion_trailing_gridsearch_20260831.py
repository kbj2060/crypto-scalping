#!/usr/bin/env python3
"""Trailing-stop cost-gate grid search for fib_extension_exhaustion (final label config
HORIZON=20(100min)/GAP=18/K=2.35/K_loss=4.70), using this repo's STANDARD backtest engine
(core.causal_futures_backtest.simulate_single_position, ATR-trailing-stop exit mode) -- same
engine/convention as every other Homer signal's cost-gate.

Per this project's default policy (feedback_trailing_stop_default_costgate_logic_20260830):
trailing stop is the FIRST exit structure tried. Tests ALL raw fires unconditional on the trained
model's own probability -- fib_extension_exhaustion uses plain single-K-pair labeling (the MFE/MAE
joint condition, not exclude-middle), so the existing classifier CSV already IS the full raw-fire
population (1,783 rows, hit=0/1 covers everything).

Note: this backtest engine's own trailing-stop exit only checks the FAVORABLE excursion path (a
resting SL a fixed distance behind the running peak) -- it does not need a separate "MAE cap" like
the classification label does, since a real trailing SL would already have exited a position long
before a -6xATR-style adverse move could develop (that is precisely what SL grid values like
2.0-4.0xATR test for). The label's MAE cap and this engine's SL are two independent, consistent
answers to the same underlying "big loss zone" concern raised by the user for this signal.

HOLDOUT (2026-04-01+) excluded entirely -- single-touch policy, not yet earned (separate from the
classification panel's own single HOLDOUT touch -- this is a distinct later exposure, same pattern
smt_divergence used).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position

FIRES_CSV = ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS = 20

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]


def log(msg: str) -> None:
    print(f"[fib_ext_trailing_gridsearch] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    return pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    klines = load_klines()
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    fires = fires.sort_values("pos").reset_index(drop=True)
    log(f"{len(klines)} klines, {len(fires)} candidate fires (excl. HOLDOUT)")

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0)
    atr = fires["atr_pct"].to_numpy()
    tp_moves_placeholder = np.full(len(fires), 999.0)

    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_BARS)
    eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON_BARS)
    val_set = set(np.flatnonzero(eligible_val).tolist())
    oos_set = set(np.flatnonzero(eligible_oos).tolist())
    val_mask = np.array([d in val_set for d in decision_indices])
    oos_mask = np.array([d in oos_set for d in decision_indices])
    log(f"VAL candidates={val_mask.sum()}  OOS candidates={oos_mask.sum()}")

    results = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                sl_moves = sl * atr
                arm_moves = arm * atr
                trail_moves = trail * atr
                row = {"sl": sl, "arm": arm, "trail": trail}
                ok = True
                for wname, mask in [("val", val_mask), ("oos", oos_mask)]:
                    result = simulate_single_position(
                        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
                        decision_indices=decision_indices[mask], scores=scores[mask],
                        tp_moves=tp_moves_placeholder[mask], sl_moves=sl_moves[mask],
                        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
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
    passing = table[table["both_positive"]].copy()
    passing["min_bp"] = passing[["val_avg_bp", "oos_avg_bp"]].min(axis=1)
    passing = passing.sort_values("min_bp", ascending=False)

    log(f"\n=== {len(passing)}/{len(table)} combos pass VAL AND OOS both positive (net of {ROUNDTRIP_COST_RATE*10000:.0f}bp cost) ===")
    for _, r in passing.head(15).iterrows():
        log(f"  SL={r['sl']:.2f} ARM={r['arm']:.2f} Trail={r['trail']:.2f}: "
            f"VAL n={int(r['val_n'])} avg={r['val_avg_bp']:+.2f}bp win={r['val_win_rate']:.3f}  "
            f"OOS n={int(r['oos_n'])} avg={r['oos_avg_bp']:+.2f}bp win={r['oos_win_rate']:.3f}")

    if len(passing) == 0:
        log("\nno combo passed both VAL and OOS -- top 10 by min(VAL,OOS) avg_bp regardless of sign:")
        table["min_bp"] = table[["val_avg_bp", "oos_avg_bp"]].min(axis=1)
        for _, r in table.sort_values("min_bp", ascending=False).head(10).iterrows():
            log(f"  SL={r['sl']:.2f} ARM={r['arm']:.2f} Trail={r['trail']:.2f}: "
                f"VAL n={int(r['val_n'])} avg={r['val_avg_bp']:+.2f}bp  OOS n={int(r['oos_n'])} avg={r['oos_avg_bp']:+.2f}bp")

    out_dir = ROOT / "tmp/eth_fib_extension_exhaustion_metalabel_tabpfn_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "trailing_gridsearch_results.csv", index=False)
    log(f"\nfull grid ({len(table)} combos) saved -> {out_dir / 'trailing_gridsearch_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

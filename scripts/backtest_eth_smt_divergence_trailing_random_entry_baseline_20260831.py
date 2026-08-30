#!/usr/bin/env python3
"""Random-entry economics baseline for smt_divergence's winning exit config (SL=4.0/ARM=2.0/
Trail=0.1xATR, HORIZON=72) -- proactive check per the orthogonal_combo deep-audit lesson (a tight
exit structure can inflate win rate on pure noise; avg bp is the metric that actually validates
edge). ARM=2.0 here is much larger than orthogonal_combo's 0.5, so a smaller inflation effect is
expected, but checked empirically rather than assumed.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path("/home/kbj20/crypto-scalping")
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines

MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS = 72
SL, ARM, TRAIL = 4.0, 2.0, 0.1
N_RANDOM = 2000
SEED = 20260831

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")


def log(msg): print(f"[smt_divergence_random_baseline] {msg}", flush=True)


def run_random(klines, atr_pct, eligible_idx, tag, rng):
    n = min(N_RANDOM, len(eligible_idx))
    decision_indices = np.sort(rng.choice(eligible_idx, size=n, replace=False))
    scores = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    atr = atr_pct[decision_indices]
    valid = np.isfinite(atr) & (atr > 0)
    decision_indices, scores, atr = decision_indices[valid], scores[valid], atr[valid]
    tp_placeholder = np.full(len(decision_indices), 999.0)
    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))
    result = simulate_single_position(
        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
        decision_indices=decision_indices, scores=scores,
        tp_moves=tp_placeholder, sl_moves=SL * atr,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=ARM * atr, trail_moves=TRAIL * atr,
    )
    ledger = result.ledger
    n_trades = int(len(ledger))
    avg_bp = float(ledger["trade_return"].mean() * 10000) if n_trades else float("nan")
    win_rate = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
    log(f"  [{tag}] candidates={len(decision_indices)} trades={n_trades} avg_trade={avg_bp:+.2f}bp win_rate={win_rate:.3f}")
    return {"tag": tag, "n_trades": n_trades, "avg_bp": avg_bp, "win_rate": win_rate}


def main():
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = klines["timestamp"]
    eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_BARS)
    eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON_BARS)
    val_idx = np.flatnonzero(eligible_val)
    oos_idx = np.flatnonzero(eligible_oos)
    rng = np.random.default_rng(SEED)
    val_r = run_random(klines, atr_pct, val_idx, "VAL random", rng)
    oos_r = run_random(klines, atr_pct, oos_idx, "OOS random", rng)
    log(f"\nREAL smt_divergence: VAL +7.00bp(win72.4%)  OOS +6.18bp(win69.6%)")
    log(f"RANDOM entries:      VAL {val_r['avg_bp']:+.2f}bp(win{val_r['win_rate']*100:.1f}%)  OOS {oos_r['avg_bp']:+.2f}bp(win{oos_r['win_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()

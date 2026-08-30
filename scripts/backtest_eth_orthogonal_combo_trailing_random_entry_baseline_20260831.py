#!/usr/bin/env python3
"""Second deep-audit baseline (companion to backtest_eth_orthogonal_combo_trailing_oscillator_only_
baseline_20260831.py, which showed the oscillator-only population ALSO gets ~95% win rate with
orthogonal_combo's winning exit config): does the SL=4.0/ARM=0.5/Trail=0.1xATR trailing stop look
profitable on genuinely RANDOM entries with no technical condition at all? If so, most of the
headline win-rate/bp is a property of this specific exit structure (tiny arm distance banks a small
win on almost any bar with ANY favorable wiggle) on ETH's general volatility/drift characteristics
over this period, not evidence of directional skill from any signal.

Uses numpy random with a fixed seed (reproducible) to sample 2000 random bar positions per window
(VAL/OOS), random long/short (50/50), sized off that bar's own real atr_pct (same Tier0 feature
used everywhere else) -- run through the IDENTICAL simulate_single_position call as the real
signal's grid-search winner.
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines

MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS = 24
SL, ARM, TRAIL = 4.0, 0.5, 0.1
N_RANDOM = 2000
SEED = 20260831

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")


def log(msg: str) -> None:
    print(f"[orthogonal_random_entry_baseline] {msg}", flush=True)


def run_random(klines: pd.DataFrame, atr_pct: np.ndarray, eligible_idx: np.ndarray, tag: str, rng: np.random.Generator) -> dict:
    n = min(N_RANDOM, len(eligible_idx))
    decision_indices = rng.choice(eligible_idx, size=n, replace=False)
    decision_indices = np.sort(decision_indices)
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
    log(f"  [{tag}] candidates={len(decision_indices)} trades={n_trades} skipped_while_open={result.skipped_while_open} "
        f"avg_trade={avg_bp:+.2f}bp win_rate={win_rate:.3f}")
    return {"tag": tag, "n_trades": n_trades, "avg_bp": avg_bp, "win_rate": win_rate}


def main() -> int:
    log("building indicator_frame for atr_pct...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = klines["timestamp"]

    eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_BARS)
    eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON_BARS)
    val_idx = np.flatnonzero(eligible_val)
    oos_idx = np.flatnonzero(eligible_oos)
    log(f"eligible bars: VAL={len(val_idx)} OOS={len(oos_idx)} -- sampling {N_RANDOM} random per window, seed={SEED}")

    rng = np.random.default_rng(SEED)
    val_r = run_random(klines, atr_pct, val_idx, "VAL (random entries)", rng)
    oos_r = run_random(klines, atr_pct, oos_idx, "OOS (random entries)", rng)

    log(f"\n=== comparison ===")
    log(f"  REAL orthogonal_combo:      VAL +9.36bp (win 91.5%)   OOS +15.13bp (win 96.0%)")
    log(f"  OSC-ONLY (no confirm):      VAL +6.47bp (win 94.6%)   OOS +7.85bp (win 95.2%)")
    log(f"  RANDOM entries (no signal): VAL {val_r['avg_bp']:+.2f}bp (win {val_r['win_rate']*100:.1f}%)   "
        f"OOS {oos_r['avg_bp']:+.2f}bp (win {oos_r['win_rate']*100:.1f}%)")

    out_dir = ROOT / "tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([val_r, oos_r]).to_csv(out_dir / "random_entry_baseline_economics.csv", index=False)
    log(f"saved -> {out_dir / 'random_entry_baseline_economics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

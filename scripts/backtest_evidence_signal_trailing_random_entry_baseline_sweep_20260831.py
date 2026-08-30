#!/usr/bin/env python3
"""Following orthogonal_combo's random-entry baseline finding (same exit structure on literally
random bars gets 83-85% win rate but NEGATIVE avg bp -- the high win rate is partly a property of
a tight-ARM trailing stop, not evidence of skill), user asked whether the same win-rate-inflation
effect applies to the OTHER 3 signals that passed their own trailing-stop cost-gate: taker_delta_
z_climax (ARM=1.5xATR), liquidity_sweep (ARM=2.0xATR), short_term_return_z (ARM=1.0xATR) -- all
have LARGER arm distances than orthogonal_combo's 0.5xATR, so the effect should be weaker, but
this checks it empirically rather than assuming from ARM size alone.

Same methodology as backtest_eth_orthogonal_combo_trailing_random_entry_baseline_20260831.py:
2000 random bar positions per VAL/OOS window, random long/short (50/50), sized off that bar's own
real atr_pct, run through the IDENTICAL simulate_single_position call as each signal's own
grid-search winner. No TabPFN/CUDA needed -- runs locally.
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
N_RANDOM = 2000
SEED = 20260831

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SIGNALS = {
    "taker_delta_z_climax": {"horizon": 24, "sl": 2.0, "arm": 1.5, "trail": 0.2,
                              "real": "VAL+OOS(combined) +8.68bp, HOLDOUT +2.17bp(win64.7%)"},
    "liquidity_sweep":      {"horizon": 30, "sl": 4.0, "arm": 2.0, "trail": 0.1,
                              "real": "VAL +10.70bp(win71.5%), OOS +14.49bp(win71.5%)"},
    "short_term_return_z":  {"horizon": 12, "sl": 2.0, "arm": 1.0, "trail": 0.2,
                              "real": "VAL +10.97bp(win76.5%), OOS +14.00bp(win79.5%)"},
}


def log(msg: str) -> None:
    print(f"[evidence_signal_random_baseline_sweep] {msg}", flush=True)


def run_random(klines: pd.DataFrame, atr_pct: np.ndarray, eligible_idx: np.ndarray, horizon: int,
               sl: float, arm: float, trail: float, tag: str, rng: np.random.Generator) -> dict:
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
        tp_moves=tp_placeholder, sl_moves=sl * atr,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=horizon,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=arm * atr, trail_moves=trail * atr,
    )
    ledger = result.ledger
    n_trades = int(len(ledger))
    avg_bp = float(ledger["trade_return"].mean() * 10000) if n_trades else float("nan")
    win_rate = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
    log(f"  [{tag}] candidates={len(decision_indices)} trades={n_trades} avg_trade={avg_bp:+.2f}bp win_rate={win_rate:.3f}")
    return {"tag": tag, "n_trades": n_trades, "avg_bp": avg_bp, "win_rate": win_rate}


def main() -> int:
    log("building indicator_frame for atr_pct (shared across all 3 signals)...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = klines["timestamp"]
    rng = np.random.default_rng(SEED)

    for name, cfg in SIGNALS.items():
        horizon = cfg["horizon"]
        eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=horizon)
        eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
        val_idx = np.flatnonzero(eligible_val)
        oos_idx = np.flatnonzero(eligible_oos)
        log(f"\n=== {name} (SL={cfg['sl']}/ARM={cfg['arm']}/Trail={cfg['trail']}xATR, HORIZON={horizon}) ===")
        val_r = run_random(klines, atr_pct, val_idx, horizon, cfg["sl"], cfg["arm"], cfg["trail"], "VAL random", rng)
        oos_r = run_random(klines, atr_pct, oos_idx, horizon, cfg["sl"], cfg["arm"], cfg["trail"], "OOS random", rng)
        log(f"  REAL signal: {cfg['real']}")
        log(f"  RANDOM     : VAL {val_r['avg_bp']:+.2f}bp(win{val_r['win_rate']*100:.1f}%)  OOS {oos_r['avg_bp']:+.2f}bp(win{oos_r['win_rate']*100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

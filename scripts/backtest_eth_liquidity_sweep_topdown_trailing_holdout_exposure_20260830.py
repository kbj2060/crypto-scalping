#!/usr/bin/env python3
"""SINGLE final HOLDOUT (2026-04-01+) exposure for the liquidity_sweep top/down trailing-stop
cost-gate -- final config SL=4.0xATR/ARM=2.0xATR/Trail=0.1xATR, picked from
backtest_eth_liquidity_sweep_topdown_trailing_gridsearch_20260830.py (91/96 combos passed VAL+OOS)
+ backtest_eth_liquidity_sweep_topdown_trailing_optimistic_crosscheck_20260830.py (robust to both
intrabar-ordering conventions, ~1-1.4bp spread). Companion to research_eth_liquidity_sweep_
topdown_metalabel_holdout_20260830.py's classification HOLDOUT touch, same day, same config --
together these are the ONE holdout exposure for this model; do not re-run with a different SL/ARM/
Trail after seeing this result.
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

FIRES_CSV = ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS = 30
SL, ARM, TRAIL = 4.0, 2.0, 0.1
HOLDOUT_START = pd.Timestamp("2026-04-01")


def log(msg: str) -> None:
    print(f"[liq_sweep_topdown_trailing_HOLDOUT] {msg}", flush=True)


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"]).sort_values("pos").reset_index(drop=True)

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0)
    atr = fires["atr_pct"].to_numpy()
    tp_placeholder = np.full(len(fires), 999.0)
    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    eligible = purged_decision_mask(ts, start=HOLDOUT_START, end=ts.max() + pd.Timedelta(minutes=5), horizon_bars=HORIZON_BARS)
    eligible_set = set(np.flatnonzero(eligible).tolist())
    mask = np.array([d in eligible_set for d in decision_indices])
    log(f"HOLDOUT (>={HOLDOUT_START.date()}) candidate fires: {mask.sum()}")

    result = simulate_single_position(
        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
        decision_indices=decision_indices[mask], scores=scores[mask],
        tp_moves=tp_placeholder[mask], sl_moves=(SL * atr)[mask],
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=(ARM * atr)[mask], trail_moves=(TRAIL * atr)[mask],
    )
    ledger = result.ledger
    n_trades = len(ledger)
    avg_bp = float(ledger["trade_return"].mean() * 10000) if n_trades else float("nan")
    win_rate = float((ledger["price_move"] > 0).mean()) if n_trades else float("nan")
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    log(f"SL={SL} ARM={ARM} Trail={TRAIL}: trades_taken={n_trades} skipped_while_open={result.skipped_while_open} "
        f"win_rate={win_rate:.3f} avg_trade={avg_bp:+.2f}bp total_account_return={total_return*100:+.3f}%")
    log(f"(for reference: VAL was +10.70bp/win71.5%, OOS was +14.49bp/win71.5% at this exact config)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

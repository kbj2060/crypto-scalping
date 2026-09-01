#!/usr/bin/env python3
"""Optimistic-intrabar-ordering cross-check for the top trailing-stop grid winners
(backtest_eth_smt_divergence_trailing_gridsearch_20260831.py: 71/96 combos passed VAL+OOS, top 4
all share Trail=0.10 -- the narrowest trail in the grid, exactly the case
feedback_intrabar_ordering_optimistic_pessimistic_bracket_20260830.md flags as needing this check
most). Same resolve_optimistic()/methodology as every other Homer signal's crosscheck script.
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

FIRES_CSV = ROOT / "data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
HORIZON_BARS = 72

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

CANDIDATES = [
    {"sl": 4.0, "arm": 2.0, "trail": 0.1},
    {"sl": 3.5, "arm": 2.0, "trail": 0.1},
    {"sl": 2.5, "arm": 2.0, "trail": 0.1},
    {"sl": 3.5, "arm": 1.5, "trail": 0.1},
]


def log(msg: str) -> None:
    print(f"[smt_divergence_optimistic_crosscheck] {msg}", flush=True)


def resolve_optimistic(side: int, entry: float, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        sl_init_move: float, arm_move: float, trail_move: float) -> tuple[float, str, int]:
    if side > 0:
        stop = entry * (1.0 - sl_init_move)
        peak = entry
        armed = False
        for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
            if bar_high > peak:
                peak = bar_high
                if not armed and (peak - entry) / entry >= arm_move:
                    armed = True
                if armed:
                    stop = max(stop, peak * (1.0 - trail_move))
            if bar_low <= stop:
                return stop / entry - 1.0, "trail_sl", offset
        return float(close[-1] / entry - 1.0), "timeout", len(close) - 1
    stop = entry * (1.0 + sl_init_move)
    peak = entry
    armed = False
    for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
        if bar_low < peak:
            peak = bar_low
            if not armed and (entry - peak) / entry >= arm_move:
                armed = True
            if armed:
                stop = min(stop, peak * (1.0 + trail_move))
        if bar_high >= stop:
            return 1.0 - stop / entry, "trail_sl", offset
    return float(1.0 - close[-1] / entry), "timeout", len(close) - 1


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    fires = fires.sort_values("pos").reset_index(drop=True)

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0)
    atr = fires["atr_pct"].to_numpy()
    tp_placeholder = np.full(len(fires), 999.0)
    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    eligible_val = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_BARS)
    eligible_oos = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON_BARS)
    val_set = set(np.flatnonzero(eligible_val).tolist())
    oos_set = set(np.flatnonzero(eligible_oos).tolist())
    val_mask = np.array([d in val_set for d in decision_indices])
    oos_mask = np.array([d in oos_set for d in decision_indices])

    ts_to_pos = {t: i for i, t in enumerate(ts)}
    atr_by_pos = dict(zip(decision_indices.tolist(), atr.tolist()))

    for cand in CANDIDATES:
        sl, arm, trail = cand["sl"], cand["arm"], cand["trail"]
        log(f"\n=== SL={sl} ARM={arm} Trail={trail} ===")
        for wname, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
            result = simulate_single_position(
                timestamps=ts, open_px=open_px, high=high, low=low, close=close,
                decision_indices=decision_indices[mask], scores=scores[mask],
                tp_moves=tp_placeholder[mask], sl_moves=(sl * atr)[mask],
                upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
                margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                arm_moves=(arm * atr)[mask], trail_moves=(trail * atr)[mask],
            )
            ledger = result.ledger
            pess_avg_bp = float(ledger["trade_return"].mean() * 10000) if len(ledger) else float("nan")
            pess_win = float((ledger["price_move"] > 0).mean()) if len(ledger) else float("nan")

            opt_price_moves = []
            for _, row in ledger.iterrows():
                entry_i = ts_to_pos[row["entry_timestamp"]]
                i = entry_i - 1
                side_sign = 1 if row["side"] > 0 else -1
                entry_px = float(open_px[entry_i])
                a = atr_by_pos[i]
                final_i = min(entry_i + HORIZON_BARS - 1, len(ts) - 1)
                fwd_high = high[entry_i:final_i + 1]
                fwd_low = low[entry_i:final_i + 1]
                fwd_close = close[entry_i:final_i + 1]
                pmove, _, _ = resolve_optimistic(side_sign, entry_px, fwd_high, fwd_low, fwd_close, sl * a, arm * a, trail * a)
                opt_price_moves.append(pmove)
            opt_price_moves = np.array(opt_price_moves)
            notional = MARGIN_FRACTION * LEVERAGE
            opt_trade_return = opt_price_moves * notional - ROUNDTRIP_COST_RATE
            opt_avg_bp = float(opt_trade_return.mean() * 10000)
            opt_win = float((opt_price_moves > 0).mean())

            log(f"  [{wname}] n={len(ledger)}  pessimistic(standard-engine)={pess_avg_bp:+.2f}bp (win={pess_win:.3f})  "
                f"optimistic={opt_avg_bp:+.2f}bp (win={opt_win:.3f})  diverge={opt_avg_bp - pess_avg_bp:+.2f}bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SINGLE-TOUCH HOLDOUT EXPOSURE for taker_delta_z_climax v5 (CLUSTER_GAP_MERGE=12), the LAST
holdout check for this signal's trailing-stop design per explicit user decision (2026-08-30):
v4 (gap=3) already used this exact HOLDOUT window once and FAILED (-0.98bp,
eth_taker_delta_climax_trailing_stop_costgate_breakthrough_20260830.md). v5 (gap=12) improved
VAL/OOS AUC (0.622/0.608 -> 0.633/0.645) and trailing-stop economics (+4.49 -> +8.68bp) using
ONLY VAL/OOS evidence to pick the gap. The user explicitly chose to spend this SAME HOLDOUT
window on v5 as a second and FINAL touch for this signal -- this script's result settles the
question either way; no v6/v7 follow-up variant is to be tried against this holdout regardless
of outcome (that would be exactly the search-until-it-passes pattern the single-touch policy
exists to prevent).

Same standard-engine methodology as the v4 holdout exposure script (backtest_taker_and_str_z_
trailing_stop_holdout_exposure_20260830.py) and the unchanged, already-validated exit config
(SL_init=2.0xATR, ARM=1.5xATR, Trail=0.2xATR, HORIZON=24 bars) -- ONLY the candidate population
(v5's gap=12 fires instead of v4's gap=3 fires) differs.
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

FIRES_CSV = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_v5_gap12_20260830/eth_5m_taker_delta_climax_metalabel_v5_gap12_features.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001

HOLDOUT_START = pd.Timestamp("2026-04-01")
HORIZON_BARS = 24
SL_INIT, ARM, TRAIL = 2.0, 1.5, 0.2  # unchanged from v4 -- only the candidate population differs


def log(msg: str) -> None:
    print(f"[HOLDOUT_EXPOSURE_v5] {msg}", flush=True)


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    holdout_end = ts.max() + pd.Timedelta(minutes=5)
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] >= HOLDOUT_START].sort_values("pos").reset_index(drop=True)
    log(f"v5 (gap=12) HOLDOUT window: [{HOLDOUT_START}, {ts.max()}] -- {len(fires)} candidate fires "
        f"(v4/gap=3 had 1,465 in the same window)")

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0)
    sl_moves = SL_INIT * fires["atr_pct"].to_numpy()
    arm_moves = ARM * fires["atr_pct"].to_numpy()
    trail_moves = TRAIL * fires["atr_pct"].to_numpy()
    tp_moves = np.zeros(len(fires))

    eligible = purged_decision_mask(ts, start=HOLDOUT_START, end=holdout_end, horizon_bars=HORIZON_BARS)
    eligible_set = set(np.flatnonzero(eligible).tolist())
    mask = np.array([d in eligible_set for d in decision_indices])

    result = simulate_single_position(
        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
        decision_indices=decision_indices[mask], scores=scores[mask],
        tp_moves=tp_moves[mask], sl_moves=sl_moves[mask],
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        arm_moves=arm_moves[mask], trail_moves=trail_moves[mask],
    )
    ledger = result.ledger
    n_candidates = int(mask.sum())
    n_trades = int(len(ledger))
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    if n_trades:
        win_rate = float((ledger["price_move"] > 0).mean())
        avg_trade_bp = float(ledger["trade_return"].mean() * 10000)
        avg_bars_held = float(ledger["bars_held"].mean())
    else:
        win_rate = avg_trade_bp = avg_bars_held = float("nan")
    log(f"candidates={n_candidates} trades_taken={n_trades} skipped_while_open={result.skipped_while_open} "
        f"win_rate={win_rate:.3f} avg_trade={avg_trade_bp:+.2f}bp avg_bars_held={avg_bars_held:.1f} "
        f"total_account_return={total_return*100:+.3f}%")
    log(f"reference: v4(gap=3) HOLDOUT was win_rate=60.8% avg_trade=-0.98bp; VAL/OOS(gap=12,this design) was avg_trade=+8.68bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SINGLE-TOUCH HOLDOUT EXPOSURE for the taker_delta_z_climax / short_term_return_z trailing-stop
designs. Per this repo's Fresh-Forward Rule (CLAUDE.md) and this thread's own discipline
(HOLDOUT untouched throughout the scratchpad discovery, the dual optimistic/pessimistic
verification, and the standard-engine VAL/OOS confirmation in
scripts/backtest_taker_and_str_z_trailing_stop_standard_engine_20260830.py), this is the FIRST
and ONLY intended exposure of HOLDOUT (2026-04-01 ~ latest available, 2026-08-28) for these two
already-finalized designs:
  - taker_delta_z_climax: SL_init=2.0xATR, ARM=1.5xATR, Trail=0.2xATR, HORIZON=24 bars (2h)
  - short_term_return_z:  SL_init=2.0xATR, ARM=1.0xATR, Trail=0.2xATR, HORIZON=12 bars (1h)
Neither config is re-tuned here or chosen based on this result -- this script exists ONLY to
report the number, using the exact same standard-engine methodology (core.causal_futures_backtest,
entry=next-bar open, single-position discipline, mark-to-market) already used for VAL/OOS.

This is deliberately a SEPARATE script from the reusable VAL/OOS validator (which continues to
exclude HOLDOUT by design, so future signals can reuse it without risk of an accidental touch).
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

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001  # 10bp standard cost

HOLDOUT_START = pd.Timestamp("2026-04-01")

SIGNALS = {
    "taker_delta_z_climax": {
        "fires_csv": ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv",
        "horizon_bars": 24,
        "sl_init": 2.0, "arm": 1.5, "trail": 0.2,
    },
    "short_term_return_z": {
        "fires_csv": ROOT / "data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv",
        "horizon_bars": 12,
        "sl_init": 2.0, "arm": 1.0, "trail": 0.2,
    },
}


def log(msg: str) -> None:
    print(f"[HOLDOUT_EXPOSURE] {msg}", flush=True)


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    holdout_end = ts.max() + pd.Timedelta(minutes=5)  # one bar past the last close, so purged_decision_mask's < end is inclusive of the true last bar
    log(f"HOLDOUT window: [{HOLDOUT_START}, {ts.max()}] ({int((ts >= HOLDOUT_START).sum())} bars)")
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    for name, cfg in SIGNALS.items():
        fires = pd.read_csv(cfg["fires_csv"], parse_dates=["timestamp"])
        fires = fires.loc[fires["timestamp"] >= HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        decision_indices = fires["pos"].to_numpy(dtype=np.int64)
        is_long = (fires["side"] == "bottom").to_numpy()
        scores = np.where(is_long, 1.0, -1.0)
        sl_moves = cfg["sl_init"] * fires["atr_pct"].to_numpy()
        arm_moves = cfg["arm"] * fires["atr_pct"].to_numpy()
        trail_moves = cfg["trail"] * fires["atr_pct"].to_numpy()
        tp_moves = np.zeros(len(fires))

        eligible = purged_decision_mask(ts, start=HOLDOUT_START, end=holdout_end, horizon_bars=cfg["horizon_bars"])
        eligible_set = set(np.flatnonzero(eligible).tolist())
        mask = np.array([d in eligible_set for d in decision_indices])
        sub_idx, sub_scores = decision_indices[mask], scores[mask]
        sub_sl, sub_arm, sub_trail, sub_tp = sl_moves[mask], arm_moves[mask], trail_moves[mask], tp_moves[mask]

        result = simulate_single_position(
            timestamps=ts, open_px=open_px, high=high, low=low, close=close,
            decision_indices=sub_idx, scores=sub_scores, tp_moves=sub_tp, sl_moves=sub_sl,
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=cfg["horizon_bars"],
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
            arm_moves=sub_arm, trail_moves=sub_trail,
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
        log(f"=== {name} (SL={cfg['sl_init']} ARM={cfg['arm']} Trail={cfg['trail']}) HOLDOUT ===")
        log(f"  candidates={n_candidates} trades_taken={n_trades} skipped_while_open={result.skipped_while_open} "
            f"win_rate={win_rate:.3f} avg_trade={avg_trade_bp:+.2f}bp avg_bars_held={avg_bars_held:.1f} "
            f"total_account_return={total_return*100:+.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

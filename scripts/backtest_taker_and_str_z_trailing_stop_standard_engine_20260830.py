#!/usr/bin/env python3
"""Validate the taker_delta_z_climax / short_term_return_z trailing-stop cost-gate findings
(eth_taker_delta_climax_trailing_stop_costgate_breakthrough_20260830.md,
eth_short_term_return_z_trailing_stop_costgate_confirmed_20260830.md -- both discovered via a
scratchpad, one-fire-at-a-time custom simulation) through this repo's own STANDARD backtest
engine (core.causal_futures_backtest.simulate_single_position), which the scratchpad sims did
NOT use. This is a materially different, more rigorous test, not a re-run of the same thing:

  1. Entry timing: the standard engine enters at the OPEN of the bar AFTER the decision bar
     (its own long-standing convention, matches V_REBOUND's own precedent) -- NOT the fire bar's
     own close, which is what the scratchpad sims used. This adds one full bar of realistic
     execution delay.
  2. Single-position discipline: the engine refuses a new decision while a prior trade is still
     open (skipped_while_open) -- a portfolio-realistic constraint the scratchpad sims did not
     have (they scored every fire independently regardless of overlap).
  3. Mark-to-market equity curve + the repo's canonical futures sizing contract (CLAUDE.md):
     notional = margin_fraction * leverage, trade_return = price_move * notional - account_cost.

core.causal_futures_backtest.py was extended (2026-08-30) with an optional ATR-trailing-stop exit
mode (_resolve_trade_trailing, activated by passing arm_moves/trail_moves) -- the existing fixed
TP/SL path (V_REBOUND's own cost-gate script) is completely unchanged/backward compatible when
those two arguments are omitted (regression-tested).

HOLDOUT (2026-04-01~) is excluded entirely -- single-touch policy, not yet earned.
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
MARGIN_FRACTION = 0.30  # this repo's standard (CLAUDE.md Futures Risk Sizing Contract, matches
LEVERAGE = 3.0          # V_REBOUND's own cost-gate script verbatim)
ROUNDTRIP_COST_RATE = 0.001  # 10bp standard cost

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")  # excluded entirely -- single-touch policy

SIGNALS = {
    "taker_delta_z_climax": {
        "fires_csv": ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv",
        "horizon_bars": 24,
        "sl_init": 2.0, "arm": 1.5, "trail": 0.2,  # validated config, breakthrough memo
    },
    "short_term_return_z": {
        "fires_csv": ROOT / "data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv",
        "horizon_bars": 12,
        "sl_init": 2.0, "arm": 1.0, "trail": 0.2,  # validated config, confirmed memo
    },
}


def log(msg: str) -> None:
    print(f"[standard_engine_trailing] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def run_signal(name: str, cfg: dict, klines: pd.DataFrame) -> None:
    fires = pd.read_csv(cfg["fires_csv"], parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)  # holdout excluded entirely
    fires = fires.sort_values("pos").reset_index(drop=True)

    decision_indices = fires["pos"].to_numpy(dtype=np.int64)
    is_long = (fires["side"] == "bottom").to_numpy()
    scores = np.where(is_long, 1.0, -1.0)
    sl_moves = cfg["sl_init"] * fires["atr_pct"].to_numpy()
    arm_moves = cfg["arm"] * fires["atr_pct"].to_numpy()
    trail_moves = cfg["trail"] * fires["atr_pct"].to_numpy()
    tp_moves = np.zeros(len(fires))  # unused in trailing mode, must be finite

    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))

    log(f"\n=== {name} (SL={cfg['sl_init']} ARM={cfg['arm']} Trail={cfg['trail']}, horizon={cfg['horizon_bars']} bars) ===")
    log(f"total candidate fires (excl. HOLDOUT): {len(fires)}")

    windows = {"VAL": (VAL_START, OOS_START), "OOS": (OOS_START, HOLDOUT_START), "VAL+OOS": (VAL_START, HOLDOUT_START)}
    for wname, (start, end) in windows.items():
        eligible = purged_decision_mask(ts, start=start, end=end, horizon_bars=cfg["horizon_bars"])
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
        log(f"  [{wname}] candidates={n_candidates} trades_taken={n_trades} "
            f"skipped_while_open={result.skipped_while_open} win_rate={win_rate:.3f} "
            f"avg_trade={avg_trade_bp:+.2f}bp avg_bars_held={avg_bars_held:.1f} "
            f"total_account_return={total_return*100:+.3f}%")


def main() -> int:
    klines = load_klines()
    log(f"{len(klines)} klines loaded")
    for name, cfg in SIGNALS.items():
        run_signal(name, cfg, klines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Follow-up to test_omega4_6_1_macro_event_veto_20260706.py: the pure entry veto was a null
result (0 trades affected) because Omega4.6.1 trades so infrequently (~1/week) that entries never
land near a scheduled macro event. BUT average hold time is ~99h (up to 282h) -- so even though
entries don't land near events, many trades are HELD OPEN through one or more events during their
life. 11 of 25 trades in the extended Jan-Jun 2026 OOS overlap at least one event window during
their hold. This tests a HAIRCUT countermeasure instead of a veto: temporarily scale down notional
during the event window portion of an already-open trade's hold (not a new-entry filter), then
restore full notional afterward.

Approximation: TP/SL/exit-head triggers are unaffected (the contract fixes them as raw price-move
barriers independent of notional, so exit TIMING doesn't change). Only the REALIZED PnL changes: a
trade's aggregate price-move is decomposed into per-bar log-returns (side-adjusted) over its hold
window using the raw OHLC series, each bar weighted by a notional multiplier (1.0 normally, the
haircut scale during any bar inside an event window), then re-aggregated. This ignores any
transaction cost from dynamically resizing (a simplification for this exploratory test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "trading_bot_modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "trading_bot_modules"))

from omega5_live import Omega5LiveAdapter  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
VETO_PRE_MIN, VETO_POST_MIN = 30, 120  # window during which the position is haircut


def build_event_calendar(years: list[int]) -> list[tuple[str, pd.Timestamp]]:
    events = []
    for y in years:
        events.extend(Omega5LiveAdapter._macro_events_for_year(y))
    return events


def event_multiplier(ts: pd.Series, events: list[tuple[str, pd.Timestamp]], haircut: float) -> np.ndarray:
    mult = np.ones(len(ts), dtype=np.float64)
    for _, event_ts in events:
        start = event_ts - pd.Timedelta(minutes=VETO_PRE_MIN)
        end = event_ts + pd.Timedelta(minutes=VETO_POST_MIN)
        hit = (ts >= start) & (ts <= end)
        mult[hit.to_numpy()] = haircut
    return mult


def recompute_trade_return(price: pd.DataFrame, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, side: int,
                            events: list[tuple[str, pd.Timestamp]], haircut: float, orig_price_move: float) -> tuple[float, float, int]:
    window = price[(price["timestamp"] > entry_ts) & (price["timestamp"] <= exit_ts)].reset_index(drop=True)
    if window.empty:
        return orig_price_move, orig_price_move, 0
    close = window["close"].astype(float).to_numpy()
    prev = np.concatenate([[price.loc[price["timestamp"] <= entry_ts, "close"].astype(float).iloc[-1]], close[:-1]])
    bar_logret = np.log(close / prev) * side
    mult = event_multiplier(window["timestamp"], events, haircut)
    n_event_bars = int((mult < 1.0).sum())
    reconstructed_total = float(np.sum(bar_logret))  # sanity check vs orig_price_move (simple vs log return, close enough for short moves)
    haircut_total = float(np.sum(bar_logret * mult))
    # scale the original (fee/slip-consistent) price move by the ratio of haircut-weighted to
    # unweighted log-return sums, preserving the original move's fee/slip-adjusted magnitude
    if abs(reconstructed_total) > 1e-9:
        scaled_price_move = orig_price_move * (haircut_total / reconstructed_total)
    else:
        scaled_price_move = orig_price_move
    return scaled_price_move, reconstructed_total, n_event_bars


def summarize(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(returns)), "wr": float((returns > 0).mean()) if len(returns) else 0.0}


def main() -> int:
    events = build_event_calendar([2025, 2026, 2027])
    price = pd.read_csv(BASE_2026, usecols=["timestamp", "close"], low_memory=False)
    price["timestamp"] = pd.to_datetime(price["timestamp"])
    price = price.sort_values("timestamp").reset_index(drop=True)

    ledger = pd.read_csv(OUT_DIR / "combined_router_duration_gated_ledger_extended.csv")
    active = ledger[ledger["notional"].astype(float) > 1e-12].copy().reset_index(drop=True)
    active["entry_dt"] = pd.to_datetime(active["entry_timestamp"])
    active["exit_dt"] = pd.to_datetime(active["exit_timestamp"])

    baseline_returns = active["trade_return"].astype(float).to_numpy()
    print(f"baseline (no haircut): {summarize(baseline_returns)}", flush=True)

    for haircut in (0.5, 0.25, 0.0):
        new_returns = []
        affected = 0
        for _, row in active.iterrows():
            scaled_move, _, n_bars = recompute_trade_return(
                price, row["entry_dt"], row["exit_dt"], int(row["side"]), events, haircut,
                float(row["raw_exit_price_move"]),
            )
            if n_bars > 0:
                affected += 1
            # re-derive trade_return preserving the original net_per_notional-to-trade_return ratio
            move_ratio = (scaled_move / row["raw_exit_price_move"]) if abs(row["raw_exit_price_move"]) > 1e-12 else 1.0
            new_net = float(row["net_per_notional"]) * move_ratio
            new_returns.append(new_net * float(row["notional"]))
        new_returns = np.array(new_returns)
        m = summarize(new_returns)
        print(f"haircut={haircut:.2f} (scale to {haircut:.0%} notional during event window bars): "
              f"{affected}/{len(active)} trades touched -> pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% "
              f"trades={m['trades']} wr={m['wr']:.3f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Follow-up to test_omega4_6_1_macro_event_haircut_20260706.py. User's proposed rule: at T-30min
before a scheduled macro event, if an open position is currently PROFITABLE, force-close it now
(lock in the gain); if it's at a loss, leave it alone (let the model's own TP/SL/exit-head logic
keep running normally). Applied sequentially per trade across however many events fall inside its
hold window -- the first T-30 checkpoint where the position is in profit force-closes it; if it's
never in profit at any checkpoint, it just runs to its original exit unchanged.

Cost accounting mirrors build_omega_plus_t12_livepass_candidate_20260630.py::apply_max_hold_time_stop
exactly: reconstruct entry_price from the close price at entry_timestamp, compute the new raw
price move at the forced-exit bar, and preserve the ORIGINAL trade's fee/slip cost delta
(raw_exit_price_move - net_per_notional) so the new exit's cost basis is consistent with how the
rest of this project's replay accounts for costs.
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
PRE_CHECK_MIN = 30  # check unrealized PnL exactly this long before each event


def build_event_calendar(years: list[int]) -> list[pd.Timestamp]:
    events = []
    for y in years:
        events.extend(ts for _, ts in Omega5LiveAdapter._macro_events_for_year(y))
    return sorted(events)


def price_at_or_before(price: pd.DataFrame, ts: pd.Timestamp) -> float:
    sub = price[price["timestamp"] <= ts]
    if sub.empty:
        raise RuntimeError(f"no price data at/before {ts}")
    return float(sub["close"].iloc[-1])


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
    print(f"baseline (no rule): {summarize(baseline_returns)}", flush=True)

    new_returns = []
    locked_count = 0
    for _, row in active.iterrows():
        entry_ts, exit_ts, side = row["entry_dt"], row["exit_dt"], int(row["side"])
        entry_price = price_at_or_before(price, entry_ts)
        old_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        checkpoints = [e - pd.Timedelta(minutes=PRE_CHECK_MIN) for e in events
                       if entry_ts < e - pd.Timedelta(minutes=PRE_CHECK_MIN) < exit_ts]
        forced = False
        for cp in sorted(checkpoints):
            p_cp = price_at_or_before(price, cp)
            unrealized = side * (p_cp / entry_price - 1.0)
            if unrealized > 0.0:
                raw_move = side * (p_cp / entry_price - 1.0)
                net = raw_move - old_cost
                new_returns.append(net * float(row["notional"]))
                forced = True
                locked_count += 1
                break
        if not forced:
            new_returns.append(float(row["trade_return"]))

    new_returns = np.array(new_returns)
    m = summarize(new_returns)
    print(f"with rule (lock profit at T-{PRE_CHECK_MIN}min, let losers ride): "
          f"{locked_count}/{len(active)} trades force-closed early -> "
          f"pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f}", flush=True)

    # detail on which trades changed
    for i, row in active.iterrows():
        if abs(new_returns[i] - row["trade_return"]) > 1e-9:
            print(f"  CHANGED: entry={row['entry_timestamp']} orig_exit={row['exit_timestamp']} "
                  f"orig_ret={row['trade_return']*100:.2f}% -> locked_ret={new_returns[i]*100:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

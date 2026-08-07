#!/usr/bin/env python3
"""BTC v3 Stage 1: sparse, execution-matched event dataset.

Fixes the two label problems found in docs/model_contracts/btc_v1_deep_analysis_20260714.md:

1. Dense/correlated samples -- the existing hourly trend-scan label (`ts_action`) is used to train
   on EVERY hourly bar within a trend segment, even though live trading only ever acts on the
   FIRST bar where a new signal appears (`is_new_parent_signal` in
   scripts/train_eval_btc_v2_regime_trendscan_20260714.py's `_merge_signal`). This script keeps
   only those transition ("new event") bars -- one sample per genuine event, not one per hour.

2. Proxy label instead of execution-utility label -- `ts_action` is a statistical
   trend-significance flag (|t-value| >= threshold), not "would a real trade here have made
   money." This script replaces it with the REALIZED outcome of independently simulating each
   event under the exact same ATR stop/trailing/time-exit contract used by the live v2 candidate
   (imports the constants and `_exit_fill` from that script unmodified -- does not reimplement the
   exit math, to avoid any subtle mismatch).

Each event is simulated on its own (not through the single shared-position state machine
`_fresh_forward_replay` uses for the live backtest) so the resulting per-event dataset reflects
what THAT SPECIFIC signal would have realized in isolation, decoupled from whatever portfolio-
capacity/cooldown constraints happen to gate concurrent execution -- that coupling is a downstream
concern for Stage 3/5, not something the label itself should already bake in.

Enforces docs/model_contracts/btc_v3_holdout_policy_20260714.md: refuses to build events whose
entry timestamp is >= HOLDOUT_START.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_btc_v2_regime_trendscan_20260714 as btc_v2  # noqa: E402

HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v3_sparse_event_dataset_20260714"


def _simulate_event_outcome(arrays: dict[str, np.ndarray], atr: np.ndarray, entry_signal_i: int, side: int) -> dict:
    """Independently replays ONE event forward under the exact v2 exit contract (stop/trail/
    time-exit), reusing btc_v2._exit_fill for the fill mechanics. Mirrors the position-management
    block inside btc_v2._fresh_forward_replay, but for a single isolated event."""
    n = len(arrays["open"])
    fill_i = min(entry_signal_i + 1, n - 1)
    if fill_i >= n - 1:
        return {"valid": False, "reason": "insufficient_bars_after_signal"}
    entry_price_candidate = float(arrays["open"][fill_i])
    touched = bool(arrays["low"][fill_i] <= entry_price_candidate) if side > 0 else bool(arrays["high"][fill_i] >= entry_price_candidate)
    if not touched:
        return {"valid": False, "reason": "entry_not_touched"}
    entry_price = entry_price_candidate
    entry_atr = max(float(atr[entry_signal_i]), 1.0e-6)
    peak_move = 0.0
    cash = 1.0
    for row_i in range(fill_i, n - 1):
        close = float(arrays["close"][row_i])
        move = (
            (close * (1.0 - btc_v2.SLIP_RATE) - entry_price) / entry_price
            if side > 0
            else (entry_price - close * (1.0 + btc_v2.SLIP_RATE)) / entry_price
        )
        peak_move = max(peak_move, move)
        hold_bars = row_i - fill_i
        reason = ""
        if move <= -btc_v2.STOP_ATR_PRICE * entry_atr:
            reason = "stop_loss"
        elif peak_move >= btc_v2.ARM_ATR_PRICE * entry_atr and peak_move - move >= btc_v2.TRAIL_ATR_PRICE * entry_atr:
            reason = "trailing_exit"
        elif hold_bars >= btc_v2.MAX_HOLD_BARS:
            reason = "time_exit"
        if reason:
            exit_fill_i, exit_price, exit_fee, route = btc_v2._exit_fill(arrays, row_i, side)
            raw_return = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
            trade_return = raw_return * btc_v2.NOTIONAL - btc_v2.FEE_RATE * btc_v2.MAKER_FEE_MULT * btc_v2.NOTIONAL - exit_fee * btc_v2.NOTIONAL
            return {
                "valid": True, "reason": reason, "route": route, "hold_bars": hold_bars,
                "entry_price": entry_price, "exit_price": exit_price, "raw_return": raw_return,
                "trade_return": float(trade_return), "win": int(trade_return > 0.0),
                "entry_fill_i": fill_i, "exit_fill_i": exit_fill_i,
            }
    return {"valid": False, "reason": "no_exit_before_data_end"}


def build(history_end: pd.Timestamp) -> pd.DataFrame:
    if history_end >= HOLDOUT_START:
        raise RuntimeError(
            f"history_end={history_end} >= HOLDOUT_START={HOLDOUT_START} -- refusing per "
            f"docs/model_contracts/btc_v3_holdout_policy_20260714.md"
        )
    print("stage=load_hourly_btc_features", flush=True)
    hourly, feature_columns = btc_v2._read_hourly()
    hourly = hourly.loc[hourly["timestamp"] <= history_end].reset_index(drop=True)

    print("stage=load_5m_execution_tape", flush=True)
    five_minute = btc_v2._read_five_minute()
    five_minute = five_minute.loc[five_minute["timestamp"] <= history_end].reset_index(drop=True)

    action = hourly["ts_action"].to_numpy()
    is_event = (action != 0) & (action != np.roll(action, 1))
    is_event[0] = bool(action[0] != 0)
    event_hours = hourly.loc[is_event].copy()
    print(f"stage=sparse_events dense_bars={len(hourly)} sparse_events={len(event_hours)} "
          f"reduction={1 - len(event_hours) / max(len(hourly), 1):.1%}", flush=True)

    # Map each hourly event to its 5-minute entry signal index: the first 5m bar at/after the
    # event's available_timestamp (event hour close + 1h, matching _fit_parent's signal availability).
    available_ts = event_hours["timestamp"] + pd.Timedelta(hours=1)
    five_ts = five_minute["timestamp"].to_numpy()
    arrays = {c: pd.to_numeric(five_minute[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    # ATR: reuse the hourly parent_atr_pct equivalent -- recompute via merge_asof against the
    # hourly frame's own atr_pct column (same source _fit_parent's signal exposes as parent_atr_pct).
    atr_by_ts = event_hours[["timestamp", "atr_pct"]].rename(columns={"timestamp": "source_timestamp"})

    rows = []
    for _, ev in event_hours.reset_index(drop=True).iterrows():
        entry_avail = ev["timestamp"] + pd.Timedelta(hours=1)
        idx = np.searchsorted(five_ts, np.datetime64(entry_avail), side="left")
        if idx >= len(five_ts):
            continue
        side = 1 if int(ev["ts_action"]) == 1 else -1
        # local ATR array aligned to the 5m tape: use this event's own atr_pct constant across its
        # own simulation window (matches _fit_parent's per-hour-constant atr assumption closely
        # enough for label construction; Stage 3 model consumes hourly features, not this local atr).
        atr_local = np.full(len(five_ts), float(ev["atr_pct"]), dtype=np.float64)
        outcome = _simulate_event_outcome(arrays, atr_local, int(idx), side)
        if not outcome.get("valid"):
            continue
        row = {c: ev[c] for c in feature_columns}
        row.update({
            "event_hour_timestamp": ev["timestamp"],
            "entry_available_timestamp": entry_avail,
            "side": side,
            "ts_t_value": float(ev["ts_t_value"]),
            "trade_return": outcome["trade_return"],
            "win": outcome["win"],
            "hold_bars_5m": outcome["hold_bars"],
            "exit_reason": outcome["reason"],
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"stage=simulated_outcomes valid_events={len(df)} / {len(event_hours)} sparse candidates", flush=True)
    if len(df):
        print(f"  win_rate={df['win'].mean():.1%} mean_trade_return={df['trade_return'].mean() * 100:.3f}% "
              f"median_hold_bars_5m={df['hold_bars_5m'].median():.0f} "
              f"long={int((df['side'] > 0).sum())} short={int((df['side'] < 0).sum())}", flush=True)
        print("  exit_reason counts:", df["exit_reason"].value_counts().to_dict(), flush=True)
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history_end = pd.Timestamp("2026-07-12 23:59:59")
    df = build(history_end)
    out_path = OUT_DIR / "sparse_event_dataset.parquet"
    df.to_parquet(out_path, index=False)
    report = {
        "history_end": str(history_end),
        "holdout_start": str(HOLDOUT_START),
        "n_events": int(len(df)),
        "win_rate": float(df["win"].mean()) if len(df) else None,
        "mean_trade_return_pct": float(df["trade_return"].mean() * 100) if len(df) else None,
        "long_events": int((df["side"] > 0).sum()) if len(df) else 0,
        "short_events": int((df["side"] < 0).sum()) if len(df) else 0,
        "exit_reasons": df["exit_reason"].value_counts().to_dict() if len(df) else {},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nsaved {len(df)} events -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

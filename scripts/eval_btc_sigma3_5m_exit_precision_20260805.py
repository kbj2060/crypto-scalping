"""Cheapest possible test of the user's "1h entry + 5m exit" idea for BTC: hold the
Sigma3 1h entry decisions EXACTLY fixed (same tape, same quality threshold/ATR
barriers, same cooldown/max-hold), and change ONLY how exits are resolved -- from
Sigma3's original `run_variant` (which checks TP/SL against the 1h bar's OWN close
price, once per hour) to a 5m intrabar walk-forward (checks TP/SL against 5m
high/low touches, matching the `_reason_and_return` convention used everywhere
else in this project's triple-barrier labels).

This is deliberately the cheapest version of the idea before building any learned
5m exit model: it isolates "does exit RESOLUTION alone matter" from "does a learned
5m exit policy add value" -- Sigma3's own numbers are close to the pass gate
(OOS cost3 -3.88%, cost1 +7.34%, near breakeven), so a purely mechanical precision
fix is worth checking before any new training. See
docs/model_contracts/sigma3_1h_trendscan_20260705_contract.md for the frozen
baseline this reproduces, and scripts/replay_omega6_v2_variants_20260704.py for the
original 1h-close-only replay engine (imported, not reimplemented, for the
baseline reproduction).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

TAPE_PATH = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705/tape_ensemble.parquet"
FIVE_MIN_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"

# Sigma3's own frozen best config (qt0.7/p0/tp1.5/sl1.0), per
# docs/model_contracts/sigma3_1h_trendscan_20260705_contract.md
QUALITY_THRESHOLD = 0.70
PERSISTENCE_BARS = 0
TP_ATR_MULT, SL_ATR_MULT = 1.5, 1.0
COOLDOWN_BARS = 3  # 1h bars
MAX_HOLD_BARS_1H = 48  # 2 days
MAX_HOLD_MINUTES = MAX_HOLD_BARS_1H * 60
FIXED_MARGIN, FIXED_LEVERAGE = 0.30, 2.0

VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-06-30 23:59:59")
# Genuinely untouched per docs/model_contracts/sigma3_1h_trendscan_20260705_contract.md's own
# recommendation ("Reserve a NEW untouched window, e.g. 2026-07+, for the next one-shot") --
# 2026-03..06 was already consumed by the ORIGINAL frozen config's one-shot look, and this
# script's 5m-exit-resolution result on that window is informative but not a clean one-shot
# (same window already observed once). This slice was never scored by any prior run.
FRESH_START, FRESH_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-20 23:59:59")

FEE = v2.FEE
SLIP = v2.SLIP


def resolve_exit_5m(five_min: pd.DataFrame, entry_ts: pd.Timestamp, side: int, entry_price: float,
                     tp_move: float, sl_move: float, max_hold_minutes: int) -> tuple[str, float, pd.Timestamp]:
    """Walk forward through 5m bars from entry_ts, checking TP/SL touches via
    high/low (not just close), matching this project's standard barrier-touch
    convention. Returns (reason, exit_price, exit_timestamp)."""
    window = five_min[(five_min["timestamp"] > entry_ts) &
                       (five_min["timestamp"] <= entry_ts + pd.Timedelta(minutes=max_hold_minutes))]
    if window.empty:
        return "no_data", entry_price, entry_ts
    if side > 0:
        tp_level = entry_price * (1.0 + tp_move)
        sl_level = entry_price * (1.0 - sl_move)
    else:
        tp_level = entry_price * (1.0 - tp_move)
        sl_level = entry_price * (1.0 + sl_move)
    for _, bar in window.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        hit_sl = (lo <= sl_level) if side > 0 else (hi >= sl_level)
        hit_tp = (hi >= tp_level) if side > 0 else (lo <= tp_level)
        if hit_sl:
            return "stop_loss", sl_level, bar["timestamp"]
        if hit_tp:
            return "take_profit", tp_level, bar["timestamp"]
    last = window.iloc[-1]
    return "time_stop", float(last["close"]), last["timestamp"]


def run_5m_exit_variant(tape: pd.DataFrame, five_min: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp,
                         fee: float = FEE, slip: float = SLIP) -> dict:
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    open_ = sub["open"].to_numpy(dtype=np.float64)
    side_arr = sub["primary_side"].to_numpy(dtype=np.int64)
    conf_arr = sub["primary_confidence"].to_numpy(dtype=np.float64)
    margin_arr = sub["primary_route_margin"].to_numpy(dtype=np.float64)
    atr_arr = sub["atr_pct"].to_numpy(dtype=np.float64)
    ts_arr = sub["timestamp"].to_numpy()

    cash, peak, mdd = 1.0, 1.0, 0.0
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if i < cooldown_until:
            i += 1
            continue
        side = int(side_arr[i])
        if side == 0:
            i += 1
            continue
        atr = max(float(atr_arr[i]), 1e-6)
        tp_move, sl_move = TP_ATR_MULT * atr, SL_ATR_MULT * atr
        entry_ts = pd.Timestamp(ts_arr[i + 1])
        entry_price = float(open_[i + 1]) * (1.0 + slip if side > 0 else 1.0 - slip)
        notional = FIXED_MARGIN * FIXED_LEVERAGE
        entry_equity = cash
        cash -= cash * fee * notional

        reason, exit_px, exit_ts = resolve_exit_5m(five_min, entry_ts, side, entry_price, tp_move, sl_move, MAX_HOLD_MINUTES)
        exit_price = exit_px * (1.0 - slip if side > 0 else 1.0 + slip)
        raw_exit = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
        cash_before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= cash_before * fee * notional
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        win = cash > entry_equity
        trades.append({"entry_ts": entry_ts, "exit_ts": exit_ts, "side": side, "reason": reason,
                        "win": bool(win), "month": str(entry_ts)[:7]})

        # advance i to the 1h bar at/after exit_ts, then apply cooldown from there
        next_i = int(np.searchsorted(ts_arr, np.datetime64(exit_ts), side="right"))
        i = max(next_i, i + 1)
        cooldown_until = i + COOLDOWN_BARS

    wins = sum(1 for t in trades if t["win"])
    reasons = {r: sum(1 for t in trades if t["reason"] == r) for r in set(t["reason"] for t in trades)} if trades else {}
    return {
        "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": len(trades),
        "wr": float(wins / len(trades)) if trades else 0.0, "reasons": reasons,
        "months": len(set(t["month"] for t in trades)),
    }


def main():
    tape_raw = pd.read_parquet(TAPE_PATH)
    tape_raw["timestamp"] = pd.to_datetime(tape_raw["timestamp"])
    tape = v2.apply_quality_threshold(tape_raw, QUALITY_THRESHOLD)

    five_min = pd.read_parquet(FIVE_MIN_PATH, columns=["timestamp", "open", "high", "low", "close"])
    five_min["timestamp"] = pd.to_datetime(five_min["timestamp"])
    five_min = five_min.sort_values("timestamp").reset_index(drop=True)

    cfg = v2.VariantConfig(
        name="sigma3_baseline_repro", tp_mode="atr_scaled", tp_atr_mult=TP_ATR_MULT, sl_atr_mult=SL_ATR_MULT,
        sizing_mode="fixed", fixed_margin=FIXED_MARGIN, fixed_leverage=FIXED_LEVERAGE,
        cooldown_bars=COOLDOWN_BARS, quality_threshold=QUALITY_THRESHOLD, persistence_bars=PERSISTENCE_BARS,
        max_hold_bars=MAX_HOLD_BARS_1H, use_fallback=False,
    )

    for split_name, start, end in [("VAL", VAL_START, VAL_END), ("OOS", OOS_START, OOS_END),
                                    ("FRESH_2026-07", FRESH_START, FRESH_END)]:
        print(f"\n=== {split_name} ({start.date()} - {end.date()}) ===")

        baseline = v2.cost_stress(tape, cfg, start=start, end=end)
        b1, b3 = baseline["cost1"], baseline["cost3"]
        print(f"  [1h close-only baseline] cost1 pnl={b1['pnl']:7.2f}% mdd={b1['mdd']:7.2f}% "
              f"trades={b1['trades']:4d} wr={b1['wr']*100:5.1f}%  |  "
              f"cost3 pnl={b3['pnl']:7.2f}% mdd={b3['mdd']:7.2f}%")
        print(f"    reasons: {b1['reasons']}")

        r1 = run_5m_exit_variant(tape, five_min, start=start, end=end, fee=FEE, slip=SLIP)
        print(f"  [5m intrabar exit, cost1] pnl={r1['pnl']:7.2f}% mdd={r1['mdd']:7.2f}% "
              f"trades={r1['trades']:4d} wr={r1['wr']*100:5.1f}%")
        print(f"    reasons: {r1['reasons']}")

        r3 = run_5m_exit_variant(tape, five_min, start=start, end=end, fee=FEE * 3, slip=SLIP * 3)
        print(f"  [5m intrabar exit, cost3] pnl={r3['pnl']:7.2f}% mdd={r3['mdd']:7.2f}% "
              f"trades={r3['trades']:4d} wr={r3['wr']*100:5.1f}%")

        print(f"  delta cost1 (5m - 1h-close): {r1['pnl']-b1['pnl']:+.2f}pp   "
              f"delta cost3 (5m - 1h-close): {r3['pnl']-b3['pnl']:+.2f}pp")
        print(f"  PASS CHECK: cost1>0 and cost3>0 -> {r1['pnl']>0 and r3['pnl']>0}")


if __name__ == "__main__":
    main()

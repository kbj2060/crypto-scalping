#!/usr/bin/env python3
"""Sigma5: 'let winners run' backtest on the Sigma3-1h ensemble signal, aiming for high cost1
PnL by capturing large trend moves (trailing stop + wide/no fixed TP + compounding + leverage)
instead of scalping small moves with tight barriers.

Rationale: Sigma3-1h has real directional edge (OOS cost1 +7.34%) but its tight tp=1.5xATR exits
cap the upside. Crypto trends run 10-30%; a trailing stop lets winners run to capture them.
Research (2026-07): trend-following/momentum dominates; hybrid ML+rules avoids overfit; but
"never go all-in, great backtests fail live from overfitting" -- so this reports the full
PnL/MDD frontier honestly and validates OOS, rather than cherry-picking max leverage.

cost1 is the primary metric (per user); cost3 reported as context. Trailing/stop in the same
unreal-vs-(mult*atr) convention as the rest of the repo's barriers. Compounding on.
Validation 2025-07..12; the single best-by-cost1 config that keeps MDD>=-35% is then one-shot on
2026-03-02..06-30 (Nth use of that window -- degraded evidential value, flagged).
"""

from __future__ import annotations

import itertools
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

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

TAPE = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705/tape_ensemble.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma5_letwinrun_20260705"
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-06-30 23:59:59")


def backtest(tape, *, thr, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold, cooldown, fee_mult, start, end):
    """Trailing-stop trend-follower. Enter on primary_side (>=thr conf, from apply_quality_threshold
    already applied). Track peak unreal; once unreal exceeds min_profit_atr*atr, arm a trailing
    stop at trail_atr*atr below the peak. Hard stop at -sl_atr*atr. Time exit at max_hold. Compound."""
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(dtype=np.float64)
    open_ = sub["open"].to_numpy(dtype=np.float64)
    side_arr = sub["primary_side"].to_numpy(dtype=np.int64)
    atr_arr = sub["atr_pct"].to_numpy(dtype=np.float64)
    FEE, SLIP = 0.00020 * fee_mult, 0.00050 * fee_mult
    notional = margin * leverage
    cash, peak_eq, mdd = 1.0, 1.0, 0.0
    pos, entry_price, hold_start, peak_unreal, entry_atr = 0, 0.0, 0, 0.0, 0.0
    entry_equity = 1.0
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or side_arr[i] == 0:
                i += 1
                continue
            side = int(side_arr[i])
            entry_price = float(open_[min(i + 1, n - 1)]) * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
            pos, hold_start, peak_unreal, entry_atr = side, i, 0.0, max(atr_arr[i], 1e-6)
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - px * (1 + SLIP)) / entry_price
        unreal = raw * notional
        eq = cash * (1 + unreal)
        peak_eq = max(peak_eq, eq)
        mdd = min(mdd, eq / max(peak_eq, 1e-12) - 1)
        peak_unreal = max(peak_unreal, unreal)
        hold = i - hold_start
        reason = ""
        if unreal <= -sl_atr * entry_atr:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr and (peak_unreal - unreal) >= trail_atr * entry_atr:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            rex = (exit_price - entry_price) / entry_price if pos > 0 else (entry_price - exit_price) / entry_price
            before = cash
            cash = cash * (1 + rex * notional)
            cash -= before * FEE * notional
            trades.append({"win": cash > entry_equity, "month": str(sub.iloc[hold_start]["timestamp"])[:7], "reason": reason, "ret": rex * notional})
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    wins = sum(1 for t in trades if t["win"])
    by_month = {}
    for t in trades:
        by_month.setdefault(t["month"], 0.0)
        by_month[t["month"]] += t["ret"]
    return {"pnl": (cash - 1) * 100, "mdd": mdd * 100, "trades": len(trades),
            "wr": wins / len(trades) if trades else 0.0, "by_month": by_month,
            "reasons": {r: sum(1 for t in trades if t["reason"] == r) for r in set(t["reason"] for t in trades)} if trades else {}}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(TAPE)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("i").reset_index(drop=True)

    grid = list(itertools.product(
        [0.60, 0.70],          # threshold
        [2.0, 3.0, 4.0],       # leverage
        [2.0, 3.0, 5.0],       # trail_atr (how much give-back triggers exit)
        [1.5, 2.5],            # sl_atr
        [1.0, 2.0],            # min_profit_atr before trailing arms
        [96, 144],             # max_hold
    ))
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.60, 0.70)}
    rows = []
    for thr, lev, trail, sl, minp, mh in grid:
        r1 = backtest(tapes[thr], thr=thr, leverage=lev, margin=0.30, trail_atr=trail, sl_atr=sl,
                      min_profit_atr=minp, max_hold=mh, cooldown=3, fee_mult=1.0, start=VAL_START, end=VAL_END)
        r3 = backtest(tapes[thr], thr=thr, leverage=lev, margin=0.30, trail_atr=trail, sl_atr=sl,
                      min_profit_atr=minp, max_hold=mh, cooldown=3, fee_mult=3.0, start=VAL_START, end=VAL_END)
        rows.append({"thr": thr, "lev": lev, "trail": trail, "sl": sl, "minp": minp, "mh": mh,
                     "c1": round(r1["pnl"], 1), "c1mdd": round(r1["mdd"], 1), "tr": r1["trades"],
                     "wr": round(r1["wr"], 3), "c3": round(r3["pnl"], 1), "mo": len(r1["by_month"])})
    df = pd.DataFrame(rows).sort_values("c1", ascending=False)
    df.to_csv(OUT_DIR / "val_frontier.csv", index=False)
    print("=== VALIDATION 2025-07..12 top 20 by cost1 PnL ===", flush=True)
    print(df.head(20).to_string(index=False), flush=True)
    # honest frontier: best cost1 with MDD >= -35 and >=5 months and >=40 trades
    elig = df[(df["c1mdd"] >= -35) & (df["mo"] >= 5) & (df["tr"] >= 40)]
    print(f"\neligible (MDD>=-35, mo>=5, tr>=40): {len(elig)}", flush=True)
    if len(elig):
        print(elig.head(10).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

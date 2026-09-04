#!/usr/bin/env python3
"""v7b FIXED TP/SL + time-exit cost-gate test -- alternative exit structure to the trailing-stop
test (backtest_eth_sweep_v_rebound_v7b_trailing_costgate_20260830.py, result: 0/205 configs pass
VAL+OOS independently).

Motivation (user, 2026-08-30): the V_REBOUND label's OWN definition is bounded and short-horizon
-- FAST_BARS=6 (30min) is the window in which the favorable excursion must occur, LOOKAHEAD_BARS=12
(60min) is when giveback_ratio is checked (research_eth_sweep_v_rebound_label_v6_binary_20260830.py,
inherited unchanged by v7/v7b -- only the exclusion threshold changed, not the windows). A trailing
stop is built for open-ended trends where you don't know how far a favorable move will run; V_REBOUND
instead predicts a fast move that resolves (one way or another) within ~30-60min. Hypothesis: a
fixed TP/SL + time-exit structure that mirrors the label's own horizon may fit this phenomenon
better than trailing, independent of the classification-error explanation already confirmed twice
(v4 and v7b) for the trailing-stop failure.

Reuses the EXACT SAME candidate population as the trailing test (same 354 VAL+OOS v7b model calls,
proba>=0.5, same entry convention = next 5m bar's OPEN, same 10bp standard round-trip cost) so the
two exit structures are directly comparable apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_sweep_v_rebound_v7b_costgate_20260830/v7b_costgate_candidates.pkl"
STANDARD_COST_BP = 10.0
FAST_BARS = 6      # 30min -- label's own fast-window (favorable excursion must occur by here)
FULL_BARS = 12     # 60min -- label's own full lookahead (giveback checked by here)


def tp_reachability_diagnostic(df: pd.DataFrame) -> None:
    print("\n=== TP reachability diagnostic (within label's own FAST_BARS=6/30min and FULL_BARS=12/60min) ===")
    print(f"{'TP(xATR)':>9} | {'winners@30m':>12} {'winners@60m':>12} | {'all-called@30m':>15} {'all-called@60m':>15}")
    winners = df[df["label"] == 1]
    for tp_mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        def reach_rate(sub: pd.DataFrame, bars: int) -> float:
            hit = 0
            for _, row in sub.iterrows():
                atr, entry, side = row["atr"], row["entry_price"], row["side"]
                tp = entry + (tp_mult * atr if side == "long" else -tp_mult * atr)
                highs = np.array(row["fwd_high"][:bars])
                lows = np.array(row["fwd_low"][:bars])
                reached = (highs >= tp).any() if side == "long" else (lows <= tp).any()
                hit += int(reached)
            return hit / len(sub) if len(sub) else float("nan")

        w30, w60 = reach_rate(winners, FAST_BARS), reach_rate(winners, FULL_BARS)
        a30, a60 = reach_rate(df, FAST_BARS), reach_rate(df, FULL_BARS)
        print(f"{tp_mult:>9.2f} | {w30:>11.1%} {w60:>11.1%} | {a30:>14.1%} {a60:>14.1%}")


def simulate_fixed(row: pd.Series, tp_mult: float, sl_mult: float, horizon_bars: int,
                    pessimistic: bool) -> float:
    """Fixed TP/SL + time-exit at close of the last bar in horizon_bars if neither is hit.
    pessimistic=True resolves a same-bar TP+SL collision in favor of the SL (worst case);
    pessimistic=False resolves it in favor of the TP (best case) -- same dual-verification
    convention as the trailing-stop script's optimistic/pessimistic intrabar ordering."""
    atr = row["atr"]
    entry = row["entry_price"]
    side = row["side"]
    sign = 1.0 if side == "long" else -1.0
    tp = entry + sign * tp_mult * atr
    sl = entry - sign * sl_mult * atr
    highs, lows, closes = row["fwd_high"], row["fwd_low"], row["fwd_close"]

    for i in range(horizon_bars):
        h, l = highs[i], lows[i]
        tp_hit = (h >= tp) if side == "long" else (l <= tp)
        sl_hit = (l <= sl) if side == "long" else (h >= sl)
        if tp_hit and sl_hit:
            return sign * ((sl if pessimistic else tp) - entry) / entry
        if tp_hit:
            return sign * (tp - entry) / entry
        if sl_hit:
            return sign * (sl - entry) / entry
    return sign * (closes[horizon_bars - 1] - entry) / entry


def grid_search(df: pd.DataFrame) -> list:
    print("\n=== Fixed TP/SL + time-exit grid (VAL+OOS combined, bp net of 10bp standard cost) ===")
    print(f"{'TP':>5} {'SL':>5} {'Hzn(bar)':>8} | {'best(bp)':>9} {'worst(bp)':>9} {'diverge':>8}")
    results = []
    for tp in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        for sl in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
            for horizon in (6, 9, 12, 18, 24):
                best_moves = df.apply(lambda r: simulate_fixed(r, tp, sl, horizon, False), axis=1)
                worst_moves = df.apply(lambda r: simulate_fixed(r, tp, sl, horizon, True), axis=1)
                best_bp = (best_moves.mean() * 1e4) - STANDARD_COST_BP
                worst_bp = (worst_moves.mean() * 1e4) - STANDARD_COST_BP
                diverge = float((np.sign(best_moves) != np.sign(worst_moves)).mean())
                results.append((tp, sl, horizon, best_bp, worst_bp, diverge))
    results.sort(key=lambda x: min(x[3], x[4]), reverse=True)
    for tp, sl, horizon, best_bp, worst_bp, diverge in results[:20]:
        print(f"{tp:>5.2f} {sl:>5.2f} {horizon:>8d} | {best_bp:>9.2f} {worst_bp:>9.2f} {diverge:>7.1%}")
    return results


def split_report(df: pd.DataFrame, tp: float, sl: float, horizon: int) -> None:
    for split_name in ("val", "oos"):
        sub = df[df["split"] == split_name]
        best = sub.apply(lambda r: simulate_fixed(r, tp, sl, horizon, False), axis=1)
        worst = sub.apply(lambda r: simulate_fixed(r, tp, sl, horizon, True), axis=1)
        print(f"  {split_name}: n={len(sub)}  best={best.mean()*1e4-STANDARD_COST_BP:+.2f}bp  "
              f"worst={worst.mean()*1e4-STANDARD_COST_BP:+.2f}bp  win_rate={float((best>0).mean()):.1%}")


def count_independently_passing(df: pd.DataFrame, results: list) -> list:
    passing = []
    for tp, sl, horizon, _, _, _ in results:
        ok = True
        for split_name in ("val", "oos"):
            sub = df[df["split"] == split_name]
            best = sub.apply(lambda r: simulate_fixed(r, tp, sl, horizon, False), axis=1)
            worst = sub.apply(lambda r: simulate_fixed(r, tp, sl, horizon, True), axis=1)
            if not (best.mean() * 1e4 - STANDARD_COST_BP > 0 and worst.mean() * 1e4 - STANDARD_COST_BP > 0):
                ok = False
                break
        if ok:
            passing.append((tp, sl, horizon))
    return passing


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    print(f"candidates: {len(df)} (val={int((df['split']=='val').sum())}, oos={int((df['split']=='oos').sum())})")
    print(f"label==1 rate within called: {df['label'].mean():.4f}")

    tp_reachability_diagnostic(df)
    results = grid_search(df)

    print("\n=== Best combined config, VAL/OOS independently ===")
    tp, sl, horizon = results[0][0], results[0][1], results[0][2]
    print(f"config: TP={tp}xATR SL={sl}xATR Horizon={horizon}bars({horizon*5}min)")
    split_report(df, tp, sl, horizon)

    passing = count_independently_passing(df, results)
    print(f"\n=== Configs with VAL AND OOS both independently positive (both orderings): {len(passing)}/{len(results)} ===")
    for tp, sl, horizon in passing[:20]:
        print(f"  TP={tp} SL={sl} Horizon={horizon}bars({horizon*5}min)")
        split_report(df, tp, sl, horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

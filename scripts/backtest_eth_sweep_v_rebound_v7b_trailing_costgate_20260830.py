#!/usr/bin/env python3
"""v7b trailing-stop cost-gate test -- explicit follow-up mandated by memory
eth_v_rebound_trailing_stop_costgate_marginal_20260830 ("재라벨링이 확정되면 이 진단(SL-race
부터)을 새 라벨로 반드시 재실행해야 한다"). Same methodology as that prior (v4-based) attempt and
the taker_delta_z_climax/short_term_return_z breakthroughs: SL-race diagnosis first, then an ATR
trailing-stop grid search, on the model's OWN out-of-sample (VAL+OOS) V자반등 calls (proba>=0.5) --
NOT ground-truth labels, matching how a live deployment would actually decide entries.

Conventions (verbatim from the prior V_REBOUND memory, reused not reimplemented since that
memory documents them precisely): entry = NEXT 5m bar's OPEN, downside-sweep+call=LONG,
upside-sweep+call=SHORT, cost = 10bp standard round-trip (this repo's universal cost-gate bar).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_sweep_v_rebound_v7b_costgate_20260830/v7b_costgate_candidates.pkl"
STANDARD_COST_BP = 10.0
SL_RACE_WINDOW_BARS = 11  # ~60min from entry (entry is already 1 bar after the sweep bar)


def sl_race_diagnostic(df: pd.DataFrame) -> None:
    winners = df[df["label"] == 1]
    print(f"\n=== SL-race diagnostic (label==1 winners within the called population, n={len(winners)}) ===")
    for sl_mult in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        raced = 0
        for _, row in winners.iterrows():
            atr = row["atr"]
            entry = row["entry_price"]
            highs = np.array(row["fwd_high"][:SL_RACE_WINDOW_BARS])
            lows = np.array(row["fwd_low"][:SL_RACE_WINDOW_BARS])
            if row["side"] == "long":
                adverse = entry - sl_mult * atr
                hit = (lows <= adverse).any()
            else:
                adverse = entry + sl_mult * atr
                hit = (highs >= adverse).any()
            raced += int(hit)
        print(f"  SL={sl_mult:.1f}x: race-loss rate {raced/len(winners):.1%} ({raced}/{len(winners)})")


def simulate_trailing(row: pd.Series, sl_mult: float, arm_mult: float, trail_mult: float,
                       pessimistic: bool) -> float:
    """Returns exit price-move as a fraction of entry (signed so positive=profit for the trade's
    own side). pessimistic=True assumes, within any single bar where BOTH the stop level and a
    new favorable extreme could plausibly occur, the WORST-case ordering (stop hit first, no
    credit for that bar's favorable excursion) -- optimistic=False assumes the opposite (favorable
    move happens first, stop only checked after)."""
    atr = row["atr"]
    entry = row["entry_price"]
    side = row["side"]
    opens, highs, lows, closes = row["fwd_open"], row["fwd_high"], row["fwd_low"], row["fwd_close"]
    sign = 1.0 if side == "long" else -1.0

    stop = entry - sign * sl_mult * atr
    armed = False
    best = entry
    for o, h, l, c in zip(opens, highs, lows, closes):
        fav_extreme = h if side == "long" else l  # this bar's most-favorable price reached
        adv_extreme = l if side == "long" else h  # this bar's most-adverse price reached

        def stop_hit() -> bool:
            return (adv_extreme <= stop) if side == "long" else (adv_extreme >= stop)

        def update_trailing() -> None:
            nonlocal armed, stop, best
            if sign * (fav_extreme - best) > 0:
                best = fav_extreme
            if not armed and sign * (best - entry) >= arm_mult * atr:
                armed = True
            if armed:
                new_stop = best - sign * trail_mult * atr
                if sign * (new_stop - stop) > 0:
                    stop = new_stop

        if pessimistic:
            if stop_hit():
                return sign * (stop - entry) / entry
            update_trailing()
        else:
            update_trailing()
            if stop_hit():
                return sign * (stop - entry) / entry
    return sign * (closes[-1] - entry) / entry  # never stopped out in the buffer -- close at last available price


def grid_search(df: pd.DataFrame) -> None:
    print("\n=== Trailing-stop grid (VAL+OOS combined, bp net of 10bp standard cost) ===")
    print(f"{'SL':>5} {'ARM':>5} {'Trail':>6} | {'opt(bp)':>9} {'pess(bp)':>9} {'diverge':>8}")
    best = []
    for sl in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        for arm in (0.10, 0.25, 0.5, 0.75, 1.0, 1.5):
            for trail in (0.10, 0.15, 0.2, 0.3, 0.5):
                if arm >= sl:
                    continue
                opt_moves = df.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
                pess_moves = df.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
                opt_bp = (opt_moves.mean() * 1e4) - STANDARD_COST_BP
                pess_bp = (pess_moves.mean() * 1e4) - STANDARD_COST_BP
                diverge = float((np.sign(opt_moves) != np.sign(pess_moves)).mean())
                best.append((sl, arm, trail, opt_bp, pess_bp, diverge))
    best.sort(key=lambda x: min(x[3], x[4]), reverse=True)
    for sl, arm, trail, opt_bp, pess_bp, diverge in best[:15]:
        print(f"{sl:>5.2f} {arm:>5.2f} {trail:>6.2f} | {opt_bp:>9.2f} {pess_bp:>9.2f} {diverge:>7.1%}")
    return best


def split_report(df: pd.DataFrame, sl: float, arm: float, trail: float) -> None:
    for split_name in ("val", "oos"):
        sub = df[df["split"] == split_name]
        opt = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
        pess = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
        print(f"  {split_name}: n={len(sub)}  opt={opt.mean()*1e4-STANDARD_COST_BP:+.2f}bp  "
              f"pess={pess.mean()*1e4-STANDARD_COST_BP:+.2f}bp  win_rate={float((opt>0).mean()):.1%}")


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    print(f"candidates: {len(df)} (val={int((df['split']=='val').sum())}, oos={int((df['split']=='oos').sum())})")
    print(f"label==1 rate within called: {df['label'].mean():.4f}")

    sl_race_diagnostic(df)
    best = grid_search(df)

    print("\n=== Best config, VAL/OOS independently ===")
    sl, arm, trail = best[0][0], best[0][1], best[0][2]
    print(f"config: SL={sl} ARM={arm} Trail={trail}")
    split_report(df, sl, arm, trail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Charts the OOS event-driven SUPPORT snapshots where real lost to placebo (paired dwell
comparison) -- 2026-08-25, user asked to see them visually. This is the clearest, most consistent
loss in today's whole battery (OOS support pairWR 0.419/0.372 vs placebo, both buffers), so it's
the default pick absent further specification.

Reproduces the EXACT placebo draws research_eth_liquidation_map_dwell_duration_test_20260825.py's
main() used for OOS (rng seed 20260825+1, support processed before resistance in evaluate_dwell()'s
side loop) by replaying the identical draw sequence -- same snapshots, same iteration order, same
rng.choice(pool) call count per level, so the specific placebo prices drawn here are the same ones
that produced the already-reported 0.419/0.372 numbers, not a fresh independent sample.

Per losing snapshot (mean real dwell < mean placebo dwell across that t0's levels, buffer=0.5%),
plots the NEAREST real support level (support_levels[0], nearest-to-price-first) against its
PAIRED placebo draw (the first placebo draw at that snapshot, matched 1:1 to the nearest level in
draw order) -- one clean pair per chart rather than all up-to-6 levels, which would be unreadable.
Window: from touch-24h to whichever of (real break, placebo break, DWELL_CAP_HOURS) is later, so
both outcomes are fully visible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_dwell_duration_test_20260825 as dwell

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "research" / "liq_map_oos_support_loss_charts_20260825"
TRAIN_FRACTION = 0.8
SEED = 20260825
SIDE = "support"
BUF = 0.005
N_CHARTS = 6


def find_touch(df_n, lows, highs, t0, level_price, side):
    fwd_end = min(df_n, t0 + 1 + base.FORWARD_HOURS)
    for i in range(t0 + 1, fwd_end):
        if side == "support" and lows[i] <= level_price:
            return i
        if side == "resistance" and highs[i] >= level_price:
            return i
    return None


def main() -> None:
    df = base.load_hourly()
    closes, lows, highs = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(df)
    snapshots = ed.simulate(df)
    split_i = int(n * TRAIN_FRACTION)
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"OOS snapshots: {len(oos_snaps)}", flush=True)

    rng = np.random.default_rng(SEED + 1)  # matches dwell script main()'s OOS seed_off=1
    key = "support_levels"
    pool = np.array([lv["distance_pct"] for s in oos_snaps for lv in s[key]])

    records = []
    for s in oos_snaps:
        cp = s["current_price"]
        real_dwells, placebo_dwells = [], []
        pair0 = None  # (real_level_price, real_touch_i, real_dwell, real_broke,
                      #  placebo_price, placebo_touch_i, placebo_dwell, placebo_broke)
        for idx, lv in enumerate(s[key]):
            ti = find_touch(n, lows, highs, s["t0"], lv["price"], SIDE)
            r_entry = None
            if ti is not None:
                d, broke = dwell.dwell_bars(closes, ti, lv["price"], SIDE, BUF)
                real_dwells.append(d)
                r_entry = (lv["price"], ti, d, broke)
            pd_ = rng.choice(pool)
            pp = cp * (1 + pd_ / 100.0)
            ti2 = find_touch(n, lows, highs, s["t0"], pp, SIDE)
            p_entry = None
            if ti2 is not None:
                d2, broke2 = dwell.dwell_bars(closes, ti2, pp, SIDE, BUF)
                placebo_dwells.append(d2)
                p_entry = (pp, ti2, d2, broke2)
            if idx == 0 and r_entry is not None and p_entry is not None:
                pair0 = (*r_entry, *p_entry)
        if real_dwells and placebo_dwells and pair0 is not None:
            r_mean, p_mean = np.mean(real_dwells), np.mean(placebo_dwells)
            pair_margin = pair0[6] - pair0[2]  # placebo_dwell - real_dwell for the SAME displayed pair
            records.append({"t0": s["t0"], "real_mean": r_mean, "placebo_mean": p_mean,
                            "snapshot_margin": p_mean - r_mean, "pair_margin": pair_margin, "pair0": pair0})

    # Selection/ranking uses pair_margin (the nearest-level pair actually drawn on the chart), NOT
    # snapshot_margin (mean across all up-to-6 levels at that t0) -- using the aggregate here would
    # let a chart's title claim a gap that isn't the one visibly plotted (caught in the first draft:
    # a chart showing 0h vs 0h labeled "margin=30.5h", driven by OTHER levels at that snapshot the
    # chart never shows). snapshot_margin is still printed below for the aggregate win/loss tally,
    # since that's the number this whole line's paired-winrate results are actually built from.
    losses = sorted([r for r in records if r["pair_margin"] > 0], key=lambda r: -r["pair_margin"])
    print(f"nearest-level pairs with real<placebo (real lost): {len(losses)} / {len(records)} paired", flush=True)
    print(f"nearest-level pairs with real>placebo (real won):  {sum(1 for r in records if r['pair_margin']<0)}", flush=True)
    print(f"nearest-level pair ties: {sum(1 for r in records if r['pair_margin']==0)}", flush=True)
    print(f"(for reference, snapshot-level -- all up-to-6 levels averaged -- "
          f"lost:{sum(1 for r in records if r['snapshot_margin']>0)} "
          f"won:{sum(1 for r in records if r['snapshot_margin']<0)} "
          f"tie:{sum(1 for r in records if r['snapshot_margin']==0)})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = df["timestamp"]

    for rank, rec in enumerate(losses[:N_CHARTS], start=1):
        t0 = rec["t0"]
        rp, rti, rd, rbroke, pp, pti, pd_, pbroke = rec["pair0"]
        span_start = max(0, min(t0, rti, pti) - 24)
        last_event = max(rti + rd, pti + pd_)
        span_end = min(n - 1, last_event + 6)
        window = df.iloc[span_start:span_end + 1]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        w = pd.Timedelta(hours=0.35)
        for _, row in window.iterrows():
            color = "#2e7d32" if row["close"] >= row["open"] else "#c62828"
            ax.plot([row["timestamp"], row["timestamp"]], [row["low"], row["high"]], color=color, linewidth=0.8)
            ax.add_patch(plt.Rectangle(
                (mdates.date2num(row["timestamp"] - w), min(row["open"], row["close"])),
                mdates.date2num(row["timestamp"] + w) - mdates.date2num(row["timestamp"] - w),
                max(abs(row["close"] - row["open"]), 1e-6),
                color=color))

        t0_ts = ts.iloc[t0]
        rti_ts, pti_ts = ts.iloc[rti], ts.iloc[pti]
        r_end_ts = ts.iloc[min(n - 1, rti + rd)]
        p_end_ts = ts.iloc[min(n - 1, pti + pd_)]

        ax.axvline(t0_ts, color="gray", linestyle=":", linewidth=1, label=f"as-of t0")
        ax.hlines(rp, rti_ts, r_end_ts, color="#1565c0", linewidth=2.2,
                  label=f"REAL support ${rp:,.2f} (dwell={rd}h, {'broke' if rbroke else 'censored'})")
        ax.hlines(pp, pti_ts, p_end_ts, color="#ef6c00", linewidth=2.2, linestyle="--",
                  label=f"PLACEBO ${pp:,.2f} (dwell={pd_}h, {'broke' if pbroke else 'censored'})")
        ax.scatter([rti_ts], [rp], color="#1565c0", marker="^", s=70, zorder=5)
        ax.scatter([pti_ts], [pp], color="#ef6c00", marker="^", s=70, zorder=5)
        if rbroke:
            ax.scatter([r_end_ts], [rp], color="#1565c0", marker="x", s=90, zorder=5)
        if pbroke:
            ax.scatter([p_end_ts], [pp], color="#ef6c00", marker="x", s=90, zorder=5)

        ax.set_title(f"#{rank} OOS support loss -- t0={t0_ts:%Y-%m-%d %H:%M} UTC  "
                     f"real={rd}h vs placebo={pd_}h  (this pair's margin={rec['pair_margin']}h)")
        ax.set_ylabel("ETHUSDT")
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = OUT_DIR / f"loss_{rank:02d}_{t0_ts:%Y%m%d_%H%M}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"wrote {out_path}", flush=True)

    print(f"\ncharts in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

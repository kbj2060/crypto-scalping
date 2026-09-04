#!/usr/bin/env python3
"""Does the LIVE liquidation map (v1, event-driven, scripts/live_liquidation_map_20260824.py::
compute_event_driven_levels) support/resist within a 1-HOUR horizon, 2026-08-25 follow-up.

User's exact framing: "나는 1시간 이내에 사고 팔기 때문에 1시간 내에 지지/저항 하는지만
테스트해주면 돼" -- every prior test in this line (today's dwell-filter/MIN_FLOOR_HOURS sweep/
Osler/volume-concentration, and 08-24's original backtest) scored hold-vs-break over
FOLLOWTHROUGH_HOURS=24 (a full day) after a touch -- far longer than this user's actual holding
period. A level could fail that 24h bar but still have genuinely held for the ~1h window a
scalper/day-trader actually cares about, or vice versa. This has never been tested at 1h before.

Same simulation (ed.simulate(), unmodified -- level generation doesn't depend on FOLLOWTHROUGH_
HOURS) and same TRAIN(80%)/OOS(20%) split-by-t0 convention as today's v2 A/B
(research_eth_liquidation_map_v2_cohort_ab_backtest_20260825.py). Only the scoring window changes.

MONKEYPATCHING NOTE (same pattern/justification as the original backtest script's own docstring):
base.evaluate_forward() reads FOLLOWTHROUGH_HOURS as a module global, not a function argument, so
this script monkeypatches base.FOLLOWTHROUGH_HOURS=1 before scoring rather than editing the shipped
module. Verified by reading evaluate_forward()'s body first (not guessed): the break-check window
is closes[touch_i : touch_i+1+FOLLOWTHROUGH_HOURS] -- i.e. inclusive of the touch bar's own close --
so FOLLOWTHROUGH_HOURS=1 checks the touch bar's close AND the next bar's close only, a correct
translation of "holds within 1 hour of being touched" at this dataset's 1h bar resolution (this
whole research line uses hourly klines; 1 hour is the finest granularity expressible here, not an
approximation of something finer).

evaluate_1h() below is copied from ed.evaluate() (not called-with-a-new-arg, since ed.evaluate()
has no followthrough parameter either) with exactly one change: the magnitude horizon dict
{24: [], 72: []} -> {1: []}, since 24h/72h-after-touch price moves aren't what this user asked
about. Touch/hold/break classification, placebo construction, paired-winrate, and agg() are
untouched copies -- see ed.evaluate()'s own docstring/comments for their rationale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap  # noqa: F401 (parity import w/ ed/base)
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_1h_holdrate_test_20260825.json"
TRAIN_FRACTION = 0.8
SEED = 20260825

base.FOLLOWTHROUGH_HOURS = 1  # see MONKEYPATCHING NOTE above


def evaluate_1h(df: pd.DataFrame, snapshots: list[dict], rng: np.random.Generator) -> dict:
    closes, lows, highs = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(df)

    def find_touch(t0, level_price, side):
        fwd_end = min(n, t0 + 1 + base.FORWARD_HOURS)
        for i in range(t0 + 1, fwd_end):
            if side == "support" and lows[i] <= level_price:
                return i
            if side == "resistance" and highs[i] >= level_price:
                return i
        return None

    def favorable_return(level_price, touch_i, side, k):
        j = min(n - 1, touch_i + k)
        p = closes[j]
        return (p - level_price) / level_price if side == "support" else (level_price - p) / level_price

    out = {}
    for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
        pool = np.array([lv["distance_pct"] for s in snapshots for lv in s[key]])
        if not len(pool):
            pool = np.array([2.0, -2.0])
        real_rows, placebo_rows = [], []
        real_ret, placebo_ret = {1: []}, {1: []}  # only change vs ed.evaluate(): {24,72} -> {1}
        for s in snapshots:
            cp = s["current_price"]
            for lv in s[key]:
                for buf in base.BUFFER_PCTS:
                    real_rows.append((buf, s["t0"], base.evaluate_forward(df, s["t0"], lv["price"], side, buf)))
                ti = find_touch(s["t0"], lv["price"], side)
                if ti is not None:
                    for h in real_ret:
                        real_ret[h].append(favorable_return(lv["price"], ti, side, h))
                pd_ = rng.choice(pool)
                pp = cp * (1 + pd_ / 100.0)
                for buf in base.BUFFER_PCTS:
                    placebo_rows.append((buf, s["t0"], base.evaluate_forward(df, s["t0"], pp, side, buf)))
                ti2 = find_touch(s["t0"], pp, side)
                if ti2 is not None:
                    for h in placebo_ret:
                        placebo_ret[h].append(favorable_return(pp, ti2, side, h))

        def agg(rows, buf):
            sub = [o for b, _, o in rows if b == buf]
            n_ = len(sub)
            touched = [o for o in sub if o != "not_touched"]
            nt = len(touched)
            nh = sum(1 for o in touched if o == "hold")
            return {"n": n_, "touch_rate": nt / n_ if n_ else None, "n_touched": nt,
                    "hold_rate": nh / nt if nt else None}

        def paired(buf):
            by_r, by_p = {}, {}
            for b, t0, o in real_rows:
                if b == buf and o != "not_touched":
                    by_r.setdefault(t0, []).append(1 if o == "hold" else 0)
            for b, t0, o in placebo_rows:
                if b == buf and o != "not_touched":
                    by_p.setdefault(t0, []).append(1 if o == "hold" else 0)
            fr = fp = tie = 0
            for t0 in set(by_r) & set(by_p):
                r, p = np.mean(by_r[t0]), np.mean(by_p[t0])
                fr += r > p
                fp += r < p
                tie += r == p
            return {"n_favor_real": fr, "n_favor_placebo": fp, "n_tie": tie,
                    "winrate": fr / (fr + fp) if (fr + fp) else None}

        out[side] = {
            "by_buffer": {str(buf): {"real": agg(real_rows, buf), "placebo": agg(placebo_rows, buf), "paired": paired(buf)}
                          for buf in base.BUFFER_PCTS},
            "magnitude": {
                str(h): {"n": len(real_ret[h]), "mean_real_pct": float(np.mean(real_ret[h]) * 100) if real_ret[h] else None,
                         "median_real_pct": float(np.median(real_ret[h]) * 100) if real_ret[h] else None,
                         "mean_placebo_pct": float(np.mean(placebo_ret[h]) * 100) if placebo_ret[h] else None,
                         "median_placebo_pct": float(np.median(placebo_ret[h]) * 100) if placebo_ret[h] else None,
                         "mean_diff_pct": float((np.mean(real_ret[h]) - np.mean(placebo_ret[h])) * 100) if real_ret[h] and placebo_ret[h] else None}
                for h in real_ret
            },
        }
    return out


def summarize(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": evaluate_1h(df, snaps, rng)}


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print(f"FOLLOWTHROUGH_HOURS (monkeypatched)={base.FOLLOWTHROUGH_HOURS}, "
          f"FORWARD_HOURS (touch window, unchanged)={base.FORWARD_HOURS}", flush=True)

    snapshots = ed.simulate(df)  # unmodified -- level generation doesn't depend on FOLLOWTHROUGH_HOURS
    print(f"snapshots: {len(snapshots)}", flush=True)

    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    train_snaps = [s for s in snapshots if s["t0"] < split_i]
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"split at bar {split_i} ({df['timestamp'].iloc[split_i]}) -- "
          f"TRAIN snapshots={len(train_snaps)}, OOS snapshots={len(oos_snaps)}", flush=True)

    results = [summarize("TRAIN", train_snaps, df, 0), summarize("OOS", oos_snaps, df, 1)]

    print(f"\n{'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'favor_r':7s} {'favor_p':7s} "
          f"{'holdR':7s} {'holdP':7s} {'touchR':7s} {'mag1h diff':11s} {'nTouch':6s}")
    for r in results:
        for side in ("support", "resistance"):
            d = r["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag1 = d["magnitude"]["1"]["mean_diff_pct"]
                print(f"{r['split']:6s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} "
                      f"{row['paired']['n_favor_real']:<7d} {row['paired']['n_favor_placebo']:<7d} "
                      f"{str(row['real']['hold_rate'])[:6]:7s} {str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{str(row['real']['touch_rate'])[:6]:7s} "
                      f"{('None' if mag1 is None else f'{mag1:+.4f}'):11s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "followthrough_hours": base.FOLLOWTHROUGH_HOURS, "forward_hours": base.FORWARD_HOURS,
        "n_bars": n, "split_bar": split_i, "split_ts": str(df["timestamp"].iloc[split_i]),
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

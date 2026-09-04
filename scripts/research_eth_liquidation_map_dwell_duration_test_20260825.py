#!/usr/bin/env python3
"""Redefines "does it support/resist" as a DURATION, not a binary hold/break-at-a-fixed-window --
2026-08-25 follow-up to the 1h-followthrough test. User's own judgment logic, verbatim: "지지/저항
한다는게 몇 개 봉동안 그 라인에 유지하는가" (whether it supports/resists = how many bars it holds
that line for). Every prior test in this line (24h, then 1h followthrough) asked "is it still
unbroken at exactly window W" -- one point on a curve. This asks for the whole curve: starting at
the touch bar, how many consecutive bars does price stay on the correct side of the level (closes
not crossing it by buffer_pct) before the level is actually broken -- right-censored at DWELL_CAP_
HOURS if it's never broken within that window. A level sliced straight through scores dwell=0; a
level that holds for many bars before eventually giving way scores high even though it "loses"
eventually -- which is the whole point of the reframing (a hold/break-at-1h verdict would have
scored a level that broke at bar 2 the same "break" as one that broke at bar 90).

Reuses, unmodified: ed.simulate() (v1 event-driven level generation -- same live formula, zero
parameter search here, per the "TRAIN was never fitting anything" clarification), find_touch's
touch definition (intrabar wick, most lenient) and the distance-matched placebo pool construction
(both copied from ed.evaluate(), same reasoning as that function's own comments), base.load_hourly()
for price, and the TRAIN(80%)/OOS(20%)-by-t0 split convention used all day.

Reporting style deliberately matches the base backtest script's own documented choice: no p-values
(paired dependency across overlapping lookback windows + a ~2-5 year single-asset sample don't
support one), just a paired per-snapshot sign count (did the real level out-dwell its paired
placebo?) alongside survival fractions at short horizons (1/3/6/12 bars = 1-12h, since that's what
actually matters for a trader whose full holding period is ~1h) and the two groups' dwell
distributions (median/mean, censoring rate).
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
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_dwell_duration_test_20260825.json"
TRAIN_FRACTION = 0.8
SEED = 20260825
DWELL_CAP_HOURS = base.FORWARD_HOURS  # 72 -- reuse the existing touch-window constant rather than
                                      # invent a new cap; generous relative to the 1-12h horizons
                                      # actually reported
SURVIVAL_CHECKPOINTS = (1, 3, 6, 12, 24, 48)  # bars after touch = hours (1h bars)


def dwell_bars(closes: np.ndarray, touch_i: int, level_price: float, side: str, buffer_pct: float) -> tuple[int, bool]:
    """Bars from the touch bar (inclusive, offset 0) until the first CLOSE breaks the level by
    buffer_pct. Returns (dwell, broke) -- broke=False means censored at DWELL_CAP_HOURS (or data
    end) without ever breaking, so dwell there is a lower bound, not an exact survival time.
    offset 0 = the touch bar's OWN close already breaks -> dwell=0, a level sliced through
    instantly with zero bars of actual defense (matches evaluate_forward()'s convention of
    including the touch bar itself in the break-check window)."""
    n = len(closes)
    end = min(n, touch_i + 1 + DWELL_CAP_HOURS)
    for i in range(touch_i, end):
        if side == "support" and closes[i] < level_price * (1 - buffer_pct):
            return i - touch_i, True
        if side == "resistance" and closes[i] > level_price * (1 + buffer_pct):
            return i - touch_i, True
    return end - touch_i, False


def evaluate_dwell(df: pd.DataFrame, snapshots: list[dict], rng: np.random.Generator) -> dict:
    closes, lows, highs = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(df)

    def find_touch(t0, level_price, side):  # verbatim from ed.evaluate()
        fwd_end = min(n, t0 + 1 + base.FORWARD_HOURS)
        for i in range(t0 + 1, fwd_end):
            if side == "support" and lows[i] <= level_price:
                return i
            if side == "resistance" and highs[i] >= level_price:
                return i
        return None

    out = {}
    for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
        pool = np.array([lv["distance_pct"] for s in snapshots for lv in s[key]])
        if not len(pool):
            pool = np.array([2.0, -2.0])

        real_by_buf = {b: [] for b in base.BUFFER_PCTS}    # list of (t0, dwell, broke)
        placebo_by_buf = {b: [] for b in base.BUFFER_PCTS}

        for s in snapshots:
            cp = s["current_price"]
            for lv in s[key]:
                ti = find_touch(s["t0"], lv["price"], side)
                if ti is not None:
                    for buf in base.BUFFER_PCTS:
                        d, broke = dwell_bars(closes, ti, lv["price"], side, buf)
                        real_by_buf[buf].append((s["t0"], d, broke))
                pd_ = rng.choice(pool)
                pp = cp * (1 + pd_ / 100.0)
                ti2 = find_touch(s["t0"], pp, side)
                if ti2 is not None:
                    for buf in base.BUFFER_PCTS:
                        d, broke = dwell_bars(closes, ti2, pp, side, buf)
                        placebo_by_buf[buf].append((s["t0"], d, broke))

        def summarize(rows: list[tuple]) -> dict:
            if not rows:
                return {"n": 0}
            dwells = np.array([d for _, d, _ in rows])
            broke = np.array([b for _, _, b in rows])
            return {
                "n": len(rows), "mean_dwell": float(dwells.mean()), "median_dwell": float(np.median(dwells)),
                "censored_pct": float((~broke).mean() * 100),
                "survival_pct": {str(k): float((dwells >= k).mean() * 100) for k in SURVIVAL_CHECKPOINTS},
            }

        def paired(real_rows: list, placebo_rows: list) -> dict:
            by_r: dict[int, list[int]] = {}
            by_p: dict[int, list[int]] = {}
            for t0, d, _ in real_rows:
                by_r.setdefault(t0, []).append(d)
            for t0, d, _ in placebo_rows:
                by_p.setdefault(t0, []).append(d)
            fr = fp = tie = 0
            for t0 in set(by_r) & set(by_p):
                r, p = np.mean(by_r[t0]), np.mean(by_p[t0])
                fr += r > p
                fp += r < p
                tie += r == p
            return {"n_favor_real": fr, "n_favor_placebo": fp, "n_tie": tie,
                    "winrate": fr / (fr + fp) if (fr + fp) else None}

        out[side] = {
            str(buf): {
                "real": summarize(real_by_buf[buf]),
                "placebo": summarize(placebo_by_buf[buf]),
                "paired_outdwell": paired(real_by_buf[buf], placebo_by_buf[buf]),
            }
            for buf in base.BUFFER_PCTS
        }
    return out


def summarize_split(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": evaluate_dwell(df, snaps, rng)}


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print(f"DWELL_CAP_HOURS={DWELL_CAP_HOURS}, survival checkpoints={SURVIVAL_CHECKPOINTS}", flush=True)

    snapshots = ed.simulate(df)
    print(f"snapshots: {len(snapshots)}", flush=True)

    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    train_snaps = [s for s in snapshots if s["t0"] < split_i]
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"split at bar {split_i} ({df['timestamp'].iloc[split_i]}) -- "
          f"TRAIN={len(train_snaps)}, OOS={len(oos_snaps)}", flush=True)

    results = [summarize_split("TRAIN", train_snaps, df, 0), summarize_split("OOS", oos_snaps, df, 1)]

    for r in results:
        print(f"\n{'='*100}\n{r['split']} (n_snapshots={r['n_snapshots']})\n{'='*100}")
        for side in ("support", "resistance"):
            for buf in ("0.005", "0.001"):
                d = r["eval"][side][buf]
                rr, pp, pw = d["real"], d["placebo"], d["paired_outdwell"]
                print(f"\n[{side} buf={float(buf)*100:.1f}%]")
                print(f"  real:    n={rr.get('n',0):4d} mean_dwell={rr.get('mean_dwell',float('nan')):5.2f}h "
                      f"median={rr.get('median_dwell',float('nan')):4.1f}h censored={rr.get('censored_pct',float('nan')):5.1f}%")
                print(f"  placebo: n={pp.get('n',0):4d} mean_dwell={pp.get('mean_dwell',float('nan')):5.2f}h "
                      f"median={pp.get('median_dwell',float('nan')):4.1f}h censored={pp.get('censored_pct',float('nan')):5.1f}%")
                if rr.get("n"):
                    surv_r = "  ".join(f"{k}h:{rr['survival_pct'][k]:.0f}%" for k in map(str, SURVIVAL_CHECKPOINTS))
                    surv_p = "  ".join(f"{k}h:{pp['survival_pct'][k]:.0f}%" for k in map(str, SURVIVAL_CHECKPOINTS))
                    print(f"  survival% real:    {surv_r}")
                    print(f"  survival% placebo: {surv_p}")
                print(f"  paired out-dwell winrate: {pw['winrate']} ({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "dwell_cap_hours": DWELL_CAP_HOURS, "n_bars": n, "split_bar": split_i,
        "split_ts": str(df["timestamp"].iloc[split_i]), "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

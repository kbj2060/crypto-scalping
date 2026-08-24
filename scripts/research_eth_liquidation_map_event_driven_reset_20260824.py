#!/usr/bin/env python3
"""Event-driven (price-triggered) support/resistance vs the fixed-time-lookback approach used all
session, 2026-08-24 5th follow-up. User's idea: "지지/저항이 뚫리면 청산된 것이니 그 때마다
옮기는 것 -- 시간 변동이 아니라 가격 변동으로 만들면?" (when support/resistance breaks, that IS a
liquidation, so regenerate it then -- build it from PRICE change, not TIME change).

Cautionary prior art checked before building this: [[eth_infotime_sampling_ab_closed_20260817]]
tested "information-time" (dollar/volume bars) resampling for DIRECTION prediction and found zero
effect on skill/economics -- the lesson there was "resampling the clock alone doesn't manufacture
information that isn't there." That is a different mechanism and target (bar-resampling for a
direction model) than this (an event-triggered LOOKBACK BOUNDARY for level selection), so it does
not block this idea, but it is the right base rate to keep in mind: don't expect the windowing
change alone to transform a small, fragile edge into a large one.

=== Mechanism (state machine, walk-forward, causal) ===
Each side (support/resistance) carries its own "reset point" -- the bar index of the last time
that side's active level set was broken. Levels are NOT recomputed on a fixed clock; they are
regenerated ONLY when a break occurs (a bar's CLOSE crosses an active level by more than
BREAK_TOLERANCE_PCT, the same close-based "invalidation" convention as every prior round), using
compute_raw_bins()/levels_from_bins() (the same production formula, unmodified) on the window
[since the side's PREVIOUS reset, current bar] -- i.e. the window length is exactly "how long
since this side last broke", clamped to [MIN_FLOOR_HOURS, MAX_LOOKBACK_HOURS] so a rapid double
-break doesn't produce a degenerate 1-bar window and a long calm stretch doesn't produce an
unbounded one. Between breaks, the level set is frozen (this is the literal "옮기는 건 뚫릴 때만"
reading, and also the cheap path -- recompute happens only at break events, not every bar).

Evaluated at the SAME spaced as-of points (every FORWARD_HOURS) and with the SAME touch/hold/break
+ placebo + magnitude machinery as every prior round this session, so results sit next to the
1d/7d/merge tables directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_event_driven_reset_20260824.json"

BREAK_TOLERANCE_PCT = 0.005  # close-based break trigger, matches the 0.5% round's user-requested tolerance
DRIFT_TOLERANCE_PCT = 0.10   # 2nd, complementary price trigger -- see the abandonment-pathology
                              # comment at its use site. 10% chosen so it fires well before distances
                              # reach the 20-70%+ range observed in the pre-fix run, still generous
                              # enough not to fire on ordinary day-to-day chop (median level distance
                              # elsewhere this session has been ~3-5%)
MIN_FLOOR_HOURS = 24         # a since-last-reset window shorter than this is extended backward to this floor
# 2026-08-24: first run used 30d here and got stuck -- once a reset produces a wide (near-30d) level
# set, wide levels rarely get touched, so resets stop happening and the window stays capped at 30d
# indefinitely (support reset only 49x, resistance 7x, in 4.7 YEARS; median staleness 16,000h =
# ~1.8 years). That is exactly the 30/45/90d regime this session's own backtests already showed has
# no edge -- capping the worst case there defeats the point. Capped at the one horizon with a
# demonstrated edge instead (7d, see eth_dashboard_liquidation_map_sr_backtest_20260824) so even a
# "stuck" stretch degrades to something already validated, not to the wide/dead regime.
MAX_LOOKBACK_HOURS = 24 * 7  # 7 days
BOOTSTRAP_HOURS = 24 * 7      # seed window before the walk-forward state machine can rely on its own history
SEED = 20260824


def simulate(df: pd.DataFrame) -> list[dict]:
    """Walk forward bar-by-bar from BOOTSTRAP_HOURS to the end. Returns a list of snapshots at
    each evaluation as-of point: {t0, current_price, support_levels, resistance_levels,
    support_window_hours, resistance_window_hours, n_support_resets, n_resistance_resets}."""
    n = len(df)
    close = df["close"].to_numpy()
    support_reset_idx = 0
    resistance_reset_idx = 0
    support_levels: list[dict] = []
    resistance_levels: list[dict] = []
    n_support_resets = 0
    n_resistance_resets = 0

    def regenerate(side: str, reset_idx: int, i: int) -> list[dict]:
        start = max(reset_idx, i - MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - MIN_FLOOR_HOURS))
        window = df.iloc[start:i + 1]
        cp = float(close[i])
        raw = liqmap.compute_raw_bins(window, cp)
        if raw is None:
            return []
        bins, bin_width, _, _ = raw
        key = "support_levels" if side == "support" else "resistance_levels"
        return liqmap.levels_from_bins(bins, bin_width, cp)[key]

    # Bootstrap: seed both sides from the first BOOTSTRAP_HOURS bars.
    support_levels = regenerate("support", 0, BOOTSTRAP_HOURS)
    resistance_levels = regenerate("resistance", 0, BOOTSTRAP_HOURS)

    eval_idxs = set(base.asof_indices(n, BOOTSTRAP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS))
    snapshots = []
    n_support_drift_resets = 0
    n_resistance_drift_resets = 0
    for i in range(BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_support = any(price < lv["price"] * (1 - BREAK_TOLERANCE_PCT) for lv in support_levels)
        broke_resistance = any(price > lv["price"] * (1 + BREAK_TOLERANCE_PCT) for lv in resistance_levels)
        # DRIFT trigger, added after the first run found a real pathology, not a bug: once support
        # anchors at an extreme (e.g. a bear-market low) and price never revisits it, "reset only on
        # break" means it NEVER refreshes again -- confirmed empirically, a support set at ~$900 in
        # 2022-08 sat frozen, un-refreshed, for the rest of the 4.7y dataset because price never fell
        # back to it (49 resets total, then zero for the remaining ~35,000 hours). Still a PRICE
        # trigger (how far price has moved from the current nearest level), not a clock -- just a
        # second, complementary price-based condition alongside "broke".
        drift_support = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > DRIFT_TOLERANCE_PCT
        drift_resistance = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > DRIFT_TOLERANCE_PCT

        if broke_support or drift_support:
            support_levels = regenerate("support", support_reset_idx, i)
            support_reset_idx = i
            n_support_resets += 1
            n_support_drift_resets += int(drift_support and not broke_support)
        if broke_resistance or drift_resistance:
            resistance_levels = regenerate("resistance", resistance_reset_idx, i)
            resistance_reset_idx = i
            n_resistance_resets += 1
            n_resistance_drift_resets += int(drift_resistance and not broke_resistance)

        if i in eval_idxs:
            snapshots.append({
                "t0": i, "current_price": float(price),
                "support_levels": support_levels, "resistance_levels": resistance_levels,
                "support_window_hours": i - support_reset_idx, "resistance_window_hours": i - resistance_reset_idx,
                "n_support_resets_so_far": n_support_resets, "n_resistance_resets_so_far": n_resistance_resets,
                "n_support_drift_resets_so_far": n_support_drift_resets,
                "n_resistance_drift_resets_so_far": n_resistance_drift_resets,
            })
    return snapshots


def evaluate(df: pd.DataFrame, snapshots: list[dict], rng: np.random.Generator) -> dict:
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
        real_ret, placebo_ret = {24: [], 72: []}, {24: [], 72: []}
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


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}", flush=True)
    snapshots = simulate(df)
    print(f"eval snapshots: {len(snapshots)}", flush=True)
    window_hours_support = [s["support_window_hours"] for s in snapshots]
    window_hours_resistance = [s["resistance_window_hours"] for s in snapshots]
    print(f"support window hours: median={np.median(window_hours_support):.0f} "
          f"p10={np.percentile(window_hours_support,10):.0f} p90={np.percentile(window_hours_support,90):.0f} "
          f"(vs fixed 24h/168h for 1d/7d)")
    print(f"resistance window hours: median={np.median(window_hours_resistance):.0f} "
          f"p10={np.percentile(window_hours_resistance,10):.0f} p90={np.percentile(window_hours_resistance,90):.0f}")
    print(f"total resets by end: support={snapshots[-1]['n_support_resets_so_far']} "
          f"(drift-triggered={snapshots[-1]['n_support_drift_resets_so_far']}), "
          f"resistance={snapshots[-1]['n_resistance_resets_so_far']} "
          f"(drift-triggered={snapshots[-1]['n_resistance_drift_resets_so_far']}) (over {len(df)} hours)")

    rng = np.random.default_rng(SEED)
    result = evaluate(df, snapshots, rng)

    print(f"\n{'side':11s} {'buf':6s} {'winrate':8s} {'favor_r':8s} {'favor_p':8s} {'hold_real':10s} {'hold_pb':8s}")
    for side, data in result.items():
        for buf, row in data["by_buffer"].items():
            p, r, pb = row["paired"], row["real"], row["placebo"]
            print(f"{side:11s} {float(buf)*100:5.1f}% {str(p['winrate'])[:6]:8s} "
                  f"{p['n_favor_real']:<8d} {p['n_favor_placebo']:<8d} "
                  f"{str(r['hold_rate'])[:8]:10s} {str(pb['hold_rate'])[:6]:8s}")
        print(f"{side:11s} magnitude: " + ", ".join(
            f"{h}h diff={data['magnitude'][h]['mean_diff_pct']:.3f}%" for h in data["magnitude"]))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_snapshots": len(snapshots), "support_window_hours_median": float(np.median(window_hours_support)),
        "resistance_window_hours_median": float(np.median(window_hours_resistance)), "eval": result,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

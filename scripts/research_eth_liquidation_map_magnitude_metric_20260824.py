#!/usr/bin/env python3
"""Magnitude metric alongside the win-rate metric used everywhere else this session, 2026-08-24
4th follow-up. User: "승률 말고 다른 척도가 있을까?" (a metric besides win-rate?) -- motivated by
the sibling session's finding that a real touch/hold win-rate edge did NOT survive conversion to
R-multiples in an actual strategy backtest (reversal magnitude too small). Win-rate only measures
FREQUENCY (did it hold more often than placebo); it says nothing about SIZE. This measures the
actual forward price move in the favorable direction after a touch, real vs placebo -- the same
"lift" idea this repo's evidence-signal scorecards use elsewhere (compare a continuous outcome,
not just a binary one), applied to the liquidation-map levels.

=== Metric ===
favorable_return(level_price, touch_i, side, k) = (close[touch_i+k] - level_price)/level_price for
support (positive = price recovered upward), or (level_price - close[touch_i+k])/level_price for
resistance (positive = price fell back down) -- k in {6, 24, 72} hours after the touch bar. Reused
episodes/levels/placebo-pool machinery from
scripts.research_eth_liquidation_map_1d7d_formula_merge_20260824 (build_episode, the same 5
configs' levels) so results sit on the identical episode set as that round's win-rate table.

Reports MEAN (what matters for expectancy/P&L -- pulled around by tail outcomes) alongside MEDIAN
(what matters for "the typical case" -- robust to tail outcomes) specifically because a mean/median
divergence is the direct fingerprint of the failure mode the sibling session found: a level that
"usually" produces a small favorable bounce (positive median, matching the win-rate edge) but
occasionally produces a large adverse break (dragging the mean down or negative) is exactly a
positive-win-rate / poor-expectancy signal -- not a contradiction, the mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.research_eth_liquidation_map_1d7d_formula_merge_20260824 as merge
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_magnitude_metric_20260824.json"
HORIZONS_HOURS = (6, 24, 72)
SEED = 20260824

CONFIGS = [
    ("1d_alone", "support", "support_levels"),
    ("7d_alone", "support", "support_levels"),
    ("lean_1d_70_30", "support", "support_levels"),
    ("1d_alone", "resistance", "resistance_levels"),
    ("7d_alone", "resistance", "resistance_levels"),
    ("equal_50_50", "resistance", "resistance_levels"),
]


def find_touch(lows: np.ndarray, highs: np.ndarray, n: int, t0: int, level_price: float, side: str) -> int | None:
    fwd_end = min(n, t0 + 1 + base.FORWARD_HOURS)
    for i in range(t0 + 1, fwd_end):
        if side == "support" and lows[i] <= level_price:
            return i
        if side == "resistance" and highs[i] >= level_price:
            return i
    return None


def favorable_return(closes: np.ndarray, n: int, level_price: float, touch_i: int, side: str, k: int) -> float:
    j = min(n - 1, touch_i + k)
    p = closes[j]
    return (p - level_price) / level_price if side == "support" else (level_price - p) / level_price


def collect(episodes: list[dict], closes, lows, highs, n, rng, cfg_name: str, side: str, key: str):
    dists = [lv["distance_pct"] for ep in episodes for lv in ep["levels"][cfg_name][key]]
    pool = np.array(dists) if dists else np.array([2.0, -2.0])
    real_by_h = {h: [] for h in HORIZONS_HOURS}
    placebo_by_h = {h: [] for h in HORIZONS_HOURS}
    for ep in episodes:
        cp = ep["current_price"]
        for lv in ep["levels"][cfg_name][key]:
            ti = find_touch(lows, highs, n, ep["t0"], lv["price"], side)
            if ti is not None:
                for h in HORIZONS_HOURS:
                    real_by_h[h].append(favorable_return(closes, n, lv["price"], ti, side, h))
            placebo_price = cp * (1 + rng.choice(pool) / 100.0)
            ti2 = find_touch(lows, highs, n, ep["t0"], placebo_price, side)
            if ti2 is not None:
                for h in HORIZONS_HOURS:
                    placebo_by_h[h].append(favorable_return(closes, n, placebo_price, ti2, side, h))
    return real_by_h, placebo_by_h


def main() -> None:
    df = base.load_hourly()
    idxs = base.asof_indices(len(df), merge.LOOKBACK_7D_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    episodes = [ep for t0 in idxs if (ep := merge.build_episode(df, t0)) is not None]
    closes, lows, highs = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(df)
    rng = np.random.default_rng(SEED)

    results = []
    print(f"{'cfg':16s} {'side':11s} {'h':4s} {'n':6s} {'mean_real%':11s} {'med_real%':10s} {'mean_pb%':10s} {'med_pb%':9s} {'mean_diff%':11s}")
    for cfg_name, side, key in CONFIGS:
        real, placebo = collect(episodes, closes, lows, highs, n, rng, cfg_name, side, key)
        for h in HORIZONS_HOURS:
            r, p = np.array(real[h]) * 100, np.array(placebo[h]) * 100
            row = {
                "cfg": cfg_name, "side": side, "horizon_hours": h, "n": len(r),
                "mean_real_pct": float(np.mean(r)), "median_real_pct": float(np.median(r)),
                "mean_placebo_pct": float(np.mean(p)), "median_placebo_pct": float(np.median(p)),
                "mean_diff_pct": float(np.mean(r) - np.mean(p)),
            }
            results.append(row)
            print(f"{cfg_name:16s} {side:11s} {h:3d}h {len(r):6d} {row['mean_real_pct']:10.3f}% "
                  f"{row['median_real_pct']:9.3f}% {row['mean_placebo_pct']:9.3f}% "
                  f"{row['median_placebo_pct']:8.3f}% {row['mean_diff_pct']:10.3f}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

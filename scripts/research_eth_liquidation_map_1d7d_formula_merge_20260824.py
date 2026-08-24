#!/usr/bin/env python3
"""Does MERGING the liquidation map's 1-day and 7-day density into one combined formula beat
either alone? 2026-08-24, 3rd follow-up on the 1/7/30/90d Coinglass-preset backtest. User's own
framing: "don't filter (confluence yes/no) -- overlay the two, or build a formula, to raise
accuracy." Also: same-day sibling session fresh-forward-backtested the liquidation-level-fade
TRADING FRAME as an actual strategy across 13 configs and closed it (all gross<=0; the one
validated component, 7d resistance, was statistically indistinguishable from a no-edge control
once converted to R-multiples -- see eth_discretionary_manual_strategy_codification_phase1_20260824).
User asked for this anyway, scoped as the pure event-level stat (not a strategy), so that's what
this measures -- a win-rate-vs-placebo comparison, not a P&L backtest, and a positive result here
would NOT by itself reopen that closed frame (same magnitude-vs-hold-rate gap that closed it could
still apply here).

=== What "merge" means here (as opposed to the confluence/filter approach explicitly ruled out) ===
compute_raw_bins() (scripts/live_liquidation_map_20260824.py, split out for exactly this) returns
the full per-price-bucket weight BEFORE the top-6/5%-floor filtering compute_liquidation_levels()
normally applies. Because bin_width depends only on current_price (identical for a 1d and a 7d
window computed at the same as-of point), the two windows' bucket indices sit on the SAME absolute
price grid and can be combined bucket-by-bucket -- a real formula over the two timeframes' evidence,
not a comparison of two pre-filtered output lists. Five merge formulas tested, all on
weight-normalized bins (each timeframe's own bins divided by its own max, so the timeframe with
more raw cumulative volume -- always 7d, having ~7x the candles -- doesn't just mechanically
dominate a raw sum):
    raw_sum          : bins_1d + bins_7d, NOT normalized (literal "overlay", as a baseline contrast)
    equal_50_50       : 0.5*norm_1d + 0.5*norm_7d
    lean_7d_30_70     : 0.3*norm_1d + 0.7*norm_7d  (7d gets more weight -- it's the one with a
                        validated resistance edge; does leaning on it help the merge more?)
    lean_1d_70_30     : 0.7*norm_1d + 0.3*norm_7d  (1d gets more weight -- fresher/tighter; does
                        leaning on recency help, given day-trading wants narrow/current levels?)
    max_of_both       : max(norm_1d, norm_7d) per bucket -- "either timeframe flags it strongly"
                        rather than a weighted blend

Baselines for comparison: 1d alone and 7d alone (levels_from_bins() on each raw/unmerged bin set),
using the SAME as-of points and the SAME touch/hold/break/placebo machinery
(scripts.research_eth_liquidation_map_support_resistance_backtest_20260824) as every prior round
this session, for direct comparability with that script's headline numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_1d7d_formula_merge_20260824.json"

LOOKBACK_1D_HOURS = 24
LOOKBACK_7D_HOURS = 24 * 7
SEED = 20260824

MERGE_FORMULAS = {
    "raw_sum": lambda n1, n7, b1, b7: {b: b1.get(b, 0.0) + b7.get(b, 0.0) for b in set(b1) | set(b7)},
    "equal_50_50": lambda n1, n7, b1, b7: {b: 0.5 * n1.get(b, 0.0) + 0.5 * n7.get(b, 0.0) for b in set(n1) | set(n7)},
    "lean_7d_30_70": lambda n1, n7, b1, b7: {b: 0.3 * n1.get(b, 0.0) + 0.7 * n7.get(b, 0.0) for b in set(n1) | set(n7)},
    "lean_1d_70_30": lambda n1, n7, b1, b7: {b: 0.7 * n1.get(b, 0.0) + 0.3 * n7.get(b, 0.0) for b in set(n1) | set(n7)},
    "max_of_both": lambda n1, n7, b1, b7: {b: max(n1.get(b, 0.0), n7.get(b, 0.0)) for b in set(n1) | set(n7)},
}


def build_episode(df: pd.DataFrame, t0: int) -> dict | None:
    current_price = float(df["close"].iloc[t0])
    window_1d = df.iloc[t0 - LOOKBACK_1D_HOURS: t0 + 1]
    window_7d = df.iloc[t0 - LOOKBACK_7D_HOURS: t0 + 1]
    raw1 = liqmap.compute_raw_bins(window_1d, current_price)
    raw7 = liqmap.compute_raw_bins(window_7d, current_price)
    if raw1 is None or raw7 is None:
        return None
    bins1, bw1, _, _ = raw1
    bins7, bw7, _, _ = raw7
    assert abs(bw1 - bw7) < 1e-9  # same current_price -> same bin_width, same grid, always true here
    norm1 = {b: w / max(bins1.values()) for b, w in bins1.items()}
    norm7 = {b: w / max(bins7.values()) for b, w in bins7.items()}

    levels_by_formula = {"1d_alone": liqmap.levels_from_bins(bins1, bw1, current_price),
                          "7d_alone": liqmap.levels_from_bins(bins7, bw1, current_price)}
    for name, fn in MERGE_FORMULAS.items():
        merged = fn(norm1, norm7, bins1, bins7)
        if not merged or not (max(merged.values()) > 0):
            levels_by_formula[name] = {"support_levels": [], "resistance_levels": []}
        else:
            levels_by_formula[name] = liqmap.levels_from_bins(merged, bw1, current_price)
    return {"t0": t0, "current_price": current_price, "levels": levels_by_formula}


def run(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    idxs = base.asof_indices(len(df), LOOKBACK_7D_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    episodes = [ep for t0 in idxs if (ep := build_episode(df, t0)) is not None]
    print(f"episodes: {len(episodes)} (of {len(idxs)} as-of points)")

    config_names = ["1d_alone", "7d_alone", *MERGE_FORMULAS.keys()]
    dist_pool = {cfg: {"support": [], "resistance": []} for cfg in config_names}
    for ep in episodes:
        for cfg in config_names:
            for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
                dist_pool[cfg][side].extend(lv["distance_pct"] for lv in ep["levels"][cfg][key])

    results = {}
    for cfg in config_names:
        side_out = {}
        for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
            pool = np.array(dist_pool[cfg][side]) if dist_pool[cfg][side] else np.array([2.0, -2.0])
            real_rows, placebo_rows = [], []
            for ep in episodes:
                levels = ep["levels"][cfg][key]
                cp = ep["current_price"]
                for lv in levels:
                    for buf in base.BUFFER_PCTS:
                        real_rows.append((buf, ep["t0"], base.evaluate_forward(df, ep["t0"], lv["price"], side, buf)))
                    pd_ = rng.choice(pool)
                    pp = cp * (1 + pd_ / 100.0)
                    for buf in base.BUFFER_PCTS:
                        placebo_rows.append((buf, ep["t0"], base.evaluate_forward(df, ep["t0"], pp, side, buf)))

            def agg(rows, buf):
                sub = [o for b, _, o in rows if b == buf]
                n = len(sub)
                touched = [o for o in sub if o != "not_touched"]
                nt = len(touched)
                nh = sum(1 for o in touched if o == "hold")
                return {"n": n, "touch_rate": nt / n if n else None, "n_touched": nt,
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

            side_out[side] = {
                str(buf): {"real": agg(real_rows, buf), "placebo": agg(placebo_rows, buf), "paired": paired(buf)}
                for buf in base.BUFFER_PCTS
            }
        results[cfg] = side_out

    return {"n_episodes": len(episodes), "configs": results}


def main() -> None:
    df = base.load_hourly()
    rng = np.random.default_rng(SEED)
    result = run(df, rng)

    print(f"\n{'config':16s} {'side':11s} {'buf':6s} {'winrate':8s} {'favor_r':8s} {'favor_p':8s} {'hold_real':10s} {'hold_pb':8s}")
    for cfg, sides in result["configs"].items():
        for side, bufs in sides.items():
            for buf_str, row in bufs.items():
                p = row["paired"]
                r, pb = row["real"], row["placebo"]
                print(f"{cfg:16s} {side:11s} {float(buf_str)*100:5.1f}% "
                      f"{str(p['winrate'])[:6]:8s} {p['n_favor_real']:<8d} {p['n_favor_placebo']:<8d} "
                      f"{str(r['hold_rate'])[:8]:10s} {str(pb['hold_rate'])[:6]:8s}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

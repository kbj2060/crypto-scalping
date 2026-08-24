#!/usr/bin/env python3
"""Does the snapshot-at-fixed-horizon magnitude metric miss "bounced, then reversed" episodes?
2026-08-24, user's follow-up on the magnitude metric: "지지했다가 다시 올라갔다가 다시 하락하는
경우는? 악재가 많으면 청산 스윕이 일어났다가 다시 하락할 수도 있잖아." (what about support-holds
-then-rallies-then-falls-again -- a sweep can happen and price can still resume falling on bad
news). Correct: scripts/research_eth_liquidation_map_magnitude_metric_20260824.py's
favorable_return() takes ONE snapshot at exactly touch_i+K -- it cannot tell "no bounce ever
happened" apart from "bounced, then a later/separate move erased it by the snapshot time." Both
read as a bad K-hour return even though they are different phenomena (one says the level carried
no information at all; the other says it did, briefly, before something else overrode it).

=== Fix: MFE alongside the existing snapshot return ===
MFE (Maximum Favorable Excursion, same concept this repo already uses for pos_mfe/pos_mae position
features) = the BEST favorable price reached at ANY point between the touch and K hours later, using
intrabar high/low (not close) since it is asking "how far did it get", not "where did it settle" --
support: max((high[j]-level_price)/level_price) over j in [touch_i, touch_i+K]; resistance:
max((level_price-low[j])/level_price) over the same range. Comparing MFE (real vs placebo) against
the existing snapshot return isolates the user's exact question:
  - If MFE_real is meaningfully bigger than the snapshot return AND bigger than MFE_placebo, a real
    bounce IS happening more than chance, it is just being erased by the time of the snapshot --
    same underlying reversal, weaker/later exit timing story, not "no information."
  - If MFE_real and the snapshot return move together (small gap, same placebo comparison), the
    snapshot number was already telling the honest story.
"given-back rate" = fraction of touches where MFE clears a real move (>=0.3%) but the K-hour
snapshot return does not hold most of it (<50% of MFE retained) -- directly counts how often the
exact pattern the user described (bounce, then give-back) occurs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as evdriven
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_mfe_givenback_20260824.json"
HORIZONS_HOURS = (24, 72)
GIVEBACK_MFE_FLOOR_PCT = 0.3   # ignore touches whose peak move never even reached this -- "gave
                                # back nothing" is not interesting if there was nothing to give back
GIVEBACK_RETENTION_THRESHOLD = 0.5  # snapshot return must retain >=50% of MFE to NOT count as given-back
SEED = 20260824


def find_touch(lows, highs, n, t0, level_price, side):
    fwd_end = min(n, t0 + 1 + base.FORWARD_HOURS)
    for i in range(t0 + 1, fwd_end):
        if side == "support" and lows[i] <= level_price:
            return i
        if side == "resistance" and highs[i] >= level_price:
            return i
    return None


def snapshot_return(closes, n, level_price, touch_i, side, k):
    j = min(n - 1, touch_i + k)
    p = closes[j]
    return (p - level_price) / level_price if side == "support" else (level_price - p) / level_price


def mfe(highs, lows, n, level_price, touch_i, side, k):
    j_end = min(n - 1, touch_i + k)
    if side == "support":
        return float((highs[touch_i:j_end + 1].max() - level_price) / level_price)
    return float((level_price - lows[touch_i:j_end + 1].min()) / level_price)


def collect_1d_7d(episodes, closes, lows, highs, n, rng, cfg_name, side, key):
    dists = [lv["distance_pct"] for ep in episodes for lv in ep["levels"][cfg_name][key]]
    pool = np.array(dists) if dists else np.array([2.0, -2.0])
    rows = {"real": {h: {"snap": [], "mfe": []} for h in HORIZONS_HOURS},
            "placebo": {h: {"snap": [], "mfe": []} for h in HORIZONS_HOURS}}
    for ep in episodes:
        cp = ep["current_price"]
        for lv in ep["levels"][cfg_name][key]:
            ti = find_touch(lows, highs, n, ep["t0"], lv["price"], side)
            if ti is not None:
                for h in HORIZONS_HOURS:
                    rows["real"][h]["snap"].append(snapshot_return(closes, n, lv["price"], ti, side, h))
                    rows["real"][h]["mfe"].append(mfe(highs, lows, n, lv["price"], ti, side, h))
            pp = cp * (1 + rng.choice(pool) / 100.0)
            ti2 = find_touch(lows, highs, n, ep["t0"], pp, side)
            if ti2 is not None:
                for h in HORIZONS_HOURS:
                    rows["placebo"][h]["snap"].append(snapshot_return(closes, n, pp, ti2, side, h))
                    rows["placebo"][h]["mfe"].append(mfe(highs, lows, n, pp, ti2, side, h))
    return rows


def summarize(rows: dict) -> dict:
    out = {}
    for kind in ("real", "placebo"):
        out[kind] = {}
        for h in HORIZONS_HOURS:
            snap = np.array(rows[kind][h]["snap"]) * 100
            mfe_arr = np.array(rows[kind][h]["mfe"]) * 100
            eligible = mfe_arr >= GIVEBACK_MFE_FLOOR_PCT
            n_eligible = int(eligible.sum())
            given_back = eligible & (snap < GIVEBACK_RETENTION_THRESHOLD * mfe_arr)
            out[kind][h] = {
                "n": len(snap), "mean_snapshot_pct": float(np.mean(snap)) if len(snap) else None,
                "mean_mfe_pct": float(np.mean(mfe_arr)) if len(mfe_arr) else None,
                "n_mfe_eligible": n_eligible,
                "given_back_rate": float(given_back.sum() / n_eligible) if n_eligible else None,
            }
    return out


def main() -> None:
    df = base.load_hourly()
    closes, lows, highs = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(df)
    rng = np.random.default_rng(SEED)

    print("=== 1d_alone / 7d_alone (static-window configs) ===")
    import scripts.research_eth_liquidation_map_1d7d_formula_merge_20260824 as merge
    idxs = base.asof_indices(n, merge.LOOKBACK_7D_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    episodes = [ep for t0 in idxs if (ep := merge.build_episode(df, t0)) is not None]

    all_results = {}
    for cfg_name, side, key in [
        ("1d_alone", "support", "support_levels"), ("7d_alone", "support", "support_levels"),
        ("1d_alone", "resistance", "resistance_levels"), ("7d_alone", "resistance", "resistance_levels"),
    ]:
        rows = collect_1d_7d(episodes, closes, lows, highs, n, rng, cfg_name, side, key)
        summary = summarize(rows)
        all_results[f"{cfg_name}/{side}"] = summary
        for h in HORIZONS_HOURS:
            r, p = summary["real"][h], summary["placebo"][h]
            print(f"{cfg_name:10s} {side:11s} {h:2d}h  n={r['n']:5d}  "
                  f"snapshot: real={r['mean_snapshot_pct']:+.3f}% placebo={p['mean_snapshot_pct']:+.3f}%  |  "
                  f"MFE: real={r['mean_mfe_pct']:.3f}% placebo={p['mean_mfe_pct']:.3f}%  |  "
                  f"given-back rate: real={r['given_back_rate']}({r['n_mfe_eligible']} eligible) "
                  f"placebo={p['given_back_rate']}({p['n_mfe_eligible']} eligible)")

    print("\n=== event_driven_reset (the config with the most extreme winrate/magnitude gap) ===")
    snaps = evdriven.simulate(df)
    ev_rows = {}
    for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
        dists = [lv["distance_pct"] for s in snaps for lv in s[key]]
        pool = np.array(dists) if dists else np.array([2.0, -2.0])
        rows = {"real": {h: {"snap": [], "mfe": []} for h in HORIZONS_HOURS},
                "placebo": {h: {"snap": [], "mfe": []} for h in HORIZONS_HOURS}}
        for s in snaps:
            cp = s["current_price"]
            for lv in s[key]:
                ti = find_touch(lows, highs, n, s["t0"], lv["price"], side)
                if ti is not None:
                    for h in HORIZONS_HOURS:
                        rows["real"][h]["snap"].append(snapshot_return(closes, n, lv["price"], ti, side, h))
                        rows["real"][h]["mfe"].append(mfe(highs, lows, n, lv["price"], ti, side, h))
                pp = cp * (1 + rng.choice(pool) / 100.0)
                ti2 = find_touch(lows, highs, n, s["t0"], pp, side)
                if ti2 is not None:
                    for h in HORIZONS_HOURS:
                        rows["placebo"][h]["snap"].append(snapshot_return(closes, n, pp, ti2, side, h))
                        rows["placebo"][h]["mfe"].append(mfe(highs, lows, n, pp, ti2, side, h))
        summary = summarize(rows)
        ev_rows[side] = summary
        for h in HORIZONS_HOURS:
            r, p = summary["real"][h], summary["placebo"][h]
            print(f"event_driven {side:11s} {h:2d}h  n={r['n']:5d}  "
                  f"snapshot: real={r['mean_snapshot_pct']:+.3f}% placebo={p['mean_snapshot_pct']:+.3f}%  |  "
                  f"MFE: real={r['mean_mfe_pct']:.3f}% placebo={p['mean_mfe_pct']:.3f}%  |  "
                  f"given-back rate: real={r['given_back_rate']}({r['n_mfe_eligible']} eligible) "
                  f"placebo={p['given_back_rate']}({p['n_mfe_eligible']} eligible)")
    all_results["event_driven"] = ev_rows

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

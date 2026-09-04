#!/usr/bin/env python3
"""Isolated single-variable A/B: does replacing v1's fixed 50:50 long/short weight split with real
direction data (taker buy-share / global long-short account ratio) help, holding v1's entry mass
(candle-close x volume x recency-halflife) and survival (alive-until-crossed-by-a-later-bar)
mechanics byte-identical otherwise?

2026-08-25's v2 ladder (docs/experiments/eth_liquidation_map_v2_oi_cohort_direction_design_20260825.md)
never actually tested this in isolation: v2a already replaced BOTH entry mass (OI-birth instead of
every-candle-volume) AND survival (OI-decline pro-rata decay instead of recency halflife) before
v2b/c added direction on top -- and v2a's OOS resistance pairWR collapse (10:8->2:5, see
eth_liquidation_map_v2_oi_cohort_ab_rejected_20260825 memory) happened with direction still fixed at
50:50 (scripts/live_liquidation_map_v2_20260825.py's "v2a": np.full(.., 0.5)), so that collapse
cannot be attributed to direction data at all -- it's confounded with the mass/survival rewrite.
This script isolates ONLY the direction variable, against the mechanism actually deployed today:
dashboard/server.py's stateless rolling-window recompute (LIQUIDATION_MAP_LOOKBACK_HOURS=24,
liqmap.compute_liquidation_levels()), NOT the event-driven state machine the original v2 ladder's
"v1" baseline used.

Pre-registered gate: adopt a direction-split variant only if OOS pairWR AND magnitude beat the
v1_live baseline (identical mechanics, long_share=0.5 fixed) AND beat placebo, on the SAME side
that regressed in the v2 ladder (resistance) without making support worse. Anything weaker is
REJECTED and v1_live (as currently deployed) stays -- no cell-picking across sides/buffers.

Variants (all share liqmap.compute_raw_bins()'s entry/survival math exactly; only long_share differs):
  v1_live  -- long_share = 0.5 fixed every bar (today's deployed symmetric assumption)
  v1_taker -- long_share = taker buy-volume share per bar (v2b's formula, reused verbatim from
              live_liquidation_map_v2_20260825.prepare_cohort_arrays() for an apples-to-apples
              direction signal vs the already-rejected v2b)
  v1_blend -- long_share = 50:50 blend of taker share and global long/short account fraction
              (v2c's formula, reused verbatim)

Mass-conservation identity (asserted at startup): compute_raw_bins_directional() with
long_share==0.5 everywhere must reproduce liqmap.compute_raw_bins() bin-for-bin. v1's
per_tier_weight already gives EACH side (long and short) the full base_weight/len(TIERS)
independently -- i.e. total mass summed over both sides is 2x base_weight, not base_weight. The x2
factor in long_weight/short_weight below exists precisely so long_share=0.5 lands on that same
total instead of silently halving the map's mass relative to v1. Without this check, a "no signal
found" result here would be ambiguous between "direction data doesn't help" and "the isolation
itself is broken."

Data/eval: reuses research_eth_liquidation_map_v2_cohort_ab_backtest_20260825.py's loader (hourly
klines + taker_buy_base, joined to the OI/long-short-ratio archive) and
research_eth_liquidation_map_event_driven_reset_20260824.py's evaluate() (touch/hold/placebo/
magnitude harness, unmodified) so results sit directly next to the existing v1(event-driven)/v2a/b/c
numbers in eth_liquidation_map_v2_cohort_ab_backtest_20260825.json -- same TRAIN(80%)/OOS(20%) split
bar, same eval grid (base.asof_indices, WARMUP_HOURS unchanged for grid comparability even though
this method's own warmup need is much shorter than 2160h -- a 24h rolling window has no cohort
accumulation lag).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.live_liquidation_map_v2_20260825 as v2
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_v1_direction_isolated_ab_20260826.json"

LOOKBACK_HOURS_LIVE = 24  # dashboard/server.py::LIQUIDATION_MAP_LOOKBACK_HOURS as of 2026-08-25
WARMUP_HOURS = v2ab.WARMUP_HOURS  # reuse identical eval grid for direct comparability to v1/v2a/b/c
TRAIN_FRACTION = v2ab.TRAIN_FRACTION
SEED = 20260826
VARIANTS = ("v1_live", "v1_taker", "v1_blend")


def compute_raw_bins_directional(df: pd.DataFrame, current_price: float, long_share: np.ndarray):
    """liqmap.compute_raw_bins() with exactly one change: the fixed 50:50 per-tier weight split
    between long_liq/short_liq is replaced by a per-bar long_share array. See module docstring for
    why the x2 factor is required for long_share=0.5 to reproduce v1 exactly."""
    if df is None or len(df) < 20 or not (current_price > 0):
        return None
    d = df.reset_index(drop=True)
    n = len(d)
    ls = np.asarray(long_share, dtype="float64")
    assert len(ls) == n, f"long_share length {len(ls)} != window length {n}"

    close = d["close"].to_numpy(dtype="float64")
    high = d["high"].to_numpy(dtype="float64")
    low = d["low"].to_numpy(dtype="float64")
    volume = d["volume"].to_numpy(dtype="float64")
    ts = pd.to_datetime(d["timestamp"], utc=True)
    now = ts.iloc[-1]
    age_hours = (now - ts).dt.total_seconds().to_numpy() / 3600.0
    recency_weight = np.exp(-age_hours / liqmap.RECENCY_HALFLIFE_HOURS)
    base_weight = volume * recency_weight

    future_min_low = np.full(n, np.inf)
    future_max_high = np.full(n, -np.inf)
    if n > 1:
        future_min_low[:-1] = liqmap._suffix_min_after(low)
        future_max_high[:-1] = liqmap._suffix_max_after(high)

    bin_width = max(current_price * liqmap.BIN_WIDTH_PCT, 1e-9)
    bins: dict[int, float] = {}

    def add(price_level: np.ndarray, weight: np.ndarray, alive: np.ndarray) -> None:
        idx = np.where(alive & (price_level > 0))[0]
        if not len(idx):
            return
        bucket = np.round(price_level[idx] / bin_width).astype("int64")
        for b, wv in zip(bucket.tolist(), weight[idx].tolist()):
            bins[b] = bins.get(b, 0.0) + wv

    long_weight = base_weight * (2.0 * ls) / len(liqmap.LEVERAGE_TIERS)
    short_weight = base_weight * (2.0 * (1.0 - ls)) / len(liqmap.LEVERAGE_TIERS)
    for lev in liqmap.LEVERAGE_TIERS:
        long_liq = close * (1.0 - 1.0 / lev + liqmap.MAINTENANCE_MARGIN_RATE)
        short_liq = close * (1.0 + 1.0 / lev - liqmap.MAINTENANCE_MARGIN_RATE)
        add(long_liq, long_weight, future_min_low > long_liq)
        add(short_liq, short_weight, future_max_high < short_liq)

    if not bins or not (max(bins.values()) > 0):
        return None
    return bins, bin_width, n, age_hours


def _identity_check(df: pd.DataFrame) -> None:
    window = df.iloc[-200:].reset_index(drop=True)
    cp = float(window["close"].iloc[-1])
    raw_v1 = liqmap.compute_raw_bins(window, cp)
    raw_dir = compute_raw_bins_directional(window, cp, np.full(len(window), 0.5))
    assert raw_v1 is not None and raw_dir is not None
    bins1, bw1, n1, _ = raw_v1
    bins2, bw2, n2, _ = raw_dir
    assert bw1 == bw2 and n1 == n2
    assert set(bins1) == set(bins2), (set(bins1) - set(bins2), set(bins2) - set(bins1))
    for k in bins1:
        assert abs(bins1[k] - bins2[k]) < 1e-6 * max(1.0, abs(bins1[k])), (k, bins1[k], bins2[k])
    print("identity check passed: long_share=0.5 reproduces liqmap.compute_raw_bins() exactly", flush=True)


def snapshots_v1_directional(df: pd.DataFrame, eval_idxs: list[int], long_share_full: np.ndarray) -> list[dict]:
    close = df["close"].to_numpy()
    out = []
    for i in eval_idxs:
        start = max(0, i - LOOKBACK_HOURS_LIVE + 1)
        window = df.iloc[start:i + 1]
        raw = compute_raw_bins_directional(window, float(close[i]), long_share_full[start:i + 1])
        if raw is None:
            continue
        bins, bin_width, _, _ = raw
        lv = liqmap.levels_from_bins(bins, bin_width, float(close[i]))
        out.append({"t0": i, "current_price": float(close[i]),
                    "support_levels": lv["support_levels"], "resistance_levels": lv["resistance_levels"]})
    return out


def summarize(name: str, split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    ev = ed.evaluate(df, snaps, rng)
    n_lv = [len(s["support_levels"]) + len(s["resistance_levels"]) for s in snaps]
    return {"variant": name, "split": split, "n_snapshots": len(snaps),
            "avg_levels_per_snapshot": round(float(np.mean(n_lv)), 2) if n_lv else 0.0,
            "eval": ev}


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)
    df = df.rename(columns={"sum_open_interest": "oi"})
    r = df["count_long_short_ratio"].ffill()
    df["long_account_frac"] = (r / (1.0 + r)).fillna(0.5)

    _identity_check(df)

    arrs = v2.prepare_cohort_arrays(df)
    long_share = {"v1_live": np.full(len(df), 0.5),
                  "v1_taker": arrs["long_share"]["v2b"],
                  "v1_blend": arrs["long_share"]["v2c"]}

    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    eval_idxs = base.asof_indices(n, WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} split at {df['timestamp'].iloc[split_i]} eval_points={len(eval_idxs)} "
          f"(train={sum(1 for i in eval_idxs if i < split_i)}, oos={sum(1 for i in eval_idxs if i >= split_i)})",
          flush=True)

    all_snaps: dict[str, list[dict]] = {}
    for var in VARIANTS:
        t = time.time()
        all_snaps[var] = snapshots_v1_directional(df, eval_idxs, long_share[var])
        print(f"{var} snapshots: {len(all_snaps[var])} ({time.time()-t:.0f}s)", flush=True)

    results = []
    for k, (name, snaps) in enumerate(all_snaps.items()):
        for split, sel in (("TRAIN", [s for s in snaps if s["t0"] < split_i]),
                           ("OOS", [s for s in snaps if s["t0"] >= split_i])):
            results.append(summarize(name, split, sel, df, seed_off=k * 10 + (0 if split == "TRAIN" else 1)))
            print(f"evaluated {name}/{split} (n={len(sel)})", flush=True)

    print(f"\n{'variant':10s} {'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24 diff':11s} {'mag72 diff':11s} {'nTouch':6s}")
    for r_ in results:
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:10s} {r_['split']:6s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                      f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{('None' if mag24 is None else f'{mag24:+.3f}'):11s} "
                      f"{('None' if mag72 is None else f'{mag72:+.3f}'):11s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "split_bar": split_i,
        "split_ts": str(df["timestamp"].iloc[split_i]), "warmup_hours": WARMUP_HOURS,
        "lookback_hours_live": LOOKBACK_HOURS_LIVE, "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolated single-variable A/B, 2026-08-26 user follow-up: replace v1's per-candle hypothetical
entry price (that candle's CLOSE) with the candle's (high+low)/2 midpoint, holding every other
mechanic -- weighting (volume x recency), the survival filter (still checked via high/low of bars
strictly AFTER the entry bar, unchanged), symmetric 50:50 long/short split, and binning -- byte-
identical to v1. Same isolation discipline as research_eth_liquidation_map_v1_direction_isolated_
ab_20260826.py and its toptrader follow-up, just a different single variable (entry-price basis
instead of direction split) -- deliberately NOT combined with any direction-split variant tested
so far, to avoid re-confounding two variables at once (the exact mistake the user already caught
in the original v2 ladder).

Rationale: a hypothetical entry priced at CLOSE is somewhat arbitrary -- it's wherever the last
trade of the hour happened to land, which can sit near a brief wick extreme. (high+low)/2 is a
commonly-used "typical price" proxy that's less sensitive to a single last-trade print. No lookahead
concern: (high+low)/2 for bar i uses only bar i's own high/low, exactly as causally available as
that bar's own close (the caller already drops the still-forming bar before any of this runs).

Pre-registered gate: adopt v1_mid only if OOS pairWR AND magnitude beat v1_live on resistance (the
side every direction-data variant tried so far has failed to improve) without making support worse;
otherwise REJECTED and v1_live (as currently deployed, entry=close) stays.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_entry_price_isolated_ab_20260826.json"

VARIANTS = ("v1_live", "v1_mid")


def compute_raw_bins_entry_price(df: pd.DataFrame, current_price: float, entry_price: np.ndarray):
    """liqmap.compute_raw_bins() with exactly one change: the per-candle hypothetical entry price
    (v1 uses that row's close) is replaced by a caller-supplied entry_price array. Weighting,
    survival filter, symmetric 50:50 split, and binning are otherwise byte-identical to v1 -- see
    module docstring."""
    if df is None or len(df) < 20 or not (current_price > 0):
        return None
    d = df.reset_index(drop=True)
    n = len(d)
    ep = np.asarray(entry_price, dtype="float64")
    assert len(ep) == n, f"entry_price length {len(ep)} != window length {n}"

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

    per_tier_weight = base_weight / len(liqmap.LEVERAGE_TIERS)
    for lev in liqmap.LEVERAGE_TIERS:
        long_liq = ep * (1.0 - 1.0 / lev + liqmap.MAINTENANCE_MARGIN_RATE)
        short_liq = ep * (1.0 + 1.0 / lev - liqmap.MAINTENANCE_MARGIN_RATE)
        add(long_liq, per_tier_weight, future_min_low > long_liq)
        add(short_liq, per_tier_weight, future_max_high < short_liq)

    if not bins or not (max(bins.values()) > 0):
        return None
    return bins, bin_width, n, age_hours


def _identity_check(df: pd.DataFrame) -> None:
    """entry_price=close must reproduce liqmap.compute_raw_bins() exactly -- proves only the price
    array changed, nothing else, before trusting any A/B result."""
    window = df.iloc[-200:].reset_index(drop=True)
    cp = float(window["close"].iloc[-1])
    raw_v1 = liqmap.compute_raw_bins(window, cp)
    raw_ep = compute_raw_bins_entry_price(window, cp, window["close"].to_numpy(dtype="float64"))
    assert raw_v1 is not None and raw_ep is not None
    bins1, bw1, n1, _ = raw_v1
    bins2, bw2, n2, _ = raw_ep
    assert bw1 == bw2 and n1 == n2
    assert set(bins1) == set(bins2), (set(bins1) - set(bins2), set(bins2) - set(bins1))
    for k in bins1:
        assert abs(bins1[k] - bins2[k]) < 1e-6 * max(1.0, abs(bins1[k])), (k, bins1[k], bins2[k])
    print("identity check passed: entry_price=close reproduces liqmap.compute_raw_bins() exactly", flush=True)


def snapshots_v1_entry_price(df: pd.DataFrame, eval_idxs: list[int], entry_price_full: np.ndarray) -> list[dict]:
    close = df["close"].to_numpy()
    out = []
    for i in eval_idxs:
        start = max(0, i - v1dir.LOOKBACK_HOURS_LIVE + 1)
        window = df.iloc[start:i + 1]
        raw = compute_raw_bins_entry_price(window, float(close[i]), entry_price_full[start:i + 1])
        if raw is None:
            continue
        bins, bin_width, _, _ = raw
        lv = liqmap.levels_from_bins(bins, bin_width, float(close[i]))
        out.append({"t0": i, "current_price": float(close[i]),
                    "support_levels": lv["support_levels"], "resistance_levels": lv["resistance_levels"]})
    return out


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)  # same windowed df as every prior variant this thread
    print(f"join: {join_stats}", flush=True)
    df = df.rename(columns={"sum_open_interest": "oi"})

    _identity_check(df)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    mid = (high + low) / 2.0
    entry_price = {"v1_live": close, "v1_mid": mid}
    diff_pct = np.abs(mid - close) / close * 100.0
    print(f"|mid-close|/close%%: mean={diff_pct.mean():.4f} median={np.median(diff_pct):.4f} "
          f"p95={np.percentile(diff_pct, 95):.4f} max={diff_pct.max():.4f}", flush=True)

    n = len(df)
    split_i = int(n * v1dir.TRAIN_FRACTION)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} split at {df['timestamp'].iloc[split_i]} eval_points={len(eval_idxs)} "
          f"(train={sum(1 for i in eval_idxs if i < split_i)}, oos={sum(1 for i in eval_idxs if i >= split_i)})",
          flush=True)

    all_snaps: dict[str, list[dict]] = {}
    for var in VARIANTS:
        t = time.time()
        all_snaps[var] = snapshots_v1_entry_price(df, eval_idxs, entry_price[var])
        print(f"{var} snapshots: {len(all_snaps[var])} ({time.time()-t:.0f}s)", flush=True)

    results = []
    for k, (name, snaps) in enumerate(all_snaps.items()):
        for split, sel in (("TRAIN", [s for s in snaps if s["t0"] < split_i]),
                           ("OOS", [s for s in snaps if s["t0"] >= split_i])):
            results.append(v1dir.summarize(name, split, sel, df, seed_off=k * 10 + (0 if split == "TRAIN" else 1)))
            print(f"evaluated {name}/{split} (n={len(sel)})", flush=True)

    print(f"\n{'variant':8s} {'split':6s} {'side':11s} {'buf%':5s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24 diff':11s} {'mag72 diff':11s} {'nTouch':6s}")
    for r_ in results:
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005", "0.001"):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:8s} {r_['split']:6s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                      f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{('None' if mag24 is None else f'{mag24:+.3f}'):11s} "
                      f"{('None' if mag72 is None else f'{mag72:+.3f}'):11s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "split_bar": split_i,
        "split_ts": str(df["timestamp"].iloc[split_i]), "warmup_hours": v1dir.WARMUP_HOURS,
        "lookback_hours_live": v1dir.LOOKBACK_HOURS_LIVE,
        "mid_vs_close_diff_pct": {"mean": float(diff_pct.mean()), "median": float(np.median(diff_pct)),
                                   "p95": float(np.percentile(diff_pct, 95)), "max": float(diff_pct.max())},
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does trading activity actually concentrate near the estimated support/resistance levels?
2026-08-25 follow-up, user request: not "does price reverse/accelerate at the level" (every prior
round in this line tested variants of that), but "is there real evidence of a contested zone --
elevated volume, elevated real liquidations -- near the level, regardless of what price does next."

Two outcome variables, same design (distance-to-nearest-level -> outcome, real level vs a matched
placebo level that resets in lockstep with the real one so both have the same sticky-between-resets
structure):
1. VOLUME, full 4.7y history (no external dependency, real statistical power).
2. Real liquidation USD (long_usd_1m + short_usd_1m from data/live/tail_risk.duckdb::tail_risk_1m,
   the actual @forceOrder websocket feed -- NOT the map's own hypothetical-liquidation bins, which
   would make this circular). Only valid from 2026-07-18 15:03 UTC (forceOrder WS endpoint fix,
   pre-that is fake-always-zero per every other script in this family) -- a ~5-6 week window, far
   less power than the volume test, and server-only (local dev has no duckdb module and a stale
   08-17 copy of the file).

Does NOT change the event-driven state machine (break/drift triggers, MIN_FLOOR/MAX_LOOKBACK,
compute_raw_bins/levels_from_bins) -- copies ed.simulate()'s exact logic but records state at EVERY
hour instead of only the sparse eval_idxs, since this test needs a continuous distance-to-level
series, not point-in-time snapshots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_volume_liquidation_concentration_20260825.json"
TAIL_RISK_DB = ROOT / "data" / "live" / "tail_risk.duckdb"
LIQ_VALID_SINCE_UTC = "2026-07-18 15:03:00+00"
SEED = ed.SEED
N_DIST_BINS = 10


def build_placebo_pool(snapshots: list[dict]) -> dict[str, np.ndarray]:
    pool = {}
    for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
        arr = np.array([lv["distance_pct"] for s in snapshots for lv in s[key]])
        pool[side] = arr if len(arr) else np.array([2.0, -2.0])
    return pool


def simulate_hourly_distance_series(df: pd.DataFrame, pool: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Per-hour distance-to-nearest-level for real levels AND a placebo level that resets in
    lockstep with the real side (redrawn from `pool` exactly when the real side resets, held fixed
    otherwise) -- keeps the placebo's "sticky for days at a time" structure matched to the real
    level's, instead of a fresh-every-hour placebo that would bias the comparison (see
    eth_liquidation_map_event_driven_dwell_filter_20260825 on how much window_hours matters here)."""
    n = len(df)
    close = df["close"].to_numpy()
    volume = df["volume"].to_numpy()
    ts = df["timestamp"]

    def regenerate(reset_idx: int, i: int, key: str) -> list[dict]:
        start = max(reset_idx, i - ed.MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - ed.MIN_FLOOR_HOURS))
        window = df.iloc[start:i + 1]
        cp = float(close[i])
        raw = liqmap.compute_raw_bins(window, cp)
        if raw is None:
            return []
        bins, bin_width, _, _ = raw
        return liqmap.levels_from_bins(bins, bin_width, cp)[key]

    support_levels = regenerate(0, ed.BOOTSTRAP_HOURS, "support_levels")
    resistance_levels = regenerate(0, ed.BOOTSTRAP_HOURS, "resistance_levels")
    support_reset_idx = resistance_reset_idx = 0
    placebo_support_price = close[ed.BOOTSTRAP_HOURS] * (1 + rng.choice(pool["support"]) / 100.0)
    placebo_resistance_price = close[ed.BOOTSTRAP_HOURS] * (1 + rng.choice(pool["resistance"]) / 100.0)

    rows = []
    for i in range(ed.BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_support = any(price < lv["price"] * (1 - ed.BREAK_TOLERANCE_PCT) for lv in support_levels)
        broke_resistance = any(price > lv["price"] * (1 + ed.BREAK_TOLERANCE_PCT) for lv in resistance_levels)
        drift_support = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > ed.DRIFT_TOLERANCE_PCT
        drift_resistance = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > ed.DRIFT_TOLERANCE_PCT

        if broke_support or drift_support:
            support_levels = regenerate(support_reset_idx, i, "support_levels")
            support_reset_idx = i
            placebo_support_price = price * (1 + rng.choice(pool["support"]) / 100.0)
        if broke_resistance or drift_resistance:
            resistance_levels = regenerate(resistance_reset_idx, i, "resistance_levels")
            resistance_reset_idx = i
            placebo_resistance_price = price * (1 + rng.choice(pool["resistance"]) / 100.0)

        nearest_support = support_levels[0]["price"] if support_levels else None
        nearest_resistance = resistance_levels[0]["price"] if resistance_levels else None
        rows.append({
            "timestamp": ts.iloc[i], "close": price, "volume": volume[i],
            "dist_real_support_pct": abs(price - nearest_support) / price * 100 if nearest_support else np.nan,
            "dist_real_resistance_pct": abs(price - nearest_resistance) / price * 100 if nearest_resistance else np.nan,
            "dist_placebo_support_pct": abs(price - placebo_support_price) / price * 100,
            "dist_placebo_resistance_pct": abs(price - placebo_resistance_price) / price * 100,
        })
    return pd.DataFrame(rows)


def bin_by_distance(dist: pd.Series, outcome: pd.Series, n_bins: int):
    valid = dist.notna() & outcome.notna()
    d, o = dist[valid], outcome[valid]
    if len(d) < n_bins * 5:
        return None
    bins = pd.qcut(d, n_bins, duplicates="drop")
    grouped = o.groupby(bins, observed=True).agg(["mean", "count"])
    spearman = float(d.reset_index(drop=True).corr(o.reset_index(drop=True), method="spearman"))
    return {
        "n": int(len(d)), "spearman_dist_vs_outcome": spearman,
        "bins": [{"bin": str(idx), "mean": float(row["mean"]), "n": int(row["count"])}
                 for idx, row in grouped.iterrows()],
    }


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}", flush=True)
    snapshots = ed.simulate(df)
    pool = build_placebo_pool(snapshots)
    rng = np.random.default_rng(SEED)
    hourly = simulate_hourly_distance_series(df, pool, rng)
    print(f"hourly distance rows: {len(hourly)}", flush=True)

    hourly["dist_real_nearest"] = hourly[["dist_real_support_pct", "dist_real_resistance_pct"]].min(axis=1)
    hourly["dist_placebo_nearest"] = hourly[["dist_placebo_support_pct", "dist_placebo_resistance_pct"]].min(axis=1)

    result: dict = {}
    print("\n=== VOLUME vs distance-to-nearest-level (full 4.7y history) ===")
    for label, col in (("real", "dist_real_nearest"), ("placebo", "dist_placebo_nearest")):
        b = bin_by_distance(hourly[col], hourly["volume"], N_DIST_BINS)
        result[f"volume_{label}"] = b
        print(f"  {label}: n={b['n'] if b else 0} spearman={b['spearman_dist_vs_outcome'] if b else None}")
        if b:
            for row in b["bins"]:
                print(f"    {row['bin']:32s} mean_volume={row['mean']:10.1f} n={row['n']}")

    if TAIL_RISK_DB.exists():
        try:
            import duckdb
            con = duckdb.connect(str(TAIL_RISK_DB), read_only=True)
            try:
                liq_df = con.execute(f"""
                    SELECT ts, long_usd_1m, short_usd_1m, valid_liq_stream, ws_stale
                    FROM tail_risk_1m
                    WHERE ts >= TIMESTAMPTZ '{LIQ_VALID_SINCE_UTC}'
                    ORDER BY ts
                """).df()
            finally:
                con.close()
            liq_df["ts"] = liq_df["ts"].dt.tz_convert("UTC")
            liq_df = liq_df[(liq_df["valid_liq_stream"] == True) & (liq_df["ws_stale"] != True)]  # noqa: E712
            liq_df["liq_usd"] = liq_df["long_usd_1m"].fillna(0.0) + liq_df["short_usd_1m"].fillna(0.0)
            liq_df["hour"] = liq_df["ts"].dt.floor("h")
            liq_hourly = liq_df.groupby("hour")["liq_usd"].sum().reset_index()

            hourly["hour"] = pd.to_datetime(hourly["timestamp"], utc=True).dt.floor("h")
            merged = hourly.merge(liq_hourly, on="hour", how="inner")
            n_bins_liq = max(2, min(N_DIST_BINS, len(merged) // 15))
            print(f"\n=== LIQUIDATION $ vs distance-to-nearest-level "
                  f"({len(merged)} hours since {LIQ_VALID_SINCE_UTC}, {n_bins_liq} bins) ===")
            for label, col in (("real", "dist_real_nearest"), ("placebo", "dist_placebo_nearest")):
                b = bin_by_distance(merged[col], merged["liq_usd"], n_bins_liq)
                result[f"liquidation_{label}"] = b
                print(f"  {label}: n={b['n'] if b else 0} spearman={b['spearman_dist_vs_outcome'] if b else None}")
                if b:
                    for row in b["bins"]:
                        print(f"    {row['bin']:32s} mean_liq_usd={row['mean']:12.1f} n={row['n']}")
        except Exception as e:  # noqa: BLE001
            print(f"liquidation analysis failed: {e}")
            result["liquidation_error"] = str(e)
    else:
        print(f"\n{TAIL_RISK_DB} not found locally -- liquidation analysis needs to run on server")
        result["liquidation_error"] = "db_missing_local"

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

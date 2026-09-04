#!/usr/bin/env python3
"""Diagnostic (not an A/B, no gate) for the 2026-08-26 user follow-up hypothesis: every isolated
single-variable change tried so far (direction source, entry-price basis, candle color) has failed
to beat v1_live's own resistance OOS pairWR (0.60, already the single best cell in the whole
series) while comfortably improving on support (which started weaker). Working hypothesis: in a
market with an up-bias over some sub-window, new highs happen more often than new lows, so
short_liq candidates (resistance, above price) get crossed-and-dropped by compute_raw_bins()'s
alive filter faster/more often than long_liq candidates (support, below price) -- leaving
resistance's surviving candidate pool thinner and more recency-concentrated, hence more sensitive
(noisier) to any reweighting we try.

diagnose_survivor_pool() instruments liqmap.compute_raw_bins()'s exact math (same constants, same
alive-filter definition, imported not reimplemented) to also expose, per snapshot: how many
(candle x leverage-tier) hypothetical positions survive on each side, their total surviving
weight, and the weight-weighted mean age (hours back from the window's last bar) of survivors on
each side. Cross-checked against liqmap.compute_raw_bins()'s own bins output (support bins are all
price<current_price i.e. long-side by construction, resistance bins all price>current_price i.e.
short-side) to confirm the instrumentation reproduces production exactly before trusting the stats.

Also reports simple trend-bias context (net price change, new-high vs new-low bar counts) for
TRAIN vs OOS so a pool-depth asymmetry finding can be tied back to an actual market-regime cause
rather than left as an unexplained number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_survivor_pool_asymmetry_diagnostic_20260826.json"


def diagnose_survivor_pool(df: pd.DataFrame, current_price: float) -> dict | None:
    if df is None or len(df) < 20 or not (current_price > 0):
        return None
    d = df.reset_index(drop=True)
    close = d["close"].to_numpy(dtype="float64")
    high = d["high"].to_numpy(dtype="float64")
    low = d["low"].to_numpy(dtype="float64")
    volume = d["volume"].to_numpy(dtype="float64")
    ts = pd.to_datetime(d["timestamp"], utc=True)
    now = ts.iloc[-1]
    age_hours = (now - ts).dt.total_seconds().to_numpy() / 3600.0
    recency_weight = np.exp(-age_hours / liqmap.RECENCY_HALFLIFE_HOURS)
    base_weight = volume * recency_weight

    n = len(d)
    future_min_low = np.full(n, np.inf)
    future_max_high = np.full(n, -np.inf)
    if n > 1:
        future_min_low[:-1] = liqmap._suffix_min_after(low)
        future_max_high[:-1] = liqmap._suffix_max_after(high)

    per_tier_weight = base_weight / len(liqmap.LEVERAGE_TIERS)
    long_alive_count = short_alive_count = 0
    long_weight_sum = short_weight_sum = 0.0
    long_age_wsum = short_age_wsum = 0.0
    for lev in liqmap.LEVERAGE_TIERS:
        long_liq = close * (1.0 - 1.0 / lev + liqmap.MAINTENANCE_MARGIN_RATE)
        short_liq = close * (1.0 + 1.0 / lev - liqmap.MAINTENANCE_MARGIN_RATE)
        long_alive = (future_min_low > long_liq) & (long_liq > 0)
        short_alive = (future_max_high < short_liq) & (short_liq > 0)
        long_alive_count += int(long_alive.sum())
        short_alive_count += int(short_alive.sum())
        long_weight_sum += float(per_tier_weight[long_alive].sum())
        short_weight_sum += float(per_tier_weight[short_alive].sum())
        long_age_wsum += float((per_tier_weight[long_alive] * age_hours[long_alive]).sum())
        short_age_wsum += float((per_tier_weight[short_alive] * age_hours[short_alive]).sum())

    return {
        "long_alive_count": long_alive_count, "short_alive_count": short_alive_count,
        "long_weight_sum": long_weight_sum, "short_weight_sum": short_weight_sum,
        "long_weighted_age_mean": (long_age_wsum / long_weight_sum) if long_weight_sum > 0 else None,
        "short_weighted_age_mean": (short_age_wsum / short_weight_sum) if short_weight_sum > 0 else None,
    }


def _cross_check(df: pd.DataFrame) -> None:
    window = df.iloc[-200:].reset_index(drop=True)
    cp = float(window["close"].iloc[-1])
    raw = liqmap.compute_raw_bins(window, cp)
    diag = diagnose_survivor_pool(window, cp)
    assert raw is not None and diag is not None
    bins, bin_width, _, _ = raw
    support_w = sum(w for b, w in bins.items() if b * bin_width < cp)
    resistance_w = sum(w for b, w in bins.items() if b * bin_width > cp)
    assert abs(support_w - diag["long_weight_sum"]) < 1e-6 * max(1.0, support_w), (support_w, diag["long_weight_sum"])
    assert abs(resistance_w - diag["short_weight_sum"]) < 1e-6 * max(1.0, resistance_w), (resistance_w, diag["short_weight_sum"])
    print("cross-check passed: diagnose_survivor_pool()'s long/short weight sums match "
          "liqmap.compute_raw_bins()'s support/resistance bin totals exactly", flush=True)


def trend_bias(df: pd.DataFrame, lo: int, hi: int) -> dict:
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    seg_close, seg_high, seg_low = close[lo:hi], high[lo:hi], low[lo:hi]
    roll_high = pd.Series(high).rolling(v1dir.LOOKBACK_HOURS_LIVE).max().to_numpy()[lo:hi]
    roll_low = pd.Series(low).rolling(v1dir.LOOKBACK_HOURS_LIVE).min().to_numpy()[lo:hi]
    new_high = seg_high >= roll_high
    new_low = seg_low <= roll_low
    return {
        "net_pct_change": float((seg_close[-1] - seg_close[0]) / seg_close[0] * 100),
        "n_bars": int(hi - lo),
        "new_24h_high_bars": int(new_high.sum()), "new_24h_low_bars": int(new_low.sum()),
        "new_high_minus_low_pct_of_bars": round(float((new_high.sum() - new_low.sum()) / (hi - lo) * 100), 3),
    }


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)

    _cross_check(df)

    n = len(df)
    split_i = int(n * v1dir.TRAIN_FRACTION)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    close = df["close"].to_numpy(dtype="float64")

    rows = []
    for i in eval_idxs:
        start = max(0, i - v1dir.LOOKBACK_HOURS_LIVE + 1)
        window = df.iloc[start:i + 1]
        diag = diagnose_survivor_pool(window, float(close[i]))
        if diag is None:
            continue
        diag["t0"] = i
        diag["split"] = "TRAIN" if i < split_i else "OOS"
        rows.append(diag)
    print(f"snapshots diagnosed: {len(rows)}", flush=True)

    def summarize(split: str) -> dict:
        sub = [r for r in rows if r["split"] == split]
        lac = np.array([r["long_alive_count"] for r in sub])
        sac = np.array([r["short_alive_count"] for r in sub])
        lw = np.array([r["long_weight_sum"] for r in sub])
        sw = np.array([r["short_weight_sum"] for r in sub])
        laa = np.array([r["long_weighted_age_mean"] for r in sub if r["long_weighted_age_mean"] is not None])
        saa = np.array([r["short_weighted_age_mean"] for r in sub if r["short_weighted_age_mean"] is not None])
        return {
            "n_snapshots": len(sub),
            "alive_count": {"long_mean": round(float(lac.mean()), 2), "short_mean": round(float(sac.mean()), 2),
                            "long_median": float(np.median(lac)), "short_median": float(np.median(sac))},
            "weight_sum": {"long_mean": round(float(lw.mean()), 2), "short_mean": round(float(sw.mean()), 2)},
            "weighted_age_hours": {"long_mean": round(float(laa.mean()), 2) if len(laa) else None,
                                    "short_mean": round(float(saa.mean()), 2) if len(saa) else None},
        }

    summary = {"TRAIN": summarize("TRAIN"), "OOS": summarize("OOS")}
    trend = {"TRAIN": trend_bias(df, 0, split_i), "OOS": trend_bias(df, split_i, n)}

    print("\n=== survivor pool: long (support-side) vs short (resistance-side) ===", flush=True)
    for split in ("TRAIN", "OOS"):
        s = summary[split]
        print(f"\n[{split}] n_snapshots={s['n_snapshots']}", flush=True)
        print(f"  alive_count   mean: long={s['alive_count']['long_mean']:.2f}  short={s['alive_count']['short_mean']:.2f}  "
              f"(short/long ratio={s['alive_count']['short_mean']/s['alive_count']['long_mean']:.3f})", flush=True)
        print(f"  weight_sum    mean: long={s['weight_sum']['long_mean']:.2f}  short={s['weight_sum']['short_mean']:.2f}  "
              f"(short/long ratio={s['weight_sum']['short_mean']/s['weight_sum']['long_mean']:.3f})", flush=True)
        la, sa = s['weighted_age_hours']['long_mean'], s['weighted_age_hours']['short_mean']
        print(f"  weighted_age(h) mean: long={la}  short={sa}  (window={v1dir.LOOKBACK_HOURS_LIVE}h)", flush=True)

    print("\n=== trend bias (market regime context) ===", flush=True)
    for split in ("TRAIN", "OOS"):
        t = trend[split]
        print(f"[{split}] net_pct_change={t['net_pct_change']:+.2f}%  n_bars={t['n_bars']}  "
              f"new_24h_high_bars={t['new_24h_high_bars']}  new_24h_low_bars={t['new_24h_low_bars']}  "
              f"(high-low)/n={t['new_high_minus_low_pct_of_bars']:+.3f}%", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "split_bar": split_i,
        "split_ts": str(df["timestamp"].iloc[split_i]), "lookback_hours_live": v1dir.LOOKBACK_HOURS_LIVE,
        "summary": summary, "trend_bias": trend,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

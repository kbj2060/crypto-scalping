#!/usr/bin/env python3
"""2026-08-26 user follow-up: "지지는 그럼 mid로 가는데 저항은 live 버전이 더 좋지 않아?" -- fair
re-read of the 4-fold data: v1_live's resistance pairWR beats v1_mid's in 3/4 folds (fold1 0.742
vs 0.567, fold2 0.458 vs 0.308, fold3 0.606 vs 0.448; only fold0 favors mid, 0.571 vs 0.519) and
v1_live's fold-average (0.581) sits comfortably above coinflip while v1_mid's (0.474) does not --
this is a real, consistent-direction gap, not pure noise as the earlier "resistance is just noisy"
framing implied. Tests a HYBRID: use mid=(high+low)/2 as the entry-price basis for the LONG-side
liquidation-price calc (feeds support, matches v1_mid) while keeping close as the entry-price basis
for the SHORT-side calc (feeds resistance, matches v1_live) -- within the SAME candle.

Structurally this should behave like a superposition of v1_mid's support + v1_live's resistance,
since long_liq/short_liq land in disjoint price bins and the survival filter is computed
independently per side -- the one possible coupling is levels_from_bins()'s MIN_LEVEL_SHARE
threshold, which normalizes against the GLOBAL max bin weight (shared across both sides), so
changing the long side's peak height could in principle shift which resistance bins clear the
threshold. Tested empirically here rather than assumed.

Reuses the identical 4-fold grid (research_eth_liquidation_map_v1_mid_walkforward_multifold_
20260826.py's fold boundaries, byte-identical since both derive from the same v1dir eval grid) so
results sit directly next to the already-reported v1_live/v1_mid fold numbers.
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
import scripts.research_eth_liquidation_map_entry_price_isolated_ab_20260826 as epdir
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_hybrid_price_multifold_20260826.json"

N_FOLDS = 4
VARIANTS = ("v1_live", "v1_mid", "v1_hybrid")


def compute_raw_bins_hybrid(df: pd.DataFrame, current_price: float,
                            long_entry_price: np.ndarray, short_entry_price: np.ndarray):
    """liqmap.compute_raw_bins() generalized to allow DIFFERENT entry-price bases for the long-side
    vs short-side liquidation-price calc. long_entry_price==short_entry_price==close reproduces v1
    exactly (see _identity_check)."""
    if df is None or len(df) < 20 or not (current_price > 0):
        return None
    d = df.reset_index(drop=True)
    n = len(d)
    lep = np.asarray(long_entry_price, dtype="float64")
    sep = np.asarray(short_entry_price, dtype="float64")
    assert len(lep) == n and len(sep) == n

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
        long_liq = lep * (1.0 - 1.0 / lev + liqmap.MAINTENANCE_MARGIN_RATE)
        short_liq = sep * (1.0 + 1.0 / lev - liqmap.MAINTENANCE_MARGIN_RATE)
        add(long_liq, per_tier_weight, future_min_low > long_liq)
        add(short_liq, per_tier_weight, future_max_high < short_liq)

    if not bins or not (max(bins.values()) > 0):
        return None
    return bins, bin_width, n, age_hours


def _identity_check(df: pd.DataFrame) -> None:
    window = df.iloc[-200:].reset_index(drop=True)
    cp = float(window["close"].iloc[-1])
    close_arr = window["close"].to_numpy(dtype="float64")
    raw_v1 = liqmap.compute_raw_bins(window, cp)
    raw_hy = compute_raw_bins_hybrid(window, cp, close_arr, close_arr)
    assert raw_v1 is not None and raw_hy is not None
    bins1, bw1, n1, _ = raw_v1
    bins2, bw2, n2, _ = raw_hy
    assert bw1 == bw2 and n1 == n2 and set(bins1) == set(bins2)
    for k in bins1:
        assert abs(bins1[k] - bins2[k]) < 1e-6 * max(1.0, abs(bins1[k]))
    print("identity check passed: long=short=close reproduces liqmap.compute_raw_bins() exactly", flush=True)


def snapshots_hybrid(df: pd.DataFrame, eval_idxs: list[int],
                     long_ep_full: np.ndarray, short_ep_full: np.ndarray) -> list[dict]:
    close = df["close"].to_numpy()
    out = []
    for i in eval_idxs:
        start = max(0, i - v1dir.LOOKBACK_HOURS_LIVE + 1)
        window = df.iloc[start:i + 1]
        raw = compute_raw_bins_hybrid(window, float(close[i]), long_ep_full[start:i + 1], short_ep_full[start:i + 1])
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
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)

    _identity_check(df)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    mid = (high + low) / 2.0

    n = len(df)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} eval_points={len(eval_idxs)}", flush=True)

    all_snaps = {
        "v1_live": epdir.snapshots_v1_entry_price(df, eval_idxs, close),
        "v1_mid": epdir.snapshots_v1_entry_price(df, eval_idxs, mid),
        "v1_hybrid": snapshots_hybrid(df, eval_idxs, mid, close),
    }
    for var in VARIANTS:
        print(f"{var} snapshots: {len(all_snaps[var])}", flush=True)

    fold_bounds = np.linspace(0, len(eval_idxs), N_FOLDS + 1).astype(int)
    folds = []
    for f in range(N_FOLDS):
        fold_eval_idxs = eval_idxs[fold_bounds[f]: fold_bounds[f + 1]]
        idx_lo, idx_hi = fold_eval_idxs[0], fold_eval_idxs[-1]
        folds.append({"fold": f, "t0_lo": idx_lo, "t0_hi": idx_hi,
                      "ts_lo": str(df["timestamp"].iloc[idx_lo]), "ts_hi": str(df["timestamp"].iloc[idx_hi])})

    results = []
    for k, var in enumerate(VARIANTS):
        snaps = all_snaps[var]
        for fold in folds:
            lo, hi = fold["t0_lo"], fold["t0_hi"]
            sel = [s for s in snaps if lo <= s["t0"] <= hi]
            rng = np.random.default_rng(20260826 + k * 100 + fold["fold"])
            ev = ed.evaluate(df, sel, rng)
            n_lv = [len(s["support_levels"]) + len(s["resistance_levels"]) for s in sel]
            results.append({"variant": var, "fold": fold["fold"], "ts_lo": fold["ts_lo"], "ts_hi": fold["ts_hi"],
                            "n_snapshots": len(sel),
                            "avg_levels_per_snapshot": round(float(np.mean(n_lv)), 2) if n_lv else 0.0, "eval": ev})
            print(f"evaluated {var} fold{fold['fold']} (n={len(sel)})", flush=True)

    print(f"\n{'variant':10s} {'fold':5s} {'period':23s} {'side':11s} {'pairWR':7s} {'holdR':7s} {'holdP':7s} "
          f"{'mag24d':8s} {'mag72d':8s} {'nTouch':6s}", flush=True)
    for r_ in results:
        period = f"{r_['ts_lo'][:10]}~{r_['ts_hi'][:10]}"
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            row = d["by_buffer"]["0.005"]
            mag24 = d["magnitude"]["24"]["mean_diff_pct"]
            mag72 = d["magnitude"]["72"]["mean_diff_pct"]
            print(f"{r_['variant']:10s} {r_['fold']:<5d} {period:23s} {side:11s} "
                  f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                  f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                  f"{('None' if mag24 is None else f'{mag24:+.3f}'):8s} "
                  f"{('None' if mag72 is None else f'{mag72:+.3f}'):8s} "
                  f"{row['real']['n_touched']:6d}")

    # Fold-average pairWR summary per variant/side, the headline comparison
    print("\n=== fold-average pairWR (buf 0.5%%) ===", flush=True)
    for var in VARIANTS:
        for side in ("support", "resistance"):
            wrs = [r["eval"][side]["by_buffer"]["0.005"]["paired"]["winrate"] for r in results
                  if r["variant"] == var and r["eval"][side]["by_buffer"]["0.005"]["paired"]["winrate"] is not None]
            print(f"{var:10s} {side:11s} mean={np.mean(wrs):.3f}  per-fold={['%.3f' % w for w in wrs]}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "n_folds": N_FOLDS, "folds": folds, "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

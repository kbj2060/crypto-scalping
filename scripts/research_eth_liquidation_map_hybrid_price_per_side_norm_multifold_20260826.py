#!/usr/bin/env python3
"""2026-08-26 user follow-up: "이걸로 시도해보자" -- test whether normalizing MIN_LEVEL_SHARE per
side (support's own max bin weight, resistance's own max bin weight) instead of liqmap.
levels_from_bins()'s current GLOBAL max (shared across both sides) fixes the cross-side coupling
that broke the naive hybrid (research_eth_liquidation_map_hybrid_price_multifold_20260826.py):
resistance's own price computation (close, unchanged) still degraded when only the support side's
entry price changed to mid, because a taller support peak raised the shared normalization floor
and silently dropped resistance bins that used to clear MIN_LEVEL_SHARE.

Deliberately does NOT touch scripts/live_liquidation_map_20260824.py (levels_from_bins() there is
what compute_liquidation_levels() actually serves live right now) -- levels_from_bins_per_side()
here is a research-only reimplementation, isolated exactly like every other variant this thread.
Only if this clears the gate would touching the production function even become a question, and
that would need its own explicit go-ahead (it changes already-deployed v1_live behavior for every
user, not just a research variant).

4 variants x 2 normalization schemes = the full comparison:
  entry price:  v1_live (close both sides) | v1_mid (mid both sides) | v1_hybrid (mid long / close short)
  normalization: global-max (production behavior) | per-side-max (this test's proposal)
v1_live_global is the already-reported production baseline; v1_live_perside is a sanity check (in
the fully-symmetric close-both-sides case, per-side vs global should barely differ, since the
survivor-pool diagnostic already found long/short total weight is ~symmetric for v1_live -- if this
sanity check moves v1_live a lot, that itself would be a red flag about the per-side idea).
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
import scripts.research_eth_liquidation_map_hybrid_price_multifold_20260826 as hy
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_hybrid_price_per_side_norm_multifold_20260826.json"

N_FOLDS = 4
ENTRY_VARIANTS = ("v1_live", "v1_mid", "v1_hybrid")
NORM_VARIANTS = ("global", "perside")


def levels_from_bins_per_side(bins: dict, bin_width: float, current_price: float) -> dict:
    """liqmap.levels_from_bins() with ONE change: MIN_LEVEL_SHARE is normalized against each SIDE's
    own max bin weight, not the global max across both sides -- see module docstring for why."""
    support_bins = {b: w for b, w in bins.items() if b * bin_width < current_price}
    resistance_bins = {b: w for b, w in bins.items() if b * bin_width > current_price}
    max_support = max(support_bins.values()) if support_bins else 0.0
    max_resistance = max(resistance_bins.values()) if resistance_bins else 0.0

    def build(side_bins: dict, max_w: float) -> list[dict]:
        if not (max_w > 0):
            return []
        return [
            {"price": b * bin_width, "weight": w, "weight_pct": w / max_w}
            for b, w in side_bins.items()
            if w / max_w >= liqmap.MIN_LEVEL_SHARE
            and abs(b * bin_width - current_price) / current_price <= liqmap.MAX_LEVEL_DISTANCE_PCT
        ]

    support = sorted(build(support_bins, max_support), key=lambda lv: -lv["weight"])[:liqmap.MAX_LEVELS_PER_SIDE]
    resistance = sorted(build(resistance_bins, max_resistance), key=lambda lv: -lv["weight"])[:liqmap.MAX_LEVELS_PER_SIDE]
    support.sort(key=lambda lv: -lv["price"])
    resistance.sort(key=lambda lv: lv["price"])

    def fmt(lv: dict) -> dict:
        return {"price": round(lv["price"], 4), "weight_pct": round(lv["weight_pct"], 4),
                "distance_pct": round((lv["price"] - current_price) / current_price * 100, 3)}

    return {"support_levels": [fmt(lv) for lv in support], "resistance_levels": [fmt(lv) for lv in resistance]}


def snapshots(df: pd.DataFrame, eval_idxs: list[int], long_ep_full: np.ndarray, short_ep_full: np.ndarray,
             norm: str) -> list[dict]:
    close = df["close"].to_numpy()
    out = []
    for i in eval_idxs:
        start = max(0, i - v1dir.LOOKBACK_HOURS_LIVE + 1)
        window = df.iloc[start:i + 1]
        raw = hy.compute_raw_bins_hybrid(window, float(close[i]), long_ep_full[start:i + 1], short_ep_full[start:i + 1])
        if raw is None:
            continue
        bins, bin_width, _, _ = raw
        cp = float(close[i])
        lv = levels_from_bins_per_side(bins, bin_width, cp) if norm == "perside" else liqmap.levels_from_bins(bins, bin_width, cp)
        out.append({"t0": i, "current_price": cp,
                    "support_levels": lv["support_levels"], "resistance_levels": lv["resistance_levels"]})
    return out


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    mid = (high + low) / 2.0
    entry_prices = {"v1_live": (close, close), "v1_mid": (mid, mid), "v1_hybrid": (mid, close)}

    n = len(df)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} eval_points={len(eval_idxs)}", flush=True)

    all_snaps = {}
    for ep_name in ENTRY_VARIANTS:
        lep, sep = entry_prices[ep_name]
        for norm in NORM_VARIANTS:
            key = f"{ep_name}_{norm}"
            all_snaps[key] = snapshots(df, eval_idxs, lep, sep, norm)
            print(f"{key} snapshots: {len(all_snaps[key])}", flush=True)

    fold_bounds = np.linspace(0, len(eval_idxs), N_FOLDS + 1).astype(int)
    folds = []
    for f in range(N_FOLDS):
        fold_eval_idxs = eval_idxs[fold_bounds[f]: fold_bounds[f + 1]]
        idx_lo, idx_hi = fold_eval_idxs[0], fold_eval_idxs[-1]
        folds.append({"fold": f, "t0_lo": idx_lo, "t0_hi": idx_hi,
                      "ts_lo": str(df["timestamp"].iloc[idx_lo]), "ts_hi": str(df["timestamp"].iloc[idx_hi])})

    results = []
    for k, key in enumerate(all_snaps):
        snaps = all_snaps[key]
        for fold in folds:
            lo, hi = fold["t0_lo"], fold["t0_hi"]
            sel = [s for s in snaps if lo <= s["t0"] <= hi]
            rng = np.random.default_rng(20260826 + k * 100 + fold["fold"])
            ev = ed.evaluate(df, sel, rng)
            results.append({"variant": key, "fold": fold["fold"], "n_snapshots": len(sel), "eval": ev})
            print(f"evaluated {key} fold{fold['fold']} (n={len(sel)})", flush=True)

    print("\n=== fold-average pairWR (buf 0.5%%) ===", flush=True)
    for key in all_snaps:
        for side in ("support", "resistance"):
            wrs = [r["eval"][side]["by_buffer"]["0.005"]["paired"]["winrate"] for r in results
                  if r["variant"] == key and r["eval"][side]["by_buffer"]["0.005"]["paired"]["winrate"] is not None]
            print(f"{key:20s} {side:11s} mean={np.mean(wrs):.3f}  per-fold={['%.3f' % w for w in wrs]}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "n_folds": N_FOLDS, "folds": folds, "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Persistence-condition follow-up for liquidity_sweep top/down metalabel (user question,
2026-08-30: "지속성 조건은 왜 뺀거야?" -- "why was persistence excluded?"). The deployed model
(research_eth_liquidity_sweep_topdown_metalabel_final_20260830.py) inherited "touch-based, no
persistence" from taker_delta_z_climax's own template WITHOUT re-testing it specifically for
liquidity_sweep -- this script closes that gap.

Two DISTINCT persistence designs, kept separate on purpose (this project's own history shows they
are not interchangeable):
  1. taker v5's design (SNAPSHOT at exactly bar+HORIZON, binary sign flip at zero) -- ALREADY
     PROVEN BAD for taker (AUC dropped 0.622/0.608/0.650 -> 0.562/0.561/0.606). NOT re-tested here
     verbatim (would just reproduce a known failure mode) -- instead this tests the two designs
     taker's own docstring flagged as untested alternatives that might avoid that noise-sensitivity:
  2. GIVEBACK-RATIO (V_REBOUND's mechanism): full-window peak vs window-END value, ratio-based
     (not a raw sign flip) -- (peak_favorable - end_favorable) / (peak_favorable - entry), gated
     at a threshold. Same "full window, not single bar's raw sign" spirit as V_REBOUND's v7b,
     approximated (not a byte-exact port of that script's formula).
  3. SMOOTHED MAJORITY (taker's own suggested untested fix): majority of the last N bars of the
     window still net-favorable relative to entry, instead of one single bar's close.

Both are ADDED AS AN AND-GATE on top of the EXISTING deployed touch condition (same fire
population -- same H=30/GAP=12 cluster-anchored fires, same K=4.0 touch requirement) -- population
size shrinks (never grows), isolating the persistence-gate's own marginal effect. This does NOT
replicate V_REBOUND's OTHER major lever (excluding the ambiguous middle into a 3rd un-labeled
bucket) -- that is a different, larger redesign (shrinks/reshapes the labeled population itself),
not what "add a persistence condition" asked for.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830 import (  # noqa: E402
    cluster_dedup_by_penetration,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SWEEP_LOOKBACK = 48
HORIZON = 30
GAP = 12
K = 4.0
GIVEBACK_THRESHOLDS = [0.20, 0.35, 0.50]
SMOOTH_N_GRID = [3, 5, 8]
GBM_SEED = 20260830


def log(msg: str) -> None:
    print(f"[liq_sweep_persistence] {msg}", flush=True)


def build_base_fires(klines: pd.DataFrame, ind: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Same cluster-anchored fire population as the deployed model, but keeps the raw forward
    high/low path (not just the touch outcome) so persistence variants can be computed on top."""
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy(); ts = sig["timestamp"].to_numpy(); n = len(sig)
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()

    rows = []
    for side, col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (ts[idx] >= np.datetime64(START))]
        idx = np.sort(idx)
        penetration = (swing_low_prior[idx] - low[idx]) if side == "bottom" else (high[idx] - swing_high_prior[idx])
        idx = cluster_dedup_by_penetration(idx, penetration, GAP)

        entry = close[idx]
        touched = np.zeros(len(idx), dtype=bool)
        giveback_ratio = np.full(len(idx), np.nan)
        smooth_favorable_frac = {n_smooth: np.full(len(idx), np.nan) for n_smooth in SMOOTH_N_GRID}

        for j, i in enumerate(idx):
            if side == "bottom":
                fwd_high = high[i + 1:i + HORIZON + 1]
                fwd_close = close[i + 1:i + HORIZON + 1]
                peak = fwd_high.max()
                touched[j] = (peak - entry[j]) / entry[j] >= K * atr_pct[i]
                if peak > entry[j]:
                    end_val = fwd_close[-1]
                    giveback_ratio[j] = (peak - end_val) / (peak - entry[j])
                for n_smooth in SMOOTH_N_GRID:
                    tail = fwd_close[-n_smooth:]
                    smooth_favorable_frac[n_smooth][j] = float((tail > entry[j]).mean())
            else:
                fwd_low = low[i + 1:i + HORIZON + 1]
                fwd_close = close[i + 1:i + HORIZON + 1]
                trough = fwd_low.min()
                touched[j] = (entry[j] - trough) / entry[j] >= K * atr_pct[i]
                if entry[j] > trough:
                    end_val = fwd_close[-1]
                    giveback_ratio[j] = (end_val - trough) / (entry[j] - trough)
                for n_smooth in SMOOTH_N_GRID:
                    tail = fwd_close[-n_smooth:]
                    smooth_favorable_frac[n_smooth][j] = float((tail < entry[j]).mean())

        feat_rows = ind.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "touched": touched.astype(float), "giveback_ratio": giveback_ratio,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for n_smooth in SMOOTH_N_GRID:
            out[f"smooth_frac_{n_smooth}"] = smooth_favorable_frac[n_smooth]
        for col_name in FEATURE_COLUMNS:
            if col_name != "is_bottom":
                out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def gbm_eval(fires: pd.DataFrame, hit_col: str, tag: str) -> dict:
    fires = fires.dropna(subset=FEATURE_COLUMNS + [hit_col]).reset_index(drop=True)
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START]
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)]
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)]
    clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
    clf.fit(train[FEATURE_COLUMNS], train[hit_col].to_numpy().astype(int))
    val_auc = roc_auc_score(val[hit_col].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos[hit_col].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])
    hit_rate = float(fires[hit_col].mean())
    log(f"  {tag}: n={len(fires)} hit_rate={hit_rate:.3f} VAL={val_auc:.4f} OOS={oos_auc:.4f} min={min(val_auc,oos_auc):.4f}")
    return {"tag": tag, "n": len(fires), "hit_rate": hit_rate, "val_auc": val_auc, "oos_auc": oos_auc}


def main() -> int:
    log("loading klines + building indicator frame + signals...")
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind) and (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()

    log("building base fires (same H=30/GAP=12 cluster-anchored population as deployed model)...")
    fires = build_base_fires(klines, ind, sig)
    log(f"{len(fires)} fires, touched(baseline hit)_rate={fires['touched'].mean():.4f} "
        f"(deployed model's K=4.0 hit_rate was 0.3054 -- should match closely)")

    results = []
    log("\n=== baseline: touch-only (deployed model, no persistence) ===")
    results.append(gbm_eval(fires, "touched", "baseline_touch_only"))

    log("\n=== variant 1: touch AND giveback_ratio<=threshold (V_REBOUND-style, full-window) ===")
    for thresh in GIVEBACK_THRESHOLDS:
        col = f"hit_giveback_{thresh}"
        # touched=False already forces the AND to False regardless of giveback_ratio (NaN-safe:
        # NaN<=thresh is False in numpy) -- population stays identical to the baseline (all fires,
        # not just touched ones), isolating the persistence-gate's own marginal effect.
        fires[col] = (fires["touched"].astype(bool) & (fires["giveback_ratio"] <= thresh)).astype(float)
        results.append(gbm_eval(fires, col, f"giveback<={thresh}"))

    log("\n=== variant 2: touch AND smoothed-majority-favorable-last-N-bars (taker's own suggested untested fix) ===")
    for n_smooth in SMOOTH_N_GRID:
        col = f"hit_smooth_{n_smooth}"
        fires[col] = fires["touched"].astype(bool) & (fires[f"smooth_frac_{n_smooth}"] > 0.5)
        results.append(gbm_eval(fires, col, f"smooth_majority_last{n_smooth}"))

    table = pd.DataFrame(results)
    table["min_val_oos"] = table[["val_auc", "oos_auc"]].min(axis=1)
    table = table.sort_values("min_val_oos", ascending=False)
    log("\n=== RANKED by min(VAL,OOS) ===")
    for _, r in table.iterrows():
        delta_vs_baseline = r["min_val_oos"] - table.loc[table["tag"] == "baseline_touch_only", "min_val_oos"].iloc[0]
        log(f"  {r['tag']:<28s} min={r['min_val_oos']:.4f} (delta vs baseline={delta_vs_baseline:+.4f}) "
            f"VAL={r['val_auc']:.4f} OOS={r['oos_auc']:.4f} hit_rate={r['hit_rate']:.3f} n={int(r['n'])}")

    out_dir = ROOT / "tmp/eth_liquidity_sweep_topdown_metalabel_20260830"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "persistence_variant_gbm_results.csv", index=False)
    log(f"\nsaved -> {out_dir / 'persistence_variant_gbm_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

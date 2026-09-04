#!/usr/bin/env python3
"""Meta-labeling for volume_wick_climax -- Project Homer signal #4, following the reusable
methodology template (docs/homer/README.md) verbatim for the Tier0 23-feature builder / TabPFN
panel / permutation-importance helpers / Fresh-Forward split, all imported from the
taker_delta_z_climax script rather than reimplemented.

Signal definition (live_evidence_signal_dashboard_20260823.py::compute_signals, via
analyze_eth_creative_reversal_evidence_signals_20260814.py::add_creative_indicators):
    vol_z (288-bar/24h rolling volume z-score) >= 2.0 AND lower_wick_ratio >= 0.5 -> bottom
    vol_z >= 2.0 AND upper_wick_ratio >= 0.5 -> top
Unlike taker (delta_z, a signed order-flow z-score) or short_term_return_z (ret3_z, a signed
lagging-return z-score), this fires on a SAME-BAR structural pattern (a volume-climax bar whose own
wick already encodes a rejected intrabar excursion), and its gating condition ANDs two variables
(volume extremity + wick shape) rather than thresholding one signed variable.

Phase 1 diagnostics (scratchpad/research_eth_volume_wick_climax_phase1_diagnostic_20260830.py, not
committed -- descriptive-only, no label decision baked in without the measurements below):
  - At-fire-bar accuracy is 27.0%/26.0% (bottom/top) -- ~2x taker's 14%/13.4%, consistent with the
    same-bar-structural-pattern hypothesis. But the majority of fires (55.5%/53.8%) still have
    their true local extreme AFTER the fire bar (median lag +2 bars/+10min, p90 +22 bars/+110min --
    almost identical tail length to taker's own p90).
  - Naive sign-only direction-hit-rate is flat/weak everywhere (51-54%), best at 15m-1h and decaying
    toward a coin flip by 2h (50.9%) -- same DECAY direction as short_term_return_z, not taker
    (which needed widening to reveal its signal).
  - Clustering is much weaker than either prior signal (median gap 59/87 bars, only 13.6%/9.3% of
    fires within 3 bars of the prior same-side fire, vs taker's 24% / short_term_return_z's 46-50%).

User explicitly asked (2026-08-30) to fold HORIZON and CLUSTER_GAP_MERGE tuning into this SAME
initial build (not decide from phase1 sign-only diagnostics alone, and not defer to a later v2 the
way taker/short_term_return_z's gap-sweeps were separate follow-up sessions). This script therefore
runs a screening grid (HORIZON in {6,12,24} bars x CLUSTER_GAP_MERGE in {3,6,12} bars, 9 combos),
re-calibrating K per combo (K's only job is to balance the hit/no-hit split near 50/50 -- it is not
itself a free hyperparameter being tuned for accuracy, since MFE scale shifts mechanically with
HORIZON) via a fine local grid search, single-seed TRAIN-fit->VAL+OOS AUC per combo (HOLDOUT is
NEVER touched during screening -- selection uses VAL as primary, OOS as secondary/confirmatory,
matching this project's established convention, e.g. the taker v4->v5 and short_term_return_z
gap-sweep decisions). The winning combo then gets the full 4-seed VAL/OOS/HOLDOUT panel +
permutation importance + baseline-lift check -- HOLDOUT is touched exactly ONCE, only for the
single already-selected final config, mirroring exactly how taker_delta_z_climax's v4 script and
short_term_return_z's v1 script each computed a HOLDOUT classification AUC inside their first
adopted run (this is the "research-stage AUC" touch, distinct from and unrelated to the separately
user-gated single-touch TRADING-ECONOMICS holdout exposure done later for both those signals).

Cluster anchor uses vol_z, this signal's own "climax intensity" variable (analogous to taker
anchoring on delta_z / short_term_return_z anchoring on ret3_z) -- but vol_z is an UNSIGNED
magnitude (large volume is "more climax" regardless of side), unlike delta_z/ret3_z which are
signed, so there is no most-negative/most-positive branch by side: both bottom and top clusters
always keep the single loudest-volume bar (idxmax).

Label (touch-based MFE, no persistence check -- established default after taker v5's rejection):
hit = intrabar MFE_pct over bars[fire+1:fire+HORIZON+1] >= K * atr_pct_at_fire. Visually verified
(scratchpad/render_eth_volume_wick_climax_metalabel_v1_20260830.py, 20-example candlestick chart,
HORIZON=12/K=1.40/gap=3 candidate) -- user reviewed, raised one question (why a specific top-side
fire that visually rallied hard was still hit=1) which traced to a genuine intra-window dip that
fully reverted afterward -- confirmed as the intended touch-vs-persistence behavior already
established by taker v5's rejection, not a bug. No label design change resulted from that review.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) -- see handoff.sh push before
executing remotely. Root path is derived dynamically, never hardcoded (dev/server use different
usernames/paths).
"""
from __future__ import annotations

import json
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
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS,
    build_indicator_frame,
    compute_permutation_importance,
    load_klines,
    run_tabpfn_panel,
)

OUT_DIR = ROOT / "data/labels/eth_5m_volume_wick_climax_metalabel_20260830"
REPORT_DIR = ROOT / "tmp/eth_volume_wick_climax_metalabel_tabpfn_20260830"

START = pd.Timestamp("2024-01-01")
VOL_Z_THRESH = 2.0
WICK_RATIO_THRESH = 0.5

HORIZON_GRID = [6, 12, 24]  # 30m/1h/2h -- phase1 sign-hit-rate best at 15m-1h, decays by 2h
GAP_GRID = [3, 6, 12]  # phase1 clustering much weaker than taker/short_term_return_z but nonzero
K_GRID = np.round(np.arange(0.30, 3.01, 0.05), 2)  # fine local search for a ~50/50 split per combo

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds as taker_delta_z_climax/short_term_return_z
SCREEN_SEED = SEEDS[0]


def log(msg: str) -> None:
    print(f"[vwc_metalabel_tabpfn] {msg}", flush=True)


def cluster_dedup_by_vol_z(idx: np.ndarray, vol_z_at_idx: np.ndarray, gap: int) -> np.ndarray:
    """vol_z is an unsigned magnitude -- always keep the cluster's idxmax, unlike taker/short_term_
    return_z's signed-z cluster_dedup (imported taker.cluster_dedup reads a module-level GAP
    constant, not a parameter -- reimplemented here, parameterized by gap, since this script sweeps
    gap as a variable)."""
    order = np.argsort(idx)
    idx_sorted, vz_sorted = idx[order], vol_z_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "vz": vz_sorted})
    keep = df.loc[df.groupby("cluster")["vz"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_raw_fires(klines: pd.DataFrame, indicator_frame: pd.DataFrame, sig: pd.DataFrame,
                     gap: int, horizon: int) -> pd.DataFrame:
    """Fires + features WITHOUT a 'hit' column yet -- pred_dir_ret/atr_pct are computed once per
    (gap, horizon) so K can be swept afterward without recomputing MFE."""
    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    vol_z_all = indicator_frame["vol_z"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_volume_wick_climax"), ("top", "top_volume_wick_climax")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup_by_vol_z(idx, vol_z_all[idx], gap)
        entry = close[idx]
        a = atr_pct[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "entry": entry, "atr_pct": a, "pred_dir_ret": pred_dir_ret,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def calibrate_k(fires_raw: pd.DataFrame) -> tuple[float, list[dict]]:
    """Pooled (both sides together) hit-rate closest to 50% -- matches this project's established
    convention (taker/short_term_return_z both calibrated K to balance the POOLED split, not each
    side independently)."""
    pred = fires_raw["pred_dir_ret"].to_numpy()
    a = fires_raw["atr_pct"].to_numpy()
    table = []
    best_k, best_diff = None, np.inf
    for K in K_GRID:
        hit_rate = float((pred >= K * a).mean())
        table.append({"K": float(K), "hit_rate": round(hit_rate, 4)})
        diff = abs(hit_rate - 0.5)
        if diff < best_diff:
            best_diff, best_k = diff, float(K)
    return best_k, table


def apply_k(fires_raw: pd.DataFrame, K: float) -> pd.DataFrame:
    fires = fires_raw.copy()
    fires["hit"] = (fires["pred_dir_ret"] >= K * fires["atr_pct"]).astype(float)
    return fires


def random_bar_baseline_wick_only(indicator_frame: pd.DataFrame, klines: pd.DataFrame, horizon: int, K: float) -> dict:
    """Analog of taker/short_term_return_z's random_bar_baseline, adapted for a two-variable AND
    condition: this signal's OWN gating threshold is vol_z>=2.0 AND wick_ratio>=0.5, so there is no
    single 'raw sign' to bet continuously the way delta_z/ret3_z's sign works. Instead this isolates
    the value of the VOLUME-CLIMAX gate specifically: keep the wick-ratio condition (it defines
    direction) but drop the vol_z>=2.0 requirement, applying the SAME MFE/K/horizon hit rule to
    every bar meeting the wick condition alone (not deduped -- matches the other 2 signals' baseline
    convention of using raw undeduped bars for the 'naive always-on' comparison)."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    n = len(klines)
    fwd_high_max = high[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1].shift(-1)
    fwd_low_min = low[::-1].rolling(window=horizon, min_periods=horizon).min()[::-1].shift(-1)
    mfe_up_pct = ((fwd_high_max - close) / close).to_numpy()
    mfe_down_pct = ((close - fwd_low_min) / close).to_numpy()

    lower_wick = indicator_frame["lower_wick_ratio"].to_numpy()
    upper_wick = indicator_frame["upper_wick_ratio"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = indicator_frame["timestamp"].to_numpy()
    valid_base = (ts >= np.datetime64(START)) & (np.arange(n) < n - horizon) & np.isfinite(atr_pct)

    bottom_idx = np.flatnonzero(valid_base & np.isfinite(lower_wick) & (lower_wick >= WICK_RATIO_THRESH))
    top_idx = np.flatnonzero(valid_base & np.isfinite(upper_wick) & (upper_wick >= WICK_RATIO_THRESH))
    bottom_hit = mfe_up_pct[bottom_idx] >= K * atr_pct[bottom_idx]
    top_hit = mfe_down_pct[top_idx] >= K * atr_pct[top_idx]
    n_total = len(bottom_idx) + len(top_idx)
    hit_rate = (bottom_hit.sum() + top_hit.sum()) / n_total
    return {"n": int(n_total), "wick_only_no_volume_gate_hit_rate": float(hit_rate)}


def split_train_val_oos(fires: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    return train, val, oos


def screen_one_combo(klines: pd.DataFrame, indicator_frame: pd.DataFrame, sig: pd.DataFrame,
                      horizon: int, gap: int) -> tuple[dict, pd.DataFrame]:
    from tabpfn import TabPFNClassifier

    fires_raw = build_raw_fires(klines, indicator_frame, sig, gap, horizon)
    n_before_dropna = len(fires_raw)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    K, _k_table = calibrate_k(fires_raw)
    fires = apply_k(fires_raw, K)

    train, val, oos = split_train_val_oos(fires)
    clf = TabPFNClassifier(device="cuda", random_state=SCREEN_SEED)
    clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
    val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])

    row = {
        "horizon": horizon, "gap": gap, "K": K,
        "n_fires_before_dropna": n_before_dropna, "n_fires": int(len(fires)),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_oos": int(len(oos)),
        "hit_rate": round(float(fires["hit"].mean()), 4),
        "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
    }
    log(f"[screen] H={horizon:>2d} gap={gap:>2d} K={K:.2f}: n={row['n_fires']} (train={row['n_train']}/val={row['n_val']}/oos={row['n_oos']}) "
        f"hit_rate={row['hit_rate']:.3f} VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f}")
    return row, fires


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines + building Tier0 indicator frame + compute_signals...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame), "row count mismatch between compute_signals and indicator_frame"
    assert (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all(), "timestamp misalignment"

    both_vz = pd.concat([sig["vol_z"], indicator_frame["vol_z"]], axis=1).dropna()
    log(f"vol_z cross-check (compute_signals vs build_indicator_frame): corr={both_vz.iloc[:,0].corr(both_vz.iloc[:,1]):.6f}, "
        f"max_abs_diff={(both_vz.iloc[:,0]-both_vz.iloc[:,1]).abs().max():.6f}")

    log(f"=== screening grid: HORIZON in {HORIZON_GRID} x CLUSTER_GAP_MERGE in {GAP_GRID} "
        f"(single seed={SCREEN_SEED}, TRAIN-fit -> VAL+OOS AUC, HOLDOUT untouched) ===")
    screening_rows = []
    fires_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for horizon in HORIZON_GRID:
        for gap in GAP_GRID:
            row, fires = screen_one_combo(klines, indicator_frame, sig, horizon, gap)
            screening_rows.append(row)
            fires_cache[(horizon, gap)] = fires

    best = max(screening_rows, key=lambda r: r["val_auc"])
    log(f"=== SELECTED (by VAL AUC): HORIZON={best['horizon']} GAP={best['gap']} K={best['K']:.2f} "
        f"(VAL_AUC={best['val_auc']:.4f}, OOS_AUC={best['oos_auc']:.4f}) ===")

    horizon_f, gap_f, K_f = best["horizon"], best["gap"], best["K"]
    fires = fires_cache[(horizon_f, gap_f)]
    log(f"final fire counts: total={len(fires)} (bottom={int((fires['side']=='bottom').sum())}, "
        f"top={int((fires['side']=='top').sum())})")

    log("running wick-only (no volume-climax gate) baseline check at the selected HORIZON/K...")
    baseline = random_bar_baseline_wick_only(indicator_frame, klines, horizon_f, K_f)
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal (vol_z>={VOL_Z_THRESH} AND wick_ratio>={WICK_RATIO_THRESH}) hit rate: {fire_hit_rate:.4f} "
        f"vs wick-only-no-volume-gate baseline: {baseline['wick_only_no_volume_gate_hit_rate']:.4f} "
        f"(lift {fire_hit_rate / baseline['wick_only_no_volume_gate_hit_rate']:.3f}x)")

    train, val, oos = split_train_val_oos(fires)
    holdout = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(OUT_DIR / "eth_5m_volume_wick_climax_metalabel_features.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"acc {val_result['accuracy_mean']:.4f}  bal_acc {val_result['balanced_accuracy_mean']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"acc {oos_result['accuracy_mean']:.4f}  bal_acc {oos_result['balanced_accuracy_mean']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (2026-04-01~latest, research-stage AUC, single touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}  "
            f"acc {holdout_result['accuracy_mean']:.4f}  bal_acc {holdout_result['balanced_accuracy_mean']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "volume_wick_climax",
        "adopted_version": "v1",
        "status": "exploratory_single_signal_below_promotion_bar",
        "summary_for_future_sessions": (
            f"v1 (this run): HORIZON/CLUSTER_GAP_MERGE selected via a 9-combo screening grid "
            f"(HORIZON in {HORIZON_GRID}, GAP in {GAP_GRID}, single-seed TRAIN-fit VAL+OOS AUC, "
            f"HOLDOUT untouched during screening) -- winner HORIZON={horizon_f}, GAP={gap_f}, "
            f"K={K_f:.2f} (K re-calibrated per combo to keep the pooled hit-rate near 50%, not "
            f"itself a free/tuned parameter). hit = touched (intrabar MFE_pct over "
            f"bars[fire+1:fire+{horizon_f}+1] >= {K_f:.2f}*atr_pct_at_fire), fires cluster-anchored "
            f"on vol_z (unsigned magnitude -> always idxmax, unlike taker/short_term_return_z's "
            f"signed-z idxmin/idxmax-by-side). Visually reviewed by user (20-example candlestick "
            f"chart, HORIZON=12/K=1.40/gap=3 candidate before the grid confirmed the final combo) -- "
            f"one question raised and resolved (a touch-then-full-reversal top-side example), no "
            f"label design change. Full screening table + phase1 diagnostics: see report fields "
            f"below / docs/homer/README.md."
        ),
        "screening_grid": screening_rows,
        "selected_horizon": horizon_f, "selected_gap": gap_f, "selected_K": K_f,
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "vol_z_crosscheck_note": "see log for compute_signals vs build_indicator_frame vol_z corr/max_abs_diff",
        "random_bar_baseline_wick_only": baseline,
        "fired_signal_hit_rate": fire_hit_rate,
        "lift_vs_wick_only_baseline": fire_hit_rate / baseline["wick_only_no_volume_gate_hit_rate"],
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
    }
    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

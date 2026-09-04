#!/usr/bin/env python3
"""Meta-labeling for orthogonal_combo -- Project Homer signal #6 (the "flagship" 3-4-condition
combo, historically the #1-lift signal of the original 8, 3.56x @1h in the 2026-08-25 rule-based
recheck). Follows the reusable methodology template (docs/homer/README.md) for the Tier0
23-feature builder / TabPFN panel / permutation-importance helpers / Fresh-Forward split, imported
verbatim from the taker_delta_z_climax script.

Signal definition (live_evidence_signal_dashboard_20260823.py::compute_signals):
    bottom = (p_fast<=0.10) & (p_slow<=0.10) & ((delta_z<=-2.0) | (funding_z<=-2.0))
    top    = (p_fast>=0.90) & (p_slow>=0.90) & (delta_z>=2.0)   # funding_z deliberately excluded
funding_z data (data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv) covers only 2025-01-01~2026-07-31 --
compute_signals()'s own merge_asof + NaN-comparison-is-False semantics already degrades bottom-leg
fires to delta_z-only gracefully outside that window (confirmed: 675/2111 = 32% of bottom fires in
phase1 predate 2025-01-01), matching the live system's own designed fallback -- not special-cased
here. funding_z is NOT added as a model feature (would force dropping the entire pre-2025 TRAIN
population via dropna); its contribution is already fully captured by the fire condition itself.

Phase1 diagnostics (scratchpad/research_eth_orthogonal_combo_phase1_diagnostic_20260830.py, not
committed) found the STRONGEST profile of any signal in this project so far: fire-bar exact-at-
extreme 9.3%/10.5%, median lag only +6 bars/+30min (p90 +22 bars/+110min -- same tail length as
every other signal checked), naive sign-hit-rate a robust 54-58% across ALL horizons 15min-4h.

Cluster anchor = p_fast+p_slow extremity (bottom: idxmax of -(p_fast+p_slow), i.e. most jointly
oversold; top: idxmax of +(p_fast+p_slow)) -- chosen because p_fast/p_slow are the ALWAYS-required
leg for both sides, unlike delta_z/funding_z which is an OR-confirmation leg that varies which one
actually fired (19.4% of bottom fires are funding_z-driven, incl. 360 funding-only).

⭐Label v2 -- EXCLUDE-MIDDLE (user request 2026-08-30, same principle as liquidity_sweep v7b):
v1 (single K threshold at the pooled-50/50 balance point) was found to have its NO_HIT population
heavily concentrated just BELOW the threshold (median NO_HIT move_atr_mult = 0.997 out of K=1.75,
only 9.2% of NO_HIT cases were "clear misses" in [0,0.3K) -- vs 18-32% for the other 4 already-
completed signals, confirmed via a cross-signal check to be a genuine outlier, not just a universal
K-calibrated-for-balance artifact). Fix: HIT(1) = move_atr_mult>=K_hi, MISS(0) = move_atr_mult<=
K_lo, EXCLUDE K_lo<move_atr_mult<K_hi entirely from training/eval. K_lo/K_hi derived from the same
single-K balance point K_center (pooled ~50% hit rate, as calibrated for every other signal) via a
fixed ratio: K_lo = K_center/1.4, K_hi = 2*K_lo (=K_center*10/7) -- reproduces the user-approved
visual-check candidate exactly at K_center=1.75 (K_lo=1.25, K_hi=2.5, 64% kept, 49.6% hit-rate-of-
kept). User explicitly declined to retrofit this fix onto the other 4 already-deployed signals
(all already spent their single-touch HOLDOUT, volume_wick_climax twice) after a cross-signal
check found them meaningfully healthier (18-32% clear-miss fraction) -- orthogonal_combo only.

Per the volume_wick_climax methodology lesson (docs/homer/README.md 5.5), HORIZON is screened with
a dense grid from the start (not a sparse 3-point one), and selection uses max(min(VAL,OOS)) (not
raw VAL-max) to avoid picking a VAL-overfit point.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN).
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
    FEATURE_COLUMNS, build_indicator_frame, compute_permutation_importance, load_klines, run_tabpfn_panel,
)

OUT_DIR = ROOT / "data/labels/eth_5m_orthogonal_combo_metalabel_20260830"
REPORT_DIR = ROOT / "tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830"
FUNDING_PATH = ROOT / "data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv"

START = pd.Timestamp("2024-01-01")
K_CENTER_GRID = np.round(np.arange(0.50, 3.51, 0.05), 2)  # search range for the pooled-50% balance point
K_LO_RATIO = 1.0 / 1.4  # K_lo = K_center * this
K_HI_RATIO = 2.0 / 1.4  # K_hi = K_center * this (= 2*K_lo)

HORIZON_GRID = [6, 8, 12, 16, 20, 24, 30, 36, 48]
GAP_GRID = [3, 6, 12]

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]
SCREEN_SEED = SEEDS[0]


def log(msg: str) -> None:
    print(f"[orthogonal_metalabel_tabpfn] {msg}", flush=True)


def load_funding_z() -> pd.DataFrame:
    """Verbatim formula from research_eth_funding_crossasset_combo_signal_20260825.py::load_funding_z."""
    f = pd.read_csv(FUNDING_PATH, parse_dates=["calc_time"])
    f = f.sort_values("calc_time").reset_index(drop=True)
    mean = f["last_funding_rate"].rolling(90, min_periods=30).mean()
    std = f["last_funding_rate"].rolling(90, min_periods=30).std()
    f["funding_z"] = (f["last_funding_rate"] - mean) / std.replace(0.0, np.nan)
    return f[["calc_time", "funding_z"]]


def cluster_dedup_oscillator(idx: np.ndarray, p_fast: np.ndarray, p_slow: np.ndarray, side: str, gap: int) -> np.ndarray:
    score = -(p_fast[idx] + p_slow[idx]) if side == "bottom" else (p_fast[idx] + p_slow[idx])
    order = np.argsort(idx)
    idx_sorted, s_sorted = idx[order], score[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "s": s_sorted})
    return np.sort(df.loc[df.groupby("cluster")["s"].idxmax()]["idx"].to_numpy())


def build_raw_fires(indicator_frame: pd.DataFrame, sig: pd.DataFrame, gap: int, horizon: int) -> pd.DataFrame:
    """Fires with move_atr_mult computed, no hit/exclude decision applied yet."""
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    p_fast = indicator_frame["p_fast"].to_numpy()
    p_slow = indicator_frame["p_slow"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_orthogonal_combo"), ("top", "top_orthogonal_combo")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup_oscillator(idx, p_fast, p_slow, side, gap)
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
            "move_atr_mult": pred_dir_ret / a,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def calibrate_k_center(fires_raw: pd.DataFrame) -> float:
    mam = fires_raw["move_atr_mult"].to_numpy()
    best_k, best_diff = None, np.inf
    for K in K_CENTER_GRID:
        diff = abs(float((mam >= K).mean()) - 0.5)
        if diff < best_diff:
            best_diff, best_k = diff, float(K)
    return best_k


def apply_exclude_middle(fires_raw: pd.DataFrame, k_center: float) -> tuple[pd.DataFrame, float, float]:
    k_lo, k_hi = round(k_center * K_LO_RATIO, 3), round(k_center * K_HI_RATIO, 3)
    mam = fires_raw["move_atr_mult"]
    is_hit, is_miss = mam >= k_hi, mam <= k_lo
    kept = fires_raw[is_hit | is_miss].copy()
    kept["hit"] = is_hit[is_hit | is_miss].astype(float).to_numpy()
    return kept, k_lo, k_hi


def random_bar_baseline_oscillator_only(indicator_frame: pd.DataFrame, klines: pd.DataFrame, horizon: int, k_lo: float, k_hi: float) -> dict:
    """Isolates the value of the delta_z/funding_z CONFIRMATION leg: keep the oscillator-extreme
    condition (p_fast/p_slow<=.10 or >=.90, it defines direction) but drop the confirmation leg,
    applying the SAME exclude-middle MFE rule to every bar meeting the oscillator condition alone."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    n = len(klines)
    fwd_high_max = high[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1].shift(-1)
    fwd_low_min = low[::-1].rolling(window=horizon, min_periods=horizon).min()[::-1].shift(-1)
    mfe_up_pct = ((fwd_high_max - close) / close).to_numpy()
    mfe_down_pct = ((close - fwd_low_min) / close).to_numpy()

    p_fast = indicator_frame["p_fast"].to_numpy()
    p_slow = indicator_frame["p_slow"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = indicator_frame["timestamp"].to_numpy()
    valid_base = (ts >= np.datetime64(START)) & (np.arange(n) < n - horizon) & np.isfinite(atr_pct)

    bottom_idx = np.flatnonzero(valid_base & (p_fast <= 0.10) & (p_slow <= 0.10))
    top_idx = np.flatnonzero(valid_base & (p_fast >= 0.90) & (p_slow >= 0.90))
    bottom_mam = mfe_up_pct[bottom_idx] / atr_pct[bottom_idx]
    top_mam = mfe_down_pct[top_idx] / atr_pct[top_idx]
    all_mam = np.concatenate([bottom_mam, top_mam])
    is_hit, is_miss = all_mam >= k_hi, all_mam <= k_lo
    kept = is_hit | is_miss
    n_total = int(kept.sum())
    hit_rate = float(is_hit.sum() / n_total) if n_total else float("nan")
    return {"n_raw": int(len(all_mam)), "n_kept": n_total, "oscillator_only_hit_rate_of_kept": hit_rate}


def split_train_val_oos(fires: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    return train, val, oos


def screen_one_combo(indicator_frame: pd.DataFrame, sig: pd.DataFrame, horizon: int, gap: int) -> tuple[dict, pd.DataFrame]:
    from tabpfn import TabPFNClassifier

    fires_raw = build_raw_fires(indicator_frame, sig, gap, horizon)
    n_before_dropna = len(fires_raw)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    k_center = calibrate_k_center(fires_raw)
    fires, k_lo, k_hi = apply_exclude_middle(fires_raw, k_center)

    train, val, oos = split_train_val_oos(fires)
    clf = TabPFNClassifier(device="cuda", random_state=SCREEN_SEED)
    clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
    val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])

    row = {
        "horizon": horizon, "gap": gap, "k_center": k_center, "k_lo": k_lo, "k_hi": k_hi,
        "n_fires_before_dropna": n_before_dropna, "n_fires_raw_after_dropna": len(fires_raw), "n_fires_kept": int(len(fires)),
        "kept_frac": round(len(fires) / len(fires_raw), 4),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_oos": int(len(oos)),
        "hit_rate": round(float(fires["hit"].mean()), 4),
        "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
        "gap_val_oos": round(abs(float(val_auc) - float(oos_auc)), 4),
    }
    log(f"[screen] H={horizon:>2d} gap={gap:>2d} k_center={k_center:.2f}(lo={k_lo:.2f}/hi={k_hi:.2f}): "
        f"kept={row['n_fires_kept']}({row['kept_frac']*100:.0f}%) train={row['n_train']}/val={row['n_val']}/oos={row['n_oos']} "
        f"hit_rate={row['hit_rate']:.3f} VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f} gap={row['gap_val_oos']:.4f}")
    return row, fires


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines + building Tier0 indicator frame + funding_z + compute_signals...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    funding_df = load_funding_z()
    log(f"funding_z: {funding_df['calc_time'].min()} ~ {funding_df['calc_time'].max()}, n={len(funding_df)}")
    sig = compute_signals(klines, btc_df=None, funding_df=funding_df).reset_index(drop=True)
    assert len(sig) == len(indicator_frame) and (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all()

    log(f"=== screening grid: HORIZON in {HORIZON_GRID} x CLUSTER_GAP_MERGE in {GAP_GRID} "
        f"({len(HORIZON_GRID)*len(GAP_GRID)} combos, single seed={SCREEN_SEED}, exclude-middle labeling, "
        f"TRAIN-fit -> VAL+OOS AUC, HOLDOUT untouched) ===")
    screening_rows = []
    fires_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for horizon in HORIZON_GRID:
        for gap in GAP_GRID:
            row, fires = screen_one_combo(indicator_frame, sig, horizon, gap)
            screening_rows.append(row)
            fires_cache[(horizon, gap)] = fires

    by_val_max = max(screening_rows, key=lambda r: r["val_auc"])
    by_min_auc = max(screening_rows, key=lambda r: min(r["val_auc"], r["oos_auc"]))
    log(f"if selected by raw VAL max: H={by_val_max['horizon']} GAP={by_val_max['gap']} "
        f"(VAL={by_val_max['val_auc']:.4f} OOS={by_val_max['oos_auc']:.4f} gap={by_val_max['gap_val_oos']:.4f})")
    log(f"if selected by max(min(VAL,OOS)): H={by_min_auc['horizon']} GAP={by_min_auc['gap']} "
        f"(VAL={by_min_auc['val_auc']:.4f} OOS={by_min_auc['oos_auc']:.4f} gap={by_min_auc['gap_val_oos']:.4f})")
    best = by_min_auc
    log(f"=== SELECTED (by max(min(VAL,OOS))): HORIZON={best['horizon']} GAP={best['gap']} "
        f"k_center={best['k_center']:.2f} (k_lo={best['k_lo']:.2f}/k_hi={best['k_hi']:.2f}) ===")

    horizon_f, gap_f = best["horizon"], best["gap"]
    k_lo_f, k_hi_f = best["k_lo"], best["k_hi"]
    fires = fires_cache[(horizon_f, gap_f)]
    log(f"final fire counts: total={len(fires)} (bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    baseline = random_bar_baseline_oscillator_only(indicator_frame, klines, horizon_f, k_lo_f, k_hi_f)
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal hit-rate-of-kept: {fire_hit_rate:.4f} vs oscillator-only-no-confirmation-leg baseline: "
        f"{baseline['oscillator_only_hit_rate_of_kept']:.4f} (lift {fire_hit_rate/baseline['oscillator_only_hit_rate_of_kept']:.3f}x)")

    train, val, oos = split_train_val_oos(fires)
    holdout = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(OUT_DIR / "eth_5m_orthogonal_combo_metalabel_features.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (single touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}")

    log("=== permutation feature importance (VAL, single seed, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f}")

    report = {
        "signal": "orthogonal_combo", "adopted_version": "v2_exclude_middle",
        "status": "exploratory_single_signal_below_promotion_bar",
        "screening_grid": screening_rows,
        "selected_by": "max(min(VAL,OOS))", "selection_alt_by_val_max": by_val_max,
        "selected_horizon": horizon_f, "selected_gap": gap_f,
        "selected_k_center": best["k_center"], "selected_k_lo": k_lo_f, "selected_k_hi": k_hi_f,
        "feature_columns": FEATURE_COLUMNS, "n_fires_total": int(len(fires)),
        "random_bar_baseline_oscillator_only": baseline,
        "fired_signal_hit_rate_of_kept": fire_hit_rate,
        "lift_vs_oscillator_only_baseline": fire_hit_rate / baseline["oscillator_only_hit_rate_of_kept"],
        "val": val_result, "oos": oos_result, "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
    }
    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

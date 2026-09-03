#!/usr/bin/env python3
"""BTC taker_delta_z_climax: HORIZON x K grid-screen + Tier0 feature analysis ONLY.

User request (2026-09-01): redo this project's Homer evidence-signal grid-screening
methodology for BTC's own `taker_delta_z_climax` trigger -- grid-screen + feature analysis
ONLY. No TabPFN training, no economic/cost-gate backtest, no HOLDOUT exposure -- those are
future work pending human review of this pass's results.

Data: data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_
tier0.csv (already built by scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py,
277,191 rows, 2024-01-01..2026-08-20, 5m BTCUSDT). Trigger columns bottom_taker_delta_z_climax /
top_taker_delta_z_climax are ALREADY computed there (verified here to be exactly delta_z<=-2.0 /
delta_z>=2.0, no clustering/dedup applied -- unlike ETH's later v4/v5 scripts, which added a
cluster-anchor dedup step on top of the same raw trigger. This script intentionally does NOT
dedup/cluster, matching the literal 5-step methodology given for this pass; clustering is a
possible refinement for a later round, not part of this grid-screen).

Methodology (mirrors ETH's phase0/v2/v3-era grid-screening approach before ETH's later
TabPFN/dedup refinements):
  1. Fresh-Forward split by date: TRAIN<2025-09-01, VAL=2025-09-01..2025-12-31 (OOS/HOLDOUT are
     defined for date math but NEVER evaluated in this script).
  2. For each (HORIZON, K) in a 5x5 grid: hit = touch-based MFE using intrabar high/low --
     bottom (predict UP): high[i+1:i+HORIZON+1].max() >= close[i] + K*atr[i]; top (predict DOWN):
     low[i+1:i+HORIZON+1].min() <= close[i] - K*atr[i]. atr is the CSV's raw price-scale ATR
     column (true_range.rolling(14).mean(), see build_eth_5m_sweep_followthrough_v2_labels_
     20260829.py::add_causal_columns) -- NOT atr_pct, so `close + K*atr` is dimensionally a price.
  3. TRAIN lift = fired-candidate pooled hit rate / random-baseline hit rate, where baseline
     draws the SAME COUNT of random non-trigger TRAIN bars (neither bottom nor top fired) and
     applies the SAME mirrored direction/threshold check (bottom-side draws checked in the "up"
     direction, top-side in "down") -- isolates whether the |delta_z|>=2 extremity threshold
     itself adds lift over an unconditional "how often does a random bar move K*ATR in HORIZON
     bars" base rate. Fixed RNG seed for reproducibility.
  4. VAL hit rate is reported for every grid cell (no resampled baseline); a VAL baseline +
     lift is additionally computed ONLY for the chosen (HORIZON,K), to confirm the TRAIN lift's
     direction/magnitude holds out of sample without re-searching on VAL.
  5. Feature analysis at the chosen (HORIZON,K): pooled bottom+top TRAIN candidates (dropna on
     the literal 21+rsi Tier0 set given for this task), (a) Pearson r (== point-biserial
     correlation when one side is the binary hit label) per feature vs hit, TRAIN only, and
     (b) a quick HistGradientBoostingClassifier fit on TRAIN -> permutation importance (roc_auc
     scoring) measured on VAL. `is_bottom` is added as one extra structural column beyond the
     task's literal 21+rsi Tier0 list (labeled separately in the report/doc, not silently folded
     into "Tier0") -- needed to let a POOLED bottom+top model/correlation distinguish direction;
     this mirrors ETH's own FEATURE_COLUMNS (research_eth_taker_delta_climax_metalabel_tabpfn_
     20260829.py), which also carries is_bottom as its first feature for the same reason.

HOLDOUT (>=2026-04-01) is never read, filtered, or computed on in this script.

Run: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_taker_delta_climax_gridscreen_20260901.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/taker_delta_climax_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")  # boundary only -- HOLDOUT itself never touched

HORIZON_GRID = [12, 18, 24, 30, 36]
K_GRID = [1.5, 2.0, 2.4, 2.8, 3.2]

# Literal "21+rsi" Tier0 feature set as specified for this task (includes raw sweep_level_low/high,
# not converted to a candidate-relative penetration distance -- per task instructions).
TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "p_fast", "p_slow",
    "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]
assert len(TIER0_FEATURES) == 22  # 21 + rsi, per task description

RNG_SEED = 20260901
MIN_TRAIN_CANDIDATES = 200  # "few hundred+ TRAIN candidates minimum" floor from the task


def log(msg: str) -> None:
    print(f"[btc_taker_gridscreen] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    usecols = list(dict.fromkeys(
        ["timestamp", "high", "low", "close", "atr",
         "bottom_taker_delta_z_climax", "top_taker_delta_z_climax"] + TIER0_FEATURES
    ))
    df = pd.read_csv(CSV_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def forward_extremes(high: np.ndarray, low: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """fwd_high_max[i] = max(high[i+1:i+horizon+1]); fwd_low_min[i] = min(low[i+1:i+horizon+1]).
    Vectorized: backward rolling(window=horizon) is right-aligned so at position i+horizon it
    equals max(high[i+1..i+horizon]); shifting the whole series by -horizon moves that value back
    to position i. Verified by hand against a direct per-row loop on a 5-element toy array."""
    fwd_high_max = pd.Series(high).rolling(horizon, min_periods=horizon).max().shift(-horizon).to_numpy()
    fwd_low_min = pd.Series(low).rolling(horizon, min_periods=horizon).min().shift(-horizon).to_numpy()
    return fwd_high_max, fwd_low_min


def random_baseline_hit(rng: np.random.Generator, pool_idx: np.ndarray, n_draw: int,
                         fwd_ext: np.ndarray, close: np.ndarray, atr: np.ndarray,
                         k: float, direction: str) -> float:
    if n_draw <= 0 or len(pool_idx) < n_draw:
        return float("nan")
    samp = rng.choice(pool_idx, size=n_draw, replace=False)
    if direction == "up":
        hit = fwd_ext[samp] >= close[samp] + k * atr[samp]
    else:
        hit = fwd_ext[samp] <= close[samp] - k * atr[samp]
    return float(hit.mean())


def main() -> int:
    log("loading BTC Tier0 candidates CSV...")
    df = load_data()
    log(f"{len(df)} rows loaded, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    ts_col = df["timestamp"]

    bottom_trig = df["bottom_taker_delta_z_climax"].fillna(False).to_numpy(dtype=bool)
    top_trig = df["top_taker_delta_z_climax"].fillna(False).to_numpy(dtype=bool)
    any_trig = bottom_trig | top_trig
    assert not (bottom_trig & top_trig).any(), "bottom/top should be mutually exclusive (delta_z can't be both <=-2 and >=2)"

    train_mask = (ts_col < VAL_START).to_numpy()
    val_mask = ((ts_col >= VAL_START) & (ts_col < OOS_START)).to_numpy()
    finite_ok = np.isfinite(close) & np.isfinite(atr)

    log(f"raw trigger counts: bottom={int(bottom_trig.sum())} top={int(top_trig.sum())} "
        f"(TRAIN bottom={int((bottom_trig & train_mask).sum())} top={int((top_trig & train_mask).sum())}, "
        f"VAL bottom={int((bottom_trig & val_mask).sum())} top={int((top_trig & val_mask).sum())})")

    grid_rows = []
    master_rng = np.random.default_rng(RNG_SEED)
    cache = {}  # horizon -> (fwd_high_max, fwd_low_min, bottom_idx_all, top_idx_all, pool_idx_train)

    for horizon in HORIZON_GRID:
        fwd_high_max, fwd_low_min = forward_extremes(high, low, horizon)
        valid_fwd = np.isfinite(fwd_high_max) & np.isfinite(fwd_low_min) & finite_ok

        bottom_idx_all = np.flatnonzero(bottom_trig & valid_fwd)
        top_idx_all = np.flatnonzero(top_trig & valid_fwd)
        pool_idx_train = np.flatnonzero(train_mask & (~any_trig) & valid_fwd)
        cache[horizon] = (fwd_high_max, fwd_low_min, bottom_idx_all, top_idx_all, pool_idx_train)

        for k in K_GRID:
            bottom_hit_all = fwd_high_max[bottom_idx_all] >= close[bottom_idx_all] + k * atr[bottom_idx_all]
            top_hit_all = fwd_low_min[top_idx_all] <= close[top_idx_all] - k * atr[top_idx_all]

            b_train_m, b_val_m = train_mask[bottom_idx_all], val_mask[bottom_idx_all]
            t_train_m, t_val_m = train_mask[top_idx_all], val_mask[top_idx_all]
            n_b_train, n_t_train = int(b_train_m.sum()), int(t_train_m.sum())
            n_b_val, n_t_val = int(b_val_m.sum()), int(t_val_m.sum())

            b_train_hitrate = float(bottom_hit_all[b_train_m].mean()) if n_b_train else float("nan")
            t_train_hitrate = float(top_hit_all[t_train_m].mean()) if n_t_train else float("nan")
            b_val_hitrate = float(bottom_hit_all[b_val_m].mean()) if n_b_val else float("nan")
            t_val_hitrate = float(top_hit_all[t_val_m].mean()) if n_t_val else float("nan")

            pooled_train_n = n_b_train + n_t_train
            pooled_val_n = n_b_val + n_t_val
            pooled_train_hit = float((bottom_hit_all[b_train_m].sum() + top_hit_all[t_train_m].sum()) / max(pooled_train_n, 1))
            pooled_val_hit = float((bottom_hit_all[b_val_m].sum() + top_hit_all[t_val_m].sum()) / max(pooled_val_n, 1))

            b_base = random_baseline_hit(master_rng, pool_idx_train, n_b_train, fwd_high_max, close, atr, k, "up")
            t_base = random_baseline_hit(master_rng, pool_idx_train, n_t_train, fwd_low_min, close, atr, k, "down")
            pooled_base = (b_base * n_b_train + t_base * n_t_train) / pooled_train_n if pooled_train_n and np.isfinite(b_base) and np.isfinite(t_base) else float("nan")

            lift_bottom = b_train_hitrate / b_base if np.isfinite(b_base) and b_base > 0 else float("nan")
            lift_top = t_train_hitrate / t_base if np.isfinite(t_base) and t_base > 0 else float("nan")
            lift_pooled = pooled_train_hit / pooled_base if np.isfinite(pooled_base) and pooled_base > 0 else float("nan")

            grid_rows.append({
                "horizon": horizon, "k": k,
                "n_train_bottom": n_b_train, "n_train_top": n_t_train,
                "n_val_bottom": n_b_val, "n_val_top": n_t_val,
                "train_hitrate_bottom": round(b_train_hitrate, 4) if np.isfinite(b_train_hitrate) else None,
                "train_hitrate_top": round(t_train_hitrate, 4) if np.isfinite(t_train_hitrate) else None,
                "train_hitrate_pooled": round(pooled_train_hit, 4),
                "train_baseline_bottom": round(b_base, 4) if np.isfinite(b_base) else None,
                "train_baseline_top": round(t_base, 4) if np.isfinite(t_base) else None,
                "train_baseline_pooled": round(pooled_base, 4) if np.isfinite(pooled_base) else None,
                "lift_bottom": round(lift_bottom, 4) if np.isfinite(lift_bottom) else None,
                "lift_top": round(lift_top, 4) if np.isfinite(lift_top) else None,
                "lift_pooled": round(lift_pooled, 4) if np.isfinite(lift_pooled) else None,
                "val_hitrate_bottom": round(b_val_hitrate, 4) if np.isfinite(b_val_hitrate) else None,
                "val_hitrate_top": round(t_val_hitrate, 4) if np.isfinite(t_val_hitrate) else None,
                "val_hitrate_pooled": round(pooled_val_hit, 4),
            })
            log(f"H={horizon:>3d} K={k:.1f}  TRAIN n(bot/top)={n_b_train}/{n_t_train}  "
                f"pooled_hit={pooled_train_hit:.4f} base={pooled_base:.4f} lift={lift_pooled:.3f}x  |  "
                f"VAL pooled_hit={pooled_val_hit:.4f} (n={pooled_val_n})")

    grid_df = pd.DataFrame(grid_rows)
    eligible = grid_df[(grid_df["n_train_bottom"] >= MIN_TRAIN_CANDIDATES) & (grid_df["n_train_top"] >= MIN_TRAIN_CANDIDATES)].copy()
    eligible = eligible.dropna(subset=["lift_pooled"])
    best_row = eligible.loc[eligible["lift_pooled"].idxmax()]
    chosen_horizon, chosen_k = int(best_row["horizon"]), float(best_row["k"])
    log(f"CHOSEN: HORIZON={chosen_horizon} K={chosen_k} (pooled TRAIN lift={best_row['lift_pooled']:.3f}x, "
        f"pooled VAL hit={best_row['val_hitrate_pooled']:.4f})")

    # ---- VAL baseline confirmation at the chosen cell only (step 4) ----
    fwd_high_max, fwd_low_min, bottom_idx_all, top_idx_all, _ = cache[chosen_horizon]
    pool_idx_val = np.flatnonzero(val_mask & (~any_trig) & np.isfinite(fwd_high_max) & np.isfinite(fwd_low_min) & finite_ok)
    b_val_m = val_mask[bottom_idx_all]
    t_val_m = val_mask[top_idx_all]
    n_b_val, n_t_val = int(b_val_m.sum()), int(t_val_m.sum())
    b_base_val = random_baseline_hit(master_rng, pool_idx_val, n_b_val, fwd_high_max, close, atr, chosen_k, "up")
    t_base_val = random_baseline_hit(master_rng, pool_idx_val, n_t_val, fwd_low_min, close, atr, chosen_k, "down")
    pooled_base_val = (b_base_val * n_b_val + t_base_val * n_t_val) / (n_b_val + n_t_val) if np.isfinite(b_base_val) and np.isfinite(t_base_val) else float("nan")
    lift_val = float(best_row["val_hitrate_pooled"]) / pooled_base_val if np.isfinite(pooled_base_val) and pooled_base_val > 0 else float("nan")
    log(f"VAL confirmation @ chosen cell: pooled_val_hit={best_row['val_hitrate_pooled']:.4f} "
        f"base={pooled_base_val:.4f} lift={lift_val:.3f}x  (TRAIN lift was {best_row['lift_pooled']:.3f}x)")

    # ---- feature analysis at chosen (horizon, K) (step 5) ----
    bottom_hit = fwd_high_max[bottom_idx_all] >= close[bottom_idx_all] + chosen_k * atr[bottom_idx_all]
    top_hit = fwd_low_min[top_idx_all] <= close[top_idx_all] - chosen_k * atr[top_idx_all]

    feat_bottom = df.iloc[bottom_idx_all][["timestamp"] + TIER0_FEATURES].copy()
    feat_bottom["hit"] = bottom_hit.astype(int)
    feat_bottom["is_bottom"] = 1
    feat_top = df.iloc[top_idx_all][["timestamp"] + TIER0_FEATURES].copy()
    feat_top["hit"] = top_hit.astype(int)
    feat_top["is_bottom"] = 0

    feat_all = pd.concat([feat_bottom, feat_top], ignore_index=True)
    n_before_dropna = len(feat_all)
    feat_all = feat_all.dropna(subset=TIER0_FEATURES + ["hit"]).reset_index(drop=True)
    log(f"feature-analysis candidates: {len(feat_all)}/{n_before_dropna} after dropna on Tier0 set")

    feat_cols = TIER0_FEATURES + ["is_bottom"]
    train_feat = feat_all[feat_all["timestamp"] < VAL_START].reset_index(drop=True)
    val_feat = feat_all[(feat_all["timestamp"] >= VAL_START) & (feat_all["timestamp"] < OOS_START)].reset_index(drop=True)
    log(f"feature-analysis split: TRAIN={len(train_feat)} VAL={len(val_feat)}")

    # (a) Pearson r == point-biserial correlation (hit is binary 0/1) per feature, TRAIN only
    y_train = train_feat["hit"].to_numpy()
    corr_rows = []
    for col in feat_cols:
        x = train_feat[col].to_numpy(dtype=float)
        r, p = pearsonr(y_train.astype(float), x)
        corr_rows.append({"feature": col, "point_biserial_r": round(float(r), 4), "p_value": round(float(p), 6)})
    corr_rows.sort(key=lambda row: -abs(row["point_biserial_r"]))
    log("=== point-biserial correlation vs hit (TRAIN) ===")
    for row in corr_rows[:10]:
        log(f"  {row['feature']:<22s} r={row['point_biserial_r']:+.4f} (p={row['p_value']:.4g})")

    # (b) quick HistGradientBoostingClassifier TRAIN-fit -> permutation importance on VAL
    clf = HistGradientBoostingClassifier(random_state=RNG_SEED, max_iter=200)
    clf.fit(train_feat[feat_cols], y_train)
    val_proba = clf.predict_proba(val_feat[feat_cols])[:, 1]
    val_auc = float(roc_auc_score(val_feat["hit"], val_proba))
    log(f"HistGBM sanity fit (NOT a promotion-grade model, feature-analysis support only): VAL AUC={val_auc:.4f}")

    perm = permutation_importance(clf, val_feat[feat_cols], val_feat["hit"], scoring="roc_auc",
                                   n_repeats=20, random_state=RNG_SEED)
    perm_rows = [
        {"feature": feat_cols[i], "importance_mean": round(float(perm.importances_mean[i]), 5),
         "importance_std": round(float(perm.importances_std[i]), 5)}
        for i in range(len(feat_cols))
    ]
    perm_rows.sort(key=lambda row: -row["importance_mean"])
    log("=== permutation importance (VAL, roc_auc scoring, 20 repeats) ===")
    for row in perm_rows[:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "taker_delta_z_climax", "asset": "BTC",
        "stage": "gridscreen_featureanalysis_only",
        "tabpfn_trained": False, "economic_cost_gate_run": False, "holdout_touched": False,
        "clustering_dedup_applied": False,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "data_source": str(CSV_PATH),
        "splits": {"train": f"<{VAL_START.date()}", "val": f"{VAL_START.date()}..{(OOS_START - pd.Timedelta(days=1)).date()}",
                   "oos_not_evaluated_this_round": f"{OOS_START.date()}..{(HOLDOUT_START - pd.Timedelta(days=1)).date()}",
                   "holdout_not_touched": f">={HOLDOUT_START.date()}"},
        "raw_trigger_counts": {
            "bottom_total": int(bottom_trig.sum()), "top_total": int(top_trig.sum()),
            "bottom_train": int((bottom_trig & train_mask).sum()), "top_train": int((top_trig & train_mask).sum()),
            "bottom_val": int((bottom_trig & val_mask).sum()), "top_val": int((top_trig & val_mask).sum()),
        },
        "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
        "grid": grid_rows,
        "chosen": {
            "horizon": chosen_horizon, "k": chosen_k,
            "train_lift_pooled": float(best_row["lift_pooled"]),
            "train_hitrate_pooled": float(best_row["train_hitrate_pooled"]),
            "train_baseline_pooled": float(best_row["train_baseline_pooled"]),
            "val_hitrate_pooled": float(best_row["val_hitrate_pooled"]),
            "val_baseline_pooled": round(pooled_base_val, 4) if np.isfinite(pooled_base_val) else None,
            "val_lift_pooled": round(lift_val, 4) if np.isfinite(lift_val) else None,
            "n_train_bottom": int(best_row["n_train_bottom"]), "n_train_top": int(best_row["n_train_top"]),
            "n_val_bottom": n_b_val, "n_val_top": n_t_val,
        },
        "feature_analysis": {
            "tier0_feature_columns_literal_21_plus_rsi": TIER0_FEATURES,
            "extra_structural_column_not_in_tier0": "is_bottom",
            "n_train": int(len(train_feat)), "n_val": int(len(val_feat)),
            "n_before_dropna": n_before_dropna,
            "point_biserial_train": corr_rows,
            "histgbm_val_auc_sanity_only": round(val_auc, 4),
            "permutation_importance_val": perm_rows,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

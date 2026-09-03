#!/usr/bin/env python3
"""Grid-screen HORIZON x K for BTC's own orthogonal_combo evidence signal (Homer project port),
plus a Tier0 feature-analysis pass at the chosen point. NO TabPFN, NO economic/cost-gate backtest,
NO holdout exposure -- those are future work pending human review (per user request 2026-09-01:
redo ETH's orthogonal_combo grid-screen+feature-analysis methodology for BTC, deliberately using a
simpler direct K*atr price-target test instead of ETH's later exclude-middle/TabPFN pipeline).

Reads the already-built shared BTC Tier0 dataset (scripts/build_btc_5m_evidence_signal_candidates_
tier0_20260901.py) -- bottom_orthogonal_combo/top_orthogonal_combo triggers are already computed
there (funding_z was used internally to build the bottom leg; not a saved column, not needed here).

Trigger (matches ETH's own design, live_evidence_signal_dashboard_20260823.py::compute_signals):
    bottom = (p_fast<=0.10) & (p_slow<=0.10) & (delta_z<=-2.0 OR funding_z<=-2.0)   -- target UP
    top    = (p_fast>=0.90) & (p_slow>=0.90) & (delta_z>=2.0)                       -- target DOWN

Methodology (per user spec, simpler than ETH's later exclude-middle/TabPFN pipeline):
  1. Fresh-forward split by DATE: TRAIN<2025-09-01, VAL=[2025-09-01,2026-01-01),
     OOS=[2026-01-01,2026-04-01). HOLDOUT(>=2026-04-01) is dropped from the working frame before
     ANY computation -- never touched, never read, never used for horizon lookahead.
  2. GAP=12 is a FIXED embargo/purge convention (not swept) -- consecutive raw fires within 12 bars
     of each other are merged into one cluster, keeping only the most oscillator-extreme row
     (idxmax of -(p_fast+p_slow) for bottom / +(p_fast+p_slow) for top). Verbatim convention from
     scripts/research_eth_orthogonal_combo_metalabel_tabpfn_20260830.py::cluster_dedup_oscillator.
  3. For each (HORIZON,K): hit = intrabar high (bottom) reaches close[i]+K*atr[i] within
     i+1..i+HORIZON (mirror: low reaches close[i]-K*atr[i] for top). `atr` is the causal ATR
     column in PRICE units (not atr_pct/atr_price, which are separate columns used by other
     signals) -- K*atr[i] is a raw price offset added to/subtracted from close[i].
  4. Lift = candidate hit-rate / random-baseline hit-rate, TRAIN only. Baseline = same COUNT of
     random non-trigger bars (bottom_orthogonal_combo==False & top_orthogonal_combo==False),
     drawn separately per side (matching real bottom/top counts) and tested against the SAME
     directional target formula, from the same scope (TRAIN pool for TRAIN lift, VAL pool for VAL
     confirmation) so baseline difficulty is comparable to the real candidates' split.
  5. Select by TRAIN lift among grid points with n_train_candidates>=MIN_TRAIN_CANDIDATES. Confirm
     (not re-search) on VAL at that exact point only, per spec. OOS at that same point is ALSO
     reported as a supplementary (not spec-required, not selection-relevant) second confirmation,
     since the OOS split is explicitly defined in the task and every prior Homer signal in this
     project reports VAL+OOS together while reserving HOLDOUT -- cheap given the same code path.
  6. Feature analysis at the chosen point: point-biserial correlation (fast, no model) AND a
     HistGradientBoostingClassifier fit on TRAIN -> permutation importance measured on VAL (both
     methods run since sklearn only, no extra infra cost -- task said "either is fine").

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_orthogonal_combo_gridscreen_20260901.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/orthogonal_combo_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

GAP = 12  # fixed embargo/purge convention across all Homer signals, not swept
HORIZON_GRID = [12, 18, 24, 30, 36]
K_GRID = [2.5, 3.0, 3.57, 4.0, 4.5]
MIN_TRAIN_CANDIDATES = 300  # "few hundred+ TRAIN candidates minimum" per task spec
RNG_SEED = 20260901

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high", "range_width_pct",
    "hour_utc", "weekday", "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z",
    "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def log(msg: str) -> None:
    print(f"[btc_orthogonal_gridscreen] {msg}", flush=True)


def cluster_dedup(idx: np.ndarray, p_fast: np.ndarray, p_slow: np.ndarray, side: str, gap: int) -> np.ndarray:
    """Merge raw fires within `gap` bars of each other into one cluster; keep only the most
    oscillator-extreme row per cluster. Verbatim convention from ETH's own orthogonal_combo screen."""
    if len(idx) == 0:
        return idx
    score = -(p_fast[idx] + p_slow[idx]) if side == "bottom" else (p_fast[idx] + p_slow[idx])
    cluster_id = np.zeros(len(idx), dtype=int)
    cid = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx, "cluster": cluster_id, "s": score})
    return np.sort(df.loc[df.groupby("cluster")["s"].idxmax(), "idx"].to_numpy())


def build_candidate_idx(frame: pd.DataFrame, horizon: int, gap: int) -> list[tuple[str, np.ndarray]]:
    """Deduped candidate row-positions for both sides at a given horizon (dedup itself is
    horizon-independent; only the `idx < n - horizon` lookahead-validity filter depends on it)."""
    n = len(frame)
    atr = frame["atr"].to_numpy()
    close = frame["close"].to_numpy()
    p_fast = frame["p_fast"].to_numpy()
    p_slow = frame["p_slow"].to_numpy()
    out = []
    for side, col in [("bottom", "bottom_orthogonal_combo"), ("top", "top_orthogonal_combo")]:
        idx = np.flatnonzero(frame[col].to_numpy())
        idx = idx[(idx < n - horizon) & np.isfinite(atr[idx]) & np.isfinite(close[idx])]
        idx = cluster_dedup(idx, p_fast, p_slow, side, gap)
        out.append((side, idx))
    return out


def compute_hits(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int, k: float) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=int)
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr = frame["atr"].to_numpy()
    entry, a = close[idx], atr[idx]
    if side == "bottom":
        target = entry + k * a
        fut = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
        return (fut >= target).astype(int)
    target = entry - k * a
    fut = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
    return (fut <= target).astype(int)


def random_baseline(frame: pd.DataFrame, side_counts: dict[str, int], horizon: int, k: float,
                     scope_mask: np.ndarray, seed: int) -> tuple[float, int]:
    """Same-count random NON-trigger bars (neither bottom nor top orthogonal_combo fired), sampled
    separately per side to match real candidates' bottom/top mix, tested with the same directional
    K*atr target. Scope-restricted (TRAIN pool for TRAIN lift, VAL pool for VAL confirmation)."""
    n = len(frame)
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr = frame["atr"].to_numpy()
    not_trigger = ~(frame["bottom_orthogonal_combo"].to_numpy() | frame["top_orthogonal_combo"].to_numpy())
    valid_pos = np.arange(n) < (n - horizon)
    eligible_mask = not_trigger & valid_pos & scope_mask & np.isfinite(atr) & np.isfinite(close)
    eligible_idx = np.flatnonzero(eligible_mask)

    rng = np.random.default_rng(seed)
    all_hits = []
    for side, count in side_counts.items():
        if count == 0:
            continue
        replace = count > len(eligible_idx)
        sampled = rng.choice(eligible_idx, size=count, replace=replace)
        hits = compute_hits(frame, sampled, side, horizon, k)
        all_hits.append(hits)
    all_hits = np.concatenate(all_hits) if all_hits else np.array([])
    if len(all_hits) == 0:
        return float("nan"), 0
    return float(all_hits.mean()), int(len(all_hits))


def scope_metrics(frame: pd.DataFrame, side_idx: list[tuple[str, np.ndarray]], scope_mask: np.ndarray,
                   horizon: int, k: float, baseline_seed: int) -> dict:
    side_counts, hits_all = {}, []
    for side, idx in side_idx:
        idx_scope = idx[scope_mask[idx]]
        hits = compute_hits(frame, idx_scope, side, horizon, k)
        side_counts[side] = len(idx_scope)
        hits_all.append(hits)
    hits_all = np.concatenate(hits_all) if hits_all else np.array([])
    n = len(hits_all)
    hit_rate = float(hits_all.mean()) if n else float("nan")
    baseline_rate, n_baseline = random_baseline(frame, side_counts, horizon, k, scope_mask, baseline_seed)
    lift = hit_rate / baseline_rate if n and baseline_rate else float("nan")
    return {
        "n": n, "n_bottom": side_counts.get("bottom", 0), "n_top": side_counts.get("top", 0),
        "hit_rate": hit_rate, "baseline_hit_rate": baseline_rate, "n_baseline": n_baseline, "lift": lift,
    }


def build_candidate_features_df(frame: pd.DataFrame, side_idx: list[tuple[str, np.ndarray]],
                                 scope_mask: np.ndarray, horizon: int, k: float) -> pd.DataFrame:
    rows = []
    for side, idx in side_idx:
        idx_scope = idx[scope_mask[idx]]
        if len(idx_scope) == 0:
            continue
        hits = compute_hits(frame, idx_scope, side, horizon, k)
        sub = frame.iloc[idx_scope][["timestamp"] + TIER0_FEATURES].copy()
        sub["side"] = side
        sub["hit"] = hits
        rows.append(sub)
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    log("loading CSV...")
    frame = pd.read_csv(
        CSV_PATH,
        usecols=["timestamp", "high", "low", "close", "atr", "p_fast", "p_slow",
                  "bottom_orthogonal_combo", "top_orthogonal_combo"] + TIER0_FEATURES,
    )
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    n_raw = len(frame)
    frame = frame.loc[frame["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    log(f"loaded {n_raw} rows; HOLDOUT (>= {HOLDOUT_START.date()}) dropped -> working frame n={len(frame)}, "
        f"range {frame['timestamp'].min()} ~ {frame['timestamp'].max()}")

    ts = frame["timestamp"]
    train_mask = (ts < VAL_START).to_numpy()
    val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts < HOLDOUT_START)).to_numpy()
    log(f"TRAIN(<{VAL_START.date()}) n={train_mask.sum()}, VAL n={val_mask.sum()}, OOS n={oos_mask.sum()}")

    log(f"=== screening grid: HORIZON in {HORIZON_GRID} x K in {K_GRID} "
        f"({len(HORIZON_GRID)*len(K_GRID)} combos, GAP={GAP} fixed, TRAIN-only lift) ===")
    candidates_cache: dict[int, list[tuple[str, np.ndarray]]] = {
        horizon: build_candidate_idx(frame, horizon, GAP) for horizon in HORIZON_GRID
    }
    for horizon, side_idx in candidates_cache.items():
        n_bottom, n_top = len(side_idx[0][1]), len(side_idx[1][1])
        log(f"  H={horizon:>2d}: deduped candidates (whole no-holdout frame) bottom={n_bottom} top={n_top}")

    grid_rows = []
    for horizon in HORIZON_GRID:
        side_idx = candidates_cache[horizon]
        for k in K_GRID:
            m = scope_metrics(frame, side_idx, train_mask, horizon, k, RNG_SEED)
            row = {
                "horizon": horizon, "k": k,
                "n_train": m["n"], "n_train_bottom": m["n_bottom"], "n_train_top": m["n_top"],
                "hit_rate_train": round(m["hit_rate"], 4) if m["n"] else None,
                "baseline_hit_rate_train": round(m["baseline_hit_rate"], 4) if m["n"] else None,
                "lift_train": round(m["lift"], 4) if m["n"] else None,
            }
            grid_rows.append(row)
            log(f"  H={horizon:>2d} K={k:.2f}: n_train={m['n']:>5d} (bottom={m['n_bottom']},top={m['n_top']}) "
                f"hit={m['hit_rate']:.3f} baseline={m['baseline_hit_rate']:.3f} lift={m['lift']:.3f}")

    eligible = [r for r in grid_rows if r["n_train"] >= MIN_TRAIN_CANDIDATES]
    pool = eligible if eligible else grid_rows
    if not eligible:
        log(f"WARNING: no grid point reached MIN_TRAIN_CANDIDATES={MIN_TRAIN_CANDIDATES}; selecting from full grid anyway")
    best = max(pool, key=lambda r: r["lift_train"])
    horizon_f, k_f = best["horizon"], best["k"]
    log(f"=== SELECTED (max TRAIN lift among n_train>={MIN_TRAIN_CANDIDATES}): "
        f"HORIZON={horizon_f} K={k_f} (lift_train={best['lift_train']:.3f}, n_train={best['n_train']}) ===")

    side_idx_f = candidates_cache[horizon_f]
    val_metrics = scope_metrics(frame, side_idx_f, val_mask, horizon_f, k_f, RNG_SEED + 1)
    log(f"VAL confirmation @ H={horizon_f} K={k_f}: n_val={val_metrics['n']} "
        f"hit={val_metrics['hit_rate']:.3f} baseline={val_metrics['baseline_hit_rate']:.3f} lift={val_metrics['lift']:.3f}")

    oos_metrics = scope_metrics(frame, side_idx_f, oos_mask, horizon_f, k_f, RNG_SEED + 2)
    log(f"OOS supplementary check @ H={horizon_f} K={k_f}: n_oos={oos_metrics['n']} "
        f"hit={oos_metrics['hit_rate']:.3f} baseline={oos_metrics['baseline_hit_rate']:.3f} lift={oos_metrics['lift']:.3f}")

    # ---- feature analysis at the chosen (HORIZON,K) ----
    log(f"=== feature analysis @ H={horizon_f} K={k_f} (Tier0 {len(TIER0_FEATURES)} features) ===")
    train_df = build_candidate_features_df(frame, side_idx_f, train_mask, horizon_f, k_f)
    val_df = build_candidate_features_df(frame, side_idx_f, val_mask, horizon_f, k_f)
    n_train_before_dropna = len(train_df)
    train_df = train_df.dropna(subset=TIER0_FEATURES).reset_index(drop=True)
    n_val_before_dropna = len(val_df)
    val_df = val_df.dropna(subset=TIER0_FEATURES).reset_index(drop=True)
    log(f"TRAIN candidates: {n_train_before_dropna} -> {len(train_df)} after dropna(Tier0 features); "
        f"VAL: {n_val_before_dropna} -> {len(val_df)}")

    corr_rows = []
    for feat in TIER0_FEATURES:
        r, p = pointbiserialr(train_df["hit"].to_numpy(), train_df[feat].to_numpy())
        corr_rows.append({"feature": feat, "point_biserial_r": round(float(r), 4), "p_value": round(float(p), 5)})
    corr_rows.sort(key=lambda r: abs(r["point_biserial_r"]), reverse=True)
    log("-- point-biserial correlation vs hit (TRAIN), ranked by |r| --")
    for r in corr_rows:
        log(f"  {r['feature']:<20s} r={r['point_biserial_r']:+.4f}  p={r['p_value']:.5f}")

    clf = HistGradientBoostingClassifier(random_state=RNG_SEED)
    clf.fit(train_df[TIER0_FEATURES], train_df["hit"].to_numpy().astype(int))
    val_proba = clf.predict_proba(val_df[TIER0_FEATURES])[:, 1]
    val_auc = roc_auc_score(val_df["hit"].to_numpy().astype(int), val_proba) if val_df["hit"].nunique() > 1 else float("nan")
    log(f"HistGradientBoostingClassifier TRAIN-fit -> VAL AUC = {val_auc:.4f} (sanity check only, not a promotion metric)")

    perm = permutation_importance(clf, val_df[TIER0_FEATURES], val_df["hit"].to_numpy().astype(int),
                                   n_repeats=20, random_state=RNG_SEED, scoring="roc_auc")
    perm_rows = [
        {"feature": feat, "importance_mean": round(float(perm.importances_mean[i]), 5),
         "importance_std": round(float(perm.importances_std[i]), 5)}
        for i, feat in enumerate(TIER0_FEATURES)
    ]
    perm_rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    log("-- permutation importance (VAL, HistGBM TRAIN-fit, 20 repeats), ranked --")
    for r in perm_rows:
        log(f"  {r['feature']:<20s} importance={r['importance_mean']:+.5f} (+/-{r['importance_std']:.5f})")

    session_timing_features = ["hour_utc", "weekday"]
    log(f"-- session-timing features check (ETH found nyse_open_flag/hour_utc/weekday added nothing; "
        f"BTC Tier0 set has no nyse_open_flag column, only {session_timing_features}) --")
    for feat in session_timing_features:
        corr_rank = next(i for i, r in enumerate(corr_rows) if r["feature"] == feat) + 1
        perm_rank = next(i for i, r in enumerate(perm_rows) if r["feature"] == feat) + 1
        log(f"  {feat}: correlation rank {corr_rank}/{len(TIER0_FEATURES)}, permutation-importance rank {perm_rank}/{len(TIER0_FEATURES)}")

    report = {
        "signal": "orthogonal_combo", "asset": "BTC", "status": "gridscreen_and_feature_analysis_only",
        "not_done_this_round": ["TabPFN training", "economic/cost-gate backtest", "holdout exposure"],
        "holdout_touched": False, "holdout_start": str(HOLDOUT_START),
        "gap_fixed": GAP, "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
        "min_train_candidates": MIN_TRAIN_CANDIDATES,
        "eth_reference_center": {"horizon": 24, "gap": 12, "k": 3.57, "source": "ETH orthogonal_combo K_hi (exclude-middle v2), see script docstring"},
        "screening_grid": grid_rows,
        "selected_horizon": horizon_f, "selected_k": k_f, "selected_by": "max(lift_train) among n_train>=min",
        "train": {**best, "note": "same fields as the matching screening_grid row"},
        "val_confirmation": val_metrics,
        "oos_supplementary": oos_metrics,
        "feature_analysis": {
            "tier0_features": TIER0_FEATURES,
            "n_train_candidates_before_dropna": n_train_before_dropna, "n_train_candidates": len(train_df),
            "n_val_candidates_before_dropna": n_val_before_dropna, "n_val_candidates": len(val_df),
            "point_biserial_correlation_train": corr_rows,
            "histgbm_val_auc": round(float(val_auc), 4) if np.isfinite(val_auc) else None,
            "permutation_importance_val": perm_rows,
            "session_timing_features": session_timing_features,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

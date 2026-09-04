#!/usr/bin/env python3
"""FINAL model + required validations for liquidity_sweep "top/down" metalabel (Homer signal #2
redo, standard touch-based-MFE template). Label locked in via
research_eth_liquidity_sweep_topdown_metalabel_{gridscreen,ksweep_tabpfn_confirm}_20260830.py:
  HORIZON=30 (150min), CLUSTER_GAP_MERGE=12 (cluster-anchor by deepest sweep penetration),
  K=4.0xATR touch-based MFE (no persistence check). TabPFN VAL 0.6587/OOS 0.6377 (4 seeds),
  smallest VAL-OOS gap of every (horizon,gap,K) candidate tried -- beats taker_delta_z_climax's
  own 0.622/0.608. Chart-verified (10 HIT/10 NO_HIT, no bugs -- fire bars mechanically always
  touch the swept level; some NO_HIT tops show later bars also grazing the old level because
  price kept trending through it instead of reversing, not a rendering bug).

This script: (1) random-bar baseline / lift, (2) full VAL+OOS TabPFN panel (4 seeds, report),
(3) permutation feature importance (VAL, 1 seed, 5 repeats), (4) vol-regime-group ablation
(atr_pct/atr_percentile_864/realized_vol_ratio -- same 3 features taker/dalton checked, since
atr_pct is used both to SET the K*atr_pct hit threshold AND as an input feature here too).
HOLDOUT (2026-04-01+) deliberately NOT touched -- reserved for a single final exposure after
the economic-gate design is also locked in, matching this project's holdout discipline.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830 import (  # noqa: E402
    cluster_dedup_by_penetration,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

OUT_DIR = ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830"
REPORT_DIR = ROOT / "tmp/eth_liquidity_sweep_topdown_metalabel_20260830"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SWEEP_LOOKBACK = 48
HORIZON = 30
GAP = 12
K = 4.0
SEEDS = [20260829, 141592, 271828, 577215]
VOL_REGIME_FEATURES = ["atr_pct", "atr_percentile_864", "realized_vol_ratio"]


def log(msg: str) -> None:
    print(f"[liq_sweep_topdown_final] {msg}", flush=True)


def build_fires(klines: pd.DataFrame, ind: pd.DataFrame, sig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy(); ts = sig["timestamp"].to_numpy(); n = len(sig)
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()

    rows = []
    counts = {}
    for side, col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (ts[idx] >= np.datetime64(START))]
        idx = np.sort(idx)
        n_raw = len(idx)
        penetration = (swing_low_prior[idx] - low[idx]) if side == "bottom" else (high[idx] - swing_high_prior[idx])
        idx = cluster_dedup_by_penetration(idx, penetration, GAP)
        counts[side] = {"raw": n_raw, "anchored": len(idx)}
        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        hit = (pred_dir_ret >= K * atr_pct[idx]).astype(float)
        feat_rows = ind.iloc[idx]
        out = pd.DataFrame({"pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
                             "hit": hit, "pred_dir_ret": pred_dir_ret, "is_bottom": 1 if side == "bottom" else 0})
        for c in FEATURE_COLUMNS:
            if c != "is_bottom":
                out[c] = feat_rows[c].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires, counts


def random_bar_baseline(ind: pd.DataFrame, klines: pd.DataFrame) -> dict:
    """Analog of taker/V_REBOUND's random_bar_baseline: applies the SAME touch-based MFE/ATR hit
    rule (K=4.0, HORIZON=30) to EVERY bar (not just actual sweep fires), using each bar's own
    is_bottom-style directional guess unavailable here (liquidity_sweep has no raw directional
    feature like delta_z) -- so this instead reports, for reference, the hit rate if a coin-flip
    direction were assigned to every bar, vs the actual fired-signal hit rate. Vectorized
    forward-rolling max/min over all ~280k bars."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    fwd_high_max = high[::-1].rolling(window=HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    fwd_low_min = low[::-1].rolling(window=HORIZON, min_periods=HORIZON).min()[::-1].shift(-1)
    mfe_up_pct = ((fwd_high_max - close) / close).to_numpy()
    mfe_down_pct = ((close - fwd_low_min) / close).to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()
    ts = ind["timestamp"].to_numpy()
    n = len(ind)
    valid = np.isfinite(atr_pct) & (atr_pct > 0) & (ts >= np.datetime64(START)) & (np.arange(n) < n - HORIZON)
    idx = np.flatnonzero(valid)
    rng = np.random.default_rng(20260830)
    coin = rng.integers(0, 2, size=len(idx)).astype(bool)
    mfe_pct = np.where(coin, mfe_up_pct[idx], mfe_down_pct[idx])
    hit = mfe_pct >= K * atr_pct[idx]
    return {"n": int(len(idx)), "random_direction_hit_rate": float(hit.mean())}


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {"auc": round(float(roc_auc_score(y, proba)), 4), "accuracy": round(float((pred == y).mean()), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4)}


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {"n_train": None, "n_eval": int(len(eval_df)), "auc_mean": round(float(table["auc"].mean()), 4),
            "auc_std": round(float(table["auc"].std(ddof=1)), 4), "accuracy_mean": round(float(table["accuracy"].mean()), 4),
            "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
            "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"], "per_seed": seed_rows}


def compute_permutation_importance(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str],
                                    seed: int = SEEDS[0], n_repeats: int = 5) -> dict:
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
    y = eval_df["hit"].to_numpy().astype(int)
    X = eval_df[feature_cols].to_numpy()
    baseline_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for j, feat in enumerate(feature_cols):
        shuffled_aucs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            shuffled_aucs.append(roc_auc_score(y, clf.predict_proba(X_perm)[:, 1]))
        importance = baseline_auc - np.mean(shuffled_aucs)
        rows.append({"feature": feat, "importance_mean": round(float(importance), 5), "importance_std": round(float(np.std(shuffled_aucs, ddof=1)), 5)})
    rows.sort(key=lambda r: -r["importance_mean"])
    return {"baseline_auc": round(float(baseline_auc), 4), "n_repeats": n_repeats, "seed": seed, "importances": rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading klines + building indicator frame + signals...")
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind) and (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()
    log(f"{len(klines)} bars ready")

    log("building liquidity_sweep top/down fires + features (final config: H=30/GAP=12/K=4.0)...")
    fires, counts = build_fires(klines, ind, sig)
    log(f"cluster-anchor: bottom {counts['bottom']['raw']}->{counts['bottom']['anchored']}, top {counts['top']['raw']}->{counts['top']['anchored']}")
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna (bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")
    fires.to_csv(OUT_DIR / "eth_5m_liquidity_sweep_topdown_metalabel_features_FINAL.csv", index=False)

    log("random-bar (coin-flip direction) baseline...")
    baseline = random_bar_baseline(ind, klines)
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal hit rate: {fire_hit_rate:.4f} vs random-direction-any-bar baseline: {baseline['random_direction_hit_rate']:.4f} "
        f"(lift {fire_hit_rate / baseline['random_direction_hit_rate']:.3f}x)")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout_n = int((ts >= HOLDOUT_START).sum())
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={holdout_n} (RESERVED, not evaluated)")

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")
    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    log("=== vol-regime-group ablation (remove atr_pct/atr_percentile_864/realized_vol_ratio) ===")
    reduced_cols = [c for c in FEATURE_COLUMNS if c not in VOL_REGIME_FEATURES]
    val_ablated = run_tabpfn_panel(train, val, reduced_cols, "VAL-ablated")
    oos_ablated = run_tabpfn_panel(train, oos, reduced_cols, "OOS-ablated")
    log(f"ablated({len(reduced_cols)} feats) VAL AUC {val_ablated['auc_mean']:.4f} (full={val_result['auc_mean']:.4f}, "
        f"delta={val_ablated['auc_mean']-val_result['auc_mean']:+.4f})  "
        f"OOS AUC {oos_ablated['auc_mean']:.4f} (full={oos_result['auc_mean']:.4f}, delta={oos_ablated['auc_mean']-oos_result['auc_mean']:+.4f})")

    report = {
        "signal": "liquidity_sweep", "label_design": "top_down_standard_touch_mfe_v1",
        "status": "final_pending_costgate",
        "label_params": {"horizon_bars": HORIZON, "cluster_gap_merge": GAP, "k_atr_mult": K,
                          "cluster_anchor_metric": "sweep_penetration_atr_level"},
        "summary_for_future_sessions": (
            f"Standard touch-based MFE template (matching taker_delta_z_climax/short_term_return_z, "
            f"NOT V_REBOUND's specialized giveback/confirmed-window/excluded-middle design). "
            f"Cluster-anchor by deepest sweep penetration (causal, non-circular, definition-intrinsic "
            f"-- never the price outcome). HORIZON=30(150min)/GAP=12/K=4.0xATR chosen via a "
            f"(horizon,gap) grid screen (GBM proxy, 24 combos) then a K sweep at the winning "
            f"(horizon,gap), both confirmed with real TabPFN (4 seeds) before locking in -- K=4.0 "
            f"beat K=1.5's naive phase1 guess by +0.05 OOS AUC (0.589->0.638). "
            f"VAL AUC {val_result['auc_mean']}, OOS AUC {oos_result['auc_mean']} -- beats "
            f"taker_delta_z_climax's 0.622/0.608 and is the smallest VAL-OOS gap "
            f"({round(abs(val_result['auc_mean']-oos_result['auc_mean']),4)}) of any candidate tried. "
            f"Chart-verified (10 HIT/10 NO_HIT, no bugs found). HOLDOUT (2026-04-01+) deliberately "
            f"NOT touched yet -- reserved for a single final exposure after cost-gate design."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)), "cluster_anchor_counts": counts,
        "random_bar_baseline": baseline, "fired_signal_hit_rate": fire_hit_rate,
        "lift_vs_random_direction_baseline": fire_hit_rate / baseline["random_direction_hit_rate"],
        "val": val_result, "oos": oos_result,
        "permutation_importance_val": perm_importance,
        "vol_regime_ablation": {"removed_features": VOL_REGIME_FEATURES, "val": val_ablated, "oos": oos_ablated},
    }
    out_path = REPORT_DIR / "final_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

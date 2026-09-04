#!/usr/bin/env python3
"""TabPFN confirmation for the K (ATR multiple) sweep at the winning HORIZON=30/CLUSTER_GAP=12
config -- research_eth_liquidity_sweep_topdown_metalabel_ksweep_20260830.py's GBM proxy found
K=1.5 (the original phase1-derived guess) is dominated by higher K (hit rate closer to/past
50/50, better min(VAL,OOS)): K=2.5 (55% hit rate, min=0.594) and K=4.0 (31% hit rate, min=0.627)
both beat K=1.5 (75% hit rate, min=0.577). Confirms K in {1.5, 2.5, 4.0} with the real model
(4 seeds) on VAL/OOS -- HOLDOUT stays untouched. Sequential refinement (HORIZON/GAP fixed first,
K refined second), matching taker_delta_z_climax's own precedent (HORIZON fixed via diagnosis,
K calibrated afterward for balance) rather than reopening the full joint grid.
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
K_CANDIDATES = [1.5, 2.5, 4.0]
SEEDS = [20260829, 141592, 271828, 577215]


def log(msg: str) -> None:
    print(f"[liq_sweep_ksweep_tabpfn_confirm] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[FEATURE_COLUMNS])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"    [{tag}] seed={seed}: auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {"n_eval": int(len(eval_df)), "auc_mean": round(float(table["auc"].mean()), 4),
            "auc_std": round(float(table["auc"].std(ddof=1)), 4),
            "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
            "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"], "per_seed": seed_rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind) and (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()
    log(f"{len(klines)} bars ready")

    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy(); ts = sig["timestamp"].to_numpy(); n = len(sig)
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()

    anchored = {}
    for side, col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (ts[idx] >= np.datetime64(START))]
        idx = np.sort(idx)
        penetration = (swing_low_prior[idx] - low[idx]) if side == "bottom" else (high[idx] - swing_high_prior[idx])
        idx = cluster_dedup_by_penetration(idx, penetration, GAP)
        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        anchored[side] = {"idx": idx, "pred_dir_ret": pred_dir_ret, "atr": atr_pct[idx]}

    results = []
    for K in K_CANDIDATES:
        tag = f"K{K}"
        log(f"\n=== {tag} (H={HORIZON}, GAP={GAP}) ===")
        rows = []
        for side in ("bottom", "top"):
            a = anchored[side]
            hit = (a["pred_dir_ret"] >= K * a["atr"]).astype(float)
            feat_rows = ind.iloc[a["idx"]]
            out = pd.DataFrame({"pos": a["idx"], "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
                                 "hit": hit, "is_bottom": 1 if side == "bottom" else 0})
            for c in FEATURE_COLUMNS:
                if c != "is_bottom":
                    out[c] = feat_rows[c].to_numpy()
            rows.append(out)
        fires = pd.concat(rows, ignore_index=True).dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
        tsf = fires["timestamp"]
        train = fires.loc[tsf < VAL_START].reset_index(drop=True)
        val = fires.loc[(tsf >= VAL_START) & (tsf < OOS_START)].reset_index(drop=True)
        oos = fires.loc[(tsf >= OOS_START) & (tsf < HOLDOUT_START)].reset_index(drop=True)
        log(f"  hit_rate={fires['hit'].mean():.3f}  n_train={len(train)}(pos={int(train['hit'].sum())}) "
            f"n_val={len(val)}(pos={int(val['hit'].sum())}) n_oos={len(oos)}(pos={int(oos['hit'].sum())}) "
            f"(HOLDOUT n={len(fires.loc[tsf >= HOLDOUT_START])}, NOT touched)")
        fires.to_csv(OUT_DIR / f"eth_5m_liquidity_sweep_topdown_metalabel_features_H{HORIZON}_GAP{GAP}_{tag}.csv", index=False)

        val_result = run_tabpfn_panel(train, val, "VAL")
        oos_result = run_tabpfn_panel(train, oos, "OOS")
        log(f"  VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
            f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")
        results.append({"K": K, "hit_rate": round(float(fires["hit"].mean()), 4),
                         "n_train": len(train), "n_val": len(val), "n_oos": len(oos),
                         "val": val_result, "oos": oos_result,
                         "val_oos_gap": round(abs(val_result["auc_mean"] - oos_result["auc_mean"]), 4),
                         "min_val_oos": round(min(val_result["auc_mean"], oos_result["auc_mean"]), 4)})

    log("\n=== SUMMARY (K sweep at H=30/GAP=12, VAL/OOS only, HOLDOUT untouched) ===")
    for r in sorted(results, key=lambda x: -x["min_val_oos"]):
        log(f"  K={r['K']}: VAL={r['val']['auc_mean']:.4f} OOS={r['oos']['auc_mean']:.4f} "
            f"min={r['min_val_oos']:.4f} |gap|={r['val_oos_gap']:.4f} hit_rate={r['hit_rate']:.3f} n_oos_pos={r['oos']['n_eval']}")

    out_path = REPORT_DIR / "ksweep_tabpfn_confirm_report.json"
    out_path.write_text(json.dumps({"horizon": HORIZON, "gap": GAP, "feature_columns": FEATURE_COLUMNS,
                                     "seeds": SEEDS, "k_candidates": results}, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

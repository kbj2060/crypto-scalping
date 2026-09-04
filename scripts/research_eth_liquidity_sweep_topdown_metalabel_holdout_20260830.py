#!/usr/bin/env python3
"""SINGLE final HOLDOUT (2026-04-01+) touch for liquidity_sweep top/down metalabel -- everything
upstream (HORIZON/GAP/K grid+confirm, chart verification, permutation importance, vol-regime
ablation, trailing-stop cost-gate grid+intrabar-ordering cross-check) is done and locked in on
TRAIN/VAL/OOS only. This is the one and only exposure of this HOLDOUT window for this model --
per this project's single-touch discipline, do NOT re-run this after seeing the result to try a
different config. TabPFN classification only (4 seeds, same TRAIN/features as VAL/OOS); the
trailing-stop economic HOLDOUT check runs separately (backtest_eth_liquidity_sweep_topdown_
trailing_holdout_exposure_20260830.py, no GPU needed) using the exact same config, same day.
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

FIRES_CSV = ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv"
REPORT_DIR = ROOT / "tmp/eth_liquidity_sweep_topdown_metalabel_20260830"
VAL_START = pd.Timestamp("2025-09-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEEDS = [20260829, 141592, 271828, 577215]

FEATURE_COLUMNS = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio", "rsi",
]


def log(msg: str) -> None:
    print(f"[liq_sweep_topdown_HOLDOUT] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {"auc": round(float(roc_auc_score(y, proba)), 4), "accuracy": round(float((pred == y).mean()), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4)}


def main() -> int:
    from tabpfn import TabPFNClassifier
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"]).dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN n={len(train)}, HOLDOUT n={len(holdout)} (SINGLE EXPOSURE)")
    log(f"HOLDOUT hit_rate={holdout['hit'].mean():.4f} (TRAIN hit_rate={train['hit'].mean():.4f})")

    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
        r = evaluate(proba, holdout["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    result = {"n_train": len(train), "n_eval": len(holdout),
              "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
              "accuracy_mean": round(float(table["accuracy"].mean()), 4),
              "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"], "per_seed": seed_rows}
    log(f"\nHOLDOUT -> AUC {result['auc_mean']:.4f}+/-{result['auc_std']:.4f} (VAL was 0.6587, OOS was 0.6372)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "holdout_report.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

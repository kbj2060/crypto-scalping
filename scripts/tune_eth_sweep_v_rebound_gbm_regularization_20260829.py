#!/usr/bin/env python3
"""Overfitting-mitigation grid search for the Tier0 V_REBOUND GBM baseline
(train_eth_sweep_v_rebound_gbm_baseline_20260829.py showed TRAIN AUC 0.772 vs VAL 0.622 --
a real gap). Grid is scored on VAL ONLY; OOS is touched exactly once at the very end with the
VAL-selected config, matching this project's own established discipline (e.g.
backtest_eth_slowk_williamsr_persistence_confluence_20260814.py: "grid-tuned on VAL only; OOS
gets a single look at the VAL-selected config").

Disclosure: this session's earlier Tier1/Omega461 incremental rounds already checked OOS AUC
13 separate times on this same window (2026-01-01..2026-03-31) -- it is not a pristine single-
touch holdout anymore, the same compromise this project's GBM2 work explicitly flagged for its
own OOS window ("이미 8회 가량 재사용됨"). This tuning pass adds exactly one more OOS look,
not the many-combinations grid itself.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_gbm_regularization_tune_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]

GRID = {
    "max_depth": [2, 3, 4, 6],
    "l2_regularization": [1.0, 3.0, 10.0],
    "min_samples_leaf": [20, 50, 100],
}
BASELINE = {"max_depth": 6, "l2_regularization": 1.0, "min_samples_leaf": 20}


def split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    return {
        "train": df.loc[ts < VAL_START],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END)],
    }


def evaluate(model, frame: pd.DataFrame) -> dict:
    X, y = frame[FEATURE_COLUMNS], frame["label"].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "naive_accuracy": round(naive_acc, 4),
    }


def fit_model(params: dict, train: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05,
        max_depth=params["max_depth"], l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
        early_stopping=True, validation_fraction=0.15, random_state=20260829,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    return model


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)
    parts = split(df)

    rows = []
    keys = list(GRID.keys())
    for combo in product(*GRID.values()):
        params = dict(zip(keys, combo))
        model = fit_model(params, parts["train"])
        train_res = evaluate(model, parts["train"])
        val_res = evaluate(model, parts["val"])
        rows.append({
            **params,
            "train_auc": train_res["auc"], "val_auc": val_res["auc"],
            "train_val_gap": round(train_res["auc"] - val_res["auc"], 4),
            "val_bal_acc": val_res["balanced_accuracy"],
        })

    table = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    print(f"grid: {len(table)} combinations, scored on VAL only\n")
    print(table.to_string(index=False))

    best = table.iloc[0].to_dict()
    best_params = {k: (int(best[k]) if k != "l2_regularization" else float(best[k])) for k in keys}
    print(f"\nbest by VAL AUC: {best_params} -> VAL AUC {best['val_auc']}, train/val gap {best['train_val_gap']}")

    print("\n=== final check: baseline params vs best params, all 3 splits (ONE OOS look) ===")
    for label, params in [("baseline (max_depth=6,l2=1,leaf=20)", BASELINE), ("tuned (best VAL AUC)", best_params)]:
        model = fit_model(params, parts["train"])
        results = {name: evaluate(model, part) for name, part in parts.items()}
        print(f"\n{label}: {params}")
        for name in ("train", "val", "oos"):
            r = results[name]
            print(f"  {name:5s} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
                  f"auc={r['auc']:.4f} naive={r['naive_accuracy']:.4f}")

    report = {
        "grid_results": table.to_dict(orient="records"),
        "best_params": best_params,
        "note": "OOS window (2026-01-01..2026-03-31) already reused ~13 times this session across Tier1/Omega461 rounds -- this pass adds one more single look with the VAL-selected config, not a fresh pristine holdout.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

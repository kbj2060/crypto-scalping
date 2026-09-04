#!/usr/bin/env python3
"""GBM baseline for the liquidity_sweep -> V_REBOUND model, Tier 0 features only
(docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md).

Fresh-Forward split per CLAUDE.md's default boundaries: TRAIN < 2025-09-01, VAL
2025-09-01..2025-12-31, OOS 2026-01-01..2026-03-31. The label data actually extends to
2026-08-28 -- everything after 2026-03-31 is deliberately RESERVED and not evaluated this
round, so a genuinely untouched slice remains for a later look rather than folding it into
TRAIN just because it happens to be available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_gbm_baseline_20260829"

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


def split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    return {
        "train": df.loc[ts < VAL_START],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END)],
        "reserved_untouched": df.loc[ts > OOS_END],
    }


def evaluate(model, frame: pd.DataFrame) -> dict:
    X, y = frame[FEATURE_COLUMNS], frame["label"].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)),
        "label_rate": float(y.mean()),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "naive_majority_class_accuracy": naive_acc,
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = split(df)
    for name, part in parts.items():
        if len(part):
            print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f} "
                  f"range={part['timestamp'].min()}..{part['timestamp'].max()}")
        else:
            print(f"{name}: n=0")

    train = parts["train"]
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=20260829,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])

    results = {name: evaluate(model, part) for name, part in parts.items()
               if name != "reserved_untouched" and len(part)}

    importance = permutation_importance(
        model, parts["val"][FEATURE_COLUMNS], parts["val"]["label"],
        n_repeats=10, random_state=20260829, scoring="roc_auc",
    )
    importance_table = pd.Series(importance.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)

    report = {
        "feature_columns": FEATURE_COLUMNS,
        "split_boundaries": {
            "val_start": str(VAL_START), "val_end": str(VAL_END),
            "oos_start": str(OOS_START), "oos_end": str(OOS_END),
        },
        "reserved_untouched_rows": int(len(parts["reserved_untouched"])),
        "results": results,
        "val_permutation_importance_auc_drop": importance_table.round(5).to_dict(),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Persist a Sigma3-1h HGB model to disk so it can be loaded for LIVE inference.

train_sigma3_1h_hgb_20260705.py trains and predicts in one process and never saves the model --
every prior Sigma3-1h/Sigma6 result in this project came from an offline batch tape, never a live
model. This script trains the IDENTICAL model (same hyperparameters, same historical feature
parquets, same seed) and joblib-dumps it plus its feature column order, so
live_sigma6_regime_tiebreak_shadow_20260801.py can load it and score new live 1h bars causally.

Uses the IDENTICAL train_mask (timestamp <= TRAIN_END = 2025-06-30) as
train_sigma3_1h_hgb_20260705.py -- the earlier version of this script fitted on the full concatenated
2024-2026 frame with no holdout at all, which silently trained the live-persisted model on the exact
VAL_2025Q4/OOS_2026H1 windows that the Sigma6/regime_tiebreak backtest numbers (and this project's
decision to run Tau1 live) were based on. Fixed 2026-08-07: this script now produces a model that is
actually identical (same rows, same target, same weights) to the one those backtest numbers describe,
not just the same hyperparameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_20260705"
OUT_DIR = ROOT / "data/ensemble/supervised/sigma3_1h_hgb_live_20260801"
SEED = 270705
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
DEFAULT_THR = 0.45  # matches train_sigma3_1h_hgb_20260705.py's own tape-quality gate
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")  # matches train_sigma3_1h_hgb_20260705.py's own holdout


def load_all() -> pd.DataFrame:
    frames = [pd.read_parquet(DATA_DIR / f"sigma3_1h_{y}.parquet") for y in (2024, 2025, 2026)]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    train_mask = df["timestamp"] <= TRAIN_END
    train_df = df.loc[train_mask]
    print(f"features: {len(feat_cols)}  rows: {len(df)}  "
          f"range: {df['timestamp'].min()}..{df['timestamp'].max()}", flush=True)
    print(f"train rows: {len(train_df)}  train range: {train_df['timestamp'].min()}.."
          f"{train_df['timestamp'].max()}  held-out rows: {len(df) - len(train_df)}", flush=True)

    Xtr = train_df[feat_cols].to_numpy(dtype=np.float64)
    ytr = train_df["ts_action"].to_numpy(dtype=np.int64)
    w = np.clip(np.abs(train_df["ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)

    clf = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.03, max_iter=400, max_depth=4,
        l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=80,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
        random_state=SEED, class_weight="balanced",
    )
    clf.fit(Xtr, ytr, sample_weight=w)
    print(f"iters: {clf.n_iter_}  classes: {list(clf.classes_)}", flush=True)

    joblib.dump(clf, OUT_DIR / "model.joblib")
    (OUT_DIR / "feature_cols.json").write_text(json.dumps({
        "feature_cols": feat_cols,
        "classes": [int(c) for c in clf.classes_],
        "seed": SEED,
        "default_thr": DEFAULT_THR,
        "train_rows": int(len(train_df)),
        "train_range": [str(train_df["timestamp"].min()), str(train_df["timestamp"].max())],
        "held_out_range": [str(TRAIN_END), str(df["timestamp"].max())],
        "held_out_rows": int(len(df) - len(train_df)),
        "source": "train_sigma3_1h_hgb_20260705.py hyperparameters AND train_mask holdout, persisted "
                   "for live inference -- identical training set to the model behind the "
                   "VAL_2025Q4/OOS_2026H1 Sigma6/regime_tiebreak backtest numbers",
    }, indent=2))
    print(f"wrote {OUT_DIR / 'model.joblib'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

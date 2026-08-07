from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR, os.path.join(_ROOT_DIR, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import make_triple_barrier_targets, median_fill_by_train, time_split_indices
from ensemble.train_unified_direction_catboost import FEATURE_COLS as CORE_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = "data/rl_training_2025_unified.csv"
DEFAULT_SAVE = "data/ensemble/supervised/unified_direction_catboost_tsfm.json"

TS_FEATURE_COLS = [
    "pred_mdjd",
    "conf_mdjd",
    "pred_patchtst",
    "conf_patchtst",
    "patchtst_median",
    "patchtst_regime_sim",
    "timesnet_cycle_sin",
    "timesnet_cycle_cos",
    "timesnet_cycle_delta",
    "dlinear_smf_ema",
    "dlinear_smf_slope",
]
FEATURE_COLS = CORE_FEATURE_COLS + TS_FEATURE_COLS


def _resolve_meta_paths(save_path: str) -> tuple[str, str]:
    meta_path = save_path
    if meta_path.endswith(".pkl"):
        meta_path = meta_path[:-4] + ".json"
    model_path = meta_path[:-5] + ".pkl" if meta_path.endswith(".json") else meta_path + ".pkl"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    return model_path, meta_path


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(args.csv_path)
    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    df = _safe_fill(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required features: {missing}")

    y = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    valid = y >= 0
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    x = df[FEATURE_COLS].copy()
    x_train = x.iloc[tr_idx].copy()
    x_val = x.iloc[va_idx].copy()
    x_test = x.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)
    y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=args.od_wait,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

    val_pred = model.predict(x_val).reshape(-1).astype(int)
    test_pred = model.predict(x_test).reshape(-1).astype(int)
    val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_pred))
    updown_mask = np.isin(y_test, [0, 2])
    dir_f1 = float(f1_score(y_test[updown_mask], test_pred[updown_mask], average="macro")) if np.any(updown_mask) else 0.0
    report = classification_report(y_test, test_pred, output_dict=True)

    model_path, meta_path = _resolve_meta_paths(args.save_path)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
    artifact = {
        "feature_cols": FEATURE_COLS,
        "core_feature_cols": CORE_FEATURE_COLS,
        "ts_feature_cols": TS_FEATURE_COLS,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "unified_direction_catboost_tsfm",
            "csv_path": args.csv_path,
            "atr_mult": args.atr_mult,
            "max_hold": args.max_hold,
            "atr_window": args.atr_window,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "val_balanced_acc": val_bal_acc,
            "test_balanced_acc": test_bal_acc,
            "test_dir_f1": dir_f1,
            "classification_report": report,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("val_balanced_acc=%.4f test_balanced_acc=%.4f test_dir_f1=%.4f", val_bal_acc, test_bal_acc, dir_f1)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TS-model-augmented CatBoost unified direction classifier")
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--save-path", default=DEFAULT_SAVE)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--atr-mult", type=float, default=0.8)
    p.add_argument("--max-hold", type=int, default=8)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--iterations", type=int, default=900)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--od-wait", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR, os.path.join(_ROOT_DIR, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import make_triple_barrier_targets, median_fill_by_train, time_split_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = "data/rl_training_2025_unified.csv"
DEFAULT_SAVE = "data/ensemble/supervised/unified_direction_hgb.json"

FEATURE_COLS = [
    "m7_prob_up",
    "m7_prob_dn",
    "m7_q50",
    "m7_qwidth",
    "m7_gmm_vol_rank",
    "m7_iso_score",
    "m7_composite_score",
    "m7_expected_ret",
    "m7_tail_risk",
    "smart_money_flow",
    "taker_acceleration",
    "trade_intensity",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
]


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


def _sample_weight(y: np.ndarray) -> np.ndarray:
    uniq, cnt = np.unique(y, return_counts=True)
    total = float(len(y))
    k = max(len(uniq), 1)
    m = {int(u): total / max(float(c) * k, 1.0) for u, c in zip(uniq, cnt)}
    return np.asarray([m[int(v)] for v in y], dtype=np.float64)


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

    y = make_triple_barrier_targets(
        df,
        atr_mult=args.atr_mult,
        max_hold=args.max_hold,
        atr_window=args.atr_window,
    )
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

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=None,
        n_iter_no_change=args.n_iter_no_change,
    )
    model.fit(x_train, y_train, sample_weight=_sample_weight(y_train))

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
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
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "unified_direction_hgb",
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
    p = argparse.ArgumentParser(description="Train supervised unified direction classifier")
    p.add_argument("--csv-path", default=DEFAULT_CSV)
    p.add_argument("--save-path", default=DEFAULT_SAVE)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--atr-mult", type=float, default=0.8)
    p.add_argument("--max-hold", type=int, default=8)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--l2-regularization", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=350)
    p.add_argument("--n-iter-no-change", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

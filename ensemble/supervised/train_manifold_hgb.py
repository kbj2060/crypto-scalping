from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.high_order_state import HIGH_ORDER_STATE_COLS
from features.selection import auto_select_features
from ensemble.artifact_utils import resolve_model_meta_paths, save_pickle
from ensemble.supervised.common import (
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
    load_feature_frame,
    make_triple_barrier_targets,
    median_fill_by_train,
    select_feature_columns,
    time_split_indices,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _select_ranked_features(df, feature_cols, tr_idx, y_train, max_features: int, must_include: list[str]) -> list[str]:
    train_df_tmp = df.iloc[tr_idx].copy()
    train_df_tmp.index = range(len(train_df_tmp))
    train_df_tmp["_label"] = y_train
    return auto_select_features(
        train_df_tmp,
        feature_cols,
        target_col="_label",
        max_features=max_features,
        corr_threshold=0.85,
        must_include=must_include,
    )


def _class_weights(y: np.ndarray) -> np.ndarray:
    uniq, cnt = np.unique(y, return_counts=True)
    total = float(len(y))
    mapping = {int(k): total / max(float(v) * len(uniq), 1.0) for k, v in zip(uniq, cnt)}
    return np.asarray([mapping.get(int(v), 1.0) for v in y], dtype=np.float64)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    df = load_feature_frame(args.data_path, args.rl_path)
    y = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    valid = y >= 0
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    must_include = [c for c in HIGH_ORDER_STATE_COLS if c in df.columns]
    feature_cols = select_feature_columns(df, must_include=must_include)
    ranked_features = _select_ranked_features(df, feature_cols, tr_idx, y[tr_idx], args.max_features, must_include)
    feature_count = min(args.feature_count, len(ranked_features))
    feature_cols = ranked_features[:feature_count]

    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
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
    sample_weight = _class_weights(y_train)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_pred))

    updown_mask = np.isin(y_test, [0, 2])
    dir_f1 = float(f1_score(y_test[updown_mask], test_pred[updown_mask], average="macro")) if np.any(updown_mask) else 0.0
    report = classification_report(y_test, test_pred, output_dict=True)

    model_path, meta_path = resolve_model_meta_paths(args.save_path)
    save_pickle({"model": model, "feature_cols": feature_cols}, model_path)
    artifact = {
        "feature_cols": feature_cols,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "manifold_hgb_direction",
            "val_balanced_acc": val_bal_acc,
            "test_balanced_acc": test_bal_acc,
            "test_dir_f1": dir_f1,
            "feature_count": len(feature_cols),
            "params": {
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "max_leaf_nodes": args.max_leaf_nodes,
                "min_samples_leaf": args.min_samples_leaf,
                "l2_regularization": args.l2_regularization,
                "max_iter": args.max_iter,
            },
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
    p = argparse.ArgumentParser(description="Train manifold-aware fast direction classifier")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/manifold_hgb.json")
    p.add_argument("--max-features", type=int, default=64)
    p.add_argument("--feature-count", type=int, default=36)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--atr-mult", type=float, default=1.5)
    p.add_argument("--max-hold", type=int, default=9)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=40)
    p.add_argument("--l2-regularization", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=450)
    p.add_argument("--n-iter-no-change", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

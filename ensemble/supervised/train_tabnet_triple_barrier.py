from __future__ import annotations

import os
import sys
import pickle
import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.optuna_helper import (
    build_config_hash,
    hash_arrays,
    hash_frame,
    load_reusable_results,
    save_training_results,
    training_results_path,
)
from ensemble.supervised.common import (
    load_feature_frame,
    select_feature_columns,
    make_triple_barrier_targets,
    time_split_indices,
    CATEGORICAL_HINTS,
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _require_tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pytorch-tabnet is required. Install with: pip install pytorch-tabnet"
        ) from e
    return TabNetClassifier


def _encode_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    cat_maps: Dict[str, Dict[str, int]] | None = None,
) -> tuple[np.ndarray, Dict[str, Dict[str, int]], np.ndarray, np.ndarray]:
    out = df[feature_cols].copy()
    cat_maps = cat_maps or {}
    cat_idxs: List[int] = []
    cat_dims: List[int] = []

    for i, c in enumerate(feature_cols):
        if c not in cat_cols:
            continue
        cat_idxs.append(i)
        if c not in cat_maps:
            values = out[c].fillna("UNK").astype(str).unique().tolist()
            mapping = {v: j for j, v in enumerate(values)}
            if "UNK" not in mapping:
                mapping["UNK"] = len(mapping)
            cat_maps[c] = mapping
        mapping = cat_maps[c]
        out[c] = out[c].fillna("UNK").astype(str).map(lambda v: mapping.get(v, mapping["UNK"]))
        cat_dims.append(len(mapping))

    med = out.median(numeric_only=True)
    out = out.fillna(med)
    return (
        out.values.astype(np.float32),
        cat_maps,
        np.array(cat_idxs, dtype=np.int64),
        np.array(cat_dims, dtype=np.int64),
    )


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "n_d": args.n_d,
        "n_a": args.n_a,
        "n_steps": args.n_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "lambda_sparse": args.lambda_sparse,
    }


def _build_model(TabNetClassifier, seed: int, params: Dict[str, Any], cat_idxs: np.ndarray, cat_dims: np.ndarray):
    return TabNetClassifier(
        n_d=int(params["n_d"]),
        n_a=int(params["n_a"]),
        n_steps=int(params["n_steps"]),
        gamma=float(params["gamma"]),
        lambda_sparse=float(params.get("lambda_sparse", 1e-5)),
        optimizer_params=dict(lr=float(params["learning_rate"])),
        seed=seed,
        verbose=0,
        cat_idxs=cat_idxs.tolist(),
        cat_dims=cat_dims.tolist(),
    )


def _val_dir_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    dir_mask = y_true != 1
    if dir_mask.sum() == 0:
        return 0.0
    return float(f1_score(y_true[dir_mask], y_pred[dir_mask], labels=[0, 2], average="macro", zero_division=0))


def _tune_params(
    args: argparse.Namespace,
    TabNetClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cat_idxs: np.ndarray,
    cat_dims: np.ndarray,
) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_d": trial.suggest_int("n_d", 16, 64, step=8),
            "n_a": trial.suggest_int("n_a", 16, 64, step=8),
            "n_steps": trial.suggest_int("n_steps", 3, 8),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 8e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
            "max_epochs": trial.suggest_int("max_epochs", 60, 220),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        }
        model = _build_model(TabNetClassifier, args.seed, params, cat_idxs, cat_dims)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_name=["val"],
            eval_metric=["balanced_accuracy"],
            max_epochs=int(params["max_epochs"]),
            patience=args.patience,
            batch_size=int(params["batch_size"]),
            virtual_batch_size=max(32, int(params["batch_size"]) // 4),
        )
        y_pred_val = model.predict(x_val)
        return _val_dir_f1(y_val, y_pred_val)

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)

    best_params = dict(study.best_params)
    if "max_epochs" in best_params:
        best_params["max_epochs"] = int(best_params["max_epochs"] * 1.1)
    return best_params, float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    TabNetClassifier = _require_tabnet()

    df = load_feature_frame(args.data_path, args.rl_path)
    y = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    valid = y >= 0
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid].astype(np.int64)

    feature_cols = select_feature_columns(df)
    cat_cols = [c for c in CATEGORICAL_HINTS if c in feature_cols]
    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)

    x_train, cat_maps_train, cat_idxs_train, cat_dims_train = _encode_frame(df.iloc[tr_idx], feature_cols, cat_cols, cat_maps=None)
    x_val, _, _, _ = _encode_frame(df.iloc[va_idx], feature_cols, cat_cols, cat_maps=cat_maps_train)

    y_train = y[tr_idx]
    y_val = y[va_idx]
    y_test = y[te_idx]

    data_hash = build_config_hash(
        {
            "x_hash": hash_frame(df[feature_cols]),
            "y_hash": hash_arrays(y.astype(np.int64)),
        }
    )
    config_hash = build_config_hash(
        {
            "atr_mult": args.atr_mult,
            "max_hold": args.max_hold,
            "atr_window": args.atr_window,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "feature_count": len(feature_cols),
            "cat_count": len(cat_cols),
        }
    )
    results_path = training_results_path(args.save_path, "tabnet_triple_barrier")

    prev = load_reusable_results(
        results_path=results_path,
        data_hash=data_hash,
        config_hash=config_hash,
        force_reuse_results=args.force_reuse_results,
        logger=logger,
    )

    if prev is not None:
        best_params = dict(prev.get("best_params", {}))
        if "max_epochs" in best_params:
            best_params["max_epochs"] = int(best_params["max_epochs"] * 1.1)
        best_val_dir_f1 = float(prev.get("best_val_dir_f1", 0.0))
    else:
        best_params, best_val_dir_f1 = _tune_params(
            args,
            TabNetClassifier,
            x_train,
            y_train,
            x_val,
            y_val,
            cat_idxs_train,
            cat_dims_train,
        )

    merged = _base_params(args)
    merged.update(best_params)

    trainval_idx = np.concatenate([tr_idx, va_idx])
    x_trainval, cat_maps_final, cat_idxs_final, cat_dims_final = _encode_frame(
        df.iloc[trainval_idx],
        feature_cols,
        cat_cols,
        cat_maps=None,
    )
    x_test, _, _, _ = _encode_frame(df.iloc[te_idx], feature_cols, cat_cols, cat_maps=cat_maps_final)
    y_trainval = y[trainval_idx]

    model = _build_model(TabNetClassifier, args.seed, merged, cat_idxs_final, cat_dims_final)
    model.fit(
        x_trainval,
        y_trainval,
        max_epochs=int(merged["max_epochs"]),
        patience=args.patience,
        batch_size=int(merged["batch_size"]),
        virtual_batch_size=max(32, int(merged["batch_size"]) // 4),
    )

    y_pred = model.predict(x_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    logger.info("TabNet Triple-Barrier test balanced_acc=%.4f", bal_acc)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_prefix = os.path.splitext(args.save_path)[0]
    model.save_model(save_prefix)

    metadata = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_maps": cat_maps_final,
        "meta": {
            "algorithm": "tabnet_triple_barrier",
            "balanced_accuracy": float(bal_acc),
            "best_val_dir_f1": best_val_dir_f1,
            "best_params": merged,
            "model_zip": f"{save_prefix}.zip",
        },
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(metadata, f)

    save_training_results(
        results_path,
        {
            "best_val_dir_f1": best_val_dir_f1,
            "test_bacc": float(bal_acc),
            "best_params": merged,
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model metadata: %s", args.save_path)
    logger.info("saved: %s", results_path)
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TabNet Triple-Barrier classifier")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/tabnet_triple_barrier.pkl")
    p.add_argument("--atr-mult", type=float, default=0.8)
    p.add_argument("--max-hold", type=int, default=12)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--n-d", type=int, default=32)
    p.add_argument("--n-a", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--lambda-sparse", type=float, default=1e-5)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

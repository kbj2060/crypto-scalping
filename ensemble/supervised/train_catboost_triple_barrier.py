from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

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
from ensemble.artifact_utils import load_best_params_from_meta, resolve_model_meta_paths, save_pickle
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


def _require_catboost():
    try:
        from catboost import CatBoostClassifier, Pool  # type: ignore
    except ImportError as e:
        raise ImportError(
            "catboost is required. Install with: pip install catboost"
        ) from e
    return CatBoostClassifier, Pool


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "bagging_temperature": args.bagging_temperature,
        "random_strength": args.random_strength,
        "border_count": args.border_count,
    }


def _prepare_catboost_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_candidates: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = df[feature_cols].copy()
    cat_cols: list[str] = []

    for col in cat_candidates:
        if col not in out.columns:
            continue
        series = out[col]
        non_null = series.dropna()
        if non_null.empty:
            out[col] = "UNK"
            cat_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            as_num = pd.to_numeric(non_null, errors="coerce")
            is_integral = np.all(np.isfinite(as_num)) and np.all(np.isclose(as_num, np.round(as_num)))
            if is_integral:
                out[col] = series.fillna(-1).round().astype(np.int64).astype(str)
                cat_cols.append(col)
            else:
                logger.info("exclude non-discrete cat hint for CatBoost: %s", col)
            continue

        out[col] = series.fillna("UNK").astype(str)
        cat_cols.append(col)

    return out, cat_cols


def _tune_params(
    args: argparse.Namespace,
    CatBoostClassifier,
    train_pool,
    val_pool,
    y_val: np.ndarray,
) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 8.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 64, 254),
        }
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        y_pred = model.predict(val_pool).reshape(-1).astype(int)
        dir_mask = y_val != 1
        if dir_mask.sum() == 0:
            return 0.0
        return float(f1_score(y_val[dir_mask], y_pred[dir_mask], labels=[0, 2], average="macro", zero_division=0))

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["iterations"] = int(best_params.get("iterations", args.iterations) * 1.1)
    return best_params, float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    CatBoostClassifier, Pool = _require_catboost()

    df = load_feature_frame(args.data_path, args.rl_path)
    y = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    valid = y >= 0
    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    feature_cols = select_feature_columns(df)
    cat_candidates = [c for c in CATEGORICAL_HINTS if c in feature_cols]

    x, cat_cols = _prepare_catboost_frame(df, feature_cols, cat_candidates)
    tr_idx, va_idx, te_idx = time_split_indices(len(x), args.train_ratio, args.val_ratio)

    x_train, y_train = x.iloc[tr_idx], y[tr_idx]
    x_val, y_val = x.iloc[va_idx], y[va_idx]
    x_test, y_test = x.iloc[te_idx], y[te_idx]

    train_pool = Pool(x_train, y_train, cat_features=cat_cols)
    val_pool = Pool(x_val, y_val, cat_features=cat_cols)
    test_pool = Pool(x_test, y_test, cat_features=cat_cols)

    data_hash = build_config_hash(
        {
            "x_hash": hash_frame(x),
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
    results_path = training_results_path(args.save_path, "catboost_triple_barrier")
    model_path, meta_path = resolve_model_meta_paths(args.save_path)

    prev = load_reusable_results(
        results_path=results_path,
        data_hash=data_hash,
        config_hash=config_hash,
        force_reuse_results=args.force_reuse_results,
        logger=logger,
    )

    if prev is not None:
        best_params = dict(prev.get("best_params", {}))
        best_params["iterations"] = int(best_params.get("iterations", args.iterations) * 1.1)
        best_val_dir_f1 = float(prev.get("best_val_dir_f1", prev.get("best_val_bacc", 0.0)))
        logger.info("reuse best_params with iterations boost x1.1 -> %d", best_params["iterations"])
    else:
        meta_params, meta_score = load_best_params_from_meta(meta_path, score_keys=["best_val_dir_f1", "best_val_bacc"])
        if meta_params is not None:
            best_params = meta_params
            best_params["iterations"] = int(best_params.get("iterations", args.iterations) * 1.1)
            best_val_dir_f1 = float(meta_score or 0.0)
            logger.info("reuse best_params from meta json: %s", meta_path)
        else:
            best_params, best_val_dir_f1 = _tune_params(args, CatBoostClassifier, train_pool, val_pool, y_val)

    merged = _base_params(args)
    merged.update(best_params)

    trainval_idx = np.concatenate([tr_idx, va_idx])
    x_trainval = x.iloc[trainval_idx]
    y_trainval = y[trainval_idx]
    trainval_pool = Pool(x_trainval, y_trainval, cat_features=cat_cols)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
        **merged,
    )
    model.fit(trainval_pool, verbose=False)
    y_pred = model.predict(test_pool).reshape(-1).astype(int)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    logger.info("CatBoost Triple-Barrier test balanced_acc=%.4f", bal_acc)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))

    save_pickle(model, model_path)
    artifact = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "catboost_triple_barrier",
            "balanced_accuracy": float(bal_acc),
            "best_val_dir_f1": float(best_val_dir_f1),
            "atr_mult": args.atr_mult,
            "max_hold": args.max_hold,
            "atr_window": args.atr_window,
            "best_params": merged,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=True)

    save_training_results(
        results_path,
        {
            "best_val_dir_f1": float(best_val_dir_f1),
            "test_bacc": float(bal_acc),
            "best_params": merged,
            "atr_mult": args.atr_mult,
            "max_hold": args.max_hold,
            "atr_window": args.atr_window,
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatBoost Triple-Barrier classifier")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/catboost_triple_barrier.pkl")
    p.add_argument("--atr-mult", type=float, default=0.8)
    p.add_argument("--max-hold", type=int, default=12)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=4.0)
    p.add_argument("--bagging-temperature", type=float, default=1.0)
    p.add_argument("--random-strength", type=float, default=1.0)
    p.add_argument("--border-count", type=int, default=128)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

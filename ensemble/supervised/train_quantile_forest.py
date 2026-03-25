from __future__ import annotations

import os
import pickle
import argparse
import logging
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from ensemble.optuna_helper import (
    build_config_hash,
    hash_arrays,
    load_reusable_results,
    save_training_results,
    training_results_path,
)
from ensemble.supervised.common import (
    load_feature_frame,
    select_feature_columns,
    make_future_return,
    time_split_indices,
    median_fill_by_train,
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _predict_quantiles(model: RandomForestRegressor, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree_preds = np.stack([est.predict(x) for est in model.estimators_], axis=1)
    q10 = np.quantile(tree_preds, 0.10, axis=1)
    q50 = np.quantile(tree_preds, 0.50, axis=1)
    q90 = np.quantile(tree_preds, 0.90, axis=1)
    return q10, q50, q90


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
    }


def _score_predictions(y_true: np.ndarray, q50: np.ndarray, flat_threshold: float) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true, q50))
    pred_dir = np.where(q50 > flat_threshold, 1, np.where(q50 < -flat_threshold, -1, 0))
    true_dir = np.where(y_true > flat_threshold, 1, np.where(y_true < -flat_threshold, -1, 0))
    dir_acc = float((pred_dir == true_dir).mean())
    return mae, dir_acc


def _tune_params(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1400),
            "max_depth": trial.suggest_int("max_depth", 4, 28),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 24),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
        flat_threshold = trial.suggest_float("flat_threshold", 1e-4, 2e-3, log=True)

        model = RandomForestRegressor(
            random_state=args.seed,
            n_jobs=args.n_jobs,
            **params,
        )
        model.fit(x_train, y_train)
        _, q50, _ = _predict_quantiles(model, x_val)
        mae, dir_acc = _score_predictions(y_val, q50, flat_threshold)
        return float(dir_acc - 50.0 * mae)

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)

    best_params = dict(study.best_params)
    if "n_estimators" in best_params:
        best_params["n_estimators"] = int(best_params["n_estimators"] * 1.1)
    return best_params, float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    df = load_feature_frame(args.data_path, args.rl_path)
    y_ret = make_future_return(df, horizon=args.horizon)
    valid = np.isfinite(y_ret)

    df = df.loc[valid].reset_index(drop=True)
    y_ret = y_ret[valid]
    feature_cols = select_feature_columns(df)
    x_all = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    x_train = x_all.iloc[tr_idx].copy()
    x_val = x_all.iloc[va_idx].copy()
    x_test = x_all.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)

    x_train_np = x_train.values
    x_val_np = x_val.values
    x_test_np = x_test.values

    y_train = y_ret[tr_idx]
    y_val = y_ret[va_idx]
    y_test = y_ret[te_idx]

    data_hash = hash_arrays(x_all.values.astype(np.float32), y_ret)
    config_hash = build_config_hash(
        {
            "horizon": args.horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "feature_count": len(feature_cols),
        }
    )
    results_path = training_results_path(args.save_path, "quantile_forest")

    prev = load_reusable_results(
        results_path=results_path,
        data_hash=data_hash,
        config_hash=config_hash,
        force_reuse_results=args.force_reuse_results,
        logger=logger,
    )

    if prev is not None:
        best_params = dict(prev.get("best_params", {}))
        if "n_estimators" in best_params:
            best_params["n_estimators"] = int(best_params["n_estimators"] * 1.1)
        best_val_score = float(prev.get("best_val_score", 0.0))
    else:
        best_params, best_val_score = _tune_params(args, x_train_np, y_train, x_val_np, y_val)

    flat_threshold = float(best_params.pop("flat_threshold", args.flat_threshold))

    merged = _base_params(args)
    merged.update(best_params)

    x_trainval_np = np.vstack([x_train_np, x_val_np])
    y_trainval = np.hstack([y_train, y_val])

    model = RandomForestRegressor(
        random_state=args.seed,
        n_jobs=args.n_jobs,
        **merged,
    )
    model.fit(x_trainval_np, y_trainval)

    q10, q50, q90 = _predict_quantiles(model, x_test_np)
    mae, dir_acc = _score_predictions(y_test, q50, flat_threshold)
    logger.info("QuantileForest MAE=%.6f dir_acc=%.4f", mae, dir_acc)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "horizon": args.horizon,
        "flat_threshold": flat_threshold,
        "meta": {
            "algorithm": "quantile_forest",
            "mae": float(mae),
            "direction_accuracy": dir_acc,
            "q10_mean": float(np.mean(q10)),
            "q50_mean": float(np.mean(q50)),
            "q90_mean": float(np.mean(q90)),
            "best_val_score": best_val_score,
            "best_params": {**merged, "flat_threshold": flat_threshold},
        },
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(artifact, f)

    save_training_results(
        results_path,
        {
            "best_val_score": best_val_score,
            "test_mae": float(mae),
            "test_dir_acc": dir_acc,
            "best_params": {**merged, "flat_threshold": flat_threshold},
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved: %s", args.save_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Quantile Regression Forest style regressor")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/quantile_forest.pkl")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--max-features", type=float, default=0.7)
    p.add_argument("--flat-threshold", type=float, default=0.0005)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

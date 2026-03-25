from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.optuna_helper import (
    build_config_hash,
    hash_arrays,
    load_reusable_results,
    save_training_results,
    training_results_path,
)
from ensemble.artifact_utils import load_best_params_from_meta, resolve_model_meta_paths, save_pickle
from ensemble.supervised.common import (
    load_feature_frame,
    select_feature_columns,
    make_triple_barrier_targets,
    make_future_return,
    time_split_indices,
    median_fill_by_train,
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _resolve_backend() -> str:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except Exception:
        return "sklearn_hgb"


def _build_models(backend: str, seed: int, n_jobs: int, params: Dict[str, Any]):
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

        common = {
            "n_estimators": int(params.get("n_estimators", 700)),
            "learning_rate": float(params.get("learning_rate", 0.03)),
            "num_leaves": int(params.get("num_leaves", 63)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "min_child_samples": int(params.get("min_child_samples", 20)),
            "reg_alpha": float(params.get("reg_alpha", 1e-3)),
            "reg_lambda": float(params.get("reg_lambda", 1e-2)),
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        cls = LGBMClassifier(objective="multiclass", num_class=3, **common)
        reg = LGBMRegressor(objective="regression_l1", **common)
        return cls, reg

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    max_iter = int(params.get("max_iter", 700))
    learning_rate = float(params.get("learning_rate", 0.03))
    max_depth = int(params.get("max_depth", 8))

    cls = HistGradientBoostingClassifier(
        random_state=seed,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    reg = HistGradientBoostingRegressor(
        random_state=seed,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        loss="absolute_error",
    )
    return cls, reg


def _base_params(backend: str) -> Dict[str, Any]:
    if backend == "lightgbm":
        return {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 1e-3,
            "reg_lambda": 1e-2,
        }
    return {
        "max_iter": 700,
        "learning_rate": 0.03,
        "max_depth": 8,
    }


def _objective_score(y_true: np.ndarray, y_pred: np.ndarray, vol_pred: np.ndarray, threshold: float, target_trade_rate: float) -> float:
    enter_mask = vol_pred >= threshold
    trade_rate = float(np.mean(enter_mask))
    if np.any(enter_mask):
        filtered_bal_acc = float(balanced_accuracy_score(y_true[enter_mask], y_pred[enter_mask]))
    else:
        filtered_bal_acc = 0.0

    coverage = min(1.0, trade_rate / max(target_trade_rate, 1e-6))
    penalty = abs(trade_rate - target_trade_rate)
    return float(filtered_bal_acc * (0.5 + 0.5 * coverage) - 0.05 * penalty)


def _tune_params(
    args: argparse.Namespace,
    backend: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train_dir: np.ndarray,
    y_val_dir: np.ndarray,
    y_train_vol: np.ndarray,
) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        if backend == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
        else:
            params = {
                "max_iter": trial.suggest_int("max_iter", 250, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
            }

        min_expected_move = trial.suggest_float("min_expected_move", 2e-4, 5e-3, log=True)

        dir_model, vol_model = _build_models(backend, args.seed, args.n_jobs, params)
        dir_model.fit(x_train, y_train_dir)
        vol_model.fit(x_train, y_train_vol)

        y_pred_dir_val = dir_model.predict(x_val)
        vol_pred_val = vol_model.predict(x_val)
        return _objective_score(
            y_true=y_val_dir,
            y_pred=y_pred_dir_val,
            vol_pred=vol_pred_val,
            threshold=min_expected_move,
            target_trade_rate=args.target_trade_rate,
        )

    logger.info("Optuna tuning start: n_trials=%d backend=%s", args.n_trials, backend)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)

    best_params = dict(study.best_params)
    if "n_estimators" in best_params:
        best_params["n_estimators"] = int(best_params["n_estimators"] * 1.1)
    if "max_iter" in best_params:
        best_params["max_iter"] = int(best_params["max_iter"] * 1.1)
    return best_params, float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    df = load_feature_frame(args.data_path, args.rl_path)

    y_dir = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    y_ret = make_future_return(df, horizon=args.horizon)
    valid = (y_dir >= 0) & np.isfinite(y_ret)

    df = df.loc[valid].reset_index(drop=True)
    y_dir = y_dir[valid]
    y_vol = np.abs(y_ret[valid])

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

    y_train_dir, y_val_dir, y_test_dir = y_dir[tr_idx], y_dir[va_idx], y_dir[te_idx]
    y_train_vol, y_val_vol, y_test_vol = y_vol[tr_idx], y_vol[va_idx], y_vol[te_idx]

    backend = _resolve_backend()
    data_hash = hash_arrays(x_all.values.astype(np.float32), y_dir.astype(np.int64), y_vol)
    config_hash = build_config_hash(
        {
            "atr_mult": args.atr_mult,
            "max_hold": args.max_hold,
            "atr_window": args.atr_window,
            "horizon": args.horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "backend": backend,
            "feature_count": len(feature_cols),
            "target_trade_rate": args.target_trade_rate,
        }
    )
    results_path = training_results_path(args.save_path, "two_stage_stacking")
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
        if "n_estimators" in best_params:
            best_params["n_estimators"] = int(best_params["n_estimators"] * 1.1)
        if "max_iter" in best_params:
            best_params["max_iter"] = int(best_params["max_iter"] * 1.1)
        best_val_score = float(prev.get("best_val_score", 0.0))
    else:
        meta_params, meta_score = load_best_params_from_meta(meta_path, score_keys=["best_val_score"])
        if meta_params is not None:
            best_params = meta_params
            if "n_estimators" in best_params:
                best_params["n_estimators"] = int(best_params["n_estimators"] * 1.1)
            if "max_iter" in best_params:
                best_params["max_iter"] = int(best_params["max_iter"] * 1.1)
            best_val_score = float(meta_score or 0.0)
            logger.info("reuse best_params from meta json: %s", meta_path)
        else:
            best_params, best_val_score = _tune_params(
                args=args,
                backend=backend,
                x_train=x_train_np,
                x_val=x_val_np,
                y_train_dir=y_train_dir,
                y_val_dir=y_val_dir,
                y_train_vol=y_train_vol,
            )

    min_expected_move = float(best_params.pop("min_expected_move", args.min_expected_move))

    merged = _base_params(backend)
    merged.update(best_params)

    x_trainval_np = np.vstack([x_train_np, x_val_np])
    y_trainval_dir = np.hstack([y_train_dir, y_val_dir])
    y_trainval_vol = np.hstack([y_train_vol, y_val_vol])

    dir_model, vol_model = _build_models(backend, args.seed, args.n_jobs, merged)
    dir_model.fit(x_trainval_np, y_trainval_dir)
    vol_model.fit(x_trainval_np, y_trainval_vol)

    y_pred_dir = dir_model.predict(x_test_np)
    dir_bal_acc = balanced_accuracy_score(y_test_dir, y_pred_dir)
    logger.info("2-Stage dir balanced_acc=%.4f backend=%s", dir_bal_acc, backend)
    logger.info("\n%s", classification_report(y_test_dir, y_pred_dir, digits=4))

    vol_pred = vol_model.predict(x_test_np)
    enter_mask = vol_pred >= min_expected_move
    trade_rate = float(np.mean(enter_mask))
    if np.any(enter_mask):
        trade_bal_acc = balanced_accuracy_score(y_test_dir[enter_mask], y_pred_dir[enter_mask])
    else:
        trade_bal_acc = 0.0
    logger.info("2-Stage filtered trade_rate=%.4f filtered_bal_acc=%.4f", trade_rate, trade_bal_acc)

    save_pickle(
        {
            "backend": backend,
            "feature_cols": feature_cols,
            "dir_model": dir_model,
            "vol_model": vol_model,
            "min_expected_move": min_expected_move,
        },
        model_path,
    )

    artifact = {
        "feature_cols": feature_cols,
        "min_expected_move": min_expected_move,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "two_stage_stacking",
            "backend": backend,
            "direction_balanced_accuracy": float(dir_bal_acc),
            "filtered_trade_rate": trade_rate,
            "filtered_balanced_accuracy": float(trade_bal_acc),
            "best_val_score": best_val_score,
            "best_params": merged,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=True)

    save_training_results(
        results_path,
        {
            "best_val_score": best_val_score,
            "test_dir_bal_acc": float(dir_bal_acc),
            "test_trade_rate": trade_rate,
            "test_filtered_bal_acc": float(trade_bal_acc),
            "best_params": {**merged, "min_expected_move": min_expected_move},
            "backend": backend,
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )

    logger.info("saved meta: %s", meta_path)
    logger.info("saved model: %s", model_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 2-stage direction+volatility stacking")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/two_stage_stacking.json")
    p.add_argument("--atr-mult", type=float, default=0.8)
    p.add_argument("--max-hold", type=int, default=12)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--min-expected-move", type=float, default=0.0015)
    p.add_argument("--target-trade-rate", type=float, default=0.25)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

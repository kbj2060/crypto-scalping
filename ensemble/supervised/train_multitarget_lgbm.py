from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score

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
        return "sklearn_rf"


def _build_models(backend: str, seed: int, n_jobs: int, params: Dict[str, Any]):
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

        common = {
            "n_estimators": int(params.get("n_estimators", 900)),
            "learning_rate": float(params.get("learning_rate", 0.03)),
            "num_leaves": int(params.get("num_leaves", 63)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "min_child_samples": int(params.get("min_child_samples", 20)),
            "reg_alpha": float(params.get("reg_alpha", 1e-3)),
            "reg_lambda": float(params.get("reg_lambda", 1e-2)),
            "random_state": seed,
            "n_jobs": n_jobs,
        }
        cls = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            verbose=-1,
            **common,
        )
        reg_quality = LGBMRegressor(
            objective="regression_l1",
            verbose=-1,
            **common,
        )
        reg_hold = LGBMRegressor(
            objective="regression_l1",
            verbose=-1,
            **common,
        )
        return cls, reg_quality, reg_hold

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    common = {
        "n_estimators": int(params.get("n_estimators", 600)),
        "max_depth": int(params.get("max_depth", 12)),
        "min_samples_leaf": int(params.get("min_samples_leaf", 5)),
        "max_features": float(params.get("max_features", 0.7)),
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    cls = RandomForestClassifier(**common)
    reg_quality = RandomForestRegressor(**common)
    reg_hold = RandomForestRegressor(**common)
    return cls, reg_quality, reg_hold


def _build_quality_and_hold_targets(df, y_dir: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    close = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64) if "high" in df.columns else close
    lows = df["low"].values.astype(np.float64) if "low" in df.columns else close
    n = len(df)

    y_quality = np.full(n, np.nan, dtype=np.float64)
    y_hold = np.full(n, np.nan, dtype=np.float64)
    for t in range(n - horizon):
        cur = max(close[t], 1e-8)
        h_win = highs[t + 1 : t + horizon + 1]
        l_win = lows[t + 1 : t + horizon + 1]
        if len(h_win) == 0:
            continue

        if y_dir[t] == 2:  # UP
            rel = h_win / cur - 1.0
        elif y_dir[t] == 0:  # DOWN
            rel = cur / np.maximum(l_win, 1e-8) - 1.0
        else:
            rel = np.zeros_like(h_win)

        best_idx = int(np.argmax(rel))
        y_quality[t] = float(rel[best_idx])
        y_hold[t] = float(best_idx + 1)

    return y_quality, y_hold


def _base_params(args: argparse.Namespace, backend: str) -> Dict[str, Any]:
    if backend == "lightgbm":
        return {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 1e-3,
            "reg_lambda": 1e-2,
        }
    return {
        "n_estimators": 600,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    }


def _tune_params(
    args: argparse.Namespace,
    backend: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train_dir: np.ndarray,
    y_val_dir: np.ndarray,
    y_train_q: np.ndarray,
    y_val_q: np.ndarray,
    y_train_h: np.ndarray,
    y_val_h: np.ndarray,
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
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
                "max_depth": trial.suggest_int("max_depth", 4, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            }

        dir_model, quality_model, hold_model = _build_models(backend, args.seed, args.n_jobs, params)
        dir_model.fit(x_train, y_train_dir)
        quality_model.fit(x_train, y_train_q)
        hold_model.fit(x_train, y_train_h)

        dir_pred = dir_model.predict(x_val)
        dir_bal_acc = balanced_accuracy_score(y_val_dir, dir_pred)
        q_mae = float(np.mean(np.abs(quality_model.predict(x_val) - y_val_q)))
        h_mae = float(np.mean(np.abs(hold_model.predict(x_val) - y_val_h)))

        return float(dir_bal_acc - 0.10 * q_mae - 0.02 * h_mae)

    logger.info("Optuna tuning start: n_trials=%d backend=%s", args.n_trials, backend)
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
    y_dir = make_triple_barrier_targets(df, atr_mult=args.atr_mult, max_hold=args.max_hold, atr_window=args.atr_window)
    y_quality, y_hold = _build_quality_and_hold_targets(df, y_dir, args.horizon)

    valid = (y_dir >= 0) & np.isfinite(y_quality) & np.isfinite(y_hold)
    df = df.loc[valid].reset_index(drop=True)
    y_dir = y_dir[valid]
    y_quality = y_quality[valid]
    y_hold = y_hold[valid]

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
    y_train_q, y_val_q, y_test_q = y_quality[tr_idx], y_quality[va_idx], y_quality[te_idx]
    y_train_h, y_val_h, y_test_h = y_hold[tr_idx], y_hold[va_idx], y_hold[te_idx]

    backend = _resolve_backend()

    data_hash = hash_arrays(x_all.values.astype(np.float32), y_dir.astype(np.int64), y_quality, y_hold)
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
        }
    )
    results_path = training_results_path(args.save_path, "multitarget_lgbm")
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
        best_val_score = float(prev.get("best_val_score", 0.0))
    else:
        meta_params, meta_score = load_best_params_from_meta(meta_path, score_keys=["best_val_score"])
        if meta_params is not None:
            best_params = meta_params
            if "n_estimators" in best_params:
                best_params["n_estimators"] = int(best_params["n_estimators"] * 1.1)
            best_val_score = float(meta_score or 0.0)
            logger.info("reuse best_params from meta json: %s", meta_path)
        else:
            best_params, best_val_score = _tune_params(
                args,
                backend,
                x_train_np,
                x_val_np,
                y_train_dir,
                y_val_dir,
                y_train_q,
                y_val_q,
                y_train_h,
                y_val_h,
            )

    merged = _base_params(args, backend)
    merged.update(best_params)

    x_trainval_np = np.vstack([x_train_np, x_val_np])
    y_trainval_dir = np.hstack([y_train_dir, y_val_dir])
    y_trainval_q = np.hstack([y_train_q, y_val_q])
    y_trainval_h = np.hstack([y_train_h, y_val_h])

    dir_model, quality_model, hold_model = _build_models(backend, args.seed, args.n_jobs, merged)
    dir_model.fit(x_trainval_np, y_trainval_dir)
    quality_model.fit(x_trainval_np, y_trainval_q)
    hold_model.fit(x_trainval_np, y_trainval_h)

    dir_pred = dir_model.predict(x_test_np)
    dir_bal_acc = balanced_accuracy_score(y_test_dir, dir_pred)
    q_pred = quality_model.predict(x_test_np)
    h_pred = hold_model.predict(x_test_np)
    q_mae = float(np.mean(np.abs(q_pred - y_test_q)))
    h_mae = float(np.mean(np.abs(h_pred - y_test_h)))

    logger.info(
        "MultiTarget backend=%s dir_bal_acc=%.4f quality_mae=%.6f hold_mae=%.4f",
        backend,
        dir_bal_acc,
        q_mae,
        h_mae,
    )

    save_pickle(
        {
            "backend": backend,
            "feature_cols": feature_cols,
            "direction_model": dir_model,
            "quality_model": quality_model,
            "hold_model": hold_model,
            "horizon": args.horizon,
        },
        model_path,
    )

    artifact = {
        "feature_cols": feature_cols,
        "horizon": args.horizon,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "multi_target_lgbm",
            "backend": backend,
            "direction_balanced_accuracy": float(dir_bal_acc),
            "quality_mae": q_mae,
            "hold_mae": h_mae,
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
            "test_quality_mae": q_mae,
            "test_hold_mae": h_mae,
            "best_params": merged,
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
    p = argparse.ArgumentParser(description="Train Multi-Target model (dir+quality+hold)")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/multi_target_lgbm.json")
    p.add_argument("--atr-mult", type=float, default=0.8)
    p.add_argument("--max-hold", type=int, default=12)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

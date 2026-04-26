from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_absolute_error

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.selection import auto_select_features
from ensemble.optuna_helper import (
    build_config_hash,
    hash_arrays,
    load_reusable_results,
    save_training_results,
    training_results_path,
)
from ensemble.artifact_utils import load_best_params_from_meta, resolve_model_meta_paths, save_pickle
from features.high_order_state import HIGH_ORDER_STATE_COLS
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


def _require_lightgbm():
    try:
        from lightgbm import LGBMRegressor  # type: ignore
    except ImportError as e:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm") from e
    return LGBMRegressor


def _build_quantile_model(LGBMRegressor, seed: int, n_jobs: int, params: Dict[str, Any], alpha: float):
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _predict_quantiles(models: Dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = models["q10"].predict(x)
    q50 = models["q50"].predict(x)
    q90 = models["q90"].predict(x)
    return q10, q50, q90


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
    }


def _select_ranked_features(
    df,
    feature_cols,
    tr_idx: np.ndarray,
    y_train: np.ndarray,
    max_features: int,
    must_include: list[str] | None = None,
) -> list[str]:
    train_df_tmp = df.iloc[tr_idx].copy()
    train_df_tmp.index = range(len(train_df_tmp))
    train_df_tmp["_target"] = (y_train > 0).astype(np.int64)
    return auto_select_features(
        train_df_tmp,
        feature_cols,
        target_col="_target",
        max_features=max_features,
        corr_threshold=0.85,
        must_include=must_include or [],
    )


def _score_predictions(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    flat_threshold: float,
) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, q50))
    pred_dir = np.where(q50 > flat_threshold, 1, np.where(q50 < -flat_threshold, -1, 0))
    true_dir = np.where(y_true > flat_threshold, 1, np.where(y_true < -flat_threshold, -1, 0))
    dir_acc = float((pred_dir == true_dir).mean())
    interval_width = float(np.mean(np.maximum(q90 - q10, 1e-8)))
    return mae, dir_acc, interval_width


def _tune_params(
    args: argparse.Namespace,
    LGBMRegressor,
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    ranked_features: list[str],
) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        feature_count = trial.suggest_int("feature_count", 16, len(ranked_features))
        selected = ranked_features[:feature_count]
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "feature_count": feature_count,
        }
        flat_threshold = trial.suggest_float("flat_threshold", 1e-4, 2e-3, log=True)

        models = {
            "q10": _build_quantile_model(LGBMRegressor, args.seed, args.n_jobs, params, alpha=0.10),
            "q50": _build_quantile_model(LGBMRegressor, args.seed, args.n_jobs, params, alpha=0.50),
            "q90": _build_quantile_model(LGBMRegressor, args.seed, args.n_jobs, params, alpha=0.90),
        }
        for model in models.values():
            model.fit(x_train[selected], y_train)

        q10, q50, q90 = _predict_quantiles(models, x_val[selected])
        mae, dir_acc, interval_width = _score_predictions(y_val, q10, q50, q90, flat_threshold)
        return float(dir_acc - 30.0 * mae - 5.0 * interval_width)

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
    LGBMRegressor = _require_lightgbm()

    df = load_feature_frame(args.data_path, args.rl_path)
    y_ret = make_future_return(df, horizon=args.horizon)
    valid = np.isfinite(y_ret)

    df = df.loc[valid].reset_index(drop=True)
    y_ret = y_ret[valid]
    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    must_include = [c for c in HIGH_ORDER_STATE_COLS if c in df.columns]
    feature_cols = select_feature_columns(df, must_include=must_include)
    ranked_features = _select_ranked_features(
        df,
        feature_cols,
        tr_idx,
        y_ret[tr_idx],
        args.max_features,
        must_include=must_include,
    )
    x_all = df[ranked_features].replace([np.inf, -np.inf], np.nan)
    x_train = x_all.iloc[tr_idx].copy()
    x_val = x_all.iloc[va_idx].copy()
    x_test = x_all.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)

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
            "feature_count": len(ranked_features),
            "model_family": "lightgbm_quantile",
        }
    )
    results_path = training_results_path(args.save_path, "quantile_forest")
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
            best_params, best_val_score = _tune_params(args, LGBMRegressor, x_train, y_train, x_val, y_val, ranked_features)

    flat_threshold = float(best_params.pop("flat_threshold", args.flat_threshold))

    merged = _base_params(args)
    merged.update(best_params)
    feature_count = int(merged.get("feature_count", len(ranked_features)))
    selected_features = ranked_features[:feature_count]

    x_trainval = x_all.iloc[np.concatenate([tr_idx, va_idx])].copy()
    x_trainval, x_test = median_fill_by_train(x_trainval, x_test)
    y_trainval = np.hstack([y_train, y_val])

    models = {
        "q10": _build_quantile_model(LGBMRegressor, args.seed, args.n_jobs, merged, alpha=0.10),
        "q50": _build_quantile_model(LGBMRegressor, args.seed, args.n_jobs, merged, alpha=0.50),
        "q90": _build_quantile_model(LGBMRegressor, args.seed, args.n_jobs, merged, alpha=0.90),
    }
    for model in models.values():
        model.fit(x_trainval[selected_features], y_trainval)

    q10, q50, q90 = _predict_quantiles(models, x_test[selected_features])
    mae, dir_acc, interval_width = _score_predictions(y_test, q10, q50, q90, flat_threshold)
    logger.info("QuantileLGBM MAE=%.6f dir_acc=%.4f interval_width=%.6f", mae, dir_acc, interval_width)

    save_pickle(
        {
            "models": models,
            "feature_cols": selected_features,
            "horizon": args.horizon,
            "flat_threshold": flat_threshold,
        },
        model_path,
    )
    artifact = {
        "feature_cols": selected_features,
        "horizon": args.horizon,
        "flat_threshold": flat_threshold,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "quantile_lgbm",
            "mae": float(mae),
            "direction_accuracy": dir_acc,
            "interval_width": interval_width,
            "q10_mean": float(np.mean(q10)),
            "q50_mean": float(np.mean(q50)),
            "q90_mean": float(np.mean(q90)),
            "best_val_score": best_val_score,
            "best_params": {**merged, "flat_threshold": flat_threshold},
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=True)

    save_training_results(
        results_path,
        {
            "best_val_score": best_val_score,
            "test_mae": float(mae),
            "test_dir_acc": dir_acc,
            "test_interval_width": interval_width,
            "best_params": {**merged, "flat_threshold": flat_threshold},
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM quantile regressors")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default="data/ensemble/supervised/quantile_forest.pkl")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--reg-alpha", type=float, default=1e-3)
    p.add_argument("--reg-lambda", type=float, default=1e-2)
    p.add_argument("--flat-threshold", type=float, default=0.0005)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--max-features", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--startup-check-only",
        action="store_true",
        help="Validate imports/arguments and exit without training",
    )
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_quantile_forest")
        raise SystemExit(0)
    train(args)

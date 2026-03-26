from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

import numpy as np
from sklearn.ensemble import IsolationForest

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
from ensemble.unsupervised.common import (
    load_unsup_frame,
    zscore_fit_transform,
    ORDERFLOW_FEATURE_HINTS,
    rank_features_by_variance,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "max_samples": args.max_samples,
        "max_features": args.max_features,
    }


def _score_anomaly_quality(score: np.ndarray, pred: np.ndarray, contamination: float) -> float:
    anomaly_ratio = float(np.mean(pred == -1))
    q50 = float(np.quantile(score, 0.50))
    q95 = float(np.quantile(score, 0.95))
    tail_gap = q95 - q50
    return float(tail_gap - 2.0 * abs(anomaly_ratio - contamination))


def _tune_params(args: argparse.Namespace, x_train: np.ndarray, x_val: np.ndarray) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        feature_count = trial.suggest_int("feature_count", args.min_features, x_train.shape[1])
        params = {
            "feature_count": feature_count,
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "contamination": trial.suggest_float("contamination", 0.01, 0.08),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        x_train_sel = x_train[:, :feature_count]
        x_val_sel = x_val[:, :feature_count]

        model = IsolationForest(
            random_state=args.seed,
            n_jobs=args.n_jobs,
            **{k: v for k, v in params.items() if k != "feature_count"},
        )
        model.fit(x_train_sel)
        pred = model.predict(x_val_sel)
        score = -model.decision_function(x_val_sel)
        return _score_anomaly_quality(score, pred, float(params["contamination"]))

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
    df = load_unsup_frame(args.data_path, args.rl_path)
    feature_cols = [c for c in ORDERFLOW_FEATURE_HINTS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.columns if df[c].dtype.kind in ("f", "i")][:20]
    feature_cols = rank_features_by_variance(df, feature_cols)
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x, mean, std = zscore_fit_transform(x)

    n_train = max(10, int(len(x) * args.train_ratio))
    n_train = min(n_train, len(x) - 1)
    x_train = x[:n_train]
    x_val = x[n_train:]
    if len(x_val) < 10:
        x_val = x_train

    data_hash = hash_arrays(x)
    config_hash = build_config_hash(
        {
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "feature_cols": feature_cols,
            "n_jobs": args.n_jobs,
        }
    )
    results_path = training_results_path(args.save_path, "isolation_forest")
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
            best_params, best_val_score = _tune_params(args, x_train, x_val)

    merged = _base_params(args)
    merged.update(best_params)
    feature_count = int(merged.get("feature_count", len(feature_cols)))
    feature_cols = feature_cols[:feature_count]
    x = x[:, :feature_count]
    mean = mean[:feature_count]
    std = std[:feature_count]

    model = IsolationForest(
        random_state=args.seed,
        n_jobs=args.n_jobs,
        **{k: v for k, v in merged.items() if k != "feature_count"},
    )
    model.fit(x)
    pred = model.predict(x)  # -1 anomaly, 1 normal
    score = -model.decision_function(x)
    anomaly_ratio = float(np.mean(pred == -1))
    logger.info("IsolationForest anomaly_ratio=%.4f score_mean=%.6f", anomaly_ratio, float(score.mean()))

    save_pickle(
        {
            "model": model,
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
        },
        model_path,
    )
    artifact = {
        "feature_cols": feature_cols,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "isolation_forest",
            "anomaly_ratio": anomaly_ratio,
            "score_mean": float(score.mean()),
            "score_std": float(score.std()),
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
            "best_params": merged,
            "anomaly_ratio": anomaly_ratio,
            "score_mean": float(score.mean()),
            "score_std": float(score.std()),
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Isolation Forest anomaly detector")
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--save-path", default="data/ensemble/unsupervised/isolation_forest.pkl")
    p.add_argument("--min-features", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--contamination", type=float, default=0.03)
    p.add_argument("--max-samples", type=float, default=1.0)
    p.add_argument("--max-features", type=float, default=1.0)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--n-trials", type=int, default=20)
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
        logger.info("startup check ok: train_isolation_forest")
        raise SystemExit(0)
    train(args)

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

import numpy as np
from joblib import dump as joblib_dump
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
from ensemble.unsupervised.common import (
    load_unsup_frame,
    zscore_fit_transform,
    ORDERFLOW_FEATURE_HINTS,
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
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1400),
            "contamination": trial.suggest_float("contamination", 0.005, 0.10),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }

        model = IsolationForest(
            random_state=args.seed,
            n_jobs=args.n_jobs,
            **params,
        )
        model.fit(x_train)
        pred = model.predict(x_val)
        score = -model.decision_function(x_val)
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
        best_params, best_val_score = _tune_params(args, x_train, x_val)

    merged = _base_params(args)
    merged.update(best_params)

    model = IsolationForest(
        random_state=args.seed,
        n_jobs=args.n_jobs,
        **merged,
    )
    model.fit(x)
    pred = model.predict(x)  # -1 anomaly, 1 normal
    score = -model.decision_function(x)
    anomaly_ratio = float(np.mean(pred == -1))
    logger.info("IsolationForest anomaly_ratio=%.4f score_mean=%.6f", anomaly_ratio, float(score.mean()))

    model_path = args.save_path if args.save_path.lower().endswith(".joblib") else os.path.splitext(args.save_path)[0] + ".joblib"
    meta_path = os.path.splitext(model_path)[0] + ".json"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib_dump(
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
    p.add_argument("--save-path", default="data/ensemble/unsupervised/isolation_forest.joblib")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--contamination", type=float, default=0.03)
    p.add_argument("--max-samples", type=float, default=1.0)
    p.add_argument("--max-features", type=float, default=1.0)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

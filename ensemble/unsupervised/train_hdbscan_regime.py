from __future__ import annotations

import os
import sys
import pickle
import argparse
import logging
from typing import Dict, Any

import numpy as np
from sklearn.metrics import silhouette_score

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
    select_numeric_features,
    zscore_fit_transform,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _require_hdbscan():
    try:
        import hdbscan  # type: ignore
    except ImportError as e:
        raise ImportError("hdbscan is required. Install with: pip install hdbscan") from e
    return hdbscan


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }


def _silhouette_with_noise(x: np.ndarray, labels: np.ndarray) -> float:
    valid = labels != -1
    if valid.sum() < 20:
        return -1.0

    labels_valid = labels[valid]
    if np.unique(labels_valid).size < 2:
        return -1.0

    x_valid = x[valid]
    if len(x_valid) > 4000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_valid), size=4000, replace=False)
        x_valid = x_valid[idx]
        labels_valid = labels_valid[idx]

    try:
        sil = float(silhouette_score(x_valid, labels_valid))
    except Exception:
        return -1.0
    noise_ratio = float(np.mean(labels == -1))
    return float(sil - 0.25 * noise_ratio)


def _tune_params(args: argparse.Namespace, hdbscan, x: np.ndarray) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    max_cluster_size = max(20, min(2000, len(x) // 2))
    min_cluster_size_lb = max(20, min(50, max_cluster_size))

    def objective(trial: "optuna.Trial") -> float:
        min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_lb, max_cluster_size)
        min_samples = trial.suggest_int("min_samples", 5, 200)
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        labels = model.fit_predict(x)
        return _silhouette_with_noise(x, labels)

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)
    return dict(study.best_params), float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    hdbscan = _require_hdbscan()

    df = load_unsup_frame(args.data_path, args.rl_path)
    feature_cols = select_numeric_features(df, min_features=args.min_features)
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x, mean, std = zscore_fit_transform(x)

    data_hash = hash_arrays(x)
    config_hash = build_config_hash(
        {
            "min_features": args.min_features,
            "seed": args.seed,
            "feature_cols": feature_cols,
        }
    )
    results_path = training_results_path(args.save_path, "hdbscan_regime")

    prev = load_reusable_results(
        results_path=results_path,
        data_hash=data_hash,
        config_hash=config_hash,
        force_reuse_results=args.force_reuse_results,
        logger=logger,
    )

    if prev is not None:
        best_params = dict(prev.get("best_params", {}))
        best_val_score = float(prev.get("best_val_score", 0.0))
    else:
        best_params, best_val_score = _tune_params(args, hdbscan, x)

    merged = _base_params(args)
    merged.update(best_params)

    model = hdbscan.HDBSCAN(
        min_cluster_size=int(merged["min_cluster_size"]),
        min_samples=int(merged["min_samples"]),
        prediction_data=True,
    )
    labels = model.fit_predict(x)
    probs = getattr(model, "probabilities_", np.zeros(len(labels), dtype=np.float32))

    unique, counts = np.unique(labels, return_counts=True)
    noise_ratio = float(np.mean(labels == -1))
    logger.info("HDBSCAN clusters=%s noise_ratio=%.4f", dict(zip(unique.tolist(), counts.tolist())), noise_ratio)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
        "meta": {
            "algorithm": "hdbscan_regime",
            "noise_ratio": noise_ratio,
            "avg_membership": float(np.mean(probs)),
            "best_val_score": best_val_score,
            "best_params": merged,
        },
    }
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(artifact, f)

    save_training_results(
        results_path,
        {
            "best_val_score": best_val_score,
            "best_params": merged,
            "noise_ratio": noise_ratio,
            "avg_membership": float(np.mean(probs)),
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved: %s", args.save_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HDBSCAN regime clustering")
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--save-path", default="data/ensemble/unsupervised/hdbscan_regime.pkl")
    p.add_argument("--min-features", type=int, default=20)
    p.add_argument("--min-cluster-size", type=int, default=300)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

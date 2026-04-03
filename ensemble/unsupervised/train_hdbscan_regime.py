from __future__ import annotations

import os
import sys
import json
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
from ensemble.artifact_utils import load_best_params_from_meta, resolve_model_meta_paths, save_pickle
from ensemble.unsupervised.common import (
    load_unsup_frame,
    select_numeric_features,
    rank_features_by_variance,
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
        "feature_count": args.max_features,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }


def _cluster_quality_score(x: np.ndarray, labels: np.ndarray) -> float:
    total = max(1, len(labels))
    valid = labels != -1
    valid_count = int(valid.sum())
    coverage = float(valid_count / total)
    noise_ratio = 1.0 - coverage

    unique_valid, counts_valid = np.unique(labels[valid], return_counts=True) if valid_count > 0 else (np.array([]), np.array([]))
    n_clusters = int(len(unique_valid))

    # Keep a continuous objective even in degenerate cases.
    if valid_count < 30 or n_clusters < 2:
        return float(-0.95 + 0.25 * coverage + 0.03 * min(n_clusters, 3))

    x_valid = x[valid]
    labels_valid = labels[valid]
    if len(x_valid) > 4000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_valid), size=4000, replace=False)
        x_valid = x_valid[idx]
        labels_valid = labels_valid[idx]

    try:
        sil = float(silhouette_score(x_valid, labels_valid))
    except Exception:
        sil = -0.2

    size_cv = float(np.std(counts_valid) / (np.mean(counts_valid) + 1e-8))
    size_penalty = min(size_cv, 2.0) / 2.0
    cluster_bonus = min(n_clusters, 8) / 8.0

    score = (
        0.65 * sil
        + 0.20 * coverage
        + 0.15 * cluster_bonus
        - 0.20 * noise_ratio
        - 0.10 * size_penalty
    )
    return float(score)


def _silhouette_with_noise(x: np.ndarray, labels: np.ndarray) -> float:
    # Backward-compatible metric retained for logging if needed.
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

    if len(x) > args.tune_max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(x), size=args.tune_max_samples, replace=False)
        x_tune = x[idx]
    else:
        x_tune = x

    logger.info("HDBSCAN tuning sample size: %d/%d", len(x_tune), len(x))

    max_feature_count = max(args.min_features, min(args.max_features, x_tune.shape[1]))
    max_cluster_size = max(20, min(600, len(x_tune) // 8))
    min_cluster_size_lb = max(20, min(50, max_cluster_size))

    def objective(trial: "optuna.Trial") -> float:
        feature_count = trial.suggest_int("feature_count", args.min_features, max_feature_count)
        min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_lb, max_cluster_size)
        min_samples_ub = max(6, min(40, min_cluster_size - 1))
        min_samples = trial.suggest_int("min_samples", 3, min_samples_ub)
        x_trial = x_tune[:, :feature_count]
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=False,
        )
        labels = model.fit_predict(x_trial)
        return _cluster_quality_score(x_trial, labels)

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
    feature_cols = rank_features_by_variance(df, select_numeric_features(df, min_features=args.min_features))
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
        best_val_score = float(prev.get("best_val_score", 0.0))
        if best_val_score <= args.retune_score_threshold:
            logger.warning(
                "previous best_val_score=%.4f <= %.4f, running Optuna again",
                best_val_score,
                args.retune_score_threshold,
            )
            best_params, best_val_score = _tune_params(args, hdbscan, x)
    else:
        meta_params, meta_score = load_best_params_from_meta(meta_path, score_keys=["best_val_score"])
        if meta_params is not None:
            best_params = meta_params
            best_val_score = float(meta_score or 0.0)
            if best_val_score <= args.retune_score_threshold:
                logger.warning(
                    "meta best_val_score=%.4f <= %.4f, running Optuna again",
                    best_val_score,
                    args.retune_score_threshold,
                )
                best_params, best_val_score = _tune_params(args, hdbscan, x)
            else:
                logger.info("reuse best_params from meta json: %s", meta_path)
        else:
            best_params, best_val_score = _tune_params(args, hdbscan, x)

    merged = _base_params(args)
    merged.update(best_params)
    feature_count = int(merged.get("feature_count", min(args.max_features, len(feature_cols))))
    feature_count = max(args.min_features, min(feature_count, len(feature_cols)))
    feature_cols = feature_cols[:feature_count]
    x = x[:, :feature_count]
    mean = mean[:feature_count]
    std = std[:feature_count]

    if len(x) > args.final_fit_max_samples:
        rng = np.random.default_rng(args.seed)
        fit_idx = np.sort(rng.choice(len(x), size=args.final_fit_max_samples, replace=False))
        x_fit = x[fit_idx]
    else:
        fit_idx = np.arange(len(x))
        x_fit = x

    logger.info("HDBSCAN final fit sample size: %d/%d", len(x_fit), len(x))

    model = hdbscan.HDBSCAN(
        min_cluster_size=int(merged["min_cluster_size"]),
        min_samples=int(merged["min_samples"]),
        prediction_data=True,
    )
    model.fit(x_fit)

    if len(x_fit) == len(x):
        labels = model.labels_
        probs = getattr(model, "probabilities_", np.zeros(len(labels), dtype=np.float32))
    else:
        try:
            labels, probs = hdbscan.approximate_predict(model, x)
            probs = probs.astype(np.float32)
        except Exception:
            labels = np.full(len(x), -1, dtype=np.int64)
            labels[fit_idx] = model.labels_
            probs = np.zeros(len(x), dtype=np.float32)
            model_probs = getattr(model, "probabilities_", np.zeros(len(fit_idx), dtype=np.float32))
            probs[fit_idx] = model_probs.astype(np.float32)

    unique, counts = np.unique(labels, return_counts=True)
    noise_ratio = float(np.mean(labels == -1))
    quality_score = _cluster_quality_score(x, labels)
    logger.info("HDBSCAN clusters=%s noise_ratio=%.4f", dict(zip(unique.tolist(), counts.tolist())), noise_ratio)

    save_pickle(
        {
            "model": model,
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
        },
        model_path,
    )
    artifact = {
        "feature_cols": feature_cols,
        "model_path": os.path.basename(model_path),
        "meta": {
            "algorithm": "hdbscan_regime",
            "noise_ratio": noise_ratio,
            "avg_membership": float(np.mean(probs)),
            "quality_score": quality_score,
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
            "noise_ratio": noise_ratio,
            "avg_membership": float(np.mean(probs)),
            "quality_score": quality_score,
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HDBSCAN regime clustering")
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--save-path", default="data/ensemble/unsupervised/hdbscan_regime.pkl")
    p.add_argument("--min-features", type=int, default=20)
    p.add_argument("--max-features", type=int, default=48)
    p.add_argument("--min-cluster-size", type=int, default=300)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--tune-max-samples", type=int, default=12000)
    p.add_argument("--final-fit-max-samples", type=int, default=80000)
    p.add_argument("--retune-score-threshold", type=float, default=-0.90)
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
        logger.info("startup check ok: train_hdbscan_regime")
        raise SystemExit(0)
    train(args)

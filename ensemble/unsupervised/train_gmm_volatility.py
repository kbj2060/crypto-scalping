from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List

import numpy as np
from sklearn.mixture import GaussianMixture
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
    zscore_fit_transform,
    VOLATILITY_FEATURE_HINTS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _pick_vol_features(df) -> List[str]:
    cols = [c for c in VOLATILITY_FEATURE_HINTS if c in df.columns]
    if cols:
        return cols
    return [c for c in df.columns if df[c].dtype.kind in ("f", "i")][:12]


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
        "reg_covar": args.reg_covar,
    }


def _silhouette_safe(x: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(x):
        return -1.0
    try:
        if len(x) > 4000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x), size=4000, replace=False)
            return float(silhouette_score(x[idx], labels[idx]))
        return float(silhouette_score(x, labels))
    except Exception:
        return -1.0


def _tune_params(args: argparse.Namespace, x_train: np.ndarray, x_val: np.ndarray) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_components": trial.suggest_int("n_components", 2, 10),
            "covariance_type": trial.suggest_categorical("covariance_type", ["full", "diag", "tied", "spherical"]),
            "reg_covar": trial.suggest_float("reg_covar", 1e-8, 1e-2, log=True),
        }
        model = GaussianMixture(
            random_state=args.seed,
            **params,
        )
        model.fit(x_train)
        val_ll = float(model.score(x_val))
        labels_val = model.predict(x_val)
        sil = _silhouette_safe(x_val, labels_val)
        return float(val_ll + 0.2 * sil)

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)
    return dict(study.best_params), float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    df = load_unsup_frame(args.data_path, args.rl_path)
    feature_cols = _pick_vol_features(df)
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
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "feature_cols": feature_cols,
        }
    )
    results_path = training_results_path(args.save_path, "gmm_volatility")

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
        best_params, best_val_score = _tune_params(args, x_train, x_val)

    merged = _base_params(args)
    merged.update(best_params)

    model = GaussianMixture(
        random_state=args.seed,
        **merged,
    )
    model.fit(x)
    labels = model.predict(x)
    probs = model.predict_proba(x)

    # rank clusters by volatility proxy: first feature after standardization
    vol_proxy = x[:, 0]
    cluster_score = {}
    for k in range(int(merged["n_components"])):
        m = labels == k
        cluster_score[k] = float(np.mean(vol_proxy[m])) if np.any(m) else -999.0
    ranked = sorted(cluster_score.keys(), key=lambda k: cluster_score[k])
    cluster_rank_map = {int(k): i for i, k in enumerate(ranked)}

    logger.info("GMM cluster_score=%s", cluster_score)
    logger.info("GMM cluster_rank_map(low->high vol)=%s", cluster_rank_map)

    model_path = args.save_path if args.save_path.lower().endswith(".npz") else os.path.splitext(args.save_path)[0] + ".npz"
    meta_path = os.path.splitext(model_path)[0] + ".json"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    cluster_keys = np.asarray(list(cluster_rank_map.keys()), dtype=np.int64)
    cluster_vals = np.asarray(list(cluster_rank_map.values()), dtype=np.int64)
    np.savez_compressed(
        model_path,
        weights=model.weights_,
        means=model.means_,
        covariances=model.covariances_,
        precisions_cholesky=model.precisions_cholesky_,
        mean=mean,
        std=std,
        cluster_rank_keys=cluster_keys,
        cluster_rank_vals=cluster_vals,
    )
    artifact = {
        "feature_cols": feature_cols,
        "model_path": os.path.basename(model_path),
        "cluster_rank_map": cluster_rank_map,
        "meta": {
            "algorithm": "gmm_volatility_regime",
            "weights": model.weights_.tolist(),
            "avg_confidence": float(np.max(probs, axis=1).mean()),
            "best_val_score": best_val_score,
            "best_params": merged,
            "covariance_type": merged["covariance_type"],
            "n_components": int(merged["n_components"]),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=True)

    save_training_results(
        results_path,
        {
            "best_val_score": best_val_score,
            "best_params": merged,
            "avg_confidence": float(np.max(probs, axis=1).mean()),
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GMM volatility regime model")
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--save-path", default="data/ensemble/unsupervised/gmm_volatility.npz")
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--covariance-type", default="full", choices=["full", "diag", "tied", "spherical"])
    p.add_argument("--reg-covar", type=float, default=1e-5)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

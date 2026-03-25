from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
    zscore_fit_transform,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _maybe_fit_umap(x: np.ndarray, n_components: int, random_state: int, n_neighbors: int, min_dist: float):
    try:
        import umap  # type: ignore
    except Exception:
        return None, None
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    emb = reducer.fit_transform(x)
    return reducer, emb


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "pca_components": args.pca_components,
        "n_clusters": args.n_clusters,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
    }


def _silhouette_safe(z: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if uniq.size < 2 or uniq.size >= len(z):
        return -1.0
    try:
        if len(z) > 4000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(z), size=4000, replace=False)
            return float(silhouette_score(z[idx], labels[idx]))
        return float(silhouette_score(z, labels))
    except Exception:
        return -1.0


def _tune_params(args: argparse.Namespace, x_train: np.ndarray, x_val: np.ndarray) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    max_pca_components = min(int(args.max_pca_components), x_train.shape[1])
    max_pca_components = max(2, max_pca_components)
    max_clusters = max(2, min(int(args.max_clusters), len(x_train) - 1))
    min_clusters = min(3, max_clusters)

    def objective(trial: "optuna.Trial") -> float:
        pca_components = trial.suggest_int("pca_components", 2, max_pca_components)
        n_clusters = trial.suggest_int("n_clusters", min_clusters, max_clusters)

        pca = PCA(n_components=pca_components, random_state=args.seed)
        z_train = pca.fit_transform(x_train)
        z_val = pca.transform(x_val)

        km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=20)
        km.fit(z_train)
        labels_val = km.predict(z_val)

        return _silhouette_safe(z_val, labels_val)

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)
    return dict(study.best_params), float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    df = load_unsup_frame(args.data_path, args.rl_path)
    feature_cols = select_numeric_features(df, min_features=args.min_features)
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x, mean, std = zscore_fit_transform(x_raw)

    n_train = max(10, int(len(x) * args.train_ratio))
    n_train = min(n_train, len(x) - 1)
    x_train = x[:n_train]
    x_val = x[n_train:]
    if len(x_val) < 10:
        x_val = x_train

    data_hash = hash_arrays(x)
    config_hash = build_config_hash(
        {
            "min_features": args.min_features,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "feature_cols": feature_cols,
            "fit_umap": args.fit_umap,
            "umap_components": args.umap_components,
            "horizon": args.horizon,
        }
    )
    results_path = training_results_path(args.save_path, "pca_umap_mapper")
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
    else:
        meta_params, meta_score = load_best_params_from_meta(meta_path, score_keys=["best_val_score"])
        if meta_params is not None:
            best_params = meta_params
            best_val_score = float(meta_score or 0.0)
            logger.info("reuse best_params from meta json: %s", meta_path)
        else:
            best_params, best_val_score = _tune_params(args, x_train, x_val)

    merged = _base_params(args)
    merged.update(best_params)

    pca = PCA(n_components=int(merged["pca_components"]), random_state=args.seed)
    z_pca = pca.fit_transform(x)

    km = KMeans(n_clusters=int(merged["n_clusters"]), random_state=args.seed, n_init=20)
    labels = km.fit_predict(z_pca)

    reducer, z_umap = _maybe_fit_umap(
        x,
        args.umap_components,
        args.seed,
        int(merged["umap_n_neighbors"]),
        float(merged["umap_min_dist"]),
    ) if args.fit_umap else (None, None)

    cluster_stats: Dict[int, Dict[str, float]] = {}
    if "close" in df.columns and len(df) > args.horizon:
        close = df["close"].values.astype(np.float64)
        fwd = np.full(len(close), np.nan)
        for i in range(len(close) - args.horizon):
            fwd[i] = close[i + args.horizon] / max(close[i], 1e-8) - 1.0
        for c in range(int(merged["n_clusters"])):
            m = (labels == c) & np.isfinite(fwd)
            if np.any(m):
                cluster_stats[c] = {
                    "count": float(np.sum(m)),
                    "fwd_ret_mean": float(np.mean(fwd[m])),
                    "fwd_ret_std": float(np.std(fwd[m])),
                }
    logger.info("PCA explained_variance_ratio=%s", np.round(pca.explained_variance_ratio_, 4).tolist())
    logger.info("cluster_stats=%s", cluster_stats)

    save_pickle(
        {
            "pca": pca,
            "kmeans": km,
            "umap": reducer,
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
            "cluster_stats": cluster_stats,
        },
        model_path,
    )
    artifact = {
        "feature_cols": feature_cols,
        "model_path": os.path.basename(model_path),
        "cluster_stats": cluster_stats,
        "meta": {
            "algorithm": "pca_umap_mapper",
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "fit_umap": bool(reducer is not None),
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
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "fit_umap": bool(reducer is not None),
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved model: %s", model_path)
    logger.info("saved meta: %s", meta_path)
    logger.info("saved: %s", results_path)
    return artifact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PCA/UMAP regime mapper")
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--save-path", default="data/ensemble/unsupervised/pca_umap_mapper.pkl")
    p.add_argument("--min-features", type=int, default=24)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--pca-components", type=int, default=3)
    p.add_argument("--max-pca-components", type=int, default=12)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--max-clusters", type=int, default=16)
    p.add_argument("--fit-umap", action="store_true")
    p.add_argument("--umap-components", type=int, default=2)
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--force-reuse-results", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

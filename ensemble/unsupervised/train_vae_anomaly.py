from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
import logging
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
from ensemble.artifact_utils import load_best_params_from_meta, resolve_model_meta_paths
from ensemble.unsupervised.common import (
    load_unsup_frame,
    select_numeric_features,
    rank_features_by_variance,
    zscore_fit_transform,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar


def _loss_fn(x, recon, mu, logvar, beta: float = 0.01):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _train_model(
    model: VAE,
    x_train: np.ndarray,
    device: str,
    learning_rate: float,
    beta: float,
    batch_size: int,
    epochs: int,
    log_prefix: str = "",
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=max(32, min(batch_size, len(x_train))),
        shuffle=True,
    )

    model.train()
    for ep in range(1, epochs + 1):
        losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)
            loss, _, _ = _loss_fn(xb, recon, mu, logvar, beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        if epochs >= 10 and ep % max(1, epochs // 10) == 0:
            logger.info("%sep=%04d loss=%.6f", log_prefix, ep, float(np.mean(losses)))


def _reconstruction_error(model: VAE, x: np.ndarray, device: str) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        xr, _, _ = model(xt)
        return torch.mean((xr - xt) ** 2, dim=1).cpu().numpy()


def _base_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "threshold_q": args.threshold_q,
    }


def _tune_params(
    args: argparse.Namespace,
    device: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[Dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    batch_candidates = [256, 512, 1024, 2048]
    batch_candidates = [b for b in batch_candidates if b <= max(256, len(x_train))]
    if not batch_candidates:
        batch_candidates = [min(256, max(32, len(x_train)))]

    tune_epochs = max(8, int(args.tune_epochs))

    def objective(trial: "optuna.Trial") -> float:
        feature_count = trial.suggest_int("feature_count", args.min_features, x_train.shape[1])
        params = {
            "feature_count": feature_count,
            "latent_dim": trial.suggest_int("latent_dim", 6, 24, step=2),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 96, 128, 192]),
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 4e-3, log=True),
            "beta": trial.suggest_float("beta", 2e-3, 0.05, log=True),
            "batch_size": trial.suggest_categorical("batch_size", batch_candidates),
            "threshold_q": trial.suggest_float("threshold_q", 0.92, 0.99),
        }

        x_train_sel = x_train[:, :feature_count]
        x_val_sel = x_val[:, :feature_count]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = VAE(input_dim=feature_count, latent_dim=int(params["latent_dim"]), hidden_dim=int(params["hidden_dim"])).to(device)
        _train_model(
            model=model,
            x_train=x_train_sel,
            device=device,
            learning_rate=float(params["learning_rate"]),
            beta=float(params["beta"]),
            batch_size=int(params["batch_size"]),
            epochs=tune_epochs,
        )

        train_err = _reconstruction_error(model, x_train_sel, device)
        val_err = _reconstruction_error(model, x_val_sel, device)
        threshold = float(np.quantile(train_err, float(params["threshold_q"])))
        val_anom_ratio = float(np.mean(val_err > threshold))

        # 낮은 재구성 오차 + 과도한 anomaly ratio 억제
        val_mse = float(np.mean(val_err))
        penalty = abs(val_anom_ratio - args.target_anomaly_ratio)
        return float(-(val_mse + 0.2 * penalty))

    logger.info("Optuna tuning start: n_trials=%d", args.n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=True)
    return dict(study.best_params), float(study.best_value)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_unsup_frame(args.data_path, args.rl_path)
    feature_cols = rank_features_by_variance(df, select_numeric_features(df, min_features=args.min_features))
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    x, mean, std = zscore_fit_transform(x_raw)

    n = len(x)
    n_train = max(10, int(n * args.train_ratio))
    n_train = min(n_train, n - 1)
    x_train = x[:n_train]
    x_val = x[n_train:]
    if len(x_val) < 10:
        x_val = x_train

    device = _resolve_device(args.device)
    logger.info("device=%s", device)

    data_hash = hash_arrays(x)
    config_hash = build_config_hash(
        {
            "min_features": args.min_features,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "device_request": args.device,
            "feature_cols": feature_cols,
            "target_anomaly_ratio": args.target_anomaly_ratio,
        }
    )
    results_path = training_results_path(args.save_path, "vae_anomaly")
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
        if "epochs" in best_params:
            best_params["epochs"] = int(best_params["epochs"] * 1.1)
        best_val_score = float(prev.get("best_val_score", 0.0))
    else:
        meta_params, meta_score = load_best_params_from_meta(meta_path, score_keys=["best_val_score"])
        if meta_params is not None:
            best_params = meta_params
            if "epochs" in best_params:
                best_params["epochs"] = int(best_params["epochs"] * 1.1)
            best_val_score = float(meta_score or 0.0)
            logger.info("reuse best_params from meta json: %s", meta_path)
        else:
            best_params, best_val_score = _tune_params(args, device, x_train, x_val)

    merged = _base_params(args)
    merged.update(best_params)
    feature_count = int(merged.get("feature_count", len(feature_cols)))
    feature_count = max(args.min_features, min(feature_count, len(feature_cols)))
    feature_cols = feature_cols[:feature_count]
    x = x[:, :feature_count]
    mean = mean[:feature_count]
    std = std[:feature_count]
    x_train_sel = x_train[:, :feature_count]
    x_val_sel = x_val[:, :feature_count]

    model = VAE(
        input_dim=feature_count,
        latent_dim=int(merged["latent_dim"]),
        hidden_dim=int(merged["hidden_dim"]),
    ).to(device)

    _train_model(
        model=model,
        x_train=x_train_sel,
        device=device,
        learning_rate=float(merged["learning_rate"]),
        beta=float(merged["beta"]),
        batch_size=int(merged["batch_size"]),
        epochs=int(merged["epochs"]),
        log_prefix="final/",
    )

    train_err = _reconstruction_error(model, x_train_sel, device)
    val_err = _reconstruction_error(model, x_val_sel, device)

    threshold = float(np.quantile(train_err, float(merged["threshold_q"])))
    val_anomaly_ratio = float(np.mean(val_err > threshold))
    logger.info("vae threshold=%.6f val_anomaly_ratio=%.4f", threshold, val_anomaly_ratio)

    payload = {
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "meta": {
            "algorithm": "vae_anomaly",
            "latent_dim": int(merged["latent_dim"]),
            "hidden_dim": int(merged["hidden_dim"]),
            "val_anomaly_ratio": val_anomaly_ratio,
            "device": device,
            "best_val_score": best_val_score,
            "best_params": merged,
        },
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "model_path": os.path.basename(model_path),
                "meta": payload["meta"] | {"threshold": threshold},
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    save_training_results(
        results_path,
        {
            "best_val_score": best_val_score,
            "best_params": merged,
            "threshold": threshold,
            "val_anomaly_ratio": val_anomaly_ratio,
            "data_hash": data_hash,
            "config_hash": config_hash,
        },
    )
    logger.info("saved: %s", model_path)
    logger.info("saved: %s", meta_path)
    logger.info("saved: %s", results_path)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAE anomaly detector")
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--save-path", default="data/ensemble/unsupervised/vae_anomaly.pkl")
    p.add_argument("--min-features", type=int, default=24)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--latent-dim", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--tune-epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--threshold-q", type=float, default=0.98)
    p.add_argument("--target-anomaly-ratio", type=float, default=0.08)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--device", default="auto")
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
        logger.info("startup check ok: train_vae_anomaly")
        raise SystemExit(0)
    train(args)

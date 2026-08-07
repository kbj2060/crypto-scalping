#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTForClassification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_ai_patchmix_direction_core_20260530 as patchmix  # noqa: E402


MODEL_ID = "ai_patchtst_tradeable_20260530"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PatchTST tradeable short/long representations and compare heads.")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--label-config", choices=("dense", "base", "fee2", "high_quality"), default="fee2")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--patch-length", type=int, default=16)
    p.add_argument("--patch-stride", type=int, default=8)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mlp-epochs", type=int, default=18)
    p.add_argument("--cat-iterations", type=int, default=700)
    p.add_argument("--cat-depth", type=int, default=5)
    p.add_argument("--cat-lr", type=float, default=0.025)
    p.add_argument("--cat-l2", type=float, default=12.0)
    p.add_argument("--random-seed", type=int, default=20260530)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _label_cfg(name: str) -> dict[str, float | str]:
    cfgs: dict[str, dict[str, float | str]] = {
        "dense": {"name": "tradeable_dense", "min_edge": 0.0010, "atr_mult": 0.16, "mae_penalty": 0.40, "cost": 0.00055, "margin": 0.00015},
        "base": {"name": "tradeable_base", "min_edge": 0.0012, "atr_mult": 0.22, "mae_penalty": 0.55, "cost": 0.00065, "margin": 0.00025},
        "fee2": {"name": "tradeable_fee2", "min_edge": 0.0016, "atr_mult": 0.26, "mae_penalty": 0.65, "cost": 0.00085, "margin": 0.00035},
        "high_quality": {"name": "tradeable_high_quality", "min_edge": 0.0020, "atr_mult": 0.30, "mae_penalty": 0.75, "cost": 0.00100, "margin": 0.00045},
    }
    return cfgs[name]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        raise KeyError(col)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _future_extreme(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    future = s.shift(-1)
    if mode == "max":
        return future[::-1].rolling(horizon, min_periods=1).max()[::-1]
    if mode == "min":
        return future[::-1].rolling(horizon, min_periods=1).min()[::-1]
    raise ValueError(mode)


def _binary_labels(frame: pd.DataFrame, *, horizon: int, cfg: dict[str, float | str]) -> pd.DataFrame:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=3).mean() / close).fillna(0.001)
    floor = np.maximum(float(cfg["min_edge"]), atr_pct.to_numpy(dtype=np.float64) * float(cfg["atr_mult"]))
    fut_high = _future_extreme(high, horizon, "max")
    fut_low = _future_extreme(low, horizon, "min")
    long_mfe = (fut_high / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    long_mae = (1.0 - fut_low / close).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    short_mfe = long_mae.copy()
    short_mae = long_mfe.copy()
    long_score = long_mfe - float(cfg["mae_penalty"]) * long_mae - float(cfg["cost"])
    short_score = short_mfe - float(cfg["mae_penalty"]) * short_mae - float(cfg["cost"])
    label = np.full(len(frame), -1, dtype=np.int64)
    label[(short_score - long_score > float(cfg["margin"])) & (short_score > floor)] = 0
    label[(long_score - short_score > float(cfg["margin"])) & (long_score > floor)] = 1
    valid = label >= 0
    valid[-int(horizon) :] = False
    return pd.DataFrame({"label": label, "valid": valid.astype(np.int8)}, index=frame.index)


def _set_profile() -> None:
    patchmix.CORE_FEATURES = (
        *patchmix.BASE_CORE_FEATURES,
        *patchmix.AUDITED_COMPACT_FEATURES,
        *patchmix.LOCAL_REGIME_FEATURES,
    )


def _read_and_channels(path: Path, limit: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = patchmix._read_frame(path, int(limit))
    core = patchmix._core_features(frame)
    channels = patchmix._patch_channels(core).to_numpy(dtype=np.float32)
    return frame, channels


class SeqDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, values: np.ndarray, indices: np.ndarray, labels: np.ndarray, *, context_length: int, mean: np.ndarray, std: np.ndarray):
        self.values = values
        self.indices = indices.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.context_length = int(context_length)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, j: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[j])
        x = self.values[idx - self.context_length : idx]
        x = (x - self.mean) / self.std
        return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(int(self.labels[j]), dtype=torch.long)


def _valid_indices(labels: pd.DataFrame, context_length: int) -> np.ndarray:
    mask = (labels["valid"].to_numpy(dtype=np.int8) > 0)
    idx = np.flatnonzero(mask)
    return idx[idx >= int(context_length)]


def _class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.maximum(np.bincount(y.astype(int), minlength=2).astype(np.float64), 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _metrics(y: np.ndarray, p_long: np.ndarray) -> dict[str, Any]:
    pred = (p_long >= 0.5).astype(np.int64)
    out: dict[str, Any] = {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "pred_counts": np.bincount(pred, minlength=2).astype(int).tolist(),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist(),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p_long))
    except Exception:
        out["auc"] = None
    return out


def _make_model(args: argparse.Namespace, n_channels: int) -> PatchTSTForClassification:
    cfg = PatchTSTConfig(
        context_length=int(args.context_length),
        patch_length=int(args.patch_length),
        patch_stride=int(args.patch_stride),
        num_input_channels=int(n_channels),
        num_targets=2,
        d_model=int(args.d_model),
        num_hidden_layers=int(args.layers),
        num_attention_heads=int(args.heads),
        ffn_dim=int(args.ffn_dim),
        pooling_type="mean",
        norm_type="batchnorm",
        use_cls_token=False,
    )
    return PatchTSTForClassification(cfg)


def _predict_model(model: PatchTSTForClassification, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(past_values=xb, return_dict=True).prediction_logits
            p = torch.softmax(logits, dim=1)[:, 1]
            ys.append(yb.numpy())
            ps.append(p.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _extract_embeddings(model: PatchTSTForClassification, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            hidden = model.model(past_values=xb, return_dict=True).last_hidden_state
            emb = hidden.mean(dim=(1, 2))
            ys.append(yb.numpy())
            embs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(ys), np.concatenate(embs)


class SmallMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Dropout(0.05), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_mlp(x_fit: np.ndarray, y_fit: np.ndarray, x_hold: np.ndarray, y_hold: np.ndarray, args: argparse.Namespace, device: torch.device) -> SmallMLP:
    model = SmallMLP(x_fit.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    weights = _class_weights(y_fit, device)
    xt = torch.as_tensor(x_fit, dtype=torch.float32)
    yt = torch.as_tensor(y_fit, dtype=torch.long)
    best_state = None
    best_bacc = -1.0
    rng = np.random.default_rng(int(args.random_seed) + 17)
    for _ in range(int(args.mlp_epochs)):
        model.train()
        order = rng.permutation(len(yt))
        for start in range(0, len(order), int(args.batch_size)):
            idx = order[start : start + int(args.batch_size)]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(torch.as_tensor(x_hold, dtype=torch.float32, device=device)), dim=1)[:, 1].detach().cpu().numpy()
        bacc = balanced_accuracy_score(y_hold, (p >= 0.5).astype(np.int64))
        if bacc > best_bacc:
            best_bacc = float(bacc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _mlp_predict(model: SmallMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.as_tensor(x[start : start + 4096], dtype=torch.float32, device=device)
            outs.append(torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(outs)


def _fit_catboost(x_fit: np.ndarray, y_fit: np.ndarray, x_hold: np.ndarray, y_hold: np.ndarray, args: argparse.Namespace) -> tuple[CatBoostClassifier, str]:
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(args.cat_iterations),
        learning_rate=float(args.cat_lr),
        depth=int(args.cat_depth),
        l2_leaf_reg=float(args.cat_l2),
        random_seed=int(args.random_seed) + 301,
        task_type=str(args.task_type),
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
    )
    model = CatBoostClassifier(**params)
    used = str(args.task_type)
    try:
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
    except Exception:
        if str(args.task_type) != "GPU":
            raise
        params["task_type"] = "CPU"
        used = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
    return model, used


def main() -> int:
    args = parse_args()
    _seed(int(args.random_seed))
    _set_profile()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.task_type) == "GPU" else "cpu")
    train, train_values = _read_and_channels(args.train_csv, int(args.limit))
    score, score_values = _read_and_channels(args.score_csv, int(args.limit))
    cfg = _label_cfg(str(args.label_config))
    train_labels = _binary_labels(train, horizon=int(args.horizon), cfg=cfg)
    score_labels = _binary_labels(score, horizon=int(args.horizon), cfg=cfg)
    train_idx_all = _valid_indices(train_labels, int(args.context_length))
    score_idx = _valid_indices(score_labels, int(args.context_length))
    split = int(len(train_idx_all) * 0.82)
    fit_idx = train_idx_all[:split]
    hold_idx = train_idx_all[split:]
    y_train_all = train_labels["label"].to_numpy(dtype=np.int64)
    y_score_all = score_labels["label"].to_numpy(dtype=np.int64)
    mean = train_values[fit_idx].mean(axis=0, keepdims=True)
    std = train_values[fit_idx].std(axis=0, keepdims=True) + 1e-6
    fit_ds = SeqDataset(train_values, fit_idx, y_train_all[fit_idx], context_length=int(args.context_length), mean=mean, std=std)
    hold_ds = SeqDataset(train_values, hold_idx, y_train_all[hold_idx], context_length=int(args.context_length), mean=mean, std=std)
    score_ds = SeqDataset(score_values, score_idx, y_score_all[score_idx], context_length=int(args.context_length), mean=mean, std=std)
    fit_loader = DataLoader(fit_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    hold_loader = DataLoader(hold_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    score_loader = DataLoader(score_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = _make_model(args, train_values.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    weights = _class_weights(y_train_all[fit_idx], device)
    best_state = None
    best_hold = -1.0
    history: list[dict[str, float]] = []
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in fit_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(past_values=xb, return_dict=True).prediction_logits
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        yh, ph = _predict_model(model, hold_loader, device)
        hold_bacc = float(balanced_accuracy_score(yh, (ph >= 0.5).astype(np.int64)))
        history.append({"epoch": float(ep), "loss": float(np.mean(losses)), "hold_bacc": hold_bacc})
        print(json.dumps(history[-1]), flush=True)
        if hold_bacc > best_hold:
            best_hold = hold_bacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    y_fit, emb_fit = _extract_embeddings(model, DataLoader(fit_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0), device)
    y_hold, emb_hold = _extract_embeddings(model, hold_loader, device)
    y_score, emb_score = _extract_embeddings(model, score_loader, device)
    yh, ph = _predict_model(model, hold_loader, device)
    ys, ps = _predict_model(model, score_loader, device)

    mlp = _fit_mlp(emb_fit, y_fit, emb_hold, y_hold, args, device)
    mlp_hold = _mlp_predict(mlp, emb_hold, device)
    mlp_score = _mlp_predict(mlp, emb_score, device)
    cat, cat_task = _fit_catboost(emb_fit, y_fit, emb_hold, y_hold, args)
    cat_hold = np.asarray(cat.predict_proba(emb_hold), dtype=np.float64)[:, 1]
    cat_score = np.asarray(cat.predict_proba(emb_score), dtype=np.float64)[:, 1]

    torch.save(model.state_dict(), args.out_dir / "patchtst_classifier.pt")
    torch.save(mlp.state_dict(), args.out_dir / "patchtst_embedding_mlp.pt")
    cat.save_model(args.out_dir / "patchtst_embedding_catboost.cbm")
    pred = pd.DataFrame(
        {
            "timestamp": score.loc[score_idx, "timestamp"].astype(str).to_numpy(),
            "label": y_score.astype(np.int64),
            "patchtst_e2e_p_long": ps.astype(np.float32),
            "patchtst_mlp_p_long": mlp_score.astype(np.float32),
            "patchtst_cat_p_long": cat_score.astype(np.float32),
        }
    )
    pred.to_csv(args.out_dir / "pred_score.csv", index=False)
    summary = {
        "type": MODEL_ID,
        "contract": "PatchTST trained from scratch on binary tradeable short/long labels; compares end-to-end classifier, frozen embedding+MLP, and frozen embedding+CatBoost.",
        "train_csv": str(args.train_csv),
        "score_csv": str(args.score_csv),
        "label_config": cfg,
        "horizon": int(args.horizon),
        "input_profile": "audit_compact_local_regime patch channels",
        "core_features": list(patchmix.CORE_FEATURES),
        "patch_channels": list(patchmix.PATCH_CHANNELS),
        "device": str(device),
        "catboost_task_type": cat_task,
        "rows": {
            "fit": int(len(fit_ds)),
            "hold": int(len(hold_ds)),
            "score": int(len(score_ds)),
            "score_tradeable_coverage": float(len(score_ds) / max(1, len(score))),
        },
        "history": history,
        "results": {
            "patchtst_end_to_end": {"hold": _metrics(yh, ph), "score": _metrics(ys, ps)},
            "patchtst_embedding_mlp": {"hold": _metrics(y_hold, mlp_hold), "score": _metrics(y_score, mlp_score)},
            "patchtst_embedding_catboost": {"hold": _metrics(y_hold, cat_hold), "score": _metrics(y_score, cat_score)},
        },
        "artifacts": {
            "classifier": str(args.out_dir / "patchtst_classifier.pt"),
            "mlp": str(args.out_dir / "patchtst_embedding_mlp.pt"),
            "catboost": str(args.out_dir / "patchtst_embedding_catboost.cbm"),
            "pred_score": str(args.out_dir / "pred_score.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"summary": str(args.out_dir / "summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

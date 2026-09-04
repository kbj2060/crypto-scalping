#!/usr/bin/env python3
"""Single-seed cheap_gate: does a TabM (BatchEnsemble MLP) beat the Tier0 GBM baseline
(train_eth_sweep_v_rebound_gbm_baseline_20260829.py, VAL AUC 0.6222 / OOS AUC 0.6425) on the
liquidity_sweep -> V_REBOUND task? Per this project's own established workflow (see
eth_odyssey4_batchensemble_collapse_and_quality_head_duplication_20260818 memory): cheap single-
seed gate first, N>=5-seed confirmation only if this looks promising -- single-seed comparisons on
this project's weak-signal 5m labels are otherwise indistinguishable from noise
([[tabm_hp_low_signal_pattern]]).

Model/training recipe reused verbatim in structure from `ExitTabMClassifier`/`_fit_binary_tabm`
(scripts/train_eval_omega1_2_tabm_exit_head_20260603.py) -- same BatchEnsemble R-only gates, same
SiLU+LayerNorm+post-norm-residual block, same AdamW/class-balanced-weighting/grad-clip/patience
recipe (this project's own N>=5-seed-confirmed finding: fancier recipes -- GCE, cosine schedule,
AdaBelief -- all lost to this plain original once fairly seed-averaged, so it is not reproduced
here). Capacity is downsized for this task's 9,137-row training set, ~8.5x smaller than this
architecture's smallest prior use in this repo (78K rows): k=8->4, hidden=192->64, layers=3->2,
batch_size=2048->512.

One correctness fix versus the reference recipe: PURGE/EMBARGO events whose 30-minute label window
crosses a split boundary (the reference recipe's own internal 0.85/0.15 split has no such gap --
flagged as a real gap in the source review). Applied at TRAIN/VAL and VAL/OOS boundaries here.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabm_cheap_gate_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=30)
SEED = 20260829

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]


@dataclass(frozen=True)
class TabMConfig:
    k: int = 4
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.12
    batch_size: int = 512
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    epochs: int = 42
    patience: int = 8


CFG = TabMConfig()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TabMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2, *, cfg: TabMConfig = CFG) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.input_scale = nn.Parameter(torch.randn(self.k, n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, n_features))
        self.in_proj = nn.Linear(n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.out = nn.Linear(int(cfg.hidden), int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return self.out(h)


def standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return ((arr - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    arr = x.to_numpy(dtype=np.float32)
    return ((arr - scaler["mean"]) / scaler["std"]).astype(np.float32)


def fit_tabm(x: pd.DataFrame, y: np.ndarray, *, seed: int, device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    x_np, scaler = standardize_fit(x)
    y_np = np.asarray(y, dtype=np.int64)
    weights = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32)

    n = len(y_np)
    split = int(n * 0.85)
    train_idx, val_idx = np.arange(split), np.arange(split, n)
    model = TabMClassifier(x_np.shape[1], 2, cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_np[train_idx]), torch.from_numpy(weights[train_idx]))
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False,
                         generator=torch.Generator().manual_seed(seed))

    best_state, best_loss, stale, last_epoch = None, float("inf"), 0, 0
    curve = []
    for epoch in range(CFG.epochs):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logits = model(xb)
            loss_k = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), yb[:, None].expand(-1, CFG.k).reshape(-1), reduction="none",
            ).reshape(-1, CFG.k)
            loss = (loss_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx, vy, vw = torch.from_numpy(x_np[val_idx]).to(device), torch.from_numpy(y_np[val_idx]).to(device), torch.from_numpy(weights[val_idx]).to(device)
            logits = model(vx)
            loss_k = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2), vy[:, None].expand(-1, CFG.k).reshape(-1), reduction="none",
            ).reshape(-1, CFG.k)
            val_loss = float(((loss_k.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        curve.append(val_loss)
        if val_loss + 1.0e-6 < best_loss:
            best_loss, best_state, stale = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= CFG.patience:
                break
    model.load_state_dict(best_state)
    return {"model": model, "scaler": scaler, "best_val_loss": best_loss, "epochs_ran": last_epoch, "val_loss_curve": curve}


@torch.no_grad()
def predict_proba(model: nn.Module, x: pd.DataFrame, scaler: dict, device: torch.device) -> np.ndarray:
    model.eval()
    x_np = standardize_apply(x, scaler)
    xb = torch.from_numpy(x_np).to(device)
    return torch.softmax(model(xb), dim=-1).mean(dim=1)[:, 1].detach().cpu().numpy()


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)), "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "naive_majority_class_accuracy": round(naive_acc, 4),
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END)],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name, part in parts.items():
        print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}")
    print(f"(purge/embargo dropped {len(df[df['timestamp'] < VAL_START]) - len(parts['train'])} train rows "
          f"and {len(df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)]) - len(parts['val'])} val rows near split boundaries)")

    fit = fit_tabm(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy(), seed=SEED, device=device)
    print(f"\nTabM fit: epochs_ran={fit['epochs_ran']} best_val_loss={fit['best_val_loss']:.4f}")
    print(f"val_loss curve: {[round(v, 4) for v in fit['val_loss_curve']]}")

    results = {}
    for name in ("train", "val", "oos"):
        proba = predict_proba(fit["model"], parts[name][FEATURE_COLUMNS], fit["scaler"], device)
        results[name] = evaluate(proba, parts[name]["label"].to_numpy())
        r = results[name]
        print(f"  {name:5s} n={r['n']:5d} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
              f"auc={r['auc']:.4f} naive={r['naive_majority_class_accuracy']:.4f} beats_naive={r['beats_naive_accuracy']}")

    gbm_baseline = {"val_auc": 0.6222, "oos_auc": 0.6425}
    print(f"\n=== vs GBM Tier0 baseline ===")
    print(f"  VAL AUC: TabM {results['val']['auc']:.4f} vs GBM {gbm_baseline['val_auc']:.4f} "
          f"(delta {results['val']['auc'] - gbm_baseline['val_auc']:+.4f})")
    print(f"  OOS AUC: TabM {results['oos']['auc']:.4f} vs GBM {gbm_baseline['oos_auc']:.4f} "
          f"(delta {results['oos']['auc'] - gbm_baseline['oos_auc']:+.4f})")

    report = {
        "seed": SEED, "config": CFG.__dict__, "epochs_ran": fit["epochs_ran"],
        "best_val_loss": fit["best_val_loss"], "val_loss_curve": fit["val_loss_curve"],
        "results": results, "gbm_baseline_for_comparison": gbm_baseline,
        "note": "single-seed cheap_gate only -- not yet N>=5-seed confirmed, per tabm_hp_low_signal_pattern",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

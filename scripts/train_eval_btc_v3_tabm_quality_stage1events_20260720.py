#!/usr/bin/env python3
"""BTC v3 follow-up: TabM-style side-split quality classifier on Stage 1's sparse event dataset.

Stage 3 (train_eval_btc_v3_quality_classifier_20260714.py) trained ONE HistGradientBoosting
classifier across LONG and SHORT events mixed together and found no separating power (inverted
quintile win-rate). Two things changed here, both directly motivated by evidence gathered before
running this:

1. 2024 excluded from training/validation. side==1 (long) win rate in 2024 was 97.7% vs side==-1
   (short) 0.0% -- an extreme, clean split that is almost certainly BTC's 2024 bull-market beta,
   not a learned signal (2025: long 23.5%/short 26.8%; 2026: long 21.5%/short 31.3% -- the
   relationship is gone or inverted). Training on 2024 would teach a spurious long/short prior
   that demonstrably does not hold in more recent data.
2. Long and short trained as SEPARATE models (side-split), matching this project's own established
   risk-sidecar convention (every promoted Omega4.6.1 risk sidecar is side-split) -- Stage 3 mixed
   both sides into one classifier.

Architecture: a single-head reduction of ThreeHeadTabM (train_eval_omega1_2_tabm_3head_20260603.py)
-- same k-expert input-scale/bias trick, same hidden/layer sizes, just one binary "win" head
instead of direction/quality/exit. Reuses the proven hyperparameters, not a new architecture search.

Splits (BTC v3 holdout policy respected -- HOLDOUT_START=2026-07-14, already enforced when Stage 1
built its event dataset):
  train: 2025-01-01 .. 2025-09-30
  validation: 2025-10-01 .. 2025-12-31
  oos: 2026-01-01 .. 2026-07-11 (dataset's own last row)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "tmp/causal_regen_20260516/btc_v3_sparse_event_dataset_20260714/sparse_event_dataset.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v3_tabm_quality_stage1events_20260720"

FEATURE_COLS = [
    "logret_1", "logret_2", "logret_3", "logret_6", "logret_12", "logret_24",
    "rvol_6", "rvol_12", "rvol_24", "rvol_48",
    "atr_pct", "rsi_14", "macd_hist", "bb_width", "bb_pos", "vol_z_48",
    "taker_imb", "body_ratio", "upper_wick", "lower_wick",
    "skew_24", "kurt_24", "dist_sma50", "hurst_proxy",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "ts_t_value",
]

TRAIN_START = pd.Timestamp("2025-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
VAL_START = pd.Timestamp("2025-10-01 00:00:00")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01 00:00:00")


@dataclass(frozen=True)
class TabMConfig:
    k: int = 8
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.15
    batch_size: int = 128
    lr: float = 1.5e-3
    weight_decay: float = 3.0e-4
    max_epochs: int = 200
    patience: int = 20
    seed: int = 7020
    n_folds: int = 5
    embargo_hours: float = 168.0  # 7 days -- covers max feature lookback (48h) and max hold (~144h)


class TabMQualityHead(nn.Module):
    """Single-head reduction of ThreeHeadTabM: same k-expert input-scale/bias ensemble trick,
    same encoder shape, one scalar (win-probability logit) output instead of 3 heads."""

    def __init__(self, n_features: int, *, cfg: TabMConfig) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.head = nn.Linear(int(cfg.hidden), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        logits = self.head(h).squeeze(-1)  # (batch, k)
        return logits.mean(dim=1)


def _standardize_fit(x: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return {"mean": mean, "std": std}


def _standardize_apply(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _quintile_diag(df: pd.DataFrame, prob: np.ndarray) -> list[dict[str, Any]]:
    out = df.copy()
    out["pred_prob"] = prob
    out["quintile"] = pd.qcut(out["pred_prob"], 5, labels=False, duplicates="drop")
    rows = []
    for q, grp in out.groupby("quintile"):
        rows.append({
            "quintile": int(q), "n": int(len(grp)),
            "win_rate": float(grp["win"].mean()),
            "mean_trade_return": float(grp["trade_return"].mean()),
            "mean_pred_prob": float(grp["pred_prob"].mean()),
        })
    return sorted(rows, key=lambda r: r["quintile"])


def _fit_predict(x_fit_raw: np.ndarray, y_fit: np.ndarray, x_pred_raw: np.ndarray, *, cfg: TabMConfig, seed: int) -> np.ndarray:
    """Fits one TabM quality head (internal random holdout of x_fit for early stopping) and
    returns predicted win-probabilities for x_pred_raw. x_pred_raw is standardized using x_fit's
    own scaler only -- never touches any statistic from the fold/split being predicted."""
    torch.manual_seed(seed)
    scaler = _standardize_fit(x_fit_raw)
    x_fit = _standardize_apply(x_fit_raw, scaler)
    x_pred = _standardize_apply(x_pred_raw, scaler)

    rng = np.random.default_rng(seed)
    n = len(x_fit)
    perm = rng.permutation(n)
    n_holdout = max(int(0.15 * n), 10)
    holdout_idx, fit_idx = perm[:n_holdout], perm[n_holdout:]

    device = torch.device("cpu")
    model = TabMQualityHead(len(FEATURE_COLS), cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.from_numpy(x_fit[fit_idx]).to(device)
    yt = torch.from_numpy(y_fit[fit_idx]).to(device)
    xh = torch.from_numpy(x_fit[holdout_idx]).to(device)
    yh = torch.from_numpy(y_fit[holdout_idx]).to(device)

    best_loss = float("inf")
    best_state = None
    patience_left = cfg.patience
    n_fit = len(xt)
    train_rng = np.random.default_rng(seed + 1)
    for _epoch in range(cfg.max_epochs):
        model.train()
        fit_perm = train_rng.permutation(n_fit)
        for start in range(0, n_fit, cfg.batch_size):
            idx = fit_perm[start:start + cfg.batch_size]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hold_loss = float(nn.functional.binary_cross_entropy_with_logits(model(xh), yh).item())
        if hold_loss < best_loss - 1e-5:
            best_loss = hold_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x_pred))).numpy()


def _purged_kfold_oof(dev_df: pd.DataFrame, *, cfg: TabMConfig) -> np.ndarray:
    """Time-ordered K-fold with a symmetric embargo around each fold's test window (purges
    training rows whose feature lookback or outcome-simulation horizon could overlap the test
    fold -- AFML-style purge+embargo, not a plain shuffled K-fold, since events are event-level
    but their features/labels span hours-to-days of surrounding bars)."""
    dev_df = dev_df.sort_values("entry_available_timestamp").reset_index(drop=True)
    n = len(dev_df)
    ts = dev_df["entry_available_timestamp"]
    fold_edges = np.linspace(0, n, cfg.n_folds + 1).astype(int)
    embargo = pd.Timedelta(hours=cfg.embargo_hours)
    oof_prob = np.full(n, np.nan, dtype=np.float32)
    for fold_i in range(cfg.n_folds):
        lo, hi = fold_edges[fold_i], fold_edges[fold_i + 1]
        test_mask = np.zeros(n, dtype=bool)
        test_mask[lo:hi] = True
        test_start, test_end = ts.iloc[lo], ts.iloc[hi - 1]
        purge_mask = (ts >= test_start - embargo) & (ts <= test_end + embargo)
        train_mask = (~test_mask) & (~purge_mask.to_numpy())
        x_fit = dev_df.loc[train_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
        y_fit = dev_df.loc[train_mask, "win"].to_numpy(dtype=np.float32)
        x_pred = dev_df.loc[test_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
        oof_prob[lo:hi] = _fit_predict(x_fit, y_fit, x_pred, cfg=cfg, seed=cfg.seed + fold_i)
    if np.isnan(oof_prob).any():
        raise RuntimeError("OOF scoring left unfilled rows")
    return oof_prob


def _train_one_side(dev_df: pd.DataFrame, oos_df: pd.DataFrame, *, side_name: str, cfg: TabMConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev_df = dev_df.sort_values("entry_available_timestamp").reset_index(drop=True)

    oof_prob = _purged_kfold_oof(dev_df, cfg=cfg)
    oof_diag = _quintile_diag(dev_df, oof_prob)

    # Final model: fit on ALL of 2025 (dev_df), score 2026 OOS once.
    x_dev_raw = dev_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_dev = dev_df["win"].to_numpy(dtype=np.float32)
    x_oos_raw = oos_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    oos_prob = _fit_predict(x_dev_raw, y_dev, x_oos_raw, cfg=cfg, seed=cfg.seed + 999)
    oos_diag = _quintile_diag(oos_df, oos_prob)

    return {
        "side": side_name,
        "n_dev_2025": int(len(dev_df)), "n_folds": cfg.n_folds, "embargo_hours": cfg.embargo_hours,
        "n_oos": int(len(oos_df)),
        "dev_baseline_mean_trade_return": float(dev_df["trade_return"].mean()),
        "dev_baseline_win_rate": float(dev_df["win"].mean()),
        "oof_quintile_diag": oof_diag,
        "oos_baseline_mean_trade_return": float(oos_df["trade_return"].mean()),
        "oos_baseline_win_rate": float(oos_df["win"].mean()),
        "oos_quintile_diag": oos_diag,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    df["entry_available_timestamp"] = pd.to_datetime(df["entry_available_timestamp"])
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    df = df[df["entry_available_timestamp"] >= TRAIN_START].reset_index(drop=True)  # excludes 2024

    cfg = TabMConfig()
    report: dict[str, Any] = {"model_id": "btc_v3_tabm_quality_stage1events_20260720_purged_kfold", "cfg": cfg.__dict__, "sides": {}}
    for side_val, side_name in ((1, "long"), (-1, "short")):
        side_df = df[df["side"] == side_val].reset_index(drop=True)
        dev_df = side_df[(side_df["entry_available_timestamp"] >= TRAIN_START) & (side_df["entry_available_timestamp"] <= VAL_END)]
        oos_df = side_df[side_df["entry_available_timestamp"] >= OOS_START]
        result = _train_one_side(dev_df, oos_df, side_name=side_name, cfg=cfg)
        report["sides"][side_name] = result
        print(f"[{side_name}] n_dev_2025={result['n_dev_2025']} n_oos={result['n_oos']}", flush=True)
        print(f"[{side_name}] OOF (2025, purged {cfg.n_folds}-fold) quintiles: {result['oof_quintile_diag']}", flush=True)
        print(f"[{side_name}] OOS (2026) quintiles: {result['oos_quintile_diag']}", flush=True)

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

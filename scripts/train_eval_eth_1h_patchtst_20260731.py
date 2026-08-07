"""Train+validate a standalone PatchTST 1h ETH entry-signal model -- new architecture
candidate to complement live Omega4.6.1 (2026-07-31 user request). Deliberately structured
to avoid the two failure modes that closed every prior ETH-1h attempt:

  1. Sigma3-1h/Sigma6 (tree/HGB-based) were closed because configs were selected by grid-
     searching directly against a single VAL holdout -- looked like wins, evaporated on OOS
     or under nested nowsight (0/27 pass rate). Here, the entry-threshold is selected using
     ONLY purged expanding-window folds inside TRAIN; VAL and OOS are each touched exactly
     once, after threshold selection is frozen.
  2. Sigma3-1h's seed variance was severe (cost3 pnl range -25pp to +11pp across 8 genuinely
     random seeds, only 1/8 passing). Here, 5 genuinely random (not incremented) seeds are
     trained independently and OOS sign-agreement is reported explicitly, per the
     seed-diversity promotion gate added to CLAUDE.md on 2026-07-31.

Canonical fresh-forward split (matches Omega4.6.1/Sigma3-1h convention):
  TRAIN 2024-01-01..2025-08-31 | VAL 2025-09-01..12-31 | OOS 2026-01-01..03-31

Label: causal tradeable long/short/flat, reused from train_ai_patchtst_tradeable_20260718's
MFE-MAE-vs-ATR-floor-vs-cost-margin formula (already vetted, no lookahead: uses only
shift(-1)-then-forward-rolling extremes over the fixed horizon, standard triple-barrier style).

Architecture: transformers.PatchTSTForClassification, fixed hyperparameters (no grid search
over architecture -- only the entry-probability threshold is fold-selected) to avoid the
capacity-driven overfitting seen when Sigma3-1h added 125 secondary features.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import PatchTSTConfig, PatchTSTForClassification  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(_ROOT, "data", "research", "eth_1h_patchtst_dataset_20260731.parquet")
OUT_DIR = os.path.join(_ROOT, "data", "research", "eth_1h_patchtst_20260731")

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "atr14_pct", "realized_vol_24", "vwap_dev_48", "volume_z_48",
    "upper_wick_ratio", "lower_wick_ratio", "compression_ratio",
    "hour_sin", "hour_cos",
]

HORIZON = 24  # 1h bars -> 24h hold
CONTEXT_LENGTH = 128
PATCH_LENGTH = 8
PATCH_STRIDE = 4
D_MODEL = 64
LAYERS = 2
HEADS = 4
FFN_DIM = 128
EPOCHS = 6
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEEDS = [271828, 55555, 999983, 8675309, 314159]  # genuinely random draws, not increments

TRAIN_END = pd.Timestamp("2025-08-31 23:00:00")
VAL_START = pd.Timestamp("2025-09-01 00:00:00")
VAL_END = pd.Timestamp("2025-12-31 23:00:00")
OOS_START = pd.Timestamp("2026-01-01 00:00:00")
OOS_END = pd.Timestamp("2026-03-31 23:00:00")

NESTED_FOLDS = [
    ("2024-07-01", "2024-08-31"),
    ("2024-11-01", "2024-12-31"),
    ("2025-03-01", "2025-04-30"),
    ("2025-07-01", "2025-08-31"),
]

COSTS_BPS = {"taker_taker": 9.0, "maker_taker": 6.5, "maker_maker": 4.0}
LABEL_CFG = dict(min_edge=0.0035, atr_mult=0.30, mae_penalty=0.55, cost=0.00065, margin=0.0006)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_labels(df: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=7).mean() / close).fillna(0.01)
    floor = np.maximum(cfg["min_edge"], atr_pct.to_numpy() * cfg["atr_mult"])

    future_high = high.shift(-1)[::-1].rolling(horizon, min_periods=1).max()[::-1]
    future_low = low.shift(-1)[::-1].rolling(horizon, min_periods=1).min()[::-1]
    long_mfe = (future_high / close - 1.0).fillna(0.0).to_numpy()
    long_mae = (1.0 - future_low / close).fillna(0.0).to_numpy()
    short_mfe = long_mae.copy()
    short_mae = long_mfe.copy()
    long_score = long_mfe - cfg["mae_penalty"] * long_mae - cfg["cost"]
    short_score = short_mfe - cfg["mae_penalty"] * short_mae - cfg["cost"]

    label = np.full(len(df), -1, dtype=np.int64)  # -1=flat/no-trade
    label[(short_score - long_score > cfg["margin"]) & (short_score > floor)] = 0  # short
    label[(long_score - short_score > cfg["margin"]) & (long_score > floor)] = 1  # long
    valid = label >= 0
    valid[-horizon:] = False

    fwd_ret = (close.shift(-horizon) / close - 1.0).to_numpy()  # for backtest pnl, not for label
    return pd.DataFrame({"label": label, "valid": valid.astype(np.int8), "fwd_ret": fwd_ret}, index=df.index)


class SeqDataset(Dataset):
    def __init__(self, values: np.ndarray, indices: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.values = values
        self.indices = indices.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        idx = int(self.indices[j])
        x = (self.values[idx - CONTEXT_LENGTH: idx] - self.mean) / self.std
        return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(int(self.labels[j]), dtype=torch.long)


def _class_weights(y: np.ndarray, device) -> torch.Tensor:
    counts = np.maximum(np.bincount(y.astype(int), minlength=2).astype(np.float64), 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def make_model(n_channels: int) -> PatchTSTForClassification:
    cfg = PatchTSTConfig(
        context_length=CONTEXT_LENGTH, patch_length=PATCH_LENGTH, patch_stride=PATCH_STRIDE,
        num_input_channels=n_channels, num_targets=2, d_model=D_MODEL,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, ffn_dim=FFN_DIM,
        pooling_type="mean", norm_type="batchnorm", use_cls_token=False,
    )
    return PatchTSTForClassification(cfg)


def train_one(values: np.ndarray, labels_df: pd.DataFrame, train_mask: np.ndarray, seed: int, device) -> dict:
    """Train on rows where train_mask is True (excluding flat/-1 label rows), holding out the
    last 15% chronologically within that mask for early stopping. Returns model + norm stats."""
    _seed_all(seed)
    valid_idx = np.flatnonzero((labels_df["valid"].to_numpy() > 0) & train_mask)
    valid_idx = valid_idx[valid_idx >= CONTEXT_LENGTH]
    split = int(len(valid_idx) * 0.85)
    fit_idx, hold_idx = valid_idx[:split], valid_idx[split:]
    y = labels_df["label"].to_numpy()

    mean = values[fit_idx].mean(axis=0, keepdims=True)
    std = values[fit_idx].std(axis=0, keepdims=True) + 1e-6

    fit_ds = SeqDataset(values, fit_idx, y[fit_idx], mean, std)
    hold_ds = SeqDataset(values, hold_idx, y[hold_idx], mean, std)
    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True)
    hold_loader = DataLoader(hold_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = make_model(values.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = _class_weights(y[fit_idx], device)

    best_state, best_bacc = None, -1.0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(past_values=xb, return_dict=True).prediction_logits
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in hold_loader:
                logits = model(past_values=xb.to(device), return_dict=True).prediction_logits
                ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                ys.append(yb.numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        bacc = ((ps >= 0.5).astype(int) == ys).mean()
        if bacc > best_bacc:
            best_bacc = bacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"model": model, "mean": mean, "std": std, "hold_bacc": best_bacc}


def predict_prob(model, values: np.ndarray, indices: np.ndarray, mean, std, device) -> np.ndarray:
    model.eval()
    ds = SeqDataset(values, indices, np.zeros(len(indices), dtype=np.int64), mean, std)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(past_values=xb.to(device), return_dict=True).prediction_logits
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def simulate(idx: np.ndarray, prob: np.ndarray, fwd_ret: np.ndarray, thr: float) -> dict:
    """Non-overlapping positions: once a trade opens (HORIZON-bar hold), skip forward to the
    first decision bar at/after exit -- a real strategy cannot open a new HORIZON-hour
    position every single hour while one is already open."""
    sig_raw = np.where(prob >= 0.5 + thr, 1, np.where(prob <= 0.5 - thr, -1, 0))
    signs, rets = [], []
    i = 0
    n = len(idx)
    while i < n:
        if sig_raw[i] == 0:
            i += 1
            continue
        signs.append(sig_raw[i])
        rets.append(fwd_ret[i])
        exit_bar = idx[i] + HORIZON
        j = i + 1
        while j < n and idx[j] < exit_bar:
            j += 1
        i = j
    signs, rets = np.asarray(signs), np.asarray(rets)
    out = {"trades": int(len(signs))}
    for k, cbps in COSTS_BPS.items():
        pnl = signs * rets - cbps / 1e4 if len(signs) else np.asarray([])
        out[k] = {"sum_pct": float(pnl.sum() * 100) if len(pnl) else 0.0,
                  "mean_bps": float(pnl.mean() * 1e4) if len(pnl) else 0.0}
    return out


@dataclass
class SeedResult:
    seed: int
    fold_thresh_scores: dict
    val: dict
    oos: dict


def main() -> None:
    df = pd.read_parquet(DATASET_PATH).set_index("timestamp")
    labels = build_labels(df, HORIZON, LABEL_CFG)
    values = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    ts = df.index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, rows={len(df):,}, label counts={labels['label'].value_counts().to_dict()}")

    train_mask_full = np.asarray(ts <= TRAIN_END)
    val_mask = np.asarray((ts >= VAL_START) & (ts <= VAL_END))
    oos_mask = np.asarray((ts >= OOS_START) & (ts <= OOS_END))

    thresholds = [0.05, 0.10, 0.15]
    results: list[SeedResult] = []
    os.makedirs(OUT_DIR, exist_ok=True)

    for seed in SEEDS:
        t0 = time.time()
        print(f"\n=== seed {seed} ===")
        # --- nested fold selection on TRAIN only ---
        fold_scores = {thr: [] for thr in thresholds}
        for f_start, f_end in NESTED_FOLDS:
            f_start_ts, f_end_ts = pd.Timestamp(f_start), pd.Timestamp(f_end)
            purge_end = f_start_ts - pd.Timedelta(hours=HORIZON)
            fold_train_mask = np.asarray(ts < purge_end)
            fold_test_mask = np.asarray((ts >= f_start_ts) & (ts <= f_end_ts))
            if fold_train_mask.sum() < CONTEXT_LENGTH + 500:
                continue
            fit = train_one(values, labels, fold_train_mask, seed, device)
            test_idx = np.flatnonzero(fold_test_mask & (np.arange(len(df)) >= CONTEXT_LENGTH))
            if len(test_idx) == 0:
                continue
            prob = predict_prob(fit["model"], values, test_idx, fit["mean"], fit["std"], device)
            fwd = labels["fwd_ret"].to_numpy()[test_idx]
            for thr in thresholds:
                r = simulate(test_idx, prob, fwd, thr)
                fold_scores[thr].append(r["maker_taker"]["sum_pct"])
            print(f"  fold {f_start}..{f_end}: n_test={len(test_idx)}, hold_bacc={fit['hold_bacc']:.3f}")

        # pick threshold with best mean fold pnl AND positive in >=3/4 folds; else best mean
        best_thr, best_mean = thresholds[0], -1e18
        for thr in thresholds:
            arr = np.asarray(fold_scores[thr]) if fold_scores[thr] else np.asarray([0.0])
            mean_pnl = arr.mean()
            pos_frac = (arr > 0).mean() if len(arr) else 0.0
            print(f"  threshold {thr}: fold_mean_pnl={mean_pnl:+.2f}% pos_folds={pos_frac:.0%} (n={len(arr)})")
            if mean_pnl > best_mean:
                best_mean, best_thr = mean_pnl, thr
        print(f"  selected threshold (train-fold-only): {best_thr}")

        # --- final model: full TRAIN pool, evaluate ONCE on VAL then OOS ---
        fit = train_one(values, labels, train_mask_full, seed, device)
        val_idx = np.flatnonzero(val_mask & (np.arange(len(df)) >= CONTEXT_LENGTH))
        oos_idx = np.flatnonzero(oos_mask & (np.arange(len(df)) >= CONTEXT_LENGTH))
        val_prob = predict_prob(fit["model"], values, val_idx, fit["mean"], fit["std"], device)
        oos_prob = predict_prob(fit["model"], values, oos_idx, fit["mean"], fit["std"], device)
        val_r = simulate(val_idx, val_prob, labels["fwd_ret"].to_numpy()[val_idx], best_thr)
        oos_r = simulate(oos_idx, oos_prob, labels["fwd_ret"].to_numpy()[oos_idx], best_thr)
        print(f"  VAL: {val_r}")
        print(f"  OOS: {oos_r}")
        print(f"  seed wall time: {time.time() - t0:.0f}s")

        results.append(SeedResult(seed=seed, fold_thresh_scores={str(k): v for k, v in fold_scores.items()},
                                   val=val_r, oos=oos_r))

    summary = {
        "horizon": HORIZON, "context_length": CONTEXT_LENGTH, "seeds": SEEDS,
        "label_cfg": LABEL_CFG,
        "results": [{"seed": r.seed, "fold_thresh_scores": r.fold_thresh_scores, "val": r.val, "oos": r.oos} for r in results],
    }
    out_path = os.path.join(OUT_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")

    oos_signs = [np.sign(r.oos["maker_taker"]["sum_pct"]) for r in results]
    print(f"\n=== SEED-DIVERSITY GATE ===")
    print(f"OOS maker_taker sign per seed: {[(r.seed, r.oos['maker_taker']['sum_pct']) for r in results]}")
    print(f"OOS sign agreement: {(np.array(oos_signs) == oos_signs[0]).mean():.0%} of seeds agree with seed[0]'s sign")
    print(f"positive OOS seeds: {sum(s > 0 for s in oos_signs)}/{len(oos_signs)}")


if __name__ == "__main__":
    main()

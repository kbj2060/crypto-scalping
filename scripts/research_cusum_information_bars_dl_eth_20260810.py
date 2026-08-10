#!/usr/bin/env python3
"""Research probe (2026-08-10, literature follow-up idea A): information-driven (CUSUM) event
sampling + triple-barrier labeling + a real neural network, on ETH 5m data.

Motivation: the user asked whether any recent paper crosses the train/OOS generalization gap
found in the overnight loop (kitchen-sink AUC 0.956 in-sample vs ~0.517 OOS). One candidate
mechanism from "Algorithmic crypto trading using information-driven bars, triple barrier
labeling and deep learning" (Financial Innovation, 2025) is EVENT-DRIVEN sampling: instead of
uniform 5-min-stride bars (where adjacent labels overlap heavily -- this repo's own
project-btc-tripbarrier-baseline-is-seed-artifact-20260807 found a 10.8x effective-N
overstatement from exactly this overlap), sample only at CUSUM-filtered "event" bars where
cumulative |return| since the last event exceeds a volatility-scaled threshold. This naturally
spaces out labels in time, directly attacking the effective-N inflation problem rather than
trying yet another feature family on the same uniformly-overlapping bars.

Barrier is SYMMETRIC (equal ATR-scaled TP/SL) specifically to avoid this repo's own
tp:sl-ratio-asymmetry bias (project-baseline-must-be-always-long-short-not-zero-20260809) --
with a symmetric barrier, always_long/always_short have no structural edge, so a real neural
net's AUC/hit-rate can be read directly.

Model: a real MLP neural network (PyTorch, GPU) trained on standard causal features computed as
of each event bar, evaluated on genuinely held-out TRAIN/DEV/VAL/OOS splits, with a GBDT
(LightGBM) side-by-side for reference (matching the tabular-DL-vs-GBDT literature framing from
tonight's search).
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402

ETH_PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
CUSUM_VOL_WINDOW = 288  # 1 day, for the volatility-scaled CUSUM threshold
CUSUM_MULTIPLIER = 3.0  # event triggers at 3x the rolling per-bar vol, cumulative
BARRIER_ATR_WINDOW = 96
BARRIER_MULTIPLIER = 2.0  # symmetric TP=SL=2x ATR
MAX_HORIZON_BARS = 96
FEATURE_WINDOWS = [6, 12, 24, 48, 96]
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_price() -> pd.DataFrame:
    df = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def cusum_events(log_close: np.ndarray, vol: np.ndarray, multiplier: float) -> np.ndarray:
    """Symmetric CUSUM filter (Lopez de Prado). Returns a boolean array marking event bars."""
    n = len(log_close)
    ret = np.diff(log_close, prepend=log_close[0])
    s_pos, s_neg = 0.0, 0.0
    events = np.zeros(n, dtype=bool)
    for i in range(1, n):
        threshold = multiplier * vol[i]
        if not np.isfinite(threshold) or threshold <= 0:
            continue
        s_pos = max(0.0, s_pos + ret[i])
        s_neg = min(0.0, s_neg + ret[i])
        if s_pos > threshold:
            events[i] = True
            s_pos = 0.0
        elif -s_neg > threshold:
            events[i] = True
            s_neg = 0.0
    return events


def triple_barrier_symmetric(high, low, close, event_idx, atr, horizon):
    """For each event index, walk forward up to `horizon` bars; outcome = 1 if +atr[event] hit
    first, 2 if -atr[event] hit first, 0 if neither within horizon (timeout)."""
    n = len(close)
    outcomes = np.zeros(len(event_idx), dtype=np.int64)
    for k, i in enumerate(event_idx):
        entry = close[i]
        band = atr[i]
        if not np.isfinite(band) or band <= 0:
            outcomes[k] = 0
            continue
        upper, lower = entry + band, entry - band
        end = min(i + horizon, n - 1)
        outcome = 0
        for j in range(i + 1, end + 1):
            if high[j] >= upper:
                outcome = 1
                break
            if low[j] <= lower:
                outcome = 2
                break
        outcomes[k] = outcome
    return outcomes


def build_causal_features(df: pd.DataFrame) -> list[str]:
    close = df["close"].to_numpy()
    log_close = np.log(close)
    bar_ret = np.diff(log_close, prepend=log_close[0])
    true_range = (df["high"].to_numpy() - df["low"].to_numpy()) / close
    vol = df["volume"].to_numpy()
    cols = []
    for w in FEATURE_WINDOWS:
        ret_w = log_close - np.roll(log_close, w); ret_w[:w] = np.nan
        rvol_w = pd.Series(bar_ret).rolling(w, min_periods=w).std().to_numpy()
        atr_w = pd.Series(true_range).rolling(w, min_periods=w).mean().to_numpy()
        vol_z = (pd.Series(vol).rolling(w, min_periods=w).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-9), raw=True)).to_numpy()
        for name, arr in [("ret", ret_w), ("rvol", rvol_w), ("atr", atr_w), ("volz", vol_z)]:
            col = f"{name}_w{w}"
            df[col] = arr
            cols.append(col)
    return cols


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, device, epochs=100, patience=10):
    model = MLP(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 512):
            idx = perm[start:start + 512]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        sched.step(val_loss)
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    df = load_price()
    close, high, low = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    log_close = np.log(close)
    bar_ret = np.diff(log_close, prepend=log_close[0])
    rolling_vol = pd.Series(bar_ret).rolling(CUSUM_VOL_WINDOW, min_periods=CUSUM_VOL_WINDOW).std().to_numpy()
    events_mask = cusum_events(log_close, rolling_vol, CUSUM_MULTIPLIER)
    true_range = (high - low) / close
    atr = pd.Series(true_range).rolling(BARRIER_ATR_WINDOW, min_periods=BARRIER_ATR_WINDOW).mean().to_numpy() * close * BARRIER_MULTIPLIER

    feat_cols = build_causal_features(df)
    valid = df[feat_cols].notna().all(axis=1).to_numpy() & np.isfinite(rolling_vol) & np.isfinite(atr)
    event_idx = np.where(events_mask & valid)[0]
    event_idx = event_idx[event_idx < len(close) - MAX_HORIZON_BARS - 1]
    print(f"Total bars: {len(df)}  CUSUM events: {len(event_idx)}  "
          f"(1 event per {len(df) / max(len(event_idx), 1):.1f} bars on average)")

    outcomes = triple_barrier_symmetric(high, low, close, event_idx, atr, MAX_HORIZON_BARS)
    events_df = df.iloc[event_idx].copy().reset_index(drop=True)
    events_df["outcome"] = outcomes
    print("Outcome distribution:", pd.Series(outcomes).value_counts(normalize=True).to_dict())

    ts = events_df["timestamp"]
    train_df = events_df[ts <= TRAIN_END]
    dev_df = events_df[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = events_df[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = events_df[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS={len(oos_df)}")

    def favored_direction(sub):
        return 1 if (sub["outcome"] == 1).mean() >= (sub["outcome"] == 2).mean() else 2

    direction = favored_direction(train_df)
    print(f"Favored direction on TRAIN (symmetric barrier -- should be close to a coin flip): {direction}, "
          f"P(long)={float((train_df['outcome']==1).mean()):.4f} P(short)={float((train_df['outcome']==2).mean()):.4f}")

    def win_label(sub):
        return (sub["outcome"] == direction).astype(int).to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_cols].to_numpy())
    X_dev = scaler.transform(dev_df[feat_cols].to_numpy())
    X_val = scaler.transform(val_df[feat_cols].to_numpy())
    X_oos = scaler.transform(oos_df[feat_cols].to_numpy())
    y_train, y_dev, y_val, y_oos = win_label(train_df), win_label(dev_df), win_label(val_df), win_label(oos_df)

    print("\n=== LightGBM (GBDT reference) ===")
    gbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                             min_child_samples=50, random_state=270705, verbosity=-1)
    gbm.fit(X_train, y_train)
    for name, X, y in [("TRAIN", X_train, y_train), ("DEV", X_dev, y_dev), ("VAL", X_val, y_val), ("OOS", X_oos, y_oos)]:
        prob = gbm.predict_proba(X)[:, 1]
        print(f"  {name}: AUC={roc_auc_score(y, prob):.4f}  n={len(y)}  win_rate={y.mean():.4f}")

    print("\n=== MLP (real neural network, PyTorch/GPU) ===")
    mlp = train_mlp(X_train, y_train, X_dev, y_dev, device)
    with torch.no_grad():
        for name, X, y in [("TRAIN", X_train, y_train), ("DEV", X_dev, y_dev), ("VAL", X_val, y_val), ("OOS", X_oos, y_oos)]:
            prob = torch.sigmoid(mlp(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()
            print(f"  {name}: AUC={roc_auc_score(y, prob):.4f}  n={len(y)}  win_rate={y.mean():.4f}")

    # Falsification audit: is the MLP's VAL edge (mean symmetric payoff, +1/-1/0) distinguishable
    # from a shuffled-label control at the same event count/spacing?
    with torch.no_grad():
        val_prob = torch.sigmoid(mlp(torch.tensor(X_val, dtype=torch.float32, device=device))).cpu().numpy()
    payoff = np.where(val_df["outcome"].to_numpy() == direction, 1.0, np.where(val_df["outcome"].to_numpy() == 0, 0.0, -1.0))
    pred_dir = (val_prob > 0.5).astype(int)
    strategy_payoff = np.where(pred_dir == 1, payoff, -payoff)
    rng = np.random.default_rng(20260810)
    null_sums = np.array([np.where(rng.permutation(pred_dir) == 1, payoff, -payoff).sum() for _ in range(2000)])
    real_sum = strategy_payoff.sum()
    percentile = float((null_sums < real_sum).mean())
    print(f"\n=== VAL falsification check: real strategy sum={real_sum:.2f} vs "
          f"{len(null_sums)} label-shuffled draws (mean={null_sums.mean():.2f}) -> percentile={percentile:.3f} ===")


if __name__ == "__main__":
    main()

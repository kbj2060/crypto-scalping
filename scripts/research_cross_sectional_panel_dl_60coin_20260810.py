#!/usr/bin/env python3
"""Research probe (2026-08-10, literature follow-up idea B): cross-sectional breadth deep
learning, mirroring the recipe behind the one clean DL-beats-GBDT case found tonight
(Financial Innovation 2026, A-share market, 5,000+ firms pooled cross-sectionally, LSTM/
Transformer beat tree models). This repo has a 60-symbol feature+label panel
(data/panel/features/*.parquet, data/panel/tripbarrier/*.parquet) already built for the BTC
Rho1 panel-transformer line (project-btc-rho1-panel-direction), which failed for BTC (ranking
head MSE at the random floor, 6/6 backtests negative). This is a distinct test: plain per-symbol
technical features (not Rho1's architecture), pooled across all 60 symbols and cross-sectionally
z-scored per timestamp (the standard cross-sectional-model normalization), training ONE neural
net, then comparing ETH-specific held-out performance against an ETH-ONLY model trained on the
exact same features/architecture with no pooling -- isolating whether BREADTH itself helps,
holding everything else fixed.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES_DIR = ROOT / "data/panel/features"
LABELS_DIR = ROOT / "data/panel/tripbarrier"
FEATURE_COLS = [
    "ret_1", "realized_vol_12", "realized_vol_48", "realized_vol_288", "rsi_14", "macd_hist",
    "bb_width_20", "atr_pct_14", "rvol_12", "rvol_48", "taker_buy_ratio", "hour_sin", "hour_cos",
    "oi_chg_288", "toptrader_ratio", "taker_long_short_vol_ratio", "funding_rate", "funding_roc_288",
]
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"
TARGET_SYMBOL = "ETHUSDT"


def load_all_symbols() -> pd.DataFrame:
    frames = []
    for feat_path in sorted(FEATURES_DIR.glob("*.parquet")):
        symbol = feat_path.stem
        label_path = LABELS_DIR / f"{symbol}.parquet"
        if not label_path.exists():
            continue
        feat = pd.read_parquet(feat_path, columns=["timestamp"] + FEATURE_COLS)
        lab = pd.read_parquet(label_path, columns=["timestamp", "trade_outcome_action", "label_valid"])
        merged = feat.merge(lab, on="timestamp", how="inner")
        merged = merged[merged["label_valid"]].dropna(subset=FEATURE_COLS)
        merged["symbol"] = symbol
        frames.append(merged)
    panel = pd.concat(frames, ignore_index=True)
    return panel


def cross_sectional_zscore(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score each feature within each timestamp's cross-section (standard factor-model
    normalization) -- this is what makes 'pooling many symbols' meaningful rather than just
    concatenating differently-scaled series."""
    panel = panel.copy()
    grouped = panel.groupby("timestamp")[cols]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    panel[cols] = ((panel[cols] - mean) / std).fillna(0.0)
    return panel


def favored_direction(sub: pd.DataFrame) -> int:
    p_long = float((sub["trade_outcome_action"] == 1).mean())
    p_short = float((sub["trade_outcome_action"] == 2).mean())
    return 1 if p_long >= p_short else 2


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_dev, y_dev, device, epochs=60, patience=8):
    model = MLP(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_dev, dtype=torch.float32, device=device)
    yv = torch.tensor(y_dev, dtype=torch.float32, device=device)
    best_loss, best_state, bad = float("inf"), None, 0
    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 4096):
            idx = perm[start:start + 4096]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        sched.step(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate(model, X, device):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    panel = load_all_symbols()
    print(f"Pooled panel: {len(panel)} rows across {panel['symbol'].nunique()} symbols "
          f"({panel['timestamp'].min()} .. {panel['timestamp'].max()})")
    panel = cross_sectional_zscore(panel, FEATURE_COLS)

    ts = panel["timestamp"]
    train_all = panel[ts <= TRAIN_END]
    dev_all = panel[(ts >= DEV_START) & (ts <= DEV_END)]
    val_all = panel[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_all = panel[ts >= OOS_START]
    print(f"POOLED: TRAIN={len(train_all)} DEV={len(dev_all)} VAL={len(val_all)} OOS={len(oos_all)}")

    direction_eth = favored_direction(train_all[train_all["symbol"] == TARGET_SYMBOL])
    print(f"ETH favored direction on TRAIN: {direction_eth}")

    def win(sub, direction):
        return (sub["trade_outcome_action"] == direction).astype(int).to_numpy()

    # --- Pooled (breadth) model: trained on ALL 60 symbols, evaluated on ETH's own rows ---
    y_train_pooled = win(train_all, direction_eth)  # NOTE: uses ETH's favored direction as the
    # target for every symbol's row too, since we want one binary target to pool on; symbols
    # whose own favored direction differs will look like harder/noisier training rows for this
    # target, which is conservative (works against the pooled model, not for it).
    X_train_pooled = train_all[FEATURE_COLS].to_numpy()
    X_dev_pooled = dev_all[FEATURE_COLS].to_numpy()
    y_dev_pooled = win(dev_all, direction_eth)

    eth_dev = dev_all[dev_all["symbol"] == TARGET_SYMBOL]
    eth_val = val_all[val_all["symbol"] == TARGET_SYMBOL]
    eth_oos = oos_all[oos_all["symbol"] == TARGET_SYMBOL]
    eth_train = train_all[train_all["symbol"] == TARGET_SYMBOL]
    print(f"ETH-only rows: TRAIN={len(eth_train)} DEV={len(eth_dev)} VAL={len(eth_val)} OOS={len(eth_oos)}")

    print("\n=== POOLED (60-symbol breadth) model, evaluated on ETH's held-out rows ===")
    pooled_mlp = train_mlp(X_train_pooled, y_train_pooled, X_dev_pooled, y_dev_pooled, device)
    for name, sub in [("ETH-DEV", eth_dev), ("ETH-VAL", eth_val), ("ETH-OOS", eth_oos)]:
        X = sub[FEATURE_COLS].to_numpy()
        y = win(sub, direction_eth)
        prob = evaluate(pooled_mlp, X, device)
        print(f"  {name}: AUC={roc_auc_score(y, prob):.4f}  n={len(y)}  win_rate={y.mean():.4f}")

    print("\n=== ETH-ONLY model, same architecture/features, no pooling (the control) ===")
    X_train_eth = eth_train[FEATURE_COLS].to_numpy()
    y_train_eth = win(eth_train, direction_eth)
    X_dev_eth = eth_dev[FEATURE_COLS].to_numpy()
    y_dev_eth = win(eth_dev, direction_eth)
    eth_only_mlp = train_mlp(X_train_eth, y_train_eth, X_dev_eth, y_dev_eth, device)
    for name, sub in [("ETH-DEV", eth_dev), ("ETH-VAL", eth_val), ("ETH-OOS", eth_oos)]:
        X = sub[FEATURE_COLS].to_numpy()
        y = win(sub, direction_eth)
        prob = evaluate(eth_only_mlp, X, device)
        print(f"  {name}: AUC={roc_auc_score(y, prob):.4f}  n={len(y)}  win_rate={y.mean():.4f}")

    print("\n=== POOLED LightGBM (GBDT breadth reference, same pooled training set) ===")
    gbm_pooled = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                    min_child_samples=200, random_state=270705, verbosity=-1)
    gbm_pooled.fit(X_train_pooled, y_train_pooled)
    for name, sub in [("ETH-DEV", eth_dev), ("ETH-VAL", eth_val), ("ETH-OOS", eth_oos)]:
        X = sub[FEATURE_COLS].to_numpy()
        y = win(sub, direction_eth)
        prob = gbm_pooled.predict_proba(X)[:, 1]
        print(f"  {name}: AUC={roc_auc_score(y, prob):.4f}  n={len(y)}  win_rate={y.mean():.4f}")


if __name__ == "__main__":
    main()

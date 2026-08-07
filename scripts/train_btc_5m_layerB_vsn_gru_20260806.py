"""BTC 5m Layer B v8: TFT-style Variable Selection Network (VSN) + GRU.

Per literature review (Oxford large-scale financial DL benchmark, 2603.01820; Adaptive TFT for
crypto, 2509.10542): generic LSTM/GRU loses to GBDT on large numeric tabular data (matches this
session's own plain-GRU result, acc 51.7% OOS vs LGBM's 63.3%), but VSN+LSTM hybrids that
explicitly gate WHICH features matter at each timestep close the gap and sometimes win on
Sharpe/downside metrics. This is the structural piece our plain GRU was missing -- it saw all 115
raw features with equal weight every timestep.

VSN (simplified, faithful to Lim et al. 2019's mechanism): per-timestep, a small GRN computes a
softmax over the d input features (context-dependent feature-selection weights), each feature is
also embedded via a shared per-scalar linear, and the GRU only sees the weighted sum -- forcing
the network to learn sparse, time-varying attention over features instead of a flat concatenation.

Runs a small sweep (seq_len x hidden x lr) and reports all configs; picks the VAL-best for full
standalone + note for combined backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    seq_len: int = 12
    d_model: int = 32
    hidden: int = 64
    vsn_hidden: int = 64
    dropout: float = 0.2
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 35
    patience: int = 6
    seed: int = 20260806


class VSNGru(nn.Module):
    def __init__(self, n_features: int, cfg: Config) -> None:
        super().__init__()
        self.n_features = n_features
        self.feature_linear = nn.Linear(1, cfg.d_model)
        self.selection = nn.Sequential(
            nn.Linear(n_features, cfg.vsn_hidden), nn.ELU(),
            nn.Linear(cfg.vsn_hidden, n_features),
        )
        self.gru = nn.GRU(cfg.d_model, cfg.hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 3)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: (batch, seq, n_features)
        weights = torch.softmax(self.selection(x), dim=-1)  # (batch, seq, n_features)
        embedded = self.feature_linear(x.unsqueeze(-1))  # (batch, seq, n_features, d_model)
        selected = (embedded * weights.unsqueeze(-1)).sum(dim=2)  # (batch, seq, d_model)
        _, h = self.gru(selected)
        last = self.dropout(h[-1])
        logits = self.head(last)
        if return_weights:
            return logits, weights
        return logits


def _standardize_fit(x: np.ndarray) -> dict:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def _standardize_apply(x: np.ndarray, scaler: dict) -> np.ndarray:
    return np.nan_to_num(((x - scaler["mean"]) / scaler["std"]), nan=0.0).astype(np.float32)


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows_view: np.ndarray, y: np.ndarray, idx: np.ndarray) -> None:
        self.windows_view, self.y, self.idx = windows_view, y, idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        return self.windows_view[j].copy(), self.y[j]


def make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    n_features = X.shape[1]
    padded = np.vstack([np.zeros((seq_len - 1, n_features), dtype=np.float32), X])
    return sliding_window_view(padded, (seq_len, n_features))[:, 0, :, :]


def train_one(cfg: Config, X: np.ndarray, y: np.ndarray, fit_idx, holdout_idx, val_idx, oos_idx, n_features):
    torch.manual_seed(cfg.seed)
    windows = make_windows(X, cfg.seq_len)

    class_counts = np.bincount(y[fit_idx], minlength=3).astype(np.float32)
    class_weight = torch.tensor(class_counts.sum() / (3 * class_counts), device=DEVICE, dtype=torch.float32)

    model = VSNGru(n_features, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    fit_loader = torch.utils.data.DataLoader(WindowDataset(windows, y, fit_idx), batch_size=cfg.batch_size, shuffle=True)
    hold_loader = torch.utils.data.DataLoader(WindowDataset(windows, y, holdout_idx), batch_size=2048, shuffle=False)

    best_loss, best_state, patience_left = float("inf"), None, cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in fit_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb, weight=class_weight)
            loss.backward()
            opt.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in hold_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                losses.append(nn.functional.cross_entropy(model(xb), yb, weight=class_weight, reduction="sum").item())
        hold_loss = sum(losses) / len(holdout_idx)
        if hold_loss < best_loss - 1e-4:
            best_loss, best_state, patience_left = hold_loss, {k: v.clone() for k, v in model.state_dict().items()}, cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(idx):
        loader = torch.utils.data.DataLoader(WindowDataset(windows, y, idx), batch_size=4096, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    val_pred, oos_pred = predict(val_idx), predict(oos_idx)
    val_acc = (y[val_idx] == val_pred).mean()
    val_f1 = f1_score(y[val_idx], val_pred, average="macro")
    oos_acc = (y[oos_idx] == oos_pred).mean()
    oos_f1 = f1_score(y[oos_idx], oos_pred, average="macro")
    return {"best_hold_loss": best_loss, "val_acc": val_acc, "val_f1": val_f1, "oos_acc": oos_acc, "oos_f1": oos_f1}, model, windows, val_pred, oos_pred


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    dvol = build_dvol_features()
    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["zigzag_action"].to_numpy(dtype=np.int64)
    ts = df["timestamp"]

    train_mask = (ts < VAL_START).to_numpy()
    scaler = _standardize_fit(X_raw[train_mask])
    X = _standardize_apply(X_raw, scaler)
    n_features = X.shape[1]

    val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts < OOS_END)).to_numpy()
    val_idx, oos_idx = np.flatnonzero(val_mask), np.flatnonzero(oos_mask)

    rng = np.random.default_rng(20260806)
    sweep_results = []
    base = Config()
    grid = [
        replace(base, seq_len=12, hidden=64, lr=1e-3),
        replace(base, seq_len=24, hidden=64, lr=1e-3),
        replace(base, seq_len=12, hidden=32, lr=5e-4),
        replace(base, seq_len=24, hidden=128, lr=5e-4),
    ]
    for i, cfg in enumerate(grid):
        train_idx_all = np.flatnonzero(train_mask & (np.arange(len(df)) >= cfg.seq_len - 1))
        perm = rng.permutation(len(train_idx_all))
        n_holdout = max(int(0.10 * len(train_idx_all)), 1000)
        holdout_idx, fit_idx = train_idx_all[perm[:n_holdout]], train_idx_all[perm[n_holdout:]]
        print(f"\n### config {i}: seq_len={cfg.seq_len} hidden={cfg.hidden} d_model={cfg.d_model} lr={cfg.lr} ###")
        res, model, windows, val_pred, oos_pred = train_one(cfg, X, y, fit_idx, holdout_idx, val_idx, oos_idx, n_features)
        print(res)
        sweep_results.append((cfg, res, val_pred, oos_pred))

    best = max(sweep_results, key=lambda r: r[1]["val_f1"])
    best_cfg, best_res, best_val_pred, best_oos_pred = best
    print(f"\n### BEST config: seq_len={best_cfg.seq_len} hidden={best_cfg.hidden} lr={best_cfg.lr} -> {best_res}")

    maj_baseline = pd.Series(y[train_mask]).value_counts(normalize=True).max()
    for name, idx, pred in [("VAL", val_idx, best_val_pred), ("OOS", oos_idx, best_oos_pred)]:
        yt = y[idx]
        print(f"\n=== {name} (best config) ===")
        print(f"baseline={maj_baseline:.4f} acc={(yt==pred).mean():.4f} macro-F1={f1_score(yt,pred,average='macro'):.4f}")
        print(classification_report(yt, pred, target_names=["CASH", "LONG", "SHORT"], digits=3))

    out = df[["timestamp"]].copy()
    all_pred = np.full(len(df), -1, dtype=int)
    all_pred[val_idx] = best_val_pred
    all_pred[oos_idx] = best_oos_pred
    out["pred"] = all_pred
    out.to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_vsngru_pred.parquet", index=False)
    print("\nwrote predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

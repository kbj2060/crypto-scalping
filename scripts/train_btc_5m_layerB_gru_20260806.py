"""BTC 5m Layer B v7: small GRU sequence model (backlog item #2) -- every model tried so far
(LightGBM, TabM single-head, TabM 3-head) sees ONE bar of features at a time. This gives Layer B
the last SEQ_LEN bars (1h lookback at 5m) as an actual sequence, testing whether direction needs
shape-over-time (how the indicators evolved over the last hour, not just their current values)
that single-row tabular models structurally can't represent.
"""
from __future__ import annotations

from dataclasses import dataclass
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
SEQ_LEN = 12  # 1h lookback at 5m


@dataclass(frozen=True)
class GRUConfig:
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.2
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 40
    patience: int = 6
    seed: int = 20260806


class GRUDirection(nn.Module):
    def __init__(self, n_features: int, *, cfg: GRUConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                           dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)  # h: (layers, batch, hidden)
        last = self.dropout(h[-1])
        return self.head(last)


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
    """Indexes into a shared (n, seq_len, n_features) sliding-window VIEW without copying until
    a batch is actually materialized."""
    def __init__(self, windows_view: np.ndarray, y: np.ndarray, idx: np.ndarray) -> None:
        self.windows_view = windows_view
        self.y = y
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        return self.windows_view[j].copy(), self.y[j]


def main() -> int:
    cfg = GRUConfig()
    torch.manual_seed(cfg.seed)
    print(f"device={DEVICE}")

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
    X = _standardize_apply(X_raw, scaler)  # (n, n_features)

    # sliding window VIEW: windows[i] = X[i-SEQ_LEN+1 : i+1]  (label at i uses bars up to and incl i)
    n_features = X.shape[1]
    padded = np.vstack([np.zeros((SEQ_LEN - 1, n_features), dtype=np.float32), X])
    windows = sliding_window_view(padded, (SEQ_LEN, n_features))[:, 0, :, :]  # (n, SEQ_LEN, n_features)
    print(f"windows shape={windows.shape}")

    val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts < OOS_END)).to_numpy()

    valid_start = SEQ_LEN - 1  # need at least SEQ_LEN real bars of history
    train_idx_all = np.flatnonzero(train_mask & (np.arange(len(df)) >= valid_start))
    val_idx = np.flatnonzero(val_mask)
    oos_idx = np.flatnonzero(oos_mask)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(train_idx_all))
    n_holdout = max(int(0.10 * len(train_idx_all)), 1000)
    holdout_idx, fit_idx = train_idx_all[perm[:n_holdout]], train_idx_all[perm[n_holdout:]]
    print(f"fit={len(fit_idx)} holdout={len(holdout_idx)} val={len(val_idx)} oos={len(oos_idx)}")

    class_counts = np.bincount(y[fit_idx], minlength=3).astype(np.float32)
    class_weight = torch.tensor(class_counts.sum() / (3 * class_counts), device=DEVICE, dtype=torch.float32)

    model = GRUDirection(n_features, cfg=cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    fit_ds = WindowDataset(windows, y, fit_idx)
    hold_ds = WindowDataset(windows, y, holdout_idx)
    fit_loader = torch.utils.data.DataLoader(fit_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    hold_loader = torch.utils.data.DataLoader(hold_ds, batch_size=2048, shuffle=False, num_workers=0)

    best_loss, best_state, patience_left = float("inf"), None, cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=class_weight)
            loss.backward()
            opt.step()
        model.eval()
        hold_losses = []
        with torch.no_grad():
            for xb, yb in hold_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                hold_losses.append(nn.functional.cross_entropy(model(xb), yb, weight=class_weight, reduction="sum").item())
        hold_loss = sum(hold_losses) / len(holdout_idx)
        print(f"epoch {epoch}: hold_loss={hold_loss:.4f}")
        if hold_loss < best_loss - 1e-4:
            best_loss, best_state, patience_left = hold_loss, {k: v.clone() for k, v in model.state_dict().items()}, cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(idx: np.ndarray) -> np.ndarray:
        ds = WindowDataset(windows, y, idx)
        loader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    maj_baseline = pd.Series(y[fit_idx]).value_counts(normalize=True).max()
    all_pred = np.full(len(df), -1, dtype=int)
    for name, idx in [("VAL", val_idx), ("OOS", oos_idx)]:
        yp = predict(idx)
        all_pred[idx] = yp
        yt = y[idx]
        acc = (yt == yp).mean()
        f1m = f1_score(yt, yp, average="macro")
        print(f"\n=== {name} (n={len(idx)}) ===")
        print(f"baseline={maj_baseline:.4f} acc={acc:.4f} macro-F1={f1m:.4f}")
        print(classification_report(yt, yp, target_names=["CASH", "LONG", "SHORT"], digits=3))

    out = df[["timestamp"]].copy()
    out["pred"] = all_pred
    out.to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_gru_pred.parquet", index=False)
    print("\nwrote predictions (train rows left as -1, unused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

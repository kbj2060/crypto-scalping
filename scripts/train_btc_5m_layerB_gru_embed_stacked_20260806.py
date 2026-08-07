"""BTC 5m Layer B v9: GRU hidden-EMBEDDING (not final prediction) stacked into LightGBM.

Per the LSTM+LightGBM+CatBoost hybrid literature (arXiv 2505.23084): the win comes from reusing
the sequence model's learned REPRESENTATION as GBDT input, not its raw classification output
(which is what the earlier Layer-A-probability stacking test used, and it barely moved the
needle). The GRU itself is a weak classifier here (51.7% OOS acc, see
train_btc_5m_layerB_gru_20260806.py) but its 64-dim hidden state might still encode SOME temporal
shape information the single-row LightGBM can't construct from instantaneous feature values alone.

Caveat: for engineering-time reasons this uses a single GRU fit on the full train split (not
K-fold out-of-fold) to generate train-row embeddings, so train-row embeddings carry mild in-sample
information the VAL/OOS embeddings don't -- watch for a train/holdout metric gap larger than the
other experiments as a symptom of this, not a genuine train-time edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 12


@dataclass(frozen=True)
class GRUConfig:
    hidden: int = 64
    dropout: float = 0.2
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 40
    patience: int = 6
    seed: int = 20260806


class GRUDirection(nn.Module):
    def __init__(self, n_features: int, cfg: GRUConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, cfg.hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 3)

    def forward(self, x, return_embed=False):
        _, h = self.gru(x)
        last = self.dropout(h[-1])
        logits = self.head(last)
        if return_embed:
            return logits, h[-1]
        return logits


def _standardize_fit(x):
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def _standardize_apply(x, scaler):
    return np.nan_to_num(((x - scaler["mean"]) / scaler["std"]), nan=0.0).astype(np.float32)


def build_dvol_features():
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows_view, y, idx):
        self.windows_view, self.y, self.idx = windows_view, y, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
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
    X = _standardize_apply(X_raw, scaler)
    n_features = X.shape[1]

    windows = np.vstack([np.zeros((SEQ_LEN - 1, n_features), dtype=np.float32), X])
    windows = sliding_window_view(windows, (SEQ_LEN, n_features))[:, 0, :, :]

    val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts < OOS_END)).to_numpy()
    train_idx_all = np.flatnonzero(train_mask & (np.arange(len(df)) >= SEQ_LEN - 1))

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(train_idx_all))
    n_holdout = max(int(0.10 * len(train_idx_all)), 1000)
    holdout_idx, fit_idx = train_idx_all[perm[:n_holdout]], train_idx_all[perm[n_holdout:]]

    class_counts = np.bincount(y[fit_idx], minlength=3).astype(np.float32)
    class_weight = torch.tensor(class_counts.sum() / (3 * class_counts), device=DEVICE, dtype=torch.float32)

    model = GRUDirection(n_features, cfg).to(DEVICE)
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
                print(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"GRU trained, best_hold_loss={best_loss:.4f}")

    # extract embeddings for ALL rows
    all_idx = np.arange(len(df))
    loader = torch.utils.data.DataLoader(WindowDataset(windows, y, all_idx), batch_size=4096, shuffle=False)
    embeds = []
    with torch.no_grad():
        for xb, _ in loader:
            _, emb = model(xb.to(DEVICE), return_embed=True)
            embeds.append(emb.cpu().numpy())
    embeds = np.concatenate(embeds, axis=0)  # (n, hidden)
    embed_cols = [f"gru_embed_{i}" for i in range(embeds.shape[1])]
    embed_df = pd.DataFrame(embeds, columns=embed_cols)

    # --- stacked LightGBM ---
    X_stacked = pd.concat([df[feature_cols].reset_index(drop=True), embed_df], axis=1)
    y_series = pd.Series(y)

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100, verbosity=-1)
    clf.fit(X_stacked[train_mask], y_series[train_mask])
    pred = clf.predict(X_stacked)
    df["pred"] = pred

    maj_baseline = y_series[train_mask].value_counts(normalize=True).max()
    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt, yp = y[mask], pred[mask]
        acc = (yt == yp).mean()
        f1m = f1_score(yt, yp, average="macro")
        print(f"\n=== {name} (n={mask.sum()}) ===")
        print(f"baseline={maj_baseline:.4f} acc={acc:.4f} macro-F1={f1m:.4f}")
        print(classification_report(yt, yp, target_names=["CASH", "LONG", "SHORT"], digits=3))

    imp = pd.Series(clf.feature_importances_, index=X_stacked.columns).sort_values(ascending=False)
    embed_ranks = [list(imp.index).index(c) for c in embed_cols]
    print(f"\nGRU embedding feature ranks (0=top, out of {len(imp)}): min={min(embed_ranks)} best-rank col={imp.index[min(embed_ranks)]}")
    print(imp.head(10).to_string())

    df[["timestamp", "pred"]].to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_gruembed_pred.parquet", index=False)
    print("\nwrote predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

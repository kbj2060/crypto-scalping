"""BTC 5m Layer A (transition detector), TabM instead of LightGBM.

Model choice: TabM ("Advancing Tabular Deep Learning with Parameter-Efficient Ensembling",
Gorishniy et al., ICLR 2025) -- a k-member ensemble MLP that shares backbone weights across
members but gives each its own input scale/bias (BatchEnsemble-style), producing deep-ensemble-
quality predictions at a fraction of the parameter/compute cost of a full ensemble. It's one of
the strongest tabular DL architectures in recent benchmarks (competitive with or beating GBDT),
and is this project's own established alternative to LightGBM for its live/near-live components
(h48qual, zig075's quality head) -- reusing the exact TabM block design from
scripts/train_eval_btc_v3_tabm_quality_stage1events_20260720.py::TabMQualityHead, just retargeted
to this session's Layer A binary target instead of that script's quality target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
PIVOT_PATH = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TabMConfig:
    k: int = 8
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.15
    batch_size: int = 512
    lr: float = 1.5e-3
    weight_decay: float = 3.0e-4
    max_epochs: int = 60
    patience: int = 8
    seed: int = 20260806


class TabMHead(nn.Module):
    def __init__(self, n_features: int, n_out: int, *, cfg: TabMConfig) -> None:
        super().__init__()
        self.k = cfg.k
        self.input_scale = nn.Parameter(torch.randn(cfg.k, n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(cfg.k, n_features))
        self.in_proj = nn.Linear(n_features, cfg.hidden)
        self.blocks = nn.ModuleList(nn.Linear(cfg.hidden, cfg.hidden) for _ in range(max(0, cfg.layers - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(cfg.k, cfg.hidden) * 0.03 + 1.0) for _ in range(max(0, cfg.layers - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hidden) for _ in range(max(0, cfg.layers)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        out = self.head(h)  # (batch, k, n_out)
        return out.mean(dim=1)


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


def main() -> int:
    cfg = TabMConfig()
    torch.manual_seed(cfg.seed)
    print(f"device={DEVICE}")

    panel = pd.read_parquet(PANEL_PATH)
    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "transition_soon"])
    dvol = build_dvol_features()
    df = panel.merge(piv, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["transition_soon"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["transition_soon"].to_numpy(dtype=np.float32)

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)).to_numpy()
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()} base_rate(train)={y[train_mask].mean():.4f}")

    scaler = _standardize_fit(X_raw[train_mask])
    X = _standardize_apply(X_raw, scaler)

    rng = np.random.default_rng(cfg.seed)
    train_idx = np.flatnonzero(train_mask)
    perm = rng.permutation(len(train_idx))
    n_holdout = max(int(0.10 * len(train_idx)), 1000)
    holdout_idx, fit_idx = train_idx[perm[:n_holdout]], train_idx[perm[n_holdout:]]

    pos_weight = torch.tensor([(1 - y[fit_idx].mean()) / max(y[fit_idx].mean(), 1e-6)], device=DEVICE)
    model = TabMHead(len(feature_cols), 1, cfg=cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.from_numpy(X[fit_idx]).to(DEVICE)
    yt = torch.from_numpy(y[fit_idx]).to(DEVICE)
    xh = torch.from_numpy(X[holdout_idx]).to(DEVICE)
    yh = torch.from_numpy(y[holdout_idx]).to(DEVICE)

    best_loss, best_state, patience_left = float("inf"), None, cfg.patience
    train_rng = np.random.default_rng(cfg.seed + 1)
    for epoch in range(cfg.max_epochs):
        model.train()
        fit_perm = train_rng.permutation(len(xt))
        for start in range(0, len(fit_perm), cfg.batch_size):
            idx = fit_perm[start:start + cfg.batch_size]
            opt.zero_grad()
            logits = model(xt[idx]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx], pos_weight=pos_weight)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hold_loss = float(nn.functional.binary_cross_entropy_with_logits(
                model(xh).squeeze(-1), yh, pos_weight=pos_weight).item())
        if hold_loss < best_loss - 1e-5:
            best_loss, best_state, patience_left = hold_loss, {k: v.clone() for k, v in model.state_dict().items()}, cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch}, best_hold_loss={best_loss:.4f}")
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        full_logits = model(torch.from_numpy(X).to(DEVICE)).squeeze(-1)
        probA = torch.sigmoid(full_logits).cpu().numpy()
    df["probA"] = probA

    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt_np, p = y[mask], probA[mask]
        auc = roc_auc_score(yt_np, p)
        ap = average_precision_score(yt_np, p)
        base_rate = yt_np.mean()
        thresh = np.quantile(p, 0.90)
        prec = yt_np[p >= thresh].mean()
        print(f"{name}: AUC={auc:.4f} AP={ap:.4f} (base={base_rate:.4f}) top-decile precision={prec:.4f}")

    df[["timestamp", "probA"]].to_parquet(ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_tabm_pred.parquet", index=False)
    print("wrote predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

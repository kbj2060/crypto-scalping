"""Stage 1 (Rho1 panel design, docs/btc_panel_crossasset_architecture_design_20260804.md):
pool all 60 panel symbols (data/panel/features/*.parquet, built by
build_panel_common_features_20260804.py) into one training set, train a from-scratch
transformer encoder (cross-time attention over an 8h window) with a symbol embedding and a
Point-Quantile-style distribution head, predicting the vol-normalized forward H-bar return.

This tests H2 from the design doc: does pooling ~60x more effective training rows (vs BTC alone)
produce a better BTC forecaster than a BTC-only-trained model of the same architecture, even
though Stage 0.5 (feature augmentation, H1) failed? Per the design doc's own scoping note added
2026-08-04, if this also comes back negative it should be read as a SECOND confirmation that this
data axis has no BTC signal, not a reason to scale the model up further.

Scope decision (documented, not hidden): full cross-symbol attention (all symbols at the same
timestamp attending to each other, i.e. a true iTransformer cross-variate layer) is deferred --
this first pass uses a learned per-symbol embedding instead, which is simpler to get training
end-to-end in one session but does not capture same-timestamp cross-symbol interaction the way
the design doc's Layer 1 describes. If this MVP shows a positive signal, upgrading to full
cross-symbol attention is the natural next step; if it doesn't, that upgrade is not worth building.

Target: vol-normalized forward H-bar log return (raw_fwd_ret / trailing realized_vol_288), so a
shared quantile scale is meaningful across symbols of very different volatility. At BTC-only eval
time (separate script), predictions are rescaled by BTC's own realized_vol_288 back to raw
price-move space for comparison against a GARCH/EWMA benchmark.
"""
from __future__ import annotations

import json
import math
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import purged_decision_mask  # noqa: E402

UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
FEATURES_DIR = ROOT / "data/panel/features"
CKPT_DIR = ROOT / "data/panel/ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "ret_1", "realized_vol_12", "realized_vol_48", "realized_vol_288",
    "rsi_14", "macd_hist", "bb_width_20", "atr_pct_14",
    "rvol_12", "rvol_48", "taker_buy_ratio",
    "hour_sin", "hour_cos",
    "funding_rate", "funding_roc_288",
    "oi_chg_288", "toptrader_ratio", "taker_long_short_vol_ratio",
]
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
WINDOW_L = 96       # 8h context
HORIZON_H = 48       # 4h forward target, matches h48qual convention elsewhere in this repo
STRIDE = 24          # one sample/2h/symbol; overlapping 8h windows add little independent data
VAL_START, OOS_START = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PanelWindowDataset(Dataset):
    def __init__(self, split: str):
        universe = json.loads(UNIVERSE_PATH.read_text())
        symbols = [row["symbol"] for row in universe["symbols"]]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}

        self.feats: dict[str, np.ndarray] = {}
        self.targets: dict[str, np.ndarray] = {}
        self.valid_starts: list[tuple[str, int]] = []

        for sym in symbols:
            df = pd.read_parquet(FEATURES_DIR / f"{sym}.parquet")
            ts = df["timestamp"]
            open_px = df["open"].to_numpy(dtype=np.float64)
            close = df["close"].to_numpy(dtype=np.float64)
            realized_vol_288 = df["realized_vol_288"].to_numpy(dtype=np.float64)
            fwd_ret = np.log(np.roll(close, -HORIZON_H) / np.roll(open_px, -1))
            fwd_ret[-HORIZON_H:] = np.nan
            # Floor the vol denominator (not just epsilon) -- illiquid symbols can have exact-zero
            # trailing realized vol over a stretch of stale/flat prints, which blew the ratio up to
            # +/-80 std-units and helped trigger the same NaN divergence noted above.
            vol_floor = 0.002 / math.sqrt(HORIZON_H)  # ~0.2%/bar floor, well below typical crypto vol
            target = fwd_ret / (np.maximum(realized_vol_288, vol_floor) * math.sqrt(HORIZON_H))
            target = np.clip(target, -15.0, 15.0)

            X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            # Blanket safety net on top of build_panel_common_features_20260804.py's own
            # per-column clipping -- a corrupt/extreme metrics-feed value caused an immediate
            # NaN divergence in the first training attempt (2026-08-04), traced to
            # taker_long_short_vol_ratio hitting 8.9e7 on a low-liquidity symbol.
            X = np.clip(X, -20.0, 20.0)
            self.feats[sym] = X
            self.targets[sym] = target.astype(np.float32)

            n = len(df)
            if split == "train":
                mask = purged_decision_mask(
                    ts, start=ts.iloc[0], end=VAL_START, horizon_bars=HORIZON_H
                )
            elif split == "val":
                mask = purged_decision_mask(
                    ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON_H
                )
            else:
                mask = (ts >= OOS_START).to_numpy()

            # nan_to_num above already guarantees every row of X is finite, so the only
            # per-row validity check needed here is the target (forward-looking, can't be
            # sanitized the same way -- NaN there means "not enough future bars yet").
            valid_target = ~np.isnan(target)
            candidate_idx = np.arange(WINDOW_L - 1, n - HORIZON_H, STRIDE)
            candidate_idx = candidate_idx[mask[candidate_idx] & valid_target[candidate_idx]]
            self.valid_starts.extend((sym, int(i)) for i in candidate_idx)

        self.symbols = symbols
        print(f"[{split}] {len(self.valid_starts)} windows across {len(symbols)} symbols", flush=True)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        sym, i = self.valid_starts[idx]
        window = self.feats[sym][i - WINDOW_L + 1:i + 1]
        target = self.targets[sym][i]
        sym_id = self.symbol_to_id[sym]
        return torch.from_numpy(window), sym_id, torch.tensor(target, dtype=torch.float32)


class Rho1Backbone(nn.Module):
    def __init__(self, n_features: int, n_symbols: int, d_model: int = 128, nhead: int = 4,
                 n_layers: int = 3, n_quantiles: int = len(QUANTILES)):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.symbol_emb = nn.Embedding(n_symbols, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, WINDOW_L, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                     dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_quantiles))

    def forward(self, x, sym_id):
        h = self.input_proj(x) + self.pos_emb
        sym_ctx = self.symbol_emb(sym_id).unsqueeze(1)
        h = h + sym_ctx
        h = self.encoder(h)
        pooled = h[:, -1, :]  # last-timestep representation
        return self.head(pooled)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    target = target.unsqueeze(1)
    errors = target - pred
    q = torch.tensor(quantiles, device=pred.device).unsqueeze(0)
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    for x, sym_id, target in loader:
        x, sym_id, target = x.to(DEVICE), sym_id.to(DEVICE), target.to(DEVICE)
        if training:
            optimizer.zero_grad()
        pred = model(x, sym_id)
        loss = pinball_loss(pred, target, QUANTILES)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> int:
    print(f"device: {DEVICE}", flush=True)
    t0 = time.time()
    train_ds = PanelWindowDataset("train")
    pooled_val_ds = PanelWindowDataset("val")
    btc_val_ds = copy.copy(pooled_val_ds)
    btc_val_ds.valid_starts = [item for item in pooled_val_ds.valid_starts if item[0] == "BTCUSDT"]
    print(f"datasets built in {time.time()-t0:.1f}s", flush=True)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    pooled_val_loader = DataLoader(pooled_val_ds, batch_size=1024, shuffle=False, num_workers=0)
    btc_val_loader = DataLoader(btc_val_ds, batch_size=1024, shuffle=False, num_workers=0)

    model = Rho1Backbone(n_features=len(FEATURE_COLS), n_symbols=len(train_ds.symbols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_val = float("inf")
    patience, bad_epochs = 3, 0
    for epoch in range(1, 9):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer)
        pooled_val_loss = run_epoch(model, pooled_val_loader, None)
        btc_val_loss = run_epoch(model, btc_val_loader, None)
        scheduler.step()
        print(f"epoch {epoch:2d}  train_pinball={train_loss:.5f}  "
              f"pooled_val_pinball={pooled_val_loss:.5f}  btc_val_pinball={btc_val_loss:.5f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if btc_val_loss < best_val - 1e-5:
            best_val = btc_val_loss
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "symbol_to_id": train_ds.symbol_to_id,
                        "feature_cols": FEATURE_COLS, "quantiles": QUANTILES,
                        "window_l": WINDOW_L, "horizon_h": HORIZON_H, "epoch": epoch,
                        "btc_val_pinball": btc_val_loss,
                        "pooled_val_pinball": pooled_val_loss,
                        "selection_metric": "btc_val_pinball"},
                       CKPT_DIR / "rho1_panel_backbone_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val={best_val:.5f})", flush=True)
                break

    print(f"done. best BTC val pinball loss: {best_val:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

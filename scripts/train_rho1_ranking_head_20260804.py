"""Stage 2 (Rho1 panel design) step 2: cross-sectional ranking head.

Reuses train_rho1_panel_backbone_20260804.py's exact backbone (same window, same per-symbol
common features, same pooling across all 60 symbols, same train/val split) but with a
single-scalar sigmoid output regressed against each symbol's own forward-H-bar cross-sectional
rank percentile (build_panel_rank_labels_20260804.py), instead of the Stage-1 quantile head's
vol-normalized-return target.

At inference (the Fresh-Forward backtest script), this model's BTC score at a given bar
approximates "where will BTC's forward return land in the cross-sectional distribution" -- a
softer, hopefully more stable proxy for direction than predicting the raw return itself, per the
design doc's Layer 2(B) rationale.
"""
from __future__ import annotations

import sys
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
import train_rho1_panel_backbone_20260804 as base  # noqa: E402

RANK_LABELS_PATH = ROOT / "data/panel/rank_labels_20260804.parquet"
CKPT_PATH = base.CKPT_DIR / "rho1_ranking_head_best.pt"


class Rho1RankDataset(Dataset):
    def __init__(self, split: str):
        import json
        universe = json.loads(base.UNIVERSE_PATH.read_text())
        symbols = [row["symbol"] for row in universe["symbols"]]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}

        rank_labels = pd.read_parquet(RANK_LABELS_PATH)

        self.feats: dict[str, np.ndarray] = {}
        self.targets: dict[str, np.ndarray] = {}
        self.valid_starts: list[tuple[str, int]] = []

        for sym in symbols:
            df = pd.read_parquet(base.FEATURES_DIR / f"{sym}.parquet")
            ts = df["timestamp"]
            X = df[base.FEATURE_COLS].to_numpy(dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -20.0, 20.0)
            self.feats[sym] = X

            sym_rank = rank_labels[rank_labels["symbol"] == sym][["timestamp", "rank_pct"]]
            merged = df[["timestamp"]].merge(sym_rank, on="timestamp", how="left")
            target = merged["rank_pct"].to_numpy(dtype=np.float32)
            self.targets[sym] = target

            n = len(df)
            if split == "train":
                mask = purged_decision_mask(
                    ts, start=ts.iloc[0], end=base.VAL_START, horizon_bars=base.HORIZON_H
                )
            elif split == "val":
                mask = purged_decision_mask(
                    ts, start=base.VAL_START, end=base.OOS_START, horizon_bars=base.HORIZON_H
                )
            else:
                mask = (ts >= base.OOS_START).to_numpy()

            valid_target = ~np.isnan(target)
            candidate_idx = np.arange(base.WINDOW_L - 1, n - base.HORIZON_H, base.STRIDE)
            candidate_idx = candidate_idx[mask[candidate_idx] & valid_target[candidate_idx]]
            self.valid_starts.extend((sym, int(i)) for i in candidate_idx)

        self.symbols = symbols
        print(f"[{split}] {len(self.valid_starts)} windows across {len(symbols)} symbols", flush=True)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        sym, i = self.valid_starts[idx]
        window = self.feats[sym][i - base.WINDOW_L + 1:i + 1]
        target = self.targets[sym][i]
        sym_id = self.symbol_to_id[sym]
        return torch.from_numpy(window), sym_id, torch.tensor(target, dtype=torch.float32)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    loss_fn = nn.MSELoss()
    for x, sym_id, target in loader:
        x, sym_id, target = x.to(base.DEVICE), sym_id.to(base.DEVICE), target.to(base.DEVICE)
        if training:
            optimizer.zero_grad()
        raw = model(x, sym_id).squeeze(-1)  # (batch, 1) -> (batch,)
        pred = torch.sigmoid(raw)
        loss = loss_fn(pred, target)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> int:
    print(f"device: {base.DEVICE}", flush=True)
    t0 = time.time()
    train_ds = Rho1RankDataset("train")
    pooled_val_ds = Rho1RankDataset("val")
    btc_val_ds = copy.copy(pooled_val_ds)
    btc_val_ds.valid_starts = [item for item in pooled_val_ds.valid_starts if item[0] == "BTCUSDT"]
    print(f"datasets built in {time.time()-t0:.1f}s", flush=True)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    pooled_val_loader = DataLoader(pooled_val_ds, batch_size=1024, shuffle=False, num_workers=0)
    btc_val_loader = DataLoader(btc_val_ds, batch_size=1024, shuffle=False, num_workers=0)

    model = base.Rho1Backbone(n_features=len(base.FEATURE_COLS), n_symbols=len(train_ds.symbols),
                              n_quantiles=1).to(base.DEVICE)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_val = float("inf")
    patience, bad_epochs = 3, 0
    btc_train_targets = np.array([
        train_ds.targets[symbol][i] for symbol, i in train_ds.valid_starts if symbol == "BTCUSDT"
    ])
    btc_val_targets = np.array([
        btc_val_ds.targets[symbol][i] for symbol, i in btc_val_ds.valid_starts
    ])
    btc_constant = float(btc_train_targets.mean())
    btc_constant_mse = float(np.mean((btc_val_targets - btc_constant) ** 2))
    print(f"BTC train-mean constant baseline: value={btc_constant:.5f} val_mse={btc_constant_mse:.5f}")
    for epoch in range(1, 9):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer)
        pooled_val_loss = run_epoch(model, pooled_val_loader, None)
        btc_val_loss = run_epoch(model, btc_val_loader, None)
        scheduler.step()
        print(f"epoch {epoch:2d}  train_mse={train_loss:.5f}  "
              f"pooled_val_mse={pooled_val_loss:.5f}  btc_val_mse={btc_val_loss:.5f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if btc_val_loss < best_val - 1e-6:
            best_val = btc_val_loss
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "symbol_to_id": train_ds.symbol_to_id,
                        "feature_cols": base.FEATURE_COLS, "window_l": base.WINDOW_L,
                        "horizon_h": base.HORIZON_H, "epoch": epoch,
                        "btc_val_mse": btc_val_loss, "pooled_val_mse": pooled_val_loss,
                        "btc_constant_baseline_mse": btc_constant_mse,
                        "selection_metric": "btc_val_mse"}, CKPT_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val={best_val:.5f})", flush=True)
                break

    print(f"done. best BTC val mse: {best_val:.5f}  constant baseline={btc_constant_mse:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

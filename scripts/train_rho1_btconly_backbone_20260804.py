"""Stage 1 fair-comparison baseline: train the EXACT SAME architecture as
train_rho1_panel_backbone_20260804.py (same window, horizon, quantiles, hyperparameters) on
BTCUSDT ALONE, so the panel-pretrained model's BTC OOS performance can be compared against a
same-architecture single-asset model, isolating whether pooling (H2) helps at all.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
import train_rho1_panel_backbone_20260804 as base  # noqa: E402

CKPT_PATH = base.CKPT_DIR / "rho1_btconly_backbone_best.pt"


class BTCOnlyDataset(base.PanelWindowDataset):
    def __init__(self, split: str):
        import json
        import numpy as np
        import pandas as pd
        self.symbol_to_id = {"BTCUSDT": 0}
        self.feats, self.targets, self.valid_starts = {}, {}, []
        sym = "BTCUSDT"
        df = pd.read_parquet(base.FEATURES_DIR / f"{sym}.parquet")
        ts = df["timestamp"]
        open_px = df["open"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        realized_vol_288 = df["realized_vol_288"].to_numpy(dtype=np.float64)
        import math
        fwd_ret = np.log(np.roll(close, -base.HORIZON_H) / np.roll(open_px, -1))
        fwd_ret[-base.HORIZON_H:] = np.nan
        vol_floor = 0.002 / math.sqrt(base.HORIZON_H)
        target = fwd_ret / (np.maximum(realized_vol_288, vol_floor) * math.sqrt(base.HORIZON_H))
        target = np.clip(target, -15.0, 15.0)

        X = df[base.FEATURE_COLS].to_numpy(dtype=__import__("numpy").float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -20.0, 20.0)
        self.feats[sym] = X
        self.targets[sym] = target.astype(np.float32)

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
        # No stride subsampling here -- BTC-only has ~60x fewer rows than the pooled set, so we
        # can afford every valid window and still finish an epoch quickly.
        candidate_idx = np.arange(base.WINDOW_L - 1, n - base.HORIZON_H, 1)
        candidate_idx = candidate_idx[mask[candidate_idx] & valid_target[candidate_idx]]
        self.valid_starts.extend((sym, int(i)) for i in candidate_idx)
        self.symbols = [sym]
        print(f"[{split}] {len(self.valid_starts)} windows (BTCUSDT only)", flush=True)


def main() -> int:
    print(f"device: {base.DEVICE}", flush=True)
    train_ds = BTCOnlyDataset("train")
    val_ds = BTCOnlyDataset("val")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = base.Rho1Backbone(n_features=len(base.FEATURE_COLS), n_symbols=1).to(base.DEVICE)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_val = float("inf")
    patience, bad_epochs = 2, 0
    for epoch in range(1, 9):
        t0 = time.time()
        train_loss = base.run_epoch(model, train_loader, optimizer)
        val_loss = base.run_epoch(model, val_loader, None)
        scheduler.step()
        print(f"epoch {epoch:2d}  train_pinball={train_loss:.5f}  val_pinball={val_loss:.5f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "symbol_to_id": {"BTCUSDT": 0},
                        "feature_cols": base.FEATURE_COLS, "quantiles": base.QUANTILES,
                        "window_l": base.WINDOW_L, "horizon_h": base.HORIZON_H, "epoch": epoch,
                        "val_pinball": val_loss}, CKPT_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val={best_val:.5f})", flush=True)
                break

    print(f"done. best val pinball loss: {best_val:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

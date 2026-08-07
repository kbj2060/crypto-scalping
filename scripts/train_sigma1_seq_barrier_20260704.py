#!/usr/bin/env python3
"""Sigma1: a NEW model lineage (not Omega/Alpha) built from this session's confirmed lessons.

Design rationale, mapped to documented findings in
docs/model_contracts/omega6_synthesis_v1_20260703_contract.md:
1. The only lever that ever produced a val+OOS pass was the persistence/hysteresis filter --
   i.e. temporal context matters. Sigma1 uses a causal GRU over a 48-bar (4h) window so the
   model can LEARN temporal persistence natively instead of having it bolted on.
2. The barrier-matched label (priority-1 test) was the right target but failed OOS due to
   per-bar path-noise overfit. Sigma1 uses a SMOOTHED version: a bar is labeled LONG/SHORT only
   if the per-bar barrier-matched label agrees for SMOOTH_BARS consecutive bars ending at that
   bar -- the exact follow-up recommended in the contract doc's priority-1 disposition.
3. The Omega L2 loader chain cannot use 2024 data (its base candidates file has no 2024
   version). Sigma1 reads data/splits/year_oos/*.csv directly, so it trains on 2024 + 2025
   (Jan-Sep) -- roughly double the data and a different market regime year, addressing the
   "materially larger training window" item from the v1 promotion-readiness verdict.
4. Level-like columns (raw prices/volumes/OI) are excluded -- 2024 price levels are far from
   2025/2026 levels and would be out-of-distribution under a train-fit standardizer. Only
   stationary engineered features are used.

Split discipline: train = 2024-01-01..2025-07-31, internal early-stop holdout =
2025-08-01..2025-09-30 (still inside the L2-convention train region, BEFORE the fresh-forward
validation boundary of 2025-10-01). Fresh-forward val (2025-10..12) is scored only by the
downstream gate-sweep script; OOS (2026-01..02) only by a one-shot check if val passes.
No lookahead: labels use offline future simulation (standard for label construction only);
model inputs at bar i are the trailing 48-bar feature window ending at i.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "sigma1_seq_barrier_20260704"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

FEATURE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_barrier_matched_20260704"

TRAIN_END = pd.Timestamp("2025-07-31 23:59:59")
HOLDOUT_START = pd.Timestamp("2025-08-01")
HOLDOUT_END = pd.Timestamp("2025-09-30 23:59:59")

WINDOW = 48
SMOOTH_BARS = 3
HIDDEN = 192
LAYERS = 2
DROPOUT = 0.10
BATCH = 512
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
EPOCHS = 10
PATIENCE = 3

# Level-like columns whose absolute scale drifts across years (price/volume/OI levels) --
# excluded so the train-fit standardizer stays in-distribution on 2026 data.
LEVEL_COLS = {
    "timestamp", "open", "high", "low", "close", "close_btc",
    "volume", "volume_btc", "quote_volume", "quote_volume_btc",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "trades",
}
FORBIDDEN_TOKENS = ("label", "target", "pnl", "zigzag", "wave3", "future", "teacher")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_year(year: int) -> pd.DataFrame:
    frame = pd.read_csv(FEATURE_FILES[year], parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    labels = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    merged = frame.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise RuntimeError(f"{year}: label join dropped rows ({len(frame)} -> {len(merged)})")
    return merged


def feature_columns(frames: list[pd.DataFrame]) -> list[str]:
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    cols = sorted(
        c for c in common
        if c not in LEVEL_COLS
        and c != "zigzag_action"
        and not any(tok in c.lower() for tok in FORBIDDEN_TOKENS)
    )
    if len(cols) < 100:
        raise RuntimeError(f"too few usable feature columns: {len(cols)}")
    return cols


def smooth_labels(raw: np.ndarray, smooth_bars: int) -> np.ndarray:
    """Label bar i with side s only if the per-bar barrier-matched label equals s for
    smooth_bars consecutive bars ending at i. Otherwise CASH. Debounces path-noise in the
    per-bar simulated-trade label (the diagnosed cause of priority-1's OOS failure)."""
    out = raw.copy()
    ok = raw != 0
    for k in range(1, smooth_bars):
        shifted = np.roll(raw, k)
        shifted[:k] = -99
        ok &= shifted == raw
    out[~ok] = 0
    return out


class Sigma1GRU(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, HIDDEN, num_layers=LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN // 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


def gather_windows(feat: np.ndarray, idx: np.ndarray, window: int) -> np.ndarray:
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    win_idx = idx[:, None] - offsets[None, :]
    return feat[win_idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260704)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR if not args.out_suffix else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    f24, f25 = load_year(2024), load_year(2025)
    cols = feature_columns([f24, f25, load_year(2026).head(100)])
    print(f"features: {len(cols)}", flush=True)

    # Concatenate 2024+2025 chronologically; window gathering never crosses the year boundary
    # (boundary indices are excluded from the sample set below).
    combined = pd.concat([f24, f25], ignore_index=True)
    ts = combined["timestamp"]
    feat = combined[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    raw_label = combined["zigzag_action"].to_numpy(dtype=np.int64)
    label = smooth_labels(raw_label, SMOOTH_BARS)

    year_arr = ts.dt.year.to_numpy()
    boundary_mask = np.ones(len(combined), dtype=bool)
    # exclude indices whose window would span the 2024->2025 file boundary
    year_change = np.flatnonzero(np.diff(year_arr) != 0) + 1
    for b in year_change:
        boundary_mask[b : b + WINDOW] = False

    train_mask = (ts <= TRAIN_END).to_numpy() & boundary_mask
    holdout_mask = ((ts >= HOLDOUT_START) & (ts <= HOLDOUT_END)).to_numpy() & boundary_mask
    valid_idx = np.arange(len(combined)) >= WINDOW - 1
    train_idx = np.flatnonzero(train_mask & valid_idx)
    holdout_idx = np.flatnonzero(holdout_mask & valid_idx)
    if int(args.max_train_samples) > 0:
        train_idx = train_idx[: int(args.max_train_samples)]
    print(f"train samples: {len(train_idx)}, holdout samples: {len(holdout_idx)}", flush=True)
    print(f"train label dist: {np.bincount(label[train_idx], minlength=3).tolist()}", flush=True)

    mean = feat[train_idx].mean(axis=0)
    std = feat[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    feat_z = np.clip((feat - mean[None, :]) / std[None, :], -10.0, 10.0).astype(np.float32)

    counts = np.bincount(label[train_idx], minlength=3).astype(np.float64)
    class_w = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) / 3.0, dtype=torch.float32).to(device)

    model = Sigma1GRU(len(cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rng = np.random.default_rng(int(args.seed))

    for epoch in range(int(args.epochs)):
        model.train()
        perm = rng.permutation(train_idx)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), BATCH):
            batch_idx = perm[start : start + BATCH]
            xb = torch.from_numpy(gather_windows(feat_z, batch_idx, WINDOW)).to(device)
            yb = torch.from_numpy(label[batch_idx]).to(device)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            h_losses = []
            for start in range(0, len(holdout_idx), BATCH * 4):
                bidx = holdout_idx[start : start + BATCH * 4]
                xb = torch.from_numpy(gather_windows(feat_z, bidx, WINDOW)).to(device)
                yb = torch.from_numpy(label[bidx]).to(device)
                h_losses.append(float(nn.functional.cross_entropy(model(xb), yb, weight=class_w).cpu()))
            hold_loss = float(np.mean(h_losses))
        print(f"epoch {epoch + 1}: train_loss={total_loss / max(n_batches, 1):.4f} holdout_loss={hold_loss:.4f}", flush=True)
        if hold_loss + 1e-6 < best_loss:
            best_loss = hold_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print("early stop", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle = {
        "model_id": MODEL_ID,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_cols": cols,
        "mean": mean,
        "std": std,
        "window": WINDOW,
        "smooth_bars": SMOOTH_BARS,
        "hidden": HIDDEN,
        "layers": LAYERS,
        "dropout": DROPOUT,
        "seed": int(args.seed),
        "best_holdout_loss": best_loss,
        "train_window": {"start": "2024-01-01", "end": str(TRAIN_END)},
        "holdout_window": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END)},
        "label_source": str(LABEL_DIR),
        "label_smoothing": f"{SMOOTH_BARS} consecutive bars must agree, else CASH",
    }
    torch.save(bundle, out_dir / "sigma1_bundle.pt")
    report: dict[str, Any] = {k: v for k, v in bundle.items() if k not in ("state_dict", "mean", "std")}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

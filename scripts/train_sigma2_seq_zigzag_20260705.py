#!/usr/bin/env python3
"""Sigma2: from-scratch m7-free architecture, designed against every documented finding in
docs/model_contracts/omega6_synthesis_v1_20260703_contract.md and sigma1_seq_barrier contract.

Design decisions, each tied to evidence:
1. LABEL = canonical zigzag_action swing label (tmp/causal_regen_20260516/zigzag_action_labels_20260531)
   -- the label behind the ONLY val+OOS pass in project history. The barrier-matched per-bar
   label (Sigma1, priority-1) is confirmed to overfit validation and flip sign OOS. A sequence
   model on the zigzag label is an untried combination.
2. ARCHITECTURE = causal GRU (64-bar window) -- persistence/temporal debouncing was the only
   lever that ever produced an OOS pass; a sequence encoder learns it natively, and the zigzag
   label's long segments suit sequence modeling better than the noisy per-bar barrier label did.
3. DATA = 2024-01-01..2025-06-30 training (18 months, ~2x the frozen winner's). 2024 is usable
   precisely BECAUSE m7 is gone (per user: m7's fit-on-2024/predict-2025 scheme was why 2024
   couldn't be used as direct training data; that scheme is now abandoned project-wide).
4. FEATURES = stationary base features + regime3 overlays (wide24 HMM current probs, cmamba-h6
   sidecar, stability-risk h6) -- all three exist for 2024/2025/2026 and were
   reproducibility-verified when extended (5.6e-16 / 2.4e-07 / 1.1e-16). EXCLUDED: all m7_*
   (unrecoverable), drift-confirmed formulas (ou_halflife, garch_vol_z, kel, dual_momentum,
   conf_patchtst, pred_patchtst), level-like columns (raw prices/volumes/OI -- 2024 levels are
   OOD vs 2026 under a train-fit standardizer), and NF ai_* (not built for 2024; avoids the
   drifted-PatchTST zone entirely).
5. SELECTION PROTOCOL = validation on 2025-07-01..12-31 (SIX months). The old 3-month Oct-Dec
   window has absorbed 5+ search rounds (900+ variants) and is statistically exhausted;
   Jul-Sep 2025 has never been used for selection by ANY prior round (it was inside older
   models' training spans, but this model does not train on it). Internal early-stop holdout =
   2025-05-01..06-30 (inside train region).
6. ONE-SHOT = 2026-03-02..06-30 (never touched by any model or search). Scored only if the
   pre-registered validation gates pass. 2026-01..02 may be reported as soft context only
   (peeked twice historically).
7. TWO SEEDS trained from the start; a config only counts as passing if BOTH seeds are
   cost1-positive at it (sign-consistency requirement, pre-registered).
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

MODEL_ID = "sigma2_seq_zigzag_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

FEATURE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
}
OVERLAYS = {
    "wide24": (
        ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530",
        "training_features_{year}_regime3_current_sensitive_hmm_wide24.csv",
    ),
    "cmamba": (
        ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601",
        "training_features_{year}_regime3_cryptomamba_h6_sidecar_20260601.csv",
    ),
    "risk": (
        ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530",
        "training_features_{year}_regime3_stability_risk_h6.csv",
    ),
}
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"

TRAIN_END = pd.Timestamp("2025-04-30 23:59:59")
HOLDOUT_START = pd.Timestamp("2025-05-01")
HOLDOUT_END = pd.Timestamp("2025-06-30 23:59:59")

WINDOW = 64
HIDDEN = 192
LAYERS = 2
DROPOUT = 0.10
BATCH = 512
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
EPOCHS = 10
PATIENCE = 3

LEVEL_COLS = {
    "timestamp", "open", "high", "low", "close", "close_btc",
    "volume", "volume_btc", "quote_volume", "quote_volume_btc",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "trades",
}
DRIFTED_COLS = {"ou_halflife", "garch_vol_z", "kel", "dual_momentum", "conf_patchtst", "pred_patchtst"}
FORBIDDEN_TOKENS = ("label", "target", "pnl", "zigzag", "wave3", "future", "teacher", "m7_")
STRING_COLS = {"regime3_cmamba_h6_sidecar_class_name"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_year(year: int) -> pd.DataFrame:
    frame = pd.read_csv(FEATURE_FILES[year], parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    year_tag = str(year) if year != 2026 else "2026_rebuilt"
    for _name, (dir_path, pattern) in OVERLAYS.items():
        path = dir_path / pattern.format(year=year_tag)
        overlay = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        cols = [c for c in overlay.columns if c != "timestamp" and c not in STRING_COLS]
        frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    labels = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    frame = frame.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    return frame


def feature_columns(frames: list[pd.DataFrame]) -> list[str]:
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    cols = sorted(
        c for c in common
        if c not in LEVEL_COLS
        and c not in DRIFTED_COLS
        and c not in STRING_COLS
        and c != "zigzag_action"
        and not any(tok in c.lower() for tok in FORBIDDEN_TOKENS)
    )
    if len(cols) < 100:
        raise RuntimeError(f"too few usable feature columns: {len(cols)}")
    return cols


class Sigma2GRU(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, HIDDEN, num_layers=LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN // 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


def gather_windows(feat: np.ndarray, idx: np.ndarray, window: int) -> np.ndarray:
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    return feat[idx[:, None] - offsets[None, :]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260705)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR if not args.out_suffix else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    f24, f25 = load_year(2024), load_year(2025)
    cols = feature_columns([f24, f25])
    print(f"features: {len(cols)}", flush=True)

    combined = pd.concat([f24, f25], ignore_index=True)
    ts = combined["timestamp"]
    feat_raw = combined[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Overlay warmup NaNs (cmamba seq_len, risk horizon tail at file edges): forward-fill within
    # the concatenated frame is unsafe across the year boundary; instead drop sample indices
    # whose CURRENT row has NaN (windows tolerate NaN in history via fillna(0) after z-scoring
    # is not ideal -- simplest safe choice: fillna(0) pre-standardization like Sigma1 did, since
    # affected rows are only the few file-edge rows).
    feat = feat_raw.fillna(0.0).to_numpy(dtype=np.float32)
    label = combined["zigzag_action"].to_numpy(dtype=np.int64)

    year_arr = ts.dt.year.to_numpy()
    boundary_mask = np.ones(len(combined), dtype=bool)
    for b in np.flatnonzero(np.diff(year_arr) != 0) + 1:
        boundary_mask[b : b + WINDOW] = False

    train_mask = (ts <= TRAIN_END).to_numpy() & boundary_mask
    holdout_mask = ((ts >= HOLDOUT_START) & (ts <= HOLDOUT_END)).to_numpy() & boundary_mask
    valid_idx = np.arange(len(combined)) >= WINDOW - 1
    train_idx = np.flatnonzero(train_mask & valid_idx)
    holdout_idx = np.flatnonzero(holdout_mask & valid_idx)
    if int(args.max_train_samples) > 0:
        train_idx = train_idx[: int(args.max_train_samples)]
    print(f"train samples: {len(train_idx)}, holdout: {len(holdout_idx)}", flush=True)
    print(f"train label dist: {np.bincount(label[train_idx], minlength=3).tolist()}", flush=True)

    mean = feat[train_idx].mean(axis=0)
    std = feat[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    feat_z = np.clip((feat - mean[None, :]) / std[None, :], -10.0, 10.0).astype(np.float32)

    counts = np.bincount(label[train_idx], minlength=3).astype(np.float64)
    class_w = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) / 3.0, dtype=torch.float32).to(device)

    model = Sigma2GRU(len(cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rng = np.random.default_rng(int(args.seed))

    for epoch in range(int(args.epochs)):
        model.train()
        perm = rng.permutation(train_idx)
        total, nb = 0.0, 0
        for start in range(0, len(perm), BATCH):
            bidx = perm[start : start + BATCH]
            xb = torch.from_numpy(gather_windows(feat_z, bidx, WINDOW)).to(device)
            yb = torch.from_numpy(label[bidx]).to(device)
            loss = nn.functional.cross_entropy(model(xb), yb, weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu())
            nb += 1
        model.eval()
        with torch.no_grad():
            hl = []
            for start in range(0, len(holdout_idx), BATCH * 4):
                bidx = holdout_idx[start : start + BATCH * 4]
                xb = torch.from_numpy(gather_windows(feat_z, bidx, WINDOW)).to(device)
                yb = torch.from_numpy(label[bidx]).to(device)
                hl.append(float(nn.functional.cross_entropy(model(xb), yb, weight=class_w).cpu()))
            hold_loss = float(np.mean(hl))
        print(f"epoch {epoch + 1}: train_loss={total / max(nb, 1):.4f} holdout_loss={hold_loss:.4f}", flush=True)
        if hold_loss + 1e-6 < best_loss:
            best_loss, stale = hold_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        "hidden": HIDDEN,
        "layers": LAYERS,
        "seed": int(args.seed),
        "best_holdout_loss": best_loss,
        "train_window": {"start": "2024-01-01", "end": str(TRAIN_END)},
        "holdout_window": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END)},
        "label_source": str(LABEL_DIR),
        "excluded": sorted(DRIFTED_COLS) + ["m7_* (all)", "NF ai_* (not built for 2024)", "level cols"],
    }
    torch.save(bundle, out_dir / "sigma2_bundle.pt")
    report: dict[str, Any] = {k: v for k, v in bundle.items() if k not in ("state_dict", "mean", "std")}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("model_id", "seed", "best_holdout_loss")}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

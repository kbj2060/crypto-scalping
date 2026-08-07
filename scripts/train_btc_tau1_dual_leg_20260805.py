"""Train Tau1 Leg A/B checkpoints without looking at calibration or OOS PnL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.btc_tau1_dual_leg_architecture_20260805 import (
    LEG_A_HORIZON_BARS, LEG_A_SEQUENCE, LEG_B_HORIZON_BARS, LEG_B_SEQUENCE,
    LegANet, LegBNet, hourly_completed_features, join_targets, load_feature_frame, purged_splits,
)

OUT = ROOT / "tmp/btc_tau1_dual_leg_training_20260805"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceDataset(Dataset):
    def __init__(self, market: np.ndarray, regime: np.ndarray, indices: np.ndarray, labels: np.ndarray, length: int) -> None:
        self.market = torch.from_numpy(market)
        self.regime = torch.from_numpy(regime)
        self.indices = torch.from_numpy(indices.astype(np.int64))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.length = length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        end = int(self.indices[row]) + 1
        return self.market[end - self.length:end], self.regime[end - self.length:end], self.labels[row]


def _epoch(model, loader, weights, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for market, regime, labels in loader:
        market, regime, labels = market.to(DEVICE), regime.to(DEVICE), labels.to(DEVICE)
        if optimizer is not None:
            optimizer.zero_grad()
        loss = F.cross_entropy(model(market, regime), labels, weight=weights)
        if optimizer is not None:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total += float(loss.item()) * len(labels); count += len(labels)
    return total / max(count, 1)


def train_leg(leg: str, max_epochs: int) -> dict:
    frame, market_cols, regime_cols = load_feature_frame()
    if leg == "B":
        frame = hourly_completed_features(frame, market_cols, regime_cols)
        regime_cols = [f"regime_input_{column}" for column in regime_cols]
    sequence, horizon, model = (LEG_A_SEQUENCE, LEG_A_HORIZON_BARS, LegANet()) if leg == "A" else (LEG_B_SEQUENCE, LEG_B_HORIZON_BARS // 12, LegBNet())
    target, _ = join_targets(frame.timestamp, leg)
    raw_market = frame[market_cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    raw_regime = frame[regime_cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    splits = purged_splits(frame.timestamp, horizon)
    # A row is only usable as a window END if the FULL trailing `sequence`-length
    # window is finite, not just the end row itself -- some columns (DVOL/on-chain
    # before their first available observation, mtf1h_ts_opt_L when trend-scan
    # doesn't converge) have NaN that forward-fill in load_feature_frame() cannot
    # backfill before their first-ever valid value, so a window can still contain a
    # NaN row even when its own end row is finite.
    finite_row = np.isfinite(raw_market).all(1) & np.isfinite(raw_regime).all(1)
    window_finite = (
        pd.Series(finite_row).rolling(sequence, min_periods=sequence).min().fillna(0).astype(bool).to_numpy()
    )
    ready = window_finite & (target >= 0)
    ready[:sequence - 1] = False
    rows = {name: np.flatnonzero(mask & ready) for name, mask in splits.items()}
    if not len(rows["train"]) or not len(rows["checkpoint"]):
        raise RuntimeError(f"{leg} has empty train/checkpoint split")
    mean_m, std_m = raw_market[rows["train"]].mean(0), raw_market[rows["train"]].std(0)
    mean_r, std_r = raw_regime[rows["train"]].mean(0), raw_regime[rows["train"]].std(0)
    std_m[std_m < 1e-6] = 1.0; std_r[std_r < 1e-6] = 1.0
    market = np.clip((raw_market - mean_m) / std_m, -10, 10).astype(np.float32)
    regime = np.clip((raw_regime - mean_r) / std_r, -10, 10).astype(np.float32)
    counts = np.bincount(target[rows["train"]], minlength=3).astype(np.float32)
    weights = 1.0 / np.sqrt(counts); weights /= weights.mean(); weights_t = torch.from_numpy(weights).to(DEVICE)
    train_loader = DataLoader(SequenceDataset(market, regime, rows["train"], target[rows["train"]], sequence), batch_size=256, shuffle=True)
    val_loader = DataLoader(SequenceDataset(market, regime, rows["checkpoint"], target[rows["checkpoint"]], sequence), batch_size=512)
    model = model.to(DEVICE); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, bad, history, best_state = float("inf"), 0, [], None
    for epoch in range(1, max_epochs + 1):
        train_loss = _epoch(model, train_loader, weights_t, optimizer)
        val_loss = _epoch(model, val_loader, weights_t)
        history.append({"epoch": epoch, "train_weighted_ce": train_loss, "checkpoint_weighted_ce": val_loss})
        print(f"leg={leg} epoch={epoch} train_ce={train_loss:.6f} checkpoint_ce={val_loss:.6f}", flush=True)
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint saved")
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"state": best_state, "market_columns": market_cols, "regime_columns": regime_cols, "market_mean": mean_m, "market_std": std_m, "regime_mean": mean_r, "regime_std": std_r, "sequence": sequence, "class_weights": weights, "history": history, "split_rows": {name: int(len(value)) for name, value in rows.items()}, "contracts": {"checkpoint_selected_on_weighted_ce_only": True, "calibration_or_oos_pnl_used_for_checkpoint": False, "future_rows_used_for_input": False, "split_targets_purged": True}}
    torch.save(payload, OUT / f"leg_{leg.lower()}_checkpoint.pt")
    report = {"leg": leg, "device": DEVICE, "best_checkpoint_weighted_ce": best, "epochs_run": len(history), "split_rows": payload["split_rows"], "class_counts_train": counts.astype(int).tolist(), "class_weights": weights.tolist(), "contracts": payload["contracts"]}
    (OUT / f"leg_{leg.lower()}_train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=["A", "B", "both"], default="both")
    parser.add_argument("--max-epochs", type=int, default=40)
    args = parser.parse_args()
    reports = [train_leg(leg, args.max_epochs) for leg in ("A", "B") if args.leg in (leg, "both")]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

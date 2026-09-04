#!/usr/bin/env python3
"""Follow-up to research_eth_sweep_v_rebound_shallow_xlstm_20260829.py's single-config run
(VAL AUC 0.6093/OOS 0.6385, REJECTED) -- per this project's own feedback_dl_needs_optimization_
before_failure_verdict ("a single lightly-tuned run is not enough evidence to call a DL approach
a failure"), that verdict was premature. Applies this project's OWN empirically-validated levers
from feedback_modern_dl_training_checklist for weak-signal financial labels, rather than guessing
blindly:
  - lr=2e-4 (10x lower than the first run's 2e-3): found in this repo to give a much wider/more
    stable good-performance window instead of a sharp peak-then-crash, in a similarly weak-signal
    setting (TabM optimizer investigation, 2026-08-16).
  - RAdam: found in this repo to be far more robust to early-stopping timing than plain Adam
    (widest good window, best late-epoch retention) in the same investigation.
  - AdamW (decoupled weight decay) as the control, replacing the first run's plain Adam+weight_decay
    (which is NOT the same regularization -- L2-via-gradient vs decoupled).
Full per-epoch val_auc curves are logged for every cell (the checklist's "diagnostic habit") so
the shape (smooth/plateau/oscillating/crashing) can be read directly, not just a best-checkpoint
summary.

2x2 sweep (optimizer x lr), single seed each (cheap-gate style, matching this project's own
convention for an initial tuning pass before any multi-seed confirmation) -- NOT yet a "properly
tuned falsification/promotion," just a fairer look before deciding whether more effort is
warranted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts/research_eth_sweep_v_rebound_shallow_xlstm_20260829.py"


def load_base():
    spec = importlib.util.spec_from_file_location("xlstm_base_20260829", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEED = 20260829
MAX_EPOCHS = 80
PATIENCE = 15
SWEEP = [
    {"name": "AdamW lr=2e-3 (original)", "opt": "adamw", "lr": 2e-3},
    {"name": "AdamW lr=2e-4", "opt": "adamw", "lr": 2e-4},
    {"name": "RAdam lr=2e-3", "opt": "radam", "lr": 2e-3},
    {"name": "RAdam lr=2e-4", "opt": "radam", "lr": 2e-4},
]


def make_optimizer(name: str, params, lr: float):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=1e-4)
    raise ValueError(name)


def run_one(base, cfg: dict, xt_train, yt_train, xt_val, y_val, xt_oos, y_oos, device: str) -> dict:
    torch.manual_seed(SEED)
    model = base.ShallowXLSTM(base.N_FEATURES, base.HIDDEN, base.N_LAYERS, base.DROPOUT).to(device)
    opt = make_optimizer(cfg["opt"], model.parameters(), cfg["lr"])

    best_val_auc, best_state, bad_epochs = -1.0, None, 0
    curve = []
    n_train = len(xt_train)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for start in range(0, n_train, base.BATCH_SIZE):
            idx = perm[start:start + base.BATCH_SIZE]
            opt.zero_grad()
            logits = model(xt_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_logits = model(xt_val).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_logits)
        curve.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc, best_state, bad_epochs = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(model(xt_val)).cpu().numpy()
        oos_proba = torch.sigmoid(model(xt_oos)).cpu().numpy()
    val_auc = roc_auc_score(y_val, val_proba)
    oos_auc = roc_auc_score(y_oos, oos_proba)
    return {
        "name": cfg["name"], "best_epoch": int(np.argmax(curve) + 1),
        "n_epochs_ran": len(curve), "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
        "curve": [round(float(v), 4) for v in curve],
    }


def main() -> int:
    base = load_base()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("building sequence dataset (shared across all sweep cells)...")
    x, y, timestamps_raw, labels_df = base.build_sequences()
    timestamps = pd.to_datetime(timestamps_raw, utc=True)
    window_end = timestamps + pd.Timedelta(minutes=30)
    train_mask = (timestamps < base.VAL_START) & (window_end < base.VAL_START)
    val_mask = (timestamps >= base.VAL_START) & (timestamps <= base.VAL_END) & (window_end < base.OOS_START)
    oos_mask = (timestamps >= base.OOS_START) & (timestamps <= base.OOS_END)
    print(f"train n={train_mask.sum()}  val n={val_mask.sum()}  oos n={oos_mask.sum()}")

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_oos, y_oos = x[oos_mask], y[oos_mask]
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0

    def to_tensor(arr):
        arr = (arr - mean) / std
        return torch.tensor(arr, dtype=torch.float32, device=device)

    xt_train = to_tensor(x_train)
    yt_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    xt_val = to_tensor(x_val)
    xt_oos = to_tensor(x_oos)

    results = []
    for cfg in SWEEP:
        print(f"\n=== {cfg['name']} ===")
        r = run_one(base, cfg, xt_train, yt_train, xt_val, y_val, xt_oos, y_oos, device)
        results.append(r)
        print(f"  ran {r['n_epochs_ran']} epochs, val_auc curve: {r['curve']}")
        print(f"  -> VAL AUC {r['val_auc']:.4f}  OOS AUC {r['oos_auc']:.4f}")

    print("\n=== SWEEP SUMMARY ===")
    for r in results:
        print(f"  {r['name']:28s} VAL {r['val_auc']:.4f}  OOS {r['oos_auc']:.4f}  ({r['n_epochs_ran']} epochs)")
    print("\n=== FOR COMPARISON ===")
    print("  TabPFN (current SOTA):  VAL AUC 0.6423   OOS AUC 0.6566")
    print("  GBM:                    VAL AUC 0.6222   OOS AUC 0.6425")
    print("  TabM (REJECTED):        VAL AUC 0.6108   OOS AUC 0.6232")
    best = max(results, key=lambda r: r["val_auc"])
    print(f"\nbest single-seed cell: {best['name']} (VAL {best['val_auc']:.4f} / OOS {best['oos_auc']:.4f}) "
          f"-- single seed only, would need N>=5 confirmation before treating as reliable "
          f"(this project's own optimizer-sweep precedent: single-seed winners have flipped under 5-seed re-test before).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

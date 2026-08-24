#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnoses WHY every faithful-TabM run's best validation checkpoint has landed
at epoch 1 (66/66 runs across Step A's single-seed cheap_gate + N>=5 seed reproduction all stopped
at exactly epoch 9 with patience=8, i.e. best_epoch=1 every single time -- see
docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md). User asked directly why.

Prior scripts only ever recorded the BEST epoch's metrics, never the full trajectory, so there was
no way to tell apart three different explanations from the summary numbers alone:
  (a) classic overfitting -- train loss keeps dropping past epoch 1 while val loss climbs
  (b) an optimization/LR issue -- BOTH train and val loss plateau immediately at epoch 1
  (c) validation-noise dominance -- the val split is small/noisy relative to signal strength, so
      epoch 1's "best" is largely a lucky draw and the val curve afterward is directionless noise,
      not a clean monotonic rise

This script logs EVERY epoch's train_loss (batch-averaged) and val_loss/dir_bacc for both
architectures on one expert/seed, disables early stopping (runs a fixed EPOCHS budget) so the full
post-epoch-1 trajectory is visible, and writes the whole curve to report.json for inspection.

Reuses research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816.py's _prepare_frames_light/
ThreeHeadTabMFull unmodified via import -- only the per-epoch logging and disabled early stopping
are new.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816 as nseed  # noqa: E402

gate = nseed.gate
base = gate.base
hard = gate.hard
CFG = gate.CFG
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816"
SEED = 260816
EXPERT = "bull"
EPOCHS = 40  # fixed budget, no early stopping -- want to SEE the trajectory past where patience=8 would have cut it off


def log(msg: str) -> None:
    print(f"[faithful_tabm_curve_diag] {msg}", flush=True)


def _fit_one_curve(model_cls, x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device):
    """Copy of gate._fit_one with early stopping REMOVED and every epoch's train+val metrics logged."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = base._standardize_fit(x_all)
    x_dir_np = base._standardize_apply(x_dir, scaler)
    x_exit_np = base._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = base._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = base._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)
    log(f"  n_dir_train={len(train_idx)} n_dir_val={len(val_idx)} n_exit_train={len(exit_train_idx)} n_exit_val={len(exit_val_idx)}")

    model = model_cls(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)

    curve = []
    for epoch in range(int(epochs)):
        model.train()
        exit_iter = iter(dl_exit)
        train_loss_sum, train_batches = 0.0, 0
        for xb, yb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            loss_qual_k = torch.nn.functional.cross_entropy(out_dir["quality"].reshape(-1, 3), yb[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1
        train_loss = train_loss_sum / max(1, train_batches)

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(CFG.quality_loss_weight) * vqual_loss + float(CFG.exit_loss_weight) * vex_loss
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
            # also compute TRAIN balanced accuracy so we can see if the model is even fitting the training data
            to = model(torch.from_numpy(x_dir_np[train_idx]).to(device))
            train_pred_k = torch.softmax(to["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            train_bacc = float(balanced_accuracy_score(y_dir_np[train_idx], train_pred_k))
        curve.append({
            "epoch": epoch + 1, "train_loss": round(train_loss, 5), "val_loss": round(vloss, 5),
            "direction_val_loss": round(vdir_loss, 5), "quality_val_loss": round(vqual_loss, 5), "exit_val_loss": round(vex_loss, 5),
            "direction_balanced_accuracy_val": round(bacc, 5), "direction_balanced_accuracy_train": round(train_bacc, 5),
        })
        log(f"    epoch={epoch+1:02d} train_loss={train_loss:.4f} val_loss={vloss:.4f} "
            f"dir_bacc_val={bacc:.4f} dir_bacc_train={train_bacc:.4f}")
    return curve


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} epochs_fixed={EPOCHS} early_stopping=disabled ===")
    frames = gate._prepare_frames_light()
    fee, slip = base.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = base._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=0,
    )
    x_exit = base._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)

    report: dict[str, Any] = {"design": "full per-epoch curve diagnostic, no early stopping, single seed/expert.", "seed": SEED, "expert": EXPERT, "epochs_fixed": EPOCHS, "curves": {}}
    for arch_name, model_cls in (("baseline_R_only", base.ThreeHeadTabM), ("full_R_S_B", gate.ThreeHeadTabMFull)):
        log(f"=== stage=train architecture={arch_name} ===")
        curve = _fit_one_curve(model_cls, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device)
        report["curves"][arch_name] = curve
        best = min(curve, key=lambda r: r["val_loss"])
        log(f"  {arch_name}: best val_loss at epoch {best['epoch']} ({best['val_loss']:.4f}); "
            f"epoch1 val_loss={curve[0]['val_loss']:.4f}; final(epoch{EPOCHS}) val_loss={curve[-1]['val_loss']:.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

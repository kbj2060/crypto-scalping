#!/usr/bin/env python3
"""RESEARCH ONLY -- best-combination grid on top of full_R_S_B + piecewise-linear embedding
(the "quarter" capacity config: hidden=96, d_embed=4, n_bins=8, 109,836 params -- fewer than the
118,552-param no-embedding baseline). User explicitly chose to keep this architecture as the fixed
base (docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md's Step B
re-verification found it's a wash vs baseline_R_only, avg delta -0.0016 across 3 experts -- not a
loss, not a win) and asked to find the best combination of the OTHER training-quality levers found
in this investigation on top of it, rather than switching back to the plainer baseline_R_only.

Fixed: architecture=full_R_S_B_embed[quarter], expert=bull, seed=260816, lr=2e-4 (this candidate's
best-evidenced LR), 40-epoch budget, NO early stopping (full curve logged, true peak taken --
avoids repeating the checkpoint-selection bug found earlier in this candidate).

Grid (2 losses x 3 optimizers = 6 runs): CE vs GCE(q=0.7, Zhang & Sabuncu arXiv:1805.07836) x
AdamW(control) vs RAdam(torch.optim.RAdam) vs AdaBelief(Zhuang et al. arXiv:2010.07468, from-scratch
implementation). None of these 6 exact combinations have been tested before -- GCE was previously
tested at lr=2e-3 on baseline_R_only, RAdam/AdaBelief were tested on baseline_R_only (not the
embedding architecture) -- this investigation's own repeated finding is that combining independently
-good techniques doesn't guarantee a combined win (GCE+ELR+mixup interaction), so this grid measures
the actual combination rather than assuming additivity.

Reuses ThreeHeadTabMFullEmbed/_quantile_bin_edges (stepb_embed), gce_loss (combo_regularizer), and
AdaBelief (optimizer_sweep) unmodified via import.
"""
from __future__ import annotations

import dataclasses
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

import research_eth_candidate_faithful_tabm_batchensemble_stepb_embed_20260816 as stepb  # noqa: E402
import research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816 as combo  # noqa: E402
import research_eth_candidate_faithful_tabm_batchensemble_optimizer_sweep_20260816 as optsweep  # noqa: E402

gate = stepb.gate
base = stepb.base
hard = stepb.hard
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_fullstepb_grid_20260816"
SEED = 260816
EXPERT = "bull"
EPOCHS = 40
LR = 2.0e-4
QUARTER = {"hidden": 96, "d_embed": 4, "n_bins": 8}
GCE_Q = combo.GCE_Q  # 0.7


def log(msg: str) -> None:
    print(f"[faithful_tabm_fullstepb_grid] {msg}", flush=True)


OPTIMIZERS = {
    "AdamW": lambda params, wd: torch.optim.AdamW(params, lr=LR, weight_decay=wd),
    "RAdam": lambda params, wd: torch.optim.RAdam(params, lr=LR, weight_decay=wd),
    "AdaBelief": lambda params, wd: optsweep.AdaBelief(params, lr=LR, weight_decay=wd),
}


def _fit_grid_cell(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device, cfg, opt_factory, use_gce):
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

    bin_edges = stepb._quantile_bin_edges(_x_np, QUARTER["n_bins"])
    model = stepb.ThreeHeadTabMFullEmbed(x_dir_np.shape[1], cfg=cfg, bin_edges=bin_edges, d_embed=QUARTER["d_embed"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = opt_factory(model.parameters(), float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)

    def cls_loss(logits_k, target):
        if use_gce:
            return combo.gce_loss(logits_k, target, q=GCE_Q)
        k = logits_k.shape[1]
        return torch.nn.functional.cross_entropy(logits_k.reshape(-1, logits_k.shape[-1]), target[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)

    curve = []
    for epoch in range(int(epochs)):
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out_dir = model(xb)
            loss_dir_k = cls_loss(out_dir["direction"], yb)
            loss_qual_k = cls_loss(out_dir["quality"], yb)
            out_exit = model(xe)
            loss_exit_k = cls_loss(out_exit["exit"], ye)
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(cfg.quality_loss_weight) * loss_qual + float(cfg.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vo = model(vx)
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
        curve.append({"epoch": epoch + 1, "direction_balanced_accuracy_val": round(bacc, 5)})
    return curve, n_params


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} lr={LR} config=quarter{QUARTER} grid=2losses x 3optimizers ===")
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
    cfg = dataclasses.replace(stepb.CFG, hidden=QUARTER["hidden"])

    report: dict[str, Any] = {"design": "full_R_S_B_embed[quarter] @ lr=2e-4, 2 losses x 3 optimizers grid, fixed epochs, no early stopping.", "seed": SEED, "expert": EXPERT, "lr": LR, "epochs_fixed": EPOCHS, "config": QUARTER, "cells": {}}
    t0 = time.time()
    for loss_name, use_gce in (("CE", False), ("GCE", True)):
        for opt_name, opt_factory in OPTIMIZERS.items():
            cell = f"{opt_name}+{loss_name}"
            log(f"=== stage=train cell={cell} ===")
            curve, n_params = _fit_grid_cell(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, cfg=cfg, opt_factory=opt_factory, use_gce=use_gce)
            report["cells"][cell] = {"n_params": n_params, "curve": curve}
            best = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
            log(f"  {cell}: peak val_bacc={best['direction_balanced_accuracy_val']:.4f} @epoch{best['epoch']}/{EPOCHS}; "
                f"final(epoch{EPOCHS})={curve[-1]['direction_balanced_accuracy_val']:.4f} (elapsed={time.time()-t0:.0f}s)")

    log("=== stage=summary (reference: baseline_R_only true peak=0.5740@epoch2; full_R_S_B_embed[quarter]+AdamW+CE true peak=0.5723@epoch9) ===")
    ranked = sorted(report["cells"].items(), key=lambda kv: max(r["direction_balanced_accuracy_val"] for r in kv[1]["curve"]), reverse=True)
    for cell, d in ranked:
        best = max(d["curve"], key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {cell:16s} peak_val_bacc={best['direction_balanced_accuracy_val']:.4f} @epoch{best['epoch']} final={d['curve'][-1]['direction_balanced_accuracy_val']:.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

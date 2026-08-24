#!/usr/bin/env python3
"""RESEARCH ONLY -- upgraded version of research_eth_candidate_faithful_tabm_batchensemble_
fullstepb_grid_20260816.py: instead of a fixed 40-epoch/no-early-stopping budget (which was a
DIAGNOSTIC choice, to see the full true curve and avoid the checkpoint-selection bug found earlier
in this candidate), this applies two training-quality upgrades from this investigation's own
findings and asks:

1. **Cosine annealing LR schedule** instead of a flat LR: `CosineAnnealingLR` from `lr=2e-4` (this
   candidate's best-evidenced flat LR) down to `eta_min=2e-6` (this candidate's own tested LR floor)
   over `T_max=60` epochs. **T_max sized deliberately per a literature check (Loshchilov & Hutter,
   SGDR, arXiv:1608.03983)**: their own ablation shows single-cycle cosine beats multi-cycle warm
   restarts on final quality (not just anytime performance), so this uses one cycle, no restarts --
   but T_max must be LENGTH-MATCHED to the actual expected run length, not set arbitrarily large.
   An earlier draft of this script used T_max=100 with patience=20; a literature-grounded review
   caught that this was wrong -- with RAdam's own flat-LR peak already at epoch 27 and patience=20,
   most runs would stop around epoch 27-47, meaning a 100-epoch cosine schedule would have barely
   decayed (still near lr_max) by the time training actually stops, defeating the point of decay.
   T_max=60 lets the schedule reach a meaningfully low LR within the range runs are actually likely
   to use. (OneCycleLR/superconvergence, Smith arXiv:1803.09820, was also researched and deliberately
   NOT used here despite being validated on short horizons -- its core mechanism relies on a LARGE
   peak LR aiding generalization, which directly contradicts this candidate's own LR sweep finding
   that a HIGHER LR memorizes faster, not slower; SGDR warm restarts were also skipped, no single-run
   quality benefit shown in the source paper's own ablation.)
2. **Corrected, metric-aligned early stopping** instead of a fixed epoch count: stops on
   `direction_val_loss` alone (NOT the combined multi-task val_loss -- that criterion was found to
   diverge from the true direction_balanced_accuracy peak specifically for embedding architectures,
   docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md's "정정" section),
   patience=20 epochs, max budget of 60 epochs (matched to the schedule's T_max above). The full
   curve is still logged throughout for independent verification that the selected checkpoint's
   direction_balanced_accuracy roughly matches the true logged peak -- if it doesn't, that's evidence
   of yet another criterion mismatch and should be reported, not hidden.

Same 2x3 grid as the fixed-epoch version (CE vs GCE(q=0.7) x AdamW vs RAdam vs AdaBelief), same
architecture=full_R_S_B_embed[quarter] (109,836 params), same expert=bull/seed=260816.
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
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_fullstepb_grid_scheduled_20260816"
SEED = 260816
EXPERT = "bull"
LR_MAX = 2.0e-4
LR_MIN = 2.0e-6
MAX_EPOCHS = 60  # length-matched to T_max per SGDR literature check -- see module docstring
PATIENCE = 20
QUARTER = {"hidden": 96, "d_embed": 4, "n_bins": 8}
GCE_Q = combo.GCE_Q  # 0.7


def log(msg: str) -> None:
    print(f"[faithful_tabm_fullstepb_sched] {msg}", flush=True)


OPTIMIZERS = {
    "AdamW": lambda params, wd: torch.optim.AdamW(params, lr=LR_MAX, weight_decay=wd),
    "RAdam": lambda params, wd: torch.optim.RAdam(params, lr=LR_MAX, weight_decay=wd),
    "AdaBelief": lambda params, wd: optsweep.AdaBelief(params, lr=LR_MAX, weight_decay=wd),
}


def _fit_grid_cell_scheduled(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, device, cfg, opt_factory, use_gce):
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR_MIN)
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
    best_dir_val_loss = float("inf")
    best_epoch = 0
    best_bacc_at_selection = None
    stale = 0
    for epoch in range(MAX_EPOCHS):
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
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vo = model(vx)
            vdir_loss = float(torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1)).detach().cpu())
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
        curve.append({"epoch": epoch + 1, "lr": round(scheduler.get_last_lr()[0], 8), "direction_val_loss": round(vdir_loss, 5), "direction_balanced_accuracy_val": round(bacc, 5)})

        if vdir_loss + 1.0e-6 < best_dir_val_loss:
            best_dir_val_loss = vdir_loss
            best_epoch = epoch + 1
            best_bacc_at_selection = bacc
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    true_peak = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
    return {
        "n_params": n_params, "curve": curve, "epochs_ran": len(curve),
        "selected_epoch": best_epoch, "selected_dir_val_loss": best_dir_val_loss, "selected_bacc": best_bacc_at_selection,
        "true_peak_epoch": true_peak["epoch"], "true_peak_bacc": true_peak["direction_balanced_accuracy_val"],
        "selection_vs_true_peak_gap": round(true_peak["direction_balanced_accuracy_val"] - (best_bacc_at_selection or 0.0), 5),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} lr={LR_MAX}->{LR_MIN} cosine max_epochs={MAX_EPOCHS} patience={PATIENCE} config=quarter{QUARTER} ===")
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

    report: dict[str, Any] = {
        "design": "full_R_S_B_embed[quarter], cosine LR schedule + direction_val_loss-based early stopping, 2 losses x 3 optimizers grid.",
        "seed": SEED, "expert": EXPERT, "lr_max": LR_MAX, "lr_min": LR_MIN, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "config": QUARTER, "cells": {},
    }
    t0 = time.time()
    for loss_name, use_gce in (("CE", False), ("GCE", True)):
        for opt_name, opt_factory in OPTIMIZERS.items():
            cell = f"{opt_name}+{loss_name}"
            log(f"=== stage=train cell={cell} ===")
            result = _fit_grid_cell_scheduled(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, device=device, cfg=cfg, opt_factory=opt_factory, use_gce=use_gce)
            report["cells"][cell] = result
            log(f"  {cell}: epochs_ran={result['epochs_ran']}/{MAX_EPOCHS} selected_epoch={result['selected_epoch']} selected_bacc={result['selected_bacc']:.4f} | "
                f"true_peak_epoch={result['true_peak_epoch']} true_peak_bacc={result['true_peak_bacc']:.4f} | "
                f"selection_gap={result['selection_vs_true_peak_gap']:+.4f} (elapsed={time.time()-t0:.0f}s)")

    log("=== stage=summary (ranked by selected_bacc -- i.e. what a real early-stopping run would actually deliver) ===")
    ranked = sorted(report["cells"].items(), key=lambda kv: kv[1]["selected_bacc"], reverse=True)
    for cell, d in ranked:
        log(f"  {cell:16s} selected_bacc={d['selected_bacc']:.4f} @epoch{d['selected_epoch']} (true_peak={d['true_peak_bacc']:.4f} @epoch{d['true_peak_epoch']}, gap={d['selection_vs_true_peak_gap']:+.4f}) epochs_ran={d['epochs_ran']}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

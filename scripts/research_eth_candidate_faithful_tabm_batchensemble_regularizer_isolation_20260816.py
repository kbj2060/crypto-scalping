#!/usr/bin/env python3
"""RESEARCH ONLY -- isolates the 3 techniques combined in
research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816.py (GCE+ELR+latent-
mixup, which underperformed plain CE: peak val_bacc 0.537@epoch8 vs plain CE's 0.574@epoch2 -- see
docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md) to distinguish 3 candidate
explanations for that negative combined result:
  (a) untuned defaults (q=0.7/lambda=3/beta=0.7/alpha=1.0 are generic literature values, not tuned
      for this project's weak-signal financial label)
  (b) the 3 techniques interfere with each other when stacked (e.g. mixup's blended targets
      conflicting with ELR's EMA-of-own-past-predictions target)
  (c) this whole noise-robust-DL technique family just doesn't fit THIS project's regime (near-zero
      label-feature mutual information, not class-conditional annotation noise -- consistent with
      [[repo_label_methodology_meta_finding]])

This script runs each technique ALONE (same architecture=baseline_R_only, same seed=260816,
expert=bull, fixed 40-epoch budget, no early stopping, full curve logged -- directly comparable to
the already-collected plain-CE baseline and the all-3-combined run):
  gce_only   : GCE (q=0.7) replaces CE on all 3 heads, no ELR, no mixup
  elr_only   : plain CE + ELR regularizer (lambda=3, beta=0.7) on direction/quality heads, no mixup
  mixup_only : plain CE with latent-space mixup (alpha=1.0) on direction/quality heads, no ELR

If NONE of the 3 individually beats the plain-CE baseline, that's evidence for (c) domain mismatch
over (b) interaction (since interaction can't explain an individual technique underperforming a
baseline it isn't combined with). If any individually beats baseline but the combo doesn't, that
points to (b). If any individually roughly matches or beats baseline only under a different
hyperparameter than the literature default, that's evidence for (a) -- this script uses the SAME
defaults as the combo run per technique; a follow-up hyperparameter sweep would be a separate step,
only run if warranted by these results.

Reuses gce_loss/elr_term from the combo regularizer script and _prepare_frames_light from
curve_diag, unmodified via import.
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

import research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816 as combo  # noqa: E402

gate = combo.gate
base = combo.base
hard = combo.hard
CFG = combo.CFG
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_regularizer_isolation_20260816"
SEED = combo.SEED
EXPERT = combo.EXPERT
EPOCHS = combo.EPOCHS

VARIANTS = [
    {"name": "gce_only", "use_gce": True, "use_elr": False, "use_mixup": False},
    {"name": "elr_only", "use_gce": False, "use_elr": True, "use_mixup": False},
    {"name": "mixup_only", "use_gce": False, "use_elr": False, "use_mixup": True},
]


def log(msg: str) -> None:
    print(f"[faithful_tabm_reg_isolation] {msg}", flush=True)


def _classification_loss(logits_k: torch.Tensor, target: torch.Tensor, use_gce: bool) -> torch.Tensor:
    """(batch, k, C), (batch,) -> (batch, k). GCE if use_gce else plain per-member CE."""
    if use_gce:
        return combo.gce_loss(logits_k, target)
    k = logits_k.shape[1]
    return torch.nn.functional.cross_entropy(logits_k.reshape(-1, logits_k.shape[-1]), target[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)


def _fit_variant(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device, use_gce, use_elr, use_mixup):
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

    model = base.ThreeHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    row_id = torch.arange(len(train_idx))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]), row_id)
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)

    n_train_dir = len(train_idx)
    ema_dir = torch.full((n_train_dir, 3), 1.0 / 3.0, dtype=torch.float32, device=device)
    ema_qual = torch.full((n_train_dir, 3), 1.0 / 3.0, dtype=torch.float32, device=device)

    curve = []
    for epoch in range(int(epochs)):
        model.train()
        exit_iter = iter(dl_exit)
        train_loss_sum, train_batches = 0.0, 0
        for xb, yb, wb, ridb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, wb, ridb = xb.to(device), yb.to(device), wb.to(device), ridb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)

            h = model.encode(xb)
            logits_dir_u = model.direction_head(h)
            logits_qual_u = model.quality_head(h)
            if use_elr:
                with torch.no_grad():
                    probs_dir_u_mean = torch.softmax(logits_dir_u, dim=-1).mean(dim=1)
                    probs_qual_u_mean = torch.softmax(logits_qual_u, dim=-1).mean(dim=1)
                    ema_dir[ridb] = combo.ELR_BETA * ema_dir[ridb] + (1.0 - combo.ELR_BETA) * probs_dir_u_mean
                    ema_qual[ridb] = combo.ELR_BETA * ema_qual[ridb] + (1.0 - combo.ELR_BETA) * probs_qual_u_mean

            bsz = xb.shape[0]
            if use_mixup:
                perm = torch.randperm(bsz, device=device)
                lam = float(np.random.beta(combo.MIXUP_ALPHA, combo.MIXUP_ALPHA))
                h_use = lam * h + (1.0 - lam) * h[perm]
            else:
                perm = torch.arange(bsz, device=device)
                lam = 1.0
                h_use = h

            logits_dir_m = model.direction_head(h_use)
            logits_qual_m = model.quality_head(h_use)
            y_perm = yb[perm]
            w_perm = wb[perm]
            cl_dir = lam * _classification_loss(logits_dir_m, yb, use_gce) + (1.0 - lam) * _classification_loss(logits_dir_m, y_perm, use_gce)
            cl_qual = lam * _classification_loss(logits_qual_m, yb, use_gce) + (1.0 - lam) * _classification_loss(logits_qual_m, y_perm, use_gce)
            w_mix = lam * wb + (1.0 - lam) * w_perm
            loss_dir = (cl_dir.mean(dim=1) * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0)
            loss_qual = (cl_qual.mean(dim=1) * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0)

            loss_elr = torch.zeros((), device=device)
            if use_elr:
                probs_dir_m_mean = torch.softmax(logits_dir_m, dim=-1).mean(dim=1)
                probs_qual_m_mean = torch.softmax(logits_qual_m, dim=-1).mean(dim=1)
                target_dir_mix = lam * ema_dir[ridb] + (1.0 - lam) * ema_dir[ridb[perm]]
                target_qual_mix = lam * ema_qual[ridb] + (1.0 - lam) * ema_qual[ridb[perm]]
                elr_dir = combo.elr_term(probs_dir_m_mean, target_dir_mix)
                elr_qual = combo.elr_term(probs_qual_m_mean, target_qual_mix)
                loss_elr = combo.ELR_LAMBDA * ((elr_dir * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0) +
                                                (elr_qual * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0))

            out_exit = model(xe)
            cl_exit = _classification_loss(out_exit["exit"], ye, use_gce)
            loss_exit = (cl_exit.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)

            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit + loss_elr
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
        curve.append({"epoch": epoch + 1, "train_loss": round(train_loss, 5), "val_loss": round(vloss, 5), "direction_balanced_accuracy_val": round(bacc, 5)})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"    epoch={epoch+1:02d} train_loss={train_loss:.4f} val_loss={vloss:.4f} val_bacc={bacc:.4f}")
    return curve


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} epochs_fixed={EPOCHS} variants={[v['name'] for v in VARIANTS]} ===")
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

    report: dict[str, Any] = {"design": "isolates GCE/ELR/mixup individually vs the all-3-combined run, fixed epochs, no early stopping.", "seed": SEED, "expert": EXPERT, "epochs_fixed": EPOCHS, "curves": {}}
    t0 = time.time()
    for v in VARIANTS:
        log(f"=== stage=train variant={v['name']} ===")
        curve = _fit_variant(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, use_gce=v["use_gce"], use_elr=v["use_elr"], use_mixup=v["use_mixup"])
        report["curves"][v["name"]] = curve
        best_bacc = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {v['name']}: best val_bacc={best_bacc['direction_balanced_accuracy_val']:.4f} at epoch {best_bacc['epoch']}/{EPOCHS}; "
            f"final(epoch{EPOCHS}) val_bacc={curve[-1]['direction_balanced_accuracy_val']:.4f} (elapsed={time.time()-t0:.0f}s)")

    log("=== stage=summary (reference: plain CE baseline peak=0.5740@epoch2 final=0.4922; all-3-combined peak=0.5368@epoch8 final=0.4829) ===")
    for v in VARIANTS:
        curve = report["curves"][v["name"]]
        best_bacc = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {v['name']:12s} peak_val_bacc={best_bacc['direction_balanced_accuracy_val']:.4f} @epoch{best_bacc['epoch']} final={curve[-1]['direction_balanced_accuracy_val']:.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

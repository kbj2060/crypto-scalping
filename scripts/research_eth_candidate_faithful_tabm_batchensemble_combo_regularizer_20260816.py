#!/usr/bin/env python3
"""RESEARCH ONLY -- combines the 3 shortlisted memorization-combating techniques from
feedback_modern_dl_training_checklist (memory) into ONE training run, on top of the LIVE
architecture (baseline_R_only == ThreeHeadTabM), tested at fixed 40 epochs / no early stopping so
the resulting curve is directly comparable to the already-collected plain-CE baseline curve
(research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816.py, seed=260816,
expert=bull: val_bacc peaked 0.574 at epoch 2, degraded to 0.492 by epoch 40 -- textbook
memorization). User asked to test all 3 together, not in isolation -- this is a single combined
cheap_gate, not a clean per-technique ablation; if it helps, a follow-up decomposition would be
needed to attribute credit. All 3 hyperparameters below are literature-common DEFAULTS, not tuned
for this project (that would need its own sweep, deferred until this combo is shown to move the
needle at all).

1. **GCE (Generalized Cross Entropy)**, Zhang & Sabuncu arXiv:1805.07836, q=0.7 (paper default).
   Replaces plain CE on ALL THREE heads (direction/quality/exit): downweights confidently-wrong
   examples instead of full-weighting them.
2. **ELR (Early-Learning Regularization)**, Liu et al. arXiv:2007.00151, lambda=3.0, beta=0.7
   (values commonly cited for the paper's CIFAR-10 config -- no universal value given in the paper,
   flagged as untuned). Applied to DIRECTION and QUALITY heads only (scope reduction -- skipped for
   exit head to limit implementation complexity/risk). Maintains a per-training-row EMA of the
   model's own past (unmixed) softmax predictions; adds -log(1 - <p, ema_target>) to the loss, which
   amplifies the early "generalizable-gradient-dominates" phase Arpit et al. describe.
3. **Latent-space mixup**, Zhang et al. arXiv:1710.09412 (adapted per arXiv:2304.04271's
   latent-mixing pattern to avoid interpolating raw OHLCV-derived features, which could produce
   physically nonsensical hybrid bars): mixes the shared BatchEnsemble embedding h = encode(x)
   (shape batch x k x hidden) between two random samples in the same batch with lambda~Beta(1,1),
   and mixes the two samples' GCE losses with the same lambda (standard mixup loss form). Applied to
   DIRECTION and QUALITY heads only (same scope reduction as ELR -- exit head keeps plain unmixed
   GCE). ELR's EMA-target dot product is likewise computed against the SAME lambda-mixed target
   (t_mix = lambda*t_i + (1-lambda)*t_j), using each original row's own EMA buffer entry.

EMA buffer update uses the model's UNMIXED prediction for each batch's rows (computed from the same
already-forward-passed embedding h before mixing -- no extra encode() call needed, only cheap extra
head-applications), consistent with ELR's original per-real-sample design; the mixed-embedding
forward pass is a separate, second application of the same heads used only for the training loss.

Reuses research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816.py's
_prepare_frames_light/frame-prep pipeline unmodified via import.
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

import research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816 as curve_diag  # noqa: E402

gate = curve_diag.gate
base = curve_diag.base
hard = curve_diag.hard
CFG = curve_diag.CFG
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816"
SEED = curve_diag.SEED
EXPERT = curve_diag.EXPERT
EPOCHS = curve_diag.EPOCHS

GCE_Q = 0.7
ELR_LAMBDA = 3.0
ELR_BETA = 0.7
MIXUP_ALPHA = 1.0
GCE_EPS = 1.0e-7
ELR_EPS = 1.0e-4


def log(msg: str) -> None:
    print(f"[faithful_tabm_combo_reg] {msg}", flush=True)


def gce_loss(logits_k: torch.Tensor, target: torch.Tensor, q: float = GCE_Q) -> torch.Tensor:
    """logits_k: (batch, k, C); target: (batch,) long. Returns (batch, k)."""
    probs_k = torch.softmax(logits_k, dim=-1)
    k = logits_k.shape[1]
    py = probs_k.gather(-1, target.view(-1, 1, 1).expand(-1, k, 1)).squeeze(-1).clamp(min=GCE_EPS)
    return (1.0 - py.pow(q)) / q


def elr_term(probs_mean: torch.Tensor, target_ema: torch.Tensor) -> torch.Tensor:
    """probs_mean/target_ema: (batch, C). Returns (batch,)."""
    dot = (probs_mean * target_ema).sum(dim=-1).clamp(max=1.0 - ELR_EPS)
    return -torch.log(1.0 - dot)


def _fit_combo(model_cls, x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device):
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

            h = model.encode(xb)  # (batch, k, hidden), requires grad
            logits_dir_u = model.direction_head(h)
            logits_qual_u = model.quality_head(h)
            with torch.no_grad():
                probs_dir_u_mean = torch.softmax(logits_dir_u, dim=-1).mean(dim=1)
                probs_qual_u_mean = torch.softmax(logits_qual_u, dim=-1).mean(dim=1)
                ema_dir[ridb] = ELR_BETA * ema_dir[ridb] + (1.0 - ELR_BETA) * probs_dir_u_mean
                ema_qual[ridb] = ELR_BETA * ema_qual[ridb] + (1.0 - ELR_BETA) * probs_qual_u_mean

            bsz = xb.shape[0]
            perm = torch.randperm(bsz, device=device)
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            h_mix = lam * h + (1.0 - lam) * h[perm]
            logits_dir_m = model.direction_head(h_mix)
            logits_qual_m = model.quality_head(h_mix)

            y_perm = yb[perm]
            w_perm = wb[perm]
            gce_dir = lam * gce_loss(logits_dir_m, yb) + (1.0 - lam) * gce_loss(logits_dir_m, y_perm)
            gce_qual = lam * gce_loss(logits_qual_m, yb) + (1.0 - lam) * gce_loss(logits_qual_m, y_perm)
            w_mix = lam * wb + (1.0 - lam) * w_perm
            loss_dir = (gce_dir.mean(dim=1) * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0)
            loss_qual = (gce_qual.mean(dim=1) * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0)

            probs_dir_m_mean = torch.softmax(logits_dir_m, dim=-1).mean(dim=1)
            probs_qual_m_mean = torch.softmax(logits_qual_m, dim=-1).mean(dim=1)
            target_dir_mix = lam * ema_dir[ridb] + (1.0 - lam) * ema_dir[ridb[perm]]
            target_qual_mix = lam * ema_qual[ridb] + (1.0 - lam) * ema_qual[ridb[perm]]
            elr_dir = elr_term(probs_dir_m_mean, target_dir_mix)
            elr_qual = elr_term(probs_qual_m_mean, target_qual_mix)
            loss_elr = ELR_LAMBDA * ((elr_dir * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0) +
                                      (elr_qual * w_mix).sum() / torch.clamp(w_mix.sum(), min=1.0))

            out_exit = model(xe)
            gce_exit = gce_loss(out_exit["exit"], ye)
            loss_exit = (gce_exit.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)

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
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} epochs_fixed={EPOCHS} "
        f"gce_q={GCE_Q} elr_lambda={ELR_LAMBDA} elr_beta={ELR_BETA} mixup_alpha={MIXUP_ALPHA} ===")
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

    t0 = time.time()
    log("=== stage=train architecture=baseline_R_only+GCE+ELR+latent_mixup ===")
    curve = _fit_combo(base.ThreeHeadTabM, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device)
    best_loss = min(curve, key=lambda r: r["val_loss"])
    best_bacc = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
    log(f"  best val_loss at epoch {best_loss['epoch']} ({best_loss['val_loss']:.4f}); "
        f"best val_bacc at epoch {best_bacc['epoch']} ({best_bacc['direction_balanced_accuracy_val']:.4f}); "
        f"final(epoch{EPOCHS}) val_bacc={curve[-1]['direction_balanced_accuracy_val']:.4f} (elapsed={time.time()-t0:.0f}s)")

    report = {
        "design": "combined GCE+ELR+latent-mixup on baseline_R_only, fixed epochs, no early stopping.",
        "seed": SEED, "expert": EXPERT, "epochs_fixed": EPOCHS,
        "hyperparams": {"gce_q": GCE_Q, "elr_lambda": ELR_LAMBDA, "elr_beta": ELR_BETA, "mixup_alpha": MIXUP_ALPHA},
        "curve": curve,
        "comparison_note": "compare against research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816's baseline_R_only curve (plain CE, same seed/expert/epochs): that run peaked val_bacc=0.574 at epoch 2, degraded to 0.492 by epoch 40.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

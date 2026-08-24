#!/usr/bin/env python3
"""RESEARCH ONLY -- tests the 2 remaining untested TabM-paper differences from this candidate's
original diff table (docs/model_contracts/eth_candidate_faithful_tabm_batchensemble_contract_
20260816.md's "발견한 차이" section), on top of the winning architecture (baseline_R_only) and
winning recipe (AdaBelief + GCE + cosine 2e-4->2e-6 + Prechelt UP_4 class-balanced-CE selection):

1. **Residual connections** (contract's 미해결 이슈 #1, explicitly deferred, never revisited):
   the paper's TabM is a standard MLP with NO residual connections; the live ThreeHeadTabM adds
   `h = h + residual` at every block, which the paper does not have. `ThreeHeadTabMNoResidual`
   below is a copy of base.ThreeHeadTabM with that line removed, everything else identical.
2. **Ensemble size k=8 vs the paper's own default k=32** (never tested this whole investigation):
   the paper sets k=32 heuristically as effective implicit regularization (arXiv:2410.24210 §3.3).
   Tested here as a straightforward cfg.k=32 swap on the unmodified base.ThreeHeadTabM.

Single-seed cheap_gate first (seed=260816, matching this whole investigation's primary seed), all
3 regime experts, fixed 60-epoch budget with Prechelt UP_4 strip-based stopping (same protocol as
the already N>=5-seed-confirmed winning recipe) -- N>=5 seed reproduction only if a variant shows a
real, non-trivial signal over the reference (baseline_R_only k=8 with residual, AdaBelief+GCE:
N=5-seed-confirmed bull=0.5534 bear=0.5570 chop=0.5617).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_baseline_grid_prechelt_20260816 as bgrid  # noqa: E402

gate = bgrid.gate
base = bgrid.base
hard = bgrid.hard
CFG = bgrid.CFG
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_residual_k32_cheap_gate_20260816"
SEED = 260816
OPT_FACTORY = bgrid.OPTIMIZERS["AdaBelief"]
USE_GCE = True


def log(msg: str) -> None:
    print(f"[faithful_tabm_residual_k32] {msg}", flush=True)


class ThreeHeadTabMNoResidual(nn.Module):
    """Copy of base.ThreeHeadTabM with the `h = h + residual` line removed -- the paper's TabM is a
    standard MLP with no residual connections; the live implementation adds one that isn't in
    arXiv:2410.24210. Everything else (R-only adapters, dropout, SiLU, LayerNorm placement) is
    identical to the live class."""

    def __init__(self, n_features: int, *, cfg=CFG) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_head = nn.Linear(int(cfg.hidden), 3)
        self.quality_head = nn.Linear(int(cfg.hidden), 3)
        self.exit_head = nn.Linear(int(cfg.hidden), 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            # NOTE: no `h = h + residual` here -- this is the only difference from base.ThreeHeadTabM
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {"direction": self.direction_head(h), "quality": self.quality_head(h), "exit": self.exit_head(h)}


VARIANTS = {
    "reference_k8_residual": {"model_cls": base.ThreeHeadTabM, "cfg": CFG},
    "no_residual_k8": {"model_cls": ThreeHeadTabMNoResidual, "cfg": CFG},
    "residual_k32": {"model_cls": base.ThreeHeadTabM, "cfg": dataclasses.replace(CFG, k=32)},
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) seed={SEED} variants={list(VARIANTS)} recipe=AdaBelief+GCE+cosine+Prechelt ===")
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

    report: dict[str, Any] = {"design": "residual removal + k=32 cheap_gate on top of the winning AdaBelief+GCE+cosine+Prechelt recipe, single seed, all 3 experts.", "seed": SEED, "results": {}}
    t0 = time.time()
    for variant_name, spec in VARIANTS.items():
        report["results"][variant_name] = {}
        for expert in hard.EXPERT_NAMES:
            expert_idx = list(hard.EXPERT_NAMES).index(expert)
            # bgrid._fit_grid_cell_prechelt hardcodes base.ThreeHeadTabM internally, so for the
            # no_residual/k32 variants we call a lightly parameterized copy instead.
            result = _fit_variant(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, device=device, model_cls=spec["model_cls"], cfg=spec["cfg"])
            report["results"][variant_name][expert] = result
            log(f"  {variant_name} {expert}: n_params={result['n_params']} selected_bacc={result['selected_bacc']:.4f} @epoch{result['selected_epoch']} "
                f"true_peak={result['true_peak_bacc']:.4f} epochs_ran={result['epochs_ran']} (elapsed={time.time()-t0:.0f}s)")

    log("=== stage=summary (reference: baseline_R_only k=8 w/ residual, N=5-seed-confirmed AdaBelief+GCE: bull=0.5534 bear=0.5570 chop=0.5617) ===")
    for variant_name in VARIANTS:
        for expert in hard.EXPERT_NAMES:
            d = report["results"][variant_name][expert]
            log(f"  {variant_name:24s} {expert:6s} selected_bacc={d['selected_bacc']:.4f} n_params={d['n_params']}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


def _fit_variant(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, device, model_cls, cfg):
    """Copy of bgrid._fit_grid_cell_prechelt, parameterized by model_cls/cfg instead of hardcoded
    base.ThreeHeadTabM/CFG -- needed since the no_residual/k32 variants aren't the exact class the
    original function closes over."""
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    from torch.utils.data import DataLoader, TensorDataset

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

    model = model_cls(x_dir_np.shape[1], cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = OPT_FACTORY(model.parameters(), float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bgrid.MAX_EPOCHS, eta_min=bgrid.LR_MIN)
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)

    def cls_loss(logits_k, target):
        if USE_GCE:
            return bgrid.combo.gce_loss(logits_k, target, q=bgrid.GCE_Q)
        k = logits_k.shape[1]
        return torch.nn.functional.cross_entropy(logits_k.reshape(-1, logits_k.shape[-1]), target[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)

    vx_t = torch.from_numpy(x_dir_np[val_idx]).to(device)
    vy_t = torch.from_numpy(y_dir_np[val_idx]).to(device)
    vw_t = torch.from_numpy(dir_w[val_idx]).to(device)

    def selection_val_loss_and_bacc():
        model.eval()
        with torch.no_grad():
            vo = model(vx_t)
            vce = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy_t[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vloss = float(((vce.mean(dim=1) * vw_t).sum() / torch.clamp(vw_t.sum(), min=1.0)).detach().cpu())
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy_t.cpu().numpy(), dir_pred_k))
        return vloss, bacc

    curve = []
    best_sel_loss = float("inf")
    best_epoch = 0
    best_bacc_at_selection = None
    last_strip_val = None
    bad_strips = 0
    for epoch in range(bgrid.MAX_EPOCHS):
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

        sel_loss, bacc = selection_val_loss_and_bacc()
        curve.append({"epoch": epoch + 1, "selection_val_loss": round(sel_loss, 5), "direction_balanced_accuracy_val": round(bacc, 5)})

        if sel_loss < best_sel_loss:
            best_sel_loss = sel_loss
            best_epoch = epoch + 1
            best_bacc_at_selection = bacc

        strip_epoch = epoch + 1
        if strip_epoch % bgrid.STRIP_LEN == 0:
            if last_strip_val is not None:
                if sel_loss > last_strip_val:
                    bad_strips += 1
                else:
                    bad_strips = 0
            last_strip_val = sel_loss
            if bad_strips >= bgrid.UP_S:
                break

    true_peak = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
    return {
        "n_params": n_params, "epochs_ran": len(curve),
        "selected_epoch": best_epoch, "selected_bacc": best_bacc_at_selection,
        "true_peak_epoch": true_peak["epoch"], "true_peak_bacc": true_peak["direction_balanced_accuracy_val"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

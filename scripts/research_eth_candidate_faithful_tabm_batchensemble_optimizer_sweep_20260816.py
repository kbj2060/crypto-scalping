#!/usr/bin/env python3
"""RESEARCH ONLY -- optimizer swap follow-up to the LR sweep result (lr=2e-4 was the best-evidenced
single lever found so far for this candidate's memorization problem -- see docs/experiments/
eth_candidate_faithful_tabm_batchensemble_20260816.md). A survey of labmlai/annotated_deep_learning
_paper_implementations (github.com/labmlai/annotated_deep_learning_paper_implementations) found two
optimizer candidates mechanistically relevant to a noisy/weak-signal gradient setting:

- RAdam (Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond", arXiv:1908.03265):
  rectifies Adam's uncontrolled early-step variance -- directly relevant since this project's own
  lr=2e-3 result showed exactly that kind of unstable, fast early spike. torch.optim.RAdam (built
  into PyTorch, confirmed available) used as-is.
- AdaBelief (Zhuang et al., "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
  Gradients", arXiv:2010.07468): replaces Adam's E[g^2] denominator with E[(g-m)^2] (variance of the
  gradient around its own momentum estimate), so steps shrink when gradients disagree with their
  recent trend -- a similar "distrust noisy per-sample signal" spirit to GCE loss (which already
  helped in this project). NOT in PyTorch core or this repo's dependencies -- implemented here
  directly from the paper's Algorithm 1 (decoupled/AdamW-style weight decay, eps=1e-16 per the
  paper's own recommendation for the belief term not to be washed out).

Both tested at lr=2e-4 (this candidate's best-evidenced LR so far) against the existing AdamW@2e-4
control, same seed=260816/expert=bull/baseline_R_only/40-epoch-no-early-stopping methodology as
every other diagnostic in this candidate, for direct comparability.
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
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_optimizer_sweep_20260816"
SEED = curve_diag.SEED
EXPERT = curve_diag.EXPERT
EPOCHS = curve_diag.EPOCHS
LR = 2.0e-4  # this candidate's best-evidenced LR so far (LR sweep, 2026-08-16)


def log(msg: str) -> None:
    print(f"[faithful_tabm_optimizer_sweep] {msg}", flush=True)


class AdaBelief(torch.optim.Optimizer):
    """Zhuang et al. 2020, arXiv:2010.07468, Algorithm 1. Decoupled (AdamW-style) weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_var"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2).add_(eps)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_var / bias_correction2).sqrt().add_(eps)
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


OPTIMIZERS = {
    "AdamW": lambda params: torch.optim.AdamW(params, lr=LR, weight_decay=float(CFG.weight_decay)),
    "RAdam": lambda params: torch.optim.RAdam(params, lr=LR, weight_decay=float(CFG.weight_decay)),
    "AdaBelief": lambda params: AdaBelief(params, lr=LR, weight_decay=float(CFG.weight_decay)),
}


def _fit_one_curve_opt(model_cls, x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device, opt_factory):
    """Copy of curve_diag._fit_one_curve, parameterized by an optimizer factory instead of hardcoded AdamW."""
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

    model = model_cls(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = opt_factory(model.parameters())
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
        curve.append({"epoch": epoch + 1, "train_loss": round(train_loss, 5), "val_loss": round(vloss, 5), "direction_balanced_accuracy_val": round(bacc, 5)})
    return curve


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} lr={LR} optimizers={list(OPTIMIZERS)} ===")
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

    report: dict[str, Any] = {"design": "optimizer swap at this candidate's best-found lr=2e-4, fixed 40 epochs, no early stopping.", "seed": SEED, "expert": EXPERT, "lr": LR, "epochs_fixed": EPOCHS, "curves": {}}
    t0 = time.time()
    for opt_name, opt_factory in OPTIMIZERS.items():
        log(f"=== stage=train optimizer={opt_name} ===")
        curve = _fit_one_curve_opt(base.ThreeHeadTabM, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, opt_factory=opt_factory)
        report["curves"][opt_name] = curve
        best_bacc = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {opt_name}: peak val_bacc={best_bacc['direction_balanced_accuracy_val']:.4f} @epoch{best_bacc['epoch']}/{EPOCHS}; "
            f"final(epoch{EPOCHS})={curve[-1]['direction_balanced_accuracy_val']:.4f} (elapsed={time.time()-t0:.0f}s)")

    log("=== stage=summary (reference: AdamW@lr=2e-3 peak=0.5740@epoch2 final=0.492; AdamW@lr=2e-4 peak=0.5714@epoch12 final=0.536) ===")
    for opt_name in OPTIMIZERS:
        curve = report["curves"][opt_name]
        best_bacc = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {opt_name:10s} peak_val_bacc={best_bacc['direction_balanced_accuracy_val']:.4f} @epoch{best_bacc['epoch']} final={curve[-1]['direction_balanced_accuracy_val']:.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

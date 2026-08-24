#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate for an explicit diversity-promoting loss term (Negative Correlation
Learning) on the canonical, UNMODIFIED ThreeHeadTabM architecture.

Follow-up to docs/experiments/eth_odyssey4_dl_reference_deep_analysis_20260816.md Section 6.8: five
independent capacity/init-based BatchEnsemble-diversity fixes (k-reduction, quality-target
separation, R-gate wide-init, lr isolation, rank-r gate generalization) were all N>=5-seed-or-
single-seed negative, and the rank-r experiment additionally confirmed (N=5 seeds, true-peak epoch)
that ensemble diversity genuinely does not increase no matter how much extra gate capacity is given
-- the loss has zero term rewarding inter-member diversity (root cause identified in section 2.3), so
giving the architecture more room to diverge doesn't make it diverge. This candidate is the one
qualitatively different axis flagged as untested: put a diversity term directly IN the loss instead
of manipulating architecture/init.

Literature (verified via OpenAlex, accessed 2026-08-17):
  - Krogh & Vedelsby (1994), "Neural network ensembles, cross validation, and active learning" --
    the classical ambiguity decomposition: ensemble squared-error = mean-member-error - ambiguity,
    where ambiguity = mean_i (f_i - f_mean)^2. Maximizing ambiguity is the diversity target.
  - Liu & Yao (1999), "Ensemble learning via negative correlation" (Neural Networks, DOI
    10.1016/s0893-6080(99)00073-8) -- Negative Correlation Learning (NCL): each member's loss gets a
    penalty term p_i = (f_i-f_mean)*sum_{j!=i}(f_j-f_mean), which (since sum_j(f_j-f_mean)=0 by
    definition of the mean) algebraically reduces to p_i = -(f_i-f_mean)^2 -- i.e. NCL's penalty is
    exactly the negative of the Krogh-Vedelsby ambiguity term for each member. Adding lambda*p_i to
    each member's loss and minimizing is equivalent to maximizing ambiguity, weighted by lambda.
  - Wang, Chen & Yao (2010), "Negative correlation learning for classification ensembles" (IJCNN,
    DOI 10.1109/ijcnn.2010.5596702) -- confirms the original NCL/ambiguity decomposition was derived
    for regression (squared-error) outputs and needs adaptation for classification. This script uses
    the common practical adaptation (also used in deep-learning NCL follow-ups such as Zhang et al.
    2019 CVPR "Nonlinear Regression via Deep Negative Correlation Learning" and Shi et al. 2018 CVPR
    "Crowd Counting with Deep Negative Correlation Learning"): apply the ambiguity term directly to
    each member's SOFTMAX PROBABILITY vector rather than a raw regression output.

Implementation: total_loss = loss_dir + qw*loss_qual + ew*loss_exit - LAMBDA * ambiguity(direction),
where ambiguity = mean_batch[ mean_k( ||p_k - p_mean||_2^2 ) ] on direction_head's per-member softmax
probabilities. At LAMBDA=0 this reduces exactly to the canonical loss (the term is computed but
multiplied by 0, contributing zero gradient) -- LAMBDA=0 is the control arm of the sweep, not a
separate sanity-check condition, since the architecture itself is completely unmodified (unlike the
rank-r candidate, there is nothing to bit-for-bit-verify here beyond "0 * anything = 0").

Protocol: single seed (260816) cheap_gate first, sweeping LAMBDA across a wide log range (ambiguity
measured ~0.01 at init vs CE-loss-scale ~1-3, so LAMBDA needs to span orders of magnitude to find
where the term actually competes with the task loss). Collapse diagnostic recorded EVERY epoch (not
just at the selected checkpoint), applying the section-6.8 lesson so true-peak-epoch diversity is
measurable without a follow-up run. If any LAMBDA shows a real signal (either direction), the natural
next step is N>=5 genuinely random seeds at that LAMBDA, matching this repo's standard gate ladder.
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
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as canon  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_ncl_diversity_loss_cheap_gate_20260817"
EXPERT = "bull"
SEED = 260816
EPOCHS = 28
LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1000.0]  # 0.0 = control (exactly canon.ThreeHeadTabM training)


def log(msg: str) -> None:
    print(f"[ncl_diversity] {msg}", flush=True)


def _ncl_ambiguity(logits_k: torch.Tensor) -> torch.Tensor:
    """Krogh & Vedelsby (1994) ambiguity / Liu & Yao (1999) NCL diversity term, on softmax
    probabilities. logits_k: (batch,k,C). Returns a scalar (batch-mean ambiguity). HIGHER = more
    diverse ensemble members."""
    probs_k = torch.softmax(logits_k, dim=-1)
    p_mean = probs_k.mean(dim=1, keepdim=True)
    ambiguity = ((probs_k - p_mean) ** 2).sum(dim=-1).mean(dim=1)
    return ambiguity.mean()


def _collapse_stats(logits_k: torch.Tensor) -> dict[str, float]:
    probs_k = torch.softmax(logits_k, dim=-1)
    n, k, c = probs_k.shape
    top_prob = probs_k.max(dim=-1).values
    top_np = top_prob.detach().cpu().numpy()
    corr = np.corrcoef(top_np.T)
    iu = np.triu_indices(k, k=1)
    pred_np = probs_k.argmax(dim=-1).detach().cpu().numpy()
    unanimous = (pred_np == pred_np[:, [0]]).all(axis=1).mean()
    return {
        "mean_pairwise_corr_of_top_confidence": float(np.mean(corr[iu])),
        "argmax_unanimity_rate": float(unanimous),
    }


def _fit_with_ncl(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, lam: float) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    cfg = canon.CFG
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = canon._standardize_fit(x_all)
    x_dir_np = canon._standardize_apply(x_dir, scaler)
    x_exit_np = canon._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = canon._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = canon._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
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

    model = canon.ThreeHeadTabM(x_dir_np.shape[1], cfg=cfg).to(device)  # UNMODIFIED architecture
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    last_epoch = 0
    curve: list[dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        train_ambiguity_sum, train_ambiguity_n = 0.0, 0
        for xb, yb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out_dir = model(xb)
            loss_dir_k = F.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_qual_k = F.cross_entropy(out_dir["quality"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            out_exit = model(xe)
            loss_exit_k = F.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            ambiguity = _ncl_ambiguity(out_dir["direction"])
            loss = loss_dir + float(cfg.quality_loss_weight) * loss_qual + float(cfg.exit_loss_weight) * loss_exit - float(lam) * ambiguity
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_ambiguity_sum += float(ambiguity.detach().cpu()) * xb.shape[0]
            train_ambiguity_n += xb.shape[0]
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
            vdir = F.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = F.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = F.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            v_ambiguity = float(_ncl_ambiguity(vo["direction"]).detach().cpu())
            # task-only val_loss (NOT including the diversity term) -- checkpoint selection must stay
            # comparable across lambda values and to every prior cheap_gate in this line, so it is
            # selected on the SAME criterion as always (task loss only), never on the diversity-biased
            # objective actually being minimized during training.
            vloss = vdir_loss + float(cfg.quality_loss_weight) * vqual_loss + float(cfg.exit_loss_weight) * vex_loss
            dir_pred = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            dir_bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred))
            collapse = _collapse_stats(vo["direction"])
        curve.append({
            "epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_bacc": round(dir_bacc, 5),
            "val_ambiguity": round(v_ambiguity, 6), "train_ambiguity_mean": round(train_ambiguity_sum / max(train_ambiguity_n, 1), 6),
            "collapse": collapse,
        })
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break

    selected = next(c for c in curve if c["epoch"] == best_epoch)
    true_peak = max(curve, key=lambda c: c["direction_bacc"])
    return {
        "lambda": float(lam),
        "epochs_ran": int(last_epoch),
        "selected_epoch": int(best_epoch),
        "selected_dir_bacc": selected["direction_bacc"],
        "selected_val_loss": selected["val_loss"],
        "selected_collapse": selected["collapse"],
        "selected_val_ambiguity": selected["val_ambiguity"],
        "true_peak_epoch": int(true_peak["epoch"]),
        "true_peak_dir_bacc": true_peak["direction_bacc"],
        "true_peak_collapse": true_peak["collapse"],
        "true_peak_val_ambiguity": true_peak["val_ambiguity"],
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(SEED)
    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seed={SEED} lambdas={LAMBDAS} ===")
    frames = truepipe.prepare_frames_true(disable_tp_sl=False)
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {
        "design": "NCL (Liu & Yao 1999) ambiguity diversity term added to the canonical loss, lambda in {0(control),1,10,100,1000}, canon.ThreeHeadTabM UNMODIFIED, single seed/expert, true 115-feature pipeline.",
        "citations": ["Krogh & Vedelsby 1994 (ambiguity decomposition)", "Liu & Yao 1999 doi:10.1016/s0893-6080(99)00073-8 (NCL)", "Wang, Chen & Yao 2010 doi:10.1109/ijcnn.2010.5596702 (NCL for classification)"],
        "seed": SEED,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "lambdas_tested": LAMBDAS,
        "runs": {},
    }
    for lam in LAMBDAS:
        res = _fit_with_ncl(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, lam=lam)
        report["runs"][f"lambda_{lam}"] = res
        cs = res["selected_collapse"]
        cp = res["true_peak_collapse"]
        log(
            f"  lambda={lam}: selected(ep{res['selected_epoch']})dir_bacc={res['selected_dir_bacc']:.4f} corr={cs['mean_pairwise_corr_of_top_confidence']:.4f} ambig={res['selected_val_ambiguity']:.5f} | "
            f"true_peak(ep{res['true_peak_epoch']})dir_bacc={res['true_peak_dir_bacc']:.4f} corr={cp['mean_pairwise_corr_of_top_confidence']:.4f} ambig={res['true_peak_val_ambiguity']:.5f} "
            f"({res['train_seconds']}s)"
        )

    baseline = report["runs"]["lambda_0.0"]
    verdict = {}
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        r = report["runs"][f"lambda_{lam}"]
        verdict[f"lambda_{lam}_vs_0"] = {
            "selected_dir_bacc_delta": r["selected_dir_bacc"] - baseline["selected_dir_bacc"],
            "true_peak_dir_bacc_delta": r["true_peak_dir_bacc"] - baseline["true_peak_dir_bacc"],
            "selected_corr_delta": r["selected_collapse"]["mean_pairwise_corr_of_top_confidence"] - baseline["selected_collapse"]["mean_pairwise_corr_of_top_confidence"],
            "true_peak_corr_delta": r["true_peak_collapse"]["mean_pairwise_corr_of_top_confidence"] - baseline["true_peak_collapse"]["mean_pairwise_corr_of_top_confidence"],
        }
    report["verdict"] = verdict
    for k, v in verdict.items():
        log(f"=== VERDICT {k}: selected_dir_bacc_delta={v['selected_dir_bacc_delta']:+.4f} true_peak_dir_bacc_delta={v['true_peak_dir_bacc_delta']:+.4f} true_peak_corr_delta={v['true_peak_corr_delta']:+.4f} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

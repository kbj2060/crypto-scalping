#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate for the BatchEnsemble wide-init-diversity candidate.

Follow-up to docs/experiments/eth_odyssey4_dl_reference_deep_analysis_20260816.md Section 2 /
memory eth_odyssey4_batchensemble_collapse_and_quality_head_duplication_20260816: a real deployed
zig075 bundle's k=8 members were measured to have collapsed to one function (pairwise top-confidence
correlation 0.997-0.999), and a fresh training run showed the diversity gates (input_scale/
expert_scale, init `randn()*0.03+1.0`) barely move from their random-init spread even after real
training -- the loss has no term rewarding inter-member diversity, so whatever diversity exists is
purely leftover random init. The k-reduction cheap_gate (accepting the collapse and shrinking k) came
back single-seed-negative (dir_bacc -0.005 to -0.01 at k in {1,2,4} vs k=8). This script tests the
OPPOSITE strategy instead: keep k=8, but widen the random-init spread of input_scale/expert_scale
(std 0.03 -> 0.1/0.2) so there is more genuine diversity to start from, since training itself doesn't
grow it. Also re-runs the member-collapse diagnostic AFTER training for each condition, to check
whether wider init actually survives training (more real diversity) or just gets ignored the same way
the current std=0.03 init does.

Single seed, single expert (bull, for consistency with the other two cheap_gates this session), true
102(+13pos)=115-feature live pipeline, plain CE, current live k=8 and same_as_direction quality target
held fixed (only the init std varies) -- isolates this one variable cleanly against the other two
already-tested axes.
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

import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as canon  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_batchensemble_wide_init_cheap_gate_20260816"
EXPERT = "bull"
SEED = 260816
EPOCHS = 28
INIT_STDS = [0.03, 0.1, 0.2]  # 0.03 = current live default (baseline)


def log(msg: str) -> None:
    print(f"[wide_init_cheap_gate] {msg}", flush=True)


def _reinit_gates(model: torch.nn.Module, std: float, seed: int) -> None:
    """Overwrite input_scale/expert_scale with a wider (or equal, for std=0.03) random spread
    around the same near-identity mean of 1.0 -- everything else about the model is untouched."""
    g = torch.Generator().manual_seed(int(seed) + 9001)
    with torch.no_grad():
        model.input_scale.copy_(torch.randn(model.input_scale.shape, generator=g) * float(std) + 1.0)
        for p in model.expert_scale:
            p.copy_(torch.randn(p.shape, generator=g) * float(std) + 1.0)


def _collapse_stats(logits_k: torch.Tensor) -> dict[str, float]:
    """Same diagnostic as diagnose_eth_odyssey4_batchensemble_member_collapse_20260816.py, inlined
    for convenience: pairwise top-confidence correlation + argmax unanimity vs an independent
    baseline, measured on the held-out val split right after training."""
    probs_k = torch.softmax(logits_k, dim=-1)
    n, k, c = probs_k.shape
    top_prob = probs_k.max(dim=-1).values
    top_np = top_prob.detach().cpu().numpy()
    corr = np.corrcoef(top_np.T)
    iu = np.triu_indices(k, k=1)
    pred_np = probs_k.argmax(dim=-1).detach().cpu().numpy()
    unanimous = (pred_np == pred_np[:, [0]]).all(axis=1).mean()
    class_freq = np.stack([(pred_np == cls).mean(axis=0) for cls in range(c)], axis=0)
    mean_class_freq = class_freq.mean(axis=1)
    independent_unanimity = float((mean_class_freq**k).sum())
    return {
        "mean_pairwise_corr_of_top_confidence": float(np.mean(corr[iu])),
        "argmax_unanimity_rate": float(unanimous),
        "independent_baseline_unanimity_rate": independent_unanimity,
        "mean_cross_member_prob_std": float(probs_k.std(dim=1).mean().item()),
    }


def _fit_with_init_std(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, init_std: float) -> dict[str, Any]:
    """canon._fit_expert_3head's exact training loop (plain CE, k=8, same_as_direction quality --
    matching the real live zig075 config), with ONE change: input_scale/expert_scale reinitialized
    at a wider std before training starts."""
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

    model = canon.ThreeHeadTabM(x_dir_np.shape[1], cfg=cfg).to(device)
    init_std_measured_before = model.input_scale.detach().std(dim=0).mean().item()
    _reinit_gates(model, init_std, seed)
    init_std_measured_after = model.input_scale.detach().std(dim=0).mean().item()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_loss = float("inf")
    best_components = None
    best_epoch = 0
    best_collapse: dict[str, float] | None = None
    stale = 0
    last_epoch = 0
    curve: list[dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
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
            loss_dir_k = torch.nn.functional.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_qual_k = torch.nn.functional.cross_entropy(out_dir["quality"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
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
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(cfg.quality_loss_weight) * vqual_loss + float(cfg.exit_loss_weight) * vex_loss
            dir_pred = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            dir_bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred))
            collapse = _collapse_stats(vo["direction"])
        curve.append({"epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_bacc": round(dir_bacc, 5)})
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": dir_bacc}
            best_epoch = epoch + 1
            best_collapse = collapse
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    gate_std_after_training = model.input_scale.detach().std(dim=0).mean().item()
    return {
        "init_std_requested": float(init_std),
        "gate_std_at_construction_default": round(init_std_measured_before, 5),
        "gate_std_after_reinit": round(init_std_measured_after, 5),
        "gate_std_after_training": round(gate_std_after_training, 5),
        "epochs_ran": int(last_epoch),
        "early_stop_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "collapse_at_best_checkpoint": best_collapse,
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(SEED)
    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seed={SEED} init_stds={INIT_STDS} ===")
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
        "design": "BatchEnsemble wide-init-diversity cheap_gate -- input_scale/expert_scale init std in {0.03(baseline),0.1,0.2}, k=8 and same_as_direction quality held fixed, single seed/expert, plain-CE, true 115-feature pipeline.",
        "seed": SEED,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "init_stds_tested": INIT_STDS,
        "runs": {},
    }
    for std in INIT_STDS:
        res = _fit_with_init_std(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, init_std=std)
        report["runs"][f"std_{std}"] = res
        bc = res["best_components"]
        cs = res["collapse_at_best_checkpoint"]
        log(
            f"  std={std} early_stop_epoch={res['early_stop_epoch']} best_val_loss={res['best_validation_loss']:.4f} "
            f"dir_bacc={bc['direction_balanced_accuracy']:.4f} gate_std_after_training={res['gate_std_after_training']:.4f} "
            f"pairwise_corr={cs['mean_pairwise_corr_of_top_confidence']:.4f} unanimity={cs['argmax_unanimity_rate']:.4f} ({res['train_seconds']}s)"
        )

    baseline = report["runs"]["std_0.03"]
    verdict = {}
    for std in INIT_STDS:
        if std == 0.03:
            continue
        r = report["runs"][f"std_{std}"]
        verdict[f"std_{std}_vs_baseline"] = {
            "dir_bacc_delta": r["best_components"]["direction_balanced_accuracy"] - baseline["best_components"]["direction_balanced_accuracy"],
            "val_loss_delta": r["best_validation_loss"] - baseline["best_validation_loss"],
            "pairwise_corr_delta": r["collapse_at_best_checkpoint"]["mean_pairwise_corr_of_top_confidence"] - baseline["collapse_at_best_checkpoint"]["mean_pairwise_corr_of_top_confidence"],
            "gate_std_survived_training": r["gate_std_after_training"],
        }
    report["verdict"] = verdict
    for k, v in verdict.items():
        log(f"=== VERDICT {k}: dir_bacc_delta={v['dir_bacc_delta']:+.4f} pairwise_corr_delta={v['pairwise_corr_delta']:+.4f} gate_std_after_training={v['gate_std_survived_training']:.4f} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

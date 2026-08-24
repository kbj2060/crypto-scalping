#!/usr/bin/env python3
"""RESEARCH ONLY -- C2 (Odyssey4 layer/parameter improvement proposal 20260816).

quality_loss_weight (0.80) and exit_loss_weight (1.15) in
scripts/train_eval_omega1_2_tabm_3head_20260603.py's ThreeHeadConfig are unexplained constants --
the layer design review (docs/experiments/eth_odyssey4_tabm_layer_design_review_20260816.md) found
no evidence of an HP search behind them, and classified them as "nuisance hyperparameters" per the
Google Deep Learning Tuning Playbook (docs/deep_learning_layer_design_and_training_reference_
20260816.md #6). This script runs an Optuna quasi-random/TPE search over both, holding everything
else (k/hidden/layers/dropout/lr/weight_decay/batch_size/patience) fixed, reusing the exact
TPESampler(seed=20260816) convention already established this session in
train_eval_eth_direction_quality_nhits_moderntcn_20260816.py's stage_hpsearch.

NOTE: A1 (GCE port into direction_head/quality_head) was tested and REVERTED
(docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md -- did not transfer, N=5 seed, 4/5
worse). The canonical script's actual training loss is plain cross_entropy; this script uses plain
CE throughout to match that real baseline.

Objective: MAXIMIZE held-out direction_balanced_accuracy (NOT the combined weighted loss) --
quality_loss_weight/exit_loss_weight directly rescale the terms of that combined loss, so using it
as the search objective would be partly self-referential (a trial could "win" by shrinking a
weight rather than by actually improving direction quality). direction_balanced_accuracy is
invariant to how the weights are chosen and is the metric this whole proposal doc already uses to
judge every other change (A1's verification, C1's cheap_gate), so it is used here too.

Single expert (bull, matching this session's other Odyssey4 GCE/embargo research scope), single
fixed seed=260816 across all trials (isolates the effect of the two weights from seed noise during
the ~20-trial search stage) and the canonical script's own epoch budget (28) / patience (8). The
BEST trial then gets a separate N>=5 genuinely-random-seed confirmation run
(research_eth_odyssey4_loss_weight_nseed_confirm_20260816.py) before any promotion claim.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402

canon = gate.base
hard = gate.hard

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_loss_weight_optuna_search_20260816"
EXPERT = "bull"
SEED = 260816
EPOCHS = 28
N_TRIALS = 20
WEIGHT_RANGE = (0.3, 2.0)  # linear, per the proposal doc (not log-scale)


def log(msg: str) -> None:
    print(f"[loss_weight_optuna] {msg}", flush=True)


def fit_one(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, cfg) -> dict[str, Any]:
    """Same as gate._fit_one (plain CE, matching the real canonical script), but takes an explicit
    cfg so quality_loss_weight/exit_loss_weight can vary per Optuna trial without mutating the
    shared module-level CFG."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
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
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_loss = float("inf")
    best_components = None
    stale = 0
    last_epoch = 0
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
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": bacc}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    return {
        "epochs_ran": int(last_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} seed={SEED} n_trials={N_TRIALS} range={WEIGHT_RANGE} ===")
    frames = gate._prepare_frames_light()
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,  # capped: dev box has 15GB RAM, unbounded build used ~13-14GB and tripped OOM once
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    trial_log: list[dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        quality_loss_weight = trial.suggest_float("quality_loss_weight", *WEIGHT_RANGE)
        exit_loss_weight = trial.suggest_float("exit_loss_weight", *WEIGHT_RANGE)
        cfg = replace(canon.CFG, quality_loss_weight=quality_loss_weight, exit_loss_weight=exit_loss_weight)
        res = fit_one(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, cfg=cfg)
        bacc = res["best_components"]["direction_balanced_accuracy"]
        trial.set_user_attr("result", res)
        trial_log.append({"trial": trial.number, "quality_loss_weight": quality_loss_weight, "exit_loss_weight": exit_loss_weight, "direction_balanced_accuracy": bacc, "best_validation_loss": res["best_validation_loss"], "train_seconds": res["train_seconds"]})
        log(f"  trial={trial.number:02d} qw={quality_loss_weight:.3f} ew={exit_loss_weight:.3f} dir_bacc={bacc:.4f} val_loss={res['best_validation_loss']:.4f} ({res['train_seconds']}s)")
        return float(bacc)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=20260816))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t0
    log(f"  Optuna {N_TRIALS} trials done ({elapsed:.0f}s) best_dir_bacc={study.best_value:.4f} best_params={study.best_params}")

    baseline_res = fit_one(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, cfg=canon.CFG)
    log(f"  baseline (qw={canon.CFG.quality_loss_weight} ew={canon.CFG.exit_loss_weight}) dir_bacc={baseline_res['best_components']['direction_balanced_accuracy']:.4f}")

    study.trials_dataframe().to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    report = {
        "design": "C2 -- Optuna TPE search over quality_loss_weight/exit_loss_weight, objective=maximize direction_balanced_accuracy.",
        "expert": EXPERT,
        "seed": SEED,
        "epochs_budget": EPOCHS,
        "n_trials": N_TRIALS,
        "weight_range": list(WEIGHT_RANGE),
        "baseline": {"quality_loss_weight": canon.CFG.quality_loss_weight, "exit_loss_weight": canon.CFG.exit_loss_weight, "direction_balanced_accuracy": baseline_res["best_components"]["direction_balanced_accuracy"], "best_validation_loss": baseline_res["best_validation_loss"]},
        "best_trial": {"params": study.best_params, "direction_balanced_accuracy": study.best_value},
        "trials": trial_log,
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

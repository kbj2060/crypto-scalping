#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate for isolating learning rate on the actual live zig075 config.

New direction after three same-day architecture-level cheap_gates (k-reduction, quality-target
separation, wide R-gate init diversity) all came back single-seed-negative -- see
eth_odyssey4_batchensemble_collapse_and_quality_head_duplication_20260816 memory. All three, run at
the current live default lr=2e-3, showed the IDENTICAL pattern: epochs_ran=9, early_stop_epoch=1 --
the best checkpoint is always the very first epoch, then val_loss degrades monotonically for 8 more
epochs before patience trips (textbook Arpit et al. arXiv:1706.05394 generalize-then-memorize dynamics,
already documented elsewhere in this project). This makes checkpoint selection extremely sensitive to
a single early snapshot.

feedback_modern_dl_training_checklist already found (single seed, on a DIFFERENT candidate script,
faithful_tabm_batchensemble) that lr=2e-4 (10x smaller than the current live lr=2e-3) keeps almost the
same peak val_bacc but crashes far more slowly -- a much wider "good" window instead of a 1-2 epoch
spike. But that finding was never cleanly isolated on the ACTUAL canonical config: every later test of
lr=2e-4 bundled it together with cosine LR schedule + AdaBelief optimizer + GCE loss + Prechelt
selection criterion, and that FULL bundle lost to the plain lr=2e-3 OLD recipe in an N=5-seed paired
test (docs/experiments -- see the "FINAL correction" note in feedback_modern_dl_training_checklist).
Bundling doesn't tell us whether lr alone (with everything else held at its current live default:
AdamW, plain CE, combined-val-loss selection, patience=8, k=8, std=0.03 init, same_as_direction
quality) would help or hurt. This script isolates exactly that one variable.

Single seed (260816), single expert (bull, for consistency with the other three cheap_gates today),
true 102(+13pos)=115-feature live pipeline, everything else at current live defaults.
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

import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as canon  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_lr_isolation_cheap_gate_20260816"
EXPERT = "bull"
SEED = 260816
EPOCHS = 40  # raised from the usual 28 -- lr=2e-4's whole point is a slower crash, needs room to show it
LR_VALUES = [2.0e-3, 2.0e-4]  # 2e-3 = current live default


def log(msg: str) -> None:
    print(f"[lr_isolation_cheap_gate] {msg}", flush=True)


def _fit_with_lr(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, lr: float) -> dict[str, Any]:
    """canon._fit_expert_3head's exact training loop (plain CE, k=8, std=0.03 default init,
    same_as_direction quality -- matching the real live zig075 config), with ONE change: CFG.lr."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    cfg = replace(canon.CFG, lr=float(lr))
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
    best_epoch = 0
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
        curve.append({"epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_bacc": round(dir_bacc, 5)})
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": dir_bacc}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    true_peak_bacc = max(c["direction_bacc"] for c in curve)
    true_peak_epoch = next(c["epoch"] for c in curve if c["direction_bacc"] == true_peak_bacc)
    return {
        "lr": float(lr),
        "epochs_ran": int(last_epoch),
        "early_stop_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "true_peak_direction_bacc": float(true_peak_bacc),
        "true_peak_epoch": int(true_peak_epoch),
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(SEED)
    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seed={SEED} lr_values={LR_VALUES} ===")
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
        "design": "LR isolation cheap_gate -- lr in {2e-3(current live default),2e-4}, everything else (AdamW, plain CE, k=8, std=0.03 init, same_as_direction quality, patience=8, combined-val-loss selection) held at current live defaults, single seed/expert, true 115-feature pipeline.",
        "seed": SEED,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "lr_values_tested": LR_VALUES,
        "runs": {},
    }
    for lr in LR_VALUES:
        res = _fit_with_lr(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, lr=lr)
        report["runs"][f"lr_{lr}"] = res
        bc = res["best_components"]
        log(
            f"  lr={lr} epochs_ran={res['epochs_ran']} early_stop_epoch={res['early_stop_epoch']} best_val_loss={res['best_validation_loss']:.4f} "
            f"selected_dir_bacc={bc['direction_balanced_accuracy']:.4f} true_peak_bacc={res['true_peak_direction_bacc']:.4f}@epoch{res['true_peak_epoch']} ({res['train_seconds']}s)"
        )

    baseline = report["runs"]["lr_0.002"]
    candidate = report["runs"]["lr_0.0002"]
    verdict = {
        "selected_dir_bacc_delta": candidate["best_components"]["direction_balanced_accuracy"] - baseline["best_components"]["direction_balanced_accuracy"],
        "true_peak_dir_bacc_delta": candidate["true_peak_direction_bacc"] - baseline["true_peak_direction_bacc"],
        "early_stop_epoch_delta": candidate["early_stop_epoch"] - baseline["early_stop_epoch"],
        "epochs_ran_delta": candidate["epochs_ran"] - baseline["epochs_ran"],
    }
    report["verdict"] = verdict
    log(f"=== VERDICT: selected_dir_bacc_delta={verdict['selected_dir_bacc_delta']:+.4f} true_peak_dir_bacc_delta={verdict['true_peak_dir_bacc_delta']:+.4f} early_stop_epoch_delta={verdict['early_stop_epoch_delta']:+d} epochs_ran_delta={verdict['epochs_ran_delta']:+d} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

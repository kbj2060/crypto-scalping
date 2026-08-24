#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate for the quality_head target-separation candidate.

CORRECTION to docs/experiments/eth_odyssey4_dl_reference_deep_analysis_20260816.md Section 3.2: that
section's claim ("quality_head duplicates direction_head's target") was based on
scripts/train_eval_omega1_2_tabm_3head_20260603.py, which turns out NOT to be the script that
produced the deployed bundle (its MODEL_ID is omega1_2_true_3head_tabm_20260603, not
omega4_3head_parent72_loose_entry_quality_20260620 -- the deployed bundle's real model_id). The real
production script is scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py, whose
--quality-mode DEFAULT is "hard_rule" (a genuinely separate target), not "same_as_direction". The
authoritative live contract (docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md
line 63) confirms h48qual already uses a real separate 48-bar ATR barrier quality target -- but
**zig075 specifically is documented there as quality_mode=same_as_direction**, a genuine, real,
live duplication (not an artifact of the one isolated "formal5seed" bundle this session happened to
diagnose first). So the candidate is still real, just narrower in scope than first framed: does
switching zig075's quality_head from same_as_direction to the already-implemented
risk_adjusted_barrier_meta_action target (same function this script already has, just an unused
--quality-mode choice for zig075) change anything?

Since the quality TARGET itself differs between conditions, quality_balanced_accuracy is not
comparable across conditions (each is scored against its own, different target) -- the fair A/B
metric is direction_balanced_accuracy (direction_head's target is unchanged: zigzag_action in both
conditions), testing whether a genuinely complementary auxiliary task helps the shared trunk versus
a redundant one. quality_balanced_accuracy against each condition's OWN target is reported as a
secondary, non-comparable diagnostic only.

Single seed, single expert (bull, for consistency with other cheap_gates this session), true
102(+13pos)=115-feature live pipeline, plain CE (matches the real live loss).
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
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent72  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_quality_target_separation_cheap_gate_20260816"
EXPERT = "bull"
SEED = 260816
EPOCHS = 28
# risk_adjusted_barrier_meta_action defaults, matching this repo's own report.json-recorded values
# (net_return_after_cost_min=0.001, mae_max=0.01, mfe_mae_min=1.2, max_hold_bars=288)
MIN_EDGE = 0.001
MAX_MAE = 0.01
MIN_MFE_MAE = 1.2
MAX_HOLD_BARS = 288


def log(msg: str) -> None:
    print(f"[quality_target_cheap_gate] {msg}", flush=True)


def _fit_with_quality_target(x_dir, y_dir, y_qual, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, condition: str) -> dict[str, Any]:
    """canon._fit_expert_3head's exact training loop (plain CE, matching the real live loss), with
    ONE change: y_qual is passed in explicitly instead of being forced equal to y_dir."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    cfg = canon.CFG
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = canon._standardize_fit(x_all)
    x_dir_np = canon._standardize_apply(x_dir, scaler)
    x_exit_np = canon._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = canon._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = canon._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
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
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]), torch.from_numpy(dir_w[train_idx]), torch.from_numpy(qual_w[train_idx]),
    )
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
        for xb, yb, yqb, wb, qwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, yqb, wb, qwb = xb.to(device), yb.to(device), yqb.to(device), wb.to(device), qwb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_qual_k = torch.nn.functional.cross_entropy(out_dir["quality"].reshape(-1, 3), yqb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
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
            vyq = torch.from_numpy(y_qual_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vyq[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(cfg.quality_loss_weight) * vqual_loss + float(cfg.exit_loss_weight) * vex_loss
            dir_pred = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            qual_pred = torch.softmax(vo["quality"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            dir_bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred))
            qual_bacc = float(balanced_accuracy_score(vyq.cpu().numpy(), qual_pred))
        curve.append({"epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_bacc": round(dir_bacc, 5), "quality_bacc_own_target": round(qual_bacc, 5)})
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {
                "direction_val_loss": vdir_loss,
                "quality_val_loss": vqual_loss,
                "exit_val_loss": vex_loss,
                "direction_balanced_accuracy": dir_bacc,
                "quality_balanced_accuracy_own_target": qual_bacc,
            }
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    return {
        "condition": condition,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "epochs_ran": int(last_epoch),
        "early_stop_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(SEED)
    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seed={SEED} ===")
    frames = truepipe.prepare_frames_true(disable_tp_sl=False)
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_dir = train_raw["zigzag_action"].to_numpy(dtype=np.int64)

    log("  building risk_adjusted_barrier_meta_action quality target (forward barrier simulation)...")
    y_qual_barrier, quality_diag = parent72._quality_target_risk_adjusted_barrier_meta_action(
        train_raw, fee=fee, slip=slip, cost_mult=3.0,
        min_edge=MIN_EDGE, max_mae=MAX_MAE, min_mfe_mae=MIN_MFE_MAE, max_hold_bars=MAX_HOLD_BARS,
    )
    log(f"  quality_diag: active_rows={quality_diag['active_rows']} positive_rate_active={quality_diag['positive_rate_active']:.4f} reason_counts={quality_diag['reason_counts']}")
    y_qual_same = y_dir.copy()
    log(f"  class counts -- same_as_direction: {np.bincount(y_qual_same, minlength=3).tolist()}, risk_adjusted_barrier: {np.bincount(y_qual_barrier, minlength=3).tolist()}")

    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {
        "design": "quality_head target-separation cheap_gate -- same_as_direction (current live zig075 config) vs risk_adjusted_barrier_meta_action (already-implemented, unused-for-zig075 alternative), single seed/expert, plain-CE, true 115-feature pipeline.",
        "seed": SEED,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "quality_target_thresholds": {"min_edge": MIN_EDGE, "max_mae": MAX_MAE, "min_mfe_mae": MIN_MFE_MAE, "max_hold_bars": MAX_HOLD_BARS},
        "quality_target_diag": quality_diag,
        "runs": {},
    }
    for condition, y_qual in (("same_as_direction", y_qual_same), ("risk_adjusted_barrier", y_qual_barrier)):
        res = _fit_with_quality_target(x_train, y_dir, y_qual, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, condition=condition)
        report["runs"][condition] = res
        bc = res["best_components"]
        log(
            f"  condition={condition} early_stop_epoch={res['early_stop_epoch']} best_val_loss={res['best_validation_loss']:.4f} "
            f"dir_bacc={bc['direction_balanced_accuracy']:.4f} qual_bacc_own_target={bc['quality_balanced_accuracy_own_target']:.4f} ({res['train_seconds']}s)"
        )

    baseline = report["runs"]["same_as_direction"]
    candidate = report["runs"]["risk_adjusted_barrier"]
    verdict = {
        "dir_bacc_delta": candidate["best_components"]["direction_balanced_accuracy"] - baseline["best_components"]["direction_balanced_accuracy"],
        "dir_val_loss_delta": candidate["best_components"]["direction_val_loss"] - baseline["best_components"]["direction_val_loss"],
        "early_stop_epoch_delta": candidate["early_stop_epoch"] - baseline["early_stop_epoch"],
        "note": "quality_balanced_accuracy is NOT comparable across conditions (different targets) -- dir_bacc_delta is the fair A/B metric.",
    }
    report["verdict"] = verdict
    log(f"=== VERDICT: dir_bacc_delta={verdict['dir_bacc_delta']:+.4f} dir_val_loss_delta={verdict['dir_val_loss_delta']:+.4f} early_stop_epoch_delta={verdict['early_stop_epoch_delta']:+d} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

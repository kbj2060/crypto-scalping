#!/usr/bin/env python3
"""RESEARCH ONLY -- capacity-reduction follow-up to Step B's decisive negative result
(docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md): the piecewise-linear
embedding on top of R+S+B (hidden=192, d_embed=8, n_bins=16 -> n_params=410,448, +224% vs the
118,552-param baseline_R_only) made direction_balanced_accuracy consistently WORSE across all 3
regime experts (bull -0.051, bear -0.041, chop -0.037), single seed. User asked to test whether
SHRINKING the model (fewer hidden units, smaller embedding dim, fewer bins) recovers some of that
loss -- i.e. is the embedding CONCEPT bad, or just the specific (oversized-for-this-project)
capacity choice on top of it, consistent with the whole session's "complexity hurts under weak
signal" pattern.

Tests a capacity sweep of (hidden, d_embed, n_bins) configs, architecture=full_R_S_B_embed only
(the one that showed the negative effect), single expert=bull (for direct comparability with the
other curve/LR-sweep/combo-regularizer diagnostics), seed=260816, FIXED 40-epoch budget with no
early stopping (full curve logged, not just a best-checkpoint summary) so the results are directly
comparable to research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816's already-
collected baseline_R_only curve (118,552 params, no embedding: val_bacc peaked 0.574 at epoch 2,
degraded to 0.492 by epoch 40) and the full_R_S_B_embed single-seed cheap_gate numbers above (those
used early stopping at patience=8, ~9-10 epochs -- this script's full curves let us see the SAME
architecture's full trajectory, not just its early-stopped snapshot).

Configs (n_params reported by the script, not hand-computed -- exact counts depend on CFG.hidden):
  current   : hidden=192, d_embed=8, n_bins=16  (already known negative, included as an in-script
              reference point re-run under fixed-epoch/no-early-stopping for a fair curve comparison)
  quarter   : hidden=96,  d_embed=4, n_bins=8
  eighth    : hidden=64,  d_embed=2, n_bins=8
  tiny      : hidden=48,  d_embed=2, n_bins=4

Reuses research_eth_candidate_faithful_tabm_batchensemble_stepb_embed_20260816.py's
PiecewiseLinearEmbedding/ThreeHeadTabMFullEmbed/_quantile_bin_edges and
curve_diag's _prepare_frames_light unmodified via import -- only the capacity sweep loop (CFG.hidden
varied via dataclasses.replace) and per-epoch curve logging (no early stopping) are new.
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

gate = stepb.gate
base = stepb.base
hard = stepb.hard
curve_diag = gate.nseed if hasattr(gate, "nseed") else None
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_stepb_capacity_sweep_20260816"
SEED = 260816
EXPERT = "bull"
EPOCHS = 40  # fixed, no early stopping -- see full trajectory like the other curve diagnostics

CONFIGS = [
    {"name": "current", "hidden": 192, "d_embed": 8, "n_bins": 16},
    {"name": "quarter", "hidden": 96, "d_embed": 4, "n_bins": 8},
    {"name": "eighth", "hidden": 64, "d_embed": 2, "n_bins": 8},
    {"name": "tiny", "hidden": 48, "d_embed": 2, "n_bins": 4},
]


def log(msg: str) -> None:
    print(f"[faithful_tabm_stepb_capacity] {msg}", flush=True)


def _fit_one_curve_embed(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device, cfg, d_embed, n_bins):
    """Full-curve (no early stopping) variant of stepb._fit_one_embed, logging every epoch."""
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

    bin_edges = stepb._quantile_bin_edges(_x_np, n_bins)
    model = stepb.ThreeHeadTabMFullEmbed(x_dir_np.shape[1], cfg=cfg, bin_edges=bin_edges, d_embed=d_embed).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)

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
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(cfg.quality_loss_weight) * vqual_loss + float(cfg.exit_loss_weight) * vex_loss
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
        curve.append({"epoch": epoch + 1, "train_loss": round(train_loss, 5), "val_loss": round(vloss, 5), "direction_balanced_accuracy_val": round(bacc, 5)})
    return curve, n_params


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} epochs_fixed={EPOCHS} configs={[c['name'] for c in CONFIGS]} ===")
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

    original_cfg = stepb.CFG
    report: dict[str, Any] = {"design": "capacity-reduction sweep on full_R_S_B_embed, fixed epochs, no early stopping.", "seed": SEED, "expert": EXPERT, "epochs_fixed": EPOCHS, "configs": CONFIGS, "curves": {}}
    t0 = time.time()
    for cfg_spec in CONFIGS:
        name = cfg_spec["name"]
        cfg = dataclasses.replace(original_cfg, hidden=int(cfg_spec["hidden"]))
        log(f"=== stage=train config={name} hidden={cfg_spec['hidden']} d_embed={cfg_spec['d_embed']} n_bins={cfg_spec['n_bins']} ===")
        curve, n_params = _fit_one_curve_embed(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, cfg=cfg, d_embed=int(cfg_spec["d_embed"]), n_bins=int(cfg_spec["n_bins"]))
        report["curves"][name] = {"n_params": n_params, "curve": curve}
        best_bacc = max(curve, key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {name}: n_params={n_params} best val_bacc={best_bacc['direction_balanced_accuracy_val']:.4f} at epoch {best_bacc['epoch']}/{EPOCHS}; "
            f"final(epoch{EPOCHS}) val_bacc={curve[-1]['direction_balanced_accuracy_val']:.4f} (elapsed={time.time()-t0:.0f}s)")

    log("=== stage=summary ===")
    for cfg_spec in CONFIGS:
        name = cfg_spec["name"]
        d = report["curves"][name]
        best_bacc = max(d["curve"], key=lambda r: r["direction_balanced_accuracy_val"])
        log(f"  {name:8s} n_params={d['n_params']:7d} peak_val_bacc={best_bacc['direction_balanced_accuracy_val']:.4f} @epoch{best_bacc['epoch']}")
    log("  reference: baseline_R_only (no embedding, 118,552 params) peaked val_bacc=0.5740 at epoch 2 (curve_diag_20260816)")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done total_seconds={time.time()-t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""RESEARCH ONLY -- N>=5 genuinely random seed confirmation for the rank-r BatchEnsemble gate
candidate (LoMETab, arXiv:2605.14365), following up on the single-seed cheap_gate
(research_eth_odyssey4_rankr_gate_cheap_gate_20260817.py,
docs/experiments/eth_odyssey4_dl_reference_deep_analysis_20260816.md §6.7).

Two things the cheap_gate left unresolved, both fixed here:

1. **Selection-criterion collapse re-confirmed only on 1 seed.** The cheap_gate found the same
   pattern as §2.5 (lr=2e-4): `combined val_loss` early-stopping selected epoch 1, but the true
   `direction_balanced_accuracy` peak was at epoch 4-5 for r=2/3 (selection gap -0.0093). This
   script reports BOTH the selected-checkpoint delta (what would actually get deployed under the
   current selection rule) and the true-peak delta (the architecture's real capability ceiling),
   across N=5 seeds, matching this repo's `feedback_modern_dl_training_checklist` methodology.

2. **Diversity was only measured at the selected (epoch-1) checkpoint, never at the true peak.**
   The cheap_gate's collapse diagnostic (`_collapse_stats`: pairwise top-confidence correlation,
   argmax unanimity) was only recomputed when val_loss hit a new best -- which for every rank
   stopped happening after epoch 1, so the diagnostic never got to see whether the newly added
   rank>=2 gate channels had grown any real diversity by the time direction_balanced_accuracy
   actually peaked (epoch 4-5). This script computes the collapse diagnostic at EVERY epoch and
   reports it at both the selected checkpoint and the true peak, closing that gap.

Protocol: N=5 genuinely random seeds (`secrets.randbelow`, not a fixed-increment cluster --
Seed-Diversity Ensemble Promotion Gate), expert=bull, plain CE, true 115-feature live pipeline,
r in {1(=current live, B frozen control), 2, 3}, sigma_init=0.03 fixed (same axis isolation as the
cheap_gate). This is the N>=5-seed confirmation stage of this repo's standard gate ladder
(cheap_gate -> N>=5 seed -> VAL/OOS fresh-forward); VAL/OOS is NOT run here.
"""
from __future__ import annotations

import json
import secrets
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

import research_eth_odyssey4_rankr_gate_cheap_gate_20260817 as rankr  # noqa: E402

canon = rankr.canon
exit_head = rankr.exit_head
hard = rankr.hard
truepipe = rankr.truepipe
RankRThreeHeadTabM = rankr.RankRThreeHeadTabM
_collapse_stats = rankr._collapse_stats

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_rankr_gate_nseed_confirm_20260817"
EXPERT = "bull"
EPOCHS = 28
N_SEEDS = 5
SEEDS = sorted(secrets.randbelow(900_000_000) + 100_000_000 for _ in range(N_SEEDS))
RANKS = [1, 2, 3]


def log(msg: str) -> None:
    print(f"[rankr_nseed] {msg}", flush=True)


def _fit_rankr_full_curve(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, rank: int) -> dict[str, Any]:
    """Same training loop as the cheap_gate's _fit_rankr, but records the collapse diagnostic at
    EVERY epoch (not just new-best-val-loss epochs), so the true-peak epoch's diversity can be
    read off after the fact regardless of which epoch the early-stopping rule selected."""
    torch.manual_seed(int(seed) + int(expert_idx) + int(rank) * 137)
    np.random.seed(int(seed) + int(expert_idx) + int(rank) * 137)
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

    model = RankRThreeHeadTabM(x_dir_np.shape[1], cfg=cfg, rank=rank, fixed_b_at_r1=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    last_epoch = 0
    curve: list[dict[str, Any]] = []  # every epoch: val_loss, dir_bacc, AND collapse stats
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
            loss_dir_k = F.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_qual_k = F.cross_entropy(out_dir["quality"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            out_exit = model(xe)
            loss_exit_k = F.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
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
            vdir = F.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = F.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = F.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(cfg.quality_loss_weight) * vqual_loss + float(cfg.exit_loss_weight) * vex_loss
            dir_pred = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            dir_bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred))
            collapse = _collapse_stats(vo["direction"])  # computed EVERY epoch now, not just new-best
        curve.append({
            "epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_bacc": round(dir_bacc, 5),
            "direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss,
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
        "rank": int(rank),
        "n_params": int(n_params),
        "epochs_ran": int(last_epoch),
        "selected_epoch": int(best_epoch),
        "selected_dir_bacc": selected["direction_bacc"],
        "selected_val_loss": selected["val_loss"],
        "selected_collapse": selected["collapse"],
        "true_peak_epoch": int(true_peak["epoch"]),
        "true_peak_dir_bacc": true_peak["direction_bacc"],
        "true_peak_collapse": true_peak["collapse"],
        "selection_gap": selected["direction_bacc"] - true_peak["direction_bacc"],
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
    }


def _delta_summary(rank_runs: list[dict[str, Any]], ctrl_runs: list[dict[str, Any]], *, metric_key: str) -> dict[str, Any]:
    deltas = [r[metric_key] - c[metric_key] for r, c in zip(rank_runs, ctrl_runs)]
    arr = np.asarray(deltas, dtype=np.float64)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    return {
        "deltas": deltas,
        "mean_delta": float(arr.mean()),
        "std_delta": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_improved": n_pos,
        "n_worsened": n_neg,
        "sign_consistent": bool(n_pos == 0 or n_neg == 0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(260816)
    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seeds={SEEDS} ranks={RANKS} ===")
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
        "design": "Rank-r BatchEnsemble gate (LoMETab, arXiv:2605.14365) N>=5 seed confirmation -- fixes 2 cheap_gate gaps: (1) reports true-peak delta alongside selected-checkpoint delta, (2) collapse diagnostic now recorded every epoch so true-peak diversity is measurable.",
        "seed_source": "secrets.randbelow (genuinely random, not fixed-increment) -- Seed-Diversity Ensemble Promotion Gate",
        "seeds": SEEDS,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "ranks_tested": RANKS,
        "runs": {f"rank_{r}": [] for r in RANKS},
    }
    t_start = time.time()
    for seed in SEEDS:
        for rank in RANKS:
            res = _fit_rankr_full_curve(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device, rank=rank)
            report["runs"][f"rank_{rank}"].append(res)
            log(
                f"  seed={seed} rank={rank}: selected(ep{res['selected_epoch']})={res['selected_dir_bacc']:.4f} "
                f"true_peak(ep{res['true_peak_epoch']})={res['true_peak_dir_bacc']:.4f} gap={res['selection_gap']:+.4f} "
                f"selected_corr={res['selected_collapse']['mean_pairwise_corr_of_top_confidence']:.4f} "
                f"true_peak_corr={res['true_peak_collapse']['mean_pairwise_corr_of_top_confidence']:.4f} "
                f"({res['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)"
            )

    ctrl_runs = report["runs"]["rank_1"]
    summary: dict[str, Any] = {}
    for rank in RANKS:
        if rank == 1:
            continue
        rank_runs = report["runs"][f"rank_{rank}"]
        sel = _delta_summary(rank_runs, ctrl_runs, metric_key="selected_dir_bacc")
        peak = _delta_summary(rank_runs, ctrl_runs, metric_key="true_peak_dir_bacc")
        corr_at_peak = [r["true_peak_collapse"]["mean_pairwise_corr_of_top_confidence"] for r in rank_runs]
        corr_at_peak_ctrl = [c["true_peak_collapse"]["mean_pairwise_corr_of_top_confidence"] for c in ctrl_runs]
        corr_deltas = [a - b for a, b in zip(corr_at_peak, corr_at_peak_ctrl)]
        summary[f"rank_{rank}_vs_rank_1"] = {
            "selected_checkpoint": sel,
            "true_peak": peak,
            "true_peak_diversity_corr_delta_mean": float(np.mean(corr_deltas)),
            "true_peak_diversity_corr_deltas": corr_deltas,
        }
        log(
            f"=== VERDICT rank_{rank}_vs_1: selected mean_delta={sel['mean_delta']:+.4f} std={sel['std_delta']:.4f} "
            f"improved={sel['n_improved']}/{N_SEEDS} sign_consistent={sel['sign_consistent']} | "
            f"true_peak mean_delta={peak['mean_delta']:+.4f} std={peak['std_delta']:.4f} "
            f"improved={peak['n_improved']}/{N_SEEDS} sign_consistent={peak['sign_consistent']} | "
            f"true_peak_diversity_corr_delta_mean={np.mean(corr_deltas):+.4f} ==="
        )
    report["summary"] = summary

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path} (total elapsed {time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

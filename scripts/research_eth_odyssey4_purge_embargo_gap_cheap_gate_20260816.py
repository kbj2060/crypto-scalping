#!/usr/bin/env python3
"""RESEARCH ONLY -- C1 cheap_gate (Odyssey4 layer/parameter improvement proposal 20260816).

_fit_expert_3head's internal 85/15 direction/quality split (train_idx=arange(split),
val_idx=arange(split,n), scripts/train_eval_omega1_2_tabm_3head_20260603.py lines ~257-259) has
zero purge/embargo gap between train and val. B2's measurement
(scripts/diagnose_odyssey4_zigzag_pivot_confirmation_delay_20260816.py) found the live
zigzag_action_labels_20260531 label set's pivot-confirmation delay has combined p95=54 bars (over
all years) -- i.e. training rows within ~54 bars of the val boundary may have zigzag_action labels
that were only confirmed using price action that reaches into the validation window.

This script is the CHEAP_GATE step: does adding a 54-bar embargo gap actually change the val-loss
curve / early-stopping epoch, for the SAME seed/expert/CFG as the canonical script? If yes, that is
evidence the current (gap=0) split has optimistic bias worth fixing with a real N>=5-seed
comparison. If no material change, that is a valid negative result to document as-is (no further
experiment needed) -- see the proposal doc's explicit gating rule for C1.

Single seed, single expert (bull, for consistency with A1/other recent research on this
architecture), 2 configs (gap=0 vs gap=54), reusing gate._prepare_frames_light() (established
bypass for the vsnlstm/chronos blocker -- see diagnose_odyssey4_expert_effective_sample_size_
20260816.py's docstring).

NOTE: A1 (GCE port) was tested and REVERTED (docs/experiments/eth_odyssey4_gce_canonical_port_
20260816.md -- did not transfer, N=5 seed, 4/5 worse). The canonical script's actual training
loss is plain cross_entropy, unchanged from before this session. This script uses plain CE to
match that real baseline (not GCE).
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

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402

canon = gate.base
hard = gate.hard
CFG = gate.CFG

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_purge_embargo_gap_cheap_gate_20260816"
EXPERT = "bull"
SEED = 260816
EPOCHS = 28
EMBARGO_GAP_BARS = 54  # B2 combined all-years p95 confirm_delay_bars


def log(msg: str) -> None:
    print(f"[purge_embargo_cheap_gate] {msg}", flush=True)


def _fit_with_gap(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, embargo_gap: int) -> dict[str, Any]:
    """gate._fit_one's training loop (plain CE, matching the real canonical script -- A1's GCE
    port was reverted), with ONE change: an embargo_gap purged between train_idx and val_idx
    (direction/quality split only -- the exit split is untouched, since exit labels come from a
    different, non-pivot-confirmation-based recipe -- see B2/C1 scope in the proposal doc)."""
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
    train_end = max(1, split - int(embargo_gap))
    train_idx = np.arange(train_end)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = canon.ThreeHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
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
        curve.append({"epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_balanced_accuracy_val": round(bacc, 5)})
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": bacc}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    return {
        "embargo_gap_bars": int(embargo_gap),
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
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} seed={SEED} embargo_gap={EMBARGO_GAP_BARS} ===")
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

    report: dict[str, Any] = {"design": "C1 cheap_gate -- embargo gap=0 vs gap=54 (B2 p95), single seed/expert, plain-CE canonical loss (A1's GCE port was reverted).", "seed": SEED, "expert": EXPERT, "epochs_budget": EPOCHS, "embargo_gap_bars_tested": EMBARGO_GAP_BARS, "runs": {}}
    for gap in (0, EMBARGO_GAP_BARS):
        res = _fit_with_gap(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, embargo_gap=gap)
        report["runs"][f"gap_{gap}"] = res
        log(f"  gap={gap:3d} early_stop_epoch={res['early_stop_epoch']} best_val_loss={res['best_validation_loss']:.4f} dir_bacc={res['best_components']['direction_balanced_accuracy']:.4f} train_rows={res['train_rows']} ({res['train_seconds']}s)")

    r0 = report["runs"]["gap_0"]
    r54 = report["runs"][f"gap_{EMBARGO_GAP_BARS}"]
    verdict = {
        "early_stop_epoch_delta": r54["early_stop_epoch"] - r0["early_stop_epoch"],
        "best_val_loss_delta": r54["best_validation_loss"] - r0["best_validation_loss"],
        "dir_bacc_delta": r54["best_components"]["direction_balanced_accuracy"] - r0["best_components"]["direction_balanced_accuracy"],
    }
    report["verdict"] = verdict
    log(f"=== VERDICT: early_stop_epoch_delta={verdict['early_stop_epoch_delta']:+d} best_val_loss_delta={verdict['best_val_loss_delta']:+.4f} dir_bacc_delta={verdict['dir_bacc_delta']:+.4f} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

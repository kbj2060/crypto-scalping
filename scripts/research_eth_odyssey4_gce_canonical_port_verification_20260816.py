#!/usr/bin/env python3
"""RESEARCH ONLY -- A1 verification (Odyssey4 layer/parameter improvement proposal 20260816).

Verifies that porting GCE (q=0.7) into scripts/train_eval_omega1_2_tabm_3head_20260603.py's
direction_head/quality_head training losses (already done in that file -- see its gce_loss
function and the two `gce_loss(out_dir[...], yb)` calls inside _fit_expert_3head) actually
transfers a real improvement on direction_balanced_accuracy IN THIS EXACT SCRIPT, rather than just
assuming the result from the isolation test that first found it
(research_eth_candidate_faithful_tabm_batchensemble_regularizer_isolation_20260816.py, val bacc
0.5758 vs 0.5740, single seed=260816, expert=bull, fixed 40-epoch budget/no early stopping).

This script instead uses the CANONICAL script's real training config end to end: same CFG
(quality_loss_weight=0.80, exit_loss_weight=1.15, patience=8 early stopping, k=8/hidden=192/
layers=3), same 85/15 internal split, same expert=bull (matching the isolation test's scope for a
fair citation), across N>=5 GENUINELY RANDOM seeds (secrets.randbelow, not a fixed-increment
cluster -- Seed-Diversity Ensemble Promotion Gate, CLAUDE.md).

Two variants trained per seed:
  - baseline_ce : plain per-member cross_entropy on direction/quality (the pre-A1 canonical loss)
  - gce_ported  : canon.gce_loss(q=0.7) on direction/quality (the post-A1 canonical loss, as now
                  actually shipped in scripts/train_eval_omega1_2_tabm_3head_20260603.py)
Both use plain cross_entropy for exit_head and for the validation-loss/early-stopping metric --
same scope as the isolation test (exit_head was explicitly out of scope there) and same
methodology (that test's own val loss was always plain CE regardless of variant, so this script's
early-stopping selection stays outcome-comparable across baseline/gce).

baseline_ce's training loop is scripts/research_eth_candidate_faithful_tabm_batchensemble_cheap_
gate_20260816.py's _fit_one() reused UNMODIFIED via import (already plain CE, already computes
direction_balanced_accuracy in best_components) -- not re-derived. gce_ported is a twin of that
same function with only the direction/quality loss calls swapped for canon.gce_loss (imported from
the now-patched canonical script, not redefined here), so any difference in the comparison is
exactly the loss-function change, matching cheap_gate's exact split/CFG/early-stopping code
otherwise line for line.

Note: exit-dataset construction uses max_samples=60000 (vs 0/unbounded upstream) -- this dev
machine has only 15GB RAM and this whole prep chain used ~13-14GB unbounded, tripping OOM against
concurrent sessions once during this run. exit_head stays plain CE in both variants regardless of
this cap, so it does not bias the direction/quality GCE-vs-CE comparison under test.

fresh_forward_bar_by_bar=n/a (classifier training only, val bacc on the internal 85/15 split --
same methodology as the original isolation test this reproduces; no backtest/portfolio ledger
touched, no promotion claim made here beyond "does GCE's win transfer into the canonical script").
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402

canon = gate.base  # train_eval_omega1_2_tabm_3head_20260603, already GCE-ported (A1)
hard = gate.hard
CFG = gate.CFG

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_gce_canonical_port_verification_20260816"
EXPERT = "bull"  # matches the original isolation test's scope, for a fair citation
EPOCHS = 28  # canonical script's own default (main()'s --epochs default), not the isolation test's fixed 40
N_SEEDS = 5
SEEDS = sorted(secrets.randbelow(900_000_000) + 100_000_000 for _ in range(N_SEEDS))  # genuinely random, not fixed-increment


def log(msg: str) -> None:
    print(f"[gce_canonical_port_verify] {msg}", flush=True)


def _fit_one_gce(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device) -> dict[str, Any]:
    """Twin of gate._fit_one, with ONLY the direction/quality training loss calls swapped from
    plain cross_entropy to canon.gce_loss (q=0.7). Split logic, CFG, exit_head loss, and the
    validation-loss/early-stopping metric are byte-identical to gate._fit_one."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    x_all_concat = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = canon._standardize_fit(x_all_concat)
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

    model = canon.ThreeHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state = None
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
            loss_dir_k = canon.gce_loss(out_dir["direction"], yb)  # <-- A1 change under test
            loss_qual_k = canon.gce_loss(out_dir["quality"], yb)  # <-- A1 change under test
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
            # Validation/early-stopping metric stays PLAIN CE regardless of variant -- same
            # methodology as the isolation test this reproduces (its val loss was always CE too).
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(CFG.quality_loss_weight) * vqual_loss + float(CFG.exit_loss_weight) * vex_loss
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": bacc}
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    return {
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "n_params": int(n_params),
        "epochs_ran": int(last_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(260816)
    log(f"=== stage=prepare_frames (light, shared) expert={EXPERT} epochs_budget={EPOCHS} seeds={SEEDS} ===")
    frames = gate._prepare_frames_light()
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {
        "design": "A1 GCE canonical-port verification -- baseline plain CE vs GCE(q=0.7) on direction/quality heads, expert=bull, canonical CFG/early-stopping.",
        "seeds": SEEDS,
        "seed_source": "secrets.randbelow (genuinely random draws, not a fixed-increment cluster) -- CLAUDE.md Seed-Diversity Ensemble Promotion Gate",
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "results": {"baseline_ce": [], "gce_ported": []},
    }
    t_start = time.time()
    for seed in SEEDS:
        res_base = gate._fit_one(canon.ThreeHeadTabM, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device)
        report["results"]["baseline_ce"].append(res_base)
        log(f"  baseline_ce seed={seed}: val_loss={res_base['best_validation_loss']:.4f} dir_bacc={res_base['best_components']['direction_balanced_accuracy']:.4f} ({res_base['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)")

        res_gce = _fit_one_gce(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=seed, epochs=EPOCHS, device=device)
        report["results"]["gce_ported"].append(res_gce)
        log(f"  gce_ported  seed={seed}: val_loss={res_gce['best_validation_loss']:.4f} dir_bacc={res_gce['best_components']['direction_balanced_accuracy']:.4f} ({res_gce['train_seconds']}s, elapsed={time.time()-t_start:.0f}s)")

    b_runs = report["results"]["baseline_ce"]
    g_runs = report["results"]["gce_ported"]
    deltas_bacc = [g["best_components"]["direction_balanced_accuracy"] - b["best_components"]["direction_balanced_accuracy"] for b, g in zip(b_runs, g_runs)]
    arr = np.asarray(deltas_bacc, dtype=np.float64)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    summary = {
        "baseline_ce_mean_dir_bacc": float(np.mean([b["best_components"]["direction_balanced_accuracy"] for b in b_runs])),
        "gce_ported_mean_dir_bacc": float(np.mean([g["best_components"]["direction_balanced_accuracy"] for g in g_runs])),
        "dir_bacc_deltas_gce_minus_baseline": deltas_bacc,
        "dir_bacc_mean_delta": float(arr.mean()),
        "dir_bacc_std_delta": float(arr.std(ddof=1)),
        "n_seeds_improved": n_pos,
        "n_seeds_worsened": n_neg,
        "sign_consistent": bool(n_pos == 0 or n_neg == 0),
    }
    report["summary"] = summary
    log(f"=== SUMMARY: baseline_mean={summary['baseline_ce_mean_dir_bacc']:.4f} gce_mean={summary['gce_ported_mean_dir_bacc']:.4f} mean_delta={summary['dir_bacc_mean_delta']:+.4f} improved={n_pos}/{len(SEEDS)} sign_consistent={summary['sign_consistent']} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

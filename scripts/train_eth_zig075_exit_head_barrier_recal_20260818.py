#!/usr/bin/env python3
"""RESEARCH ONLY -- retrain zig075's OWN exit head (frozen zig075 parent encoder/direction/
quality, exit_head only) with THREE dense-labeling parameters recalibrated together, following up
on [[eth_odyssey4_zig075_exit_head_threshold_review_20260817]] (deployed-scale giveback_min=0.65
confirmed to fire only after ~97.6% average MFE giveback -- "developing exit_head engagement" was
explicitly rejected as a goal there) and today's [[eth_odyssey4_exit_head_liveatr_barrier_and_
label_reaudit_20260818]] (adverse_unreal/min_mfe_for_giveback were never rescaled after the
barrier moved from the old fixed 2.6%/1.4% recipe to the current ATR floor-dominated ~7.5%/4.0%
one -- confirmed today the floor binds 96-99% of the time in every regime, so ~7.5%/4.0% IS the
practical barrier width now).

Recalibration (reasoning in the module docstring, not re-derived here -- see the 2026-08-18
session's chat record / experiment doc):
  adverse_unreal:        -0.010 -> -0.020  (25% of new 4.0% SL floor -> 50%, closer to the old
                          recipe's 71%-of-SL intent but more conservative)
  min_mfe_for_giveback:  +0.006 -> +0.015  (8% of new 7.5% TP floor -> 20%, restoring
                          approximately the old recipe's 23%-of-TP intent)
  giveback_min:           0.65  -> 0.45    (deployed 0.65 fires too late (97.6% empirical
                          giveback); a DIFFERENT parent/encoder at 0.25 fired 0% of the time
                          (eth_candidate Phase2, 2026-08-18) -- 0.45 is an unexplored midpoint,
                          not a re-test of either known extreme)
  terminal_window:        3 (unchanged -- not implicated by either finding above, held fixed so
                          any effect observed here is attributable to the other 3 changes only)

Reuses research_eth_omega461_exit_head_liveatr_relabel_20260813.py's _build_exit_dataset_entry_
label_live_atr_barrier UNMODIFIED (imported, not copied or edited) -- inherits BOTH the 2026-08-17
pos_tp/pos_sl fix and the 2026-08-18 pos_unrealized/pos_mfe/pos_mae raw-scale fix for free. Does
NOT further edit that shared file (a concurrent session may be editing it right now for the
remaining reaudit findings -- this script only imports and calls it with different keyword
arguments, avoiding any write conflict).

Frozen parent: zig075's OWN currently-deployed bundle (research_eth_omega461_exit_sweep_20260721.
COMPONENTS["zig075"]["bundle"]) -- NOT the eth_candidate unified quality-B parent used by the
sibling Phase 2 script. direction+quality heads are frozen (loaded, never updated); only the exit
head is retrained per expert (pricemove_retrain._fit_exit_head_only, unchanged).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. No live/shadow files touched. Does not overwrite the
deployed zig075 bundle (writes to a new OUT_DIR).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as pricemove_retrain  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_head_liveatr_relabel_20260813 as liveatr  # noqa: E402 -- reused unmodified, not copied
import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402

MODEL_ID = "eth_zig075_exit_head_barrier_recal_20260818"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_BUNDLE = base_sweep.COMPONENTS["zig075"]["bundle"]
ADVERSE_UNREAL = -0.020
MIN_MFE_FOR_GIVEBACK = 0.015
GIVEBACK_MIN = 0.45
TERMINAL_WINDOW = 3


def _fit_exit_head_unfrozen_encoder(
    baseline_payload: dict[str, Any], x_exit, y_exit, exit_route_frame, *,
    expert_idx: int, seed: int, epochs: int, device: torch.device, model_path: Path,
    encoder_lr_mult: float = 0.1,
) -> dict[str, Any]:
    """User request 2026-08-18: 'encoder도 freeze하지 말고 exit_head와 함께 학습'. Byte-for-byte
    copy of pricemove_retrain._fit_exit_head_only's data prep/train loop, except the encoder
    (input_scale/input_bias/in_proj/blocks/expert_scale/norms) is ALSO unfrozen and included in the
    optimizer, at encoder_lr_mult x the exit_head's own LR (standard fine-tune practice, mitigates
    the overfitting risk flagged when this was discussed earlier this session: ~1500 effectively-
    independent candidate segments is a small fine-tuning set relative to the now-much-larger
    trainable parameter count). direction_head/quality_head stay frozen -- this loop has no
    direction/quality loss term, so unfreezing them would leave them receiving zero gradient
    (dead weight, not a true joint multi-task retrain -- that would require reconstructing the
    original 3-head joint training loop, out of scope tonight, see chat record)."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    if list(x_exit.columns) != list(baseline_payload["scaler"]["columns"]):
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} feature column contract mismatch for exit-only retrain")
    x_np = parent._standardize_apply(x_exit, baseline_payload["scaler"])
    y_np = np.asarray(y_exit, dtype=np.int64)
    classes = sorted(np.unique(y_np).astype(int).tolist())
    if classes != [0, 1]:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} exit labels need both classes [0,1], got {classes}")
    route_probs = parent._route_probs(exit_route_frame)
    route_w = route_probs[:, int(expert_idx)].astype(np.float32)
    weights = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32) * route_w
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid exit-only sample weights")

    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 256))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_np[train_idx]), torch.from_numpy(weights[train_idx]))
    dl = DataLoader(ds, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    model = parent.ThreeHeadTabM(int(baseline_payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(baseline_payload["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    encoder_params = list(itertools.chain(
        [model.input_scale, model.input_bias], model.in_proj.parameters(),
        model.blocks.parameters(), model.expert_scale.parameters(), model.norms.parameters(),
    ))
    for p in encoder_params:
        p.requires_grad_(True)
    for p in model.exit_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": encoder_params, "lr": float(parent.CFG.lr) * float(encoder_lr_mult)},
        {"params": model.exit_head.parameters(), "lr": float(parent.CFG.lr)},
    ], weight_decay=float(parent.CFG.weight_decay))
    trainable = encoder_params + list(model.exit_head.parameters())
    best_state = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), wb.to(device, non_blocking=True)
            out = model(xb)
            loss = (pricemove_retrain._ce_tabm(out["exit"], yb) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx, vy, vw = torch.from_numpy(x_np[val_idx]).to(device), torch.from_numpy(y_np[val_idx]).to(device), torch.from_numpy(weights[val_idx]).to(device)
            vo = model(vx)
            val_loss = float(((pricemove_retrain._ce_tabm(vo["exit"], vy) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        **baseline_payload, "model_id": MODEL_ID, "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_exit_validation_loss": float(best_loss), "exit_epochs_ran": int(last_epoch),
        "frozen_contract": "direction_quality_frozen_encoder_and_exit_head_jointly_retrained",
        "encoder_lr_mult": float(encoder_lr_mult),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    return payload


def _retrain_exit_head_only(
    x_exit_raw, y_exit, frame_exit, *, seed: int, epochs: int, device: torch.device, out_dir: Path,
    parent_bundle: Path, unfreeze_encoder: bool = False, encoder_lr_mult: float = 0.1,
) -> dict[str, Any]:
    """Same freeze-encoder/direction/quality, retrain-exit-head-only pattern as the sibling Phase 2
    script's _retrain_exit_head_only -- reimplemented locally (not imported) only so saved model
    filenames say 'zig075_barrier_recal' and don't collide with any other candidate's artifacts.
    unfreeze_encoder=True dispatches to _fit_exit_head_unfrozen_encoder instead (user request
    2026-08-18)."""
    bundle = torch.load(parent_bundle, map_location=device, weights_only=False)
    baseline_models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    fit_fn = _fit_exit_head_unfrozen_encoder if unfreeze_encoder else pricemove_retrain._fit_exit_head_only
    suffix = "zig075_barrier_recal_unfrozen" if unfreeze_encoder else "zig075_barrier_recal"
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        model_path = out_dir / "models" / f"{expert}_3head_tabm_exit_{suffix}.pt"
        kwargs = {"encoder_lr_mult": encoder_lr_mult} if unfreeze_encoder else {}
        payload = fit_fn(
            baseline_models[expert], x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(seed), epochs=int(epochs), device=device, model_path=model_path,
            **kwargs,
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(model_path),
            "exit_epochs_ran": int(payload["exit_epochs_ran"]),
            "best_exit_validation_loss": float(payload["best_exit_validation_loss"]),
        }

    bundle_path = out_dir / "true_3head_tabm_bundle.pt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_id": MODEL_ID},
        bundle_path,
    )
    return {"parent_bundle": str(parent_bundle), "new_bundle": str(bundle_path), "summaries": summaries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=1500)
    ap.add_argument("--max-horizon-bars", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--adverse-unreal", type=float, default=ADVERSE_UNREAL)
    ap.add_argument("--min-mfe-for-giveback", type=float, default=MIN_MFE_FOR_GIVEBACK)
    ap.add_argument("--giveback-min", type=float, default=GIVEBACK_MIN)
    ap.add_argument("--terminal-window", type=int, default=TERMINAL_WINDOW)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--parent-bundle", type=Path, default=PARENT_BUNDLE)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--unfreeze-encoder", action="store_true",
                     help="Also train the encoder (in_proj/blocks/norms) alongside exit_head, at "
                          "--encoder-lr-mult x lr. direction_head/quality_head stay frozen (no loss "
                          "term for them in this loop). Default: encoder frozen, matches every other "
                          "exit-head-only retrain in this repo.")
    ap.add_argument("--encoder-lr-mult", type=float, default=0.1)
    args = ap.parse_args()

    liveatr._seed_everything(int(args.seed))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"stage=prepare_frames parent_bundle={args.parent_bundle}", flush=True)
    t0 = time.time()
    frames = liveatr.omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=liveatr.DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = liveatr.omega._load_fee_slip()
    print(f"  train_df rows={len(frames['train_df'])} elapsed={time.time() - t0:.1f}s", flush=True)

    print("stage=timescale_checkpoint", flush=True)
    t0 = time.time()
    tc = liveatr._fast_timescale_checkpoint(frames["train_df"], atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars))
    long_median = tc["long_bars_stats"].get("median", 0.0)
    short_median = tc["short_bars_stats"].get("median", 0.0)
    gate_pass = bool(long_median >= liveatr.TIMESCALE_GATE_MIN_MEDIAN_BARS and short_median >= liveatr.TIMESCALE_GATE_MIN_MEDIAN_BARS)
    print(f"  long_median={long_median:.1f} short_median={short_median:.1f} gate_pass={gate_pass} elapsed={time.time() - t0:.1f}s", flush=True)
    if not gate_pass:
        print("stage=ABORT gate_pass=False", flush=True)
        return 1

    rng = np.random.default_rng(int(args.seed))
    valid_idx = np.asarray(tc["valid_candidate_idx"], dtype=np.int64)
    n_sample = min(int(args.max_candidates), len(valid_idx))
    candidate_idx = np.sort(rng.choice(valid_idx, size=n_sample, replace=False))
    print(f"stage=build_exit_dataset candidates_sampled={len(candidate_idx)}/{len(valid_idx)} "
          f"adverse_unreal={args.adverse_unreal} min_mfe_for_giveback={args.min_mfe_for_giveback} "
          f"giveback_min={args.giveback_min} terminal_window={args.terminal_window}", flush=True)
    print("stage=risk_sizing component=zig075", flush=True)
    risk_margin, risk_leverage = liveatr._risk_sizing_for_component("zig075", frames["train_df"], seed=int(args.seed))

    t0 = time.time()
    x_exit_raw, y_exit, frame_exit, exit_diag = liveatr._build_exit_dataset_entry_label_live_atr_barrier(
        frames["train_df"], frames["s_train_label"],
        candidate_idx=candidate_idx, risk_margin=risk_margin, risk_leverage=risk_leverage,
        fee=fee, slip=slip, cost_mult=float(args.cost_mult),
        atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars),
        terminal_window=int(args.terminal_window), adverse_unreal=float(args.adverse_unreal),
        min_mfe_for_giveback=float(args.min_mfe_for_giveback), giveback_min=float(args.giveback_min),
    )
    build_elapsed = time.time() - t0
    print(f"  rows={exit_diag['rows']} used_candidates={exit_diag['used_candidates']} positive_rate={exit_diag['positive_rate']:.4f} "
          f"reason_counts={exit_diag['continued_exit_reasons']} elapsed={build_elapsed:.1f}s", flush=True)
    exit_diag["build_elapsed_sec"] = build_elapsed

    print("stage=retrain_exit_head", flush=True)
    t0 = time.time()
    retrain_info = _retrain_exit_head_only(
        x_exit_raw, y_exit, frame_exit, seed=int(args.seed), epochs=int(args.epochs), device=device, out_dir=out_dir,
        parent_bundle=args.parent_bundle, unfreeze_encoder=bool(args.unfreeze_encoder),
        encoder_lr_mult=float(args.encoder_lr_mult),
    )
    print(f"  retrain elapsed={time.time() - t0:.1f}s", flush=True)

    report = {
        "model_id": MODEL_ID, "parent_bundle": str(args.parent_bundle),
        "adverse_unreal": float(args.adverse_unreal), "min_mfe_for_giveback": float(args.min_mfe_for_giveback),
        "giveback_min": float(args.giveback_min), "terminal_window": int(args.terminal_window),
        "unfreeze_encoder": bool(args.unfreeze_encoder), "encoder_lr_mult": float(args.encoder_lr_mult),
        "seed": int(args.seed), "checkpoint": tc, "dataset": exit_diag, "retrain": retrain_info,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=liveatr._json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

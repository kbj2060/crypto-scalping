#!/usr/bin/env python3
"""RESEARCH ONLY -- Phase 2 of eth_candidate_unified_single_component_redesign (design doc
section 3-C / 5, step "Phase 2"): retrain ONLY the exit head on top of the Phase-1-confirmed
Variant B (quality_mode=same_as_direction) parent, using the SAME already-validated ATR-barrier
exit-label machinery as research_eth_omega461_exit_head_liveatr_relabel_20260813.py (imported,
not copied) -- that script's feature-barrier-mismatch bug (pos_tp/pos_sl features computed from a
stale BASE_TEMPLATE constant instead of the real per-candidate ATR barrier) was already fixed in
this session (docs/experiments/eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817.md),
so this Phase 2 script inherits the fix for free by reusing that module's functions unmodified.

Why NOT reuse the exit head Phase 1 already trained as a side effect: that one used the parent
script's own --exit-label-mode entry_label_terminal_giveback, which is NOT ATR-barrier-based --
it defines the trade's lifecycle as "hold until the zigzag_action SEGMENT ends" and labels near
the segment end as exit=1 regardless of giveback. That is structurally the SAME oracle-segment-
boundary label already confirmed broken this session (99.86% positive from terminal_window_exit,
docs/experiments/ilias_eth_exit_head_passivity_root_cause_20260817.md) -- lowering giveback_min
would not fix it, since the giveback branch is not what dominates. This script instead builds a
genuinely ATR-barrier-resolved exit label (same formula as the live floor) and lowers giveback_min
from the deployed default (0.65, confirmed this session to fire only after ~97.6% average
giveback -- docs/experiments/eth_odyssey4_zig075_exit_head_threshold_review_20260817.md) to 0.25 --
a single recalibrated constant, no new free parameters, same design discipline as the rest of
this line.

Frozen parent: the FIRST Phase 1 quality_B_samedir seed (2559205075) -- arbitrary but fixed
choice, consistent with how the ORIGINAL liveATR-relabel script always freezes ONE parent per
component. direction+quality heads are frozen (loaded, never updated); only the exit head is
retrained per expert (pricemove_retrain._fit_exit_head_only, unchanged).

N=5 genuinely random seeds for the EXIT HEAD retrain specifically (parent-model seed variance was
already assessed in Phase 1 and is not re-tested here -- this isolates "does the giveback_min
recalibration fix the exit head's timing" from parent-seed noise).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. No live/shadow files touched. Does not overwrite
the Phase 1 parent bundle (writes to a new OUT_DIR).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as pricemove_retrain  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_head_liveatr_relabel_20260813 as liveatr  # noqa: E402 -- reused unmodified, not copied

MODEL_ID = "eth_candidate_unified_phase2_exit_head_giveback_recal_20260817"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_eth_candidate_unified_phase1_quality_B_samedir_seed2559205075/true_3head_tabm_bundle.pt"
GIVEBACK_MIN = 0.25  # recalibrated from the deployed 0.65 (design doc section 3-C); see module docstring


def _retrain_exit_head_only(
    x_exit_raw, y_exit, frame_exit, *, seed: int, epochs: int, device: torch.device, out_dir: Path,
    parent_bundle: Path, hard_regime_filter: bool = False,
) -> dict[str, Any]:
    """Adapted from liveatr._retrain_component_exit_head_liveatr: identical per-expert
    freeze-encoder/retrain-exit-only pattern, but loads parent_bundle directly instead of looking
    up a component name in sweep.COMPONENTS (which the Phase 1 candidate is not registered in)."""
    bundle = torch.load(parent_bundle, map_location=device, weights_only=False)
    baseline_models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        model_path = out_dir / "models" / f"{expert}_3head_tabm_exit_giveback_recal.pt"
        payload = pricemove_retrain._fit_exit_head_only(
            baseline_models[expert], x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(seed), epochs=int(epochs), device=device, model_path=model_path,
            hard_regime_filter=bool(hard_regime_filter),
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
    ap.add_argument("--max-candidates", type=int, default=1500)  # matches the deployed h48qual/zig075 liveATR-relabel scale
    ap.add_argument("--max-horizon-bars", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--giveback-min", type=float, default=GIVEBACK_MIN)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--parent-bundle", type=Path, default=PARENT_BUNDLE)
    ap.add_argument("--hard-regime-filter", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    liveatr._seed_everything(int(args.seed))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"stage=prepare_frames parent_bundle={args.parent_bundle} hard_regime_filter={args.hard_regime_filter}", flush=True)
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
    print(f"stage=build_exit_dataset candidates_sampled={len(candidate_idx)}/{len(valid_idx)} giveback_min={args.giveback_min}", flush=True)
    t0 = time.time()
    # This Phase 1 candidate parent has no registered risk sidecar (sweep.COMPONENTS only has
    # "h48qual"/"zig075" -- see module docstring), so there is no real per-candidate sizing to
    # source yet. risk_margin=None/risk_leverage=None is an explicit opt-in to the fixed
    # BASE_TEMPLATE fallback (recorded as risk_sizing_source="base_template_constant_no_sidecar_
    # available" in exit_diag) rather than a silent default -- 2026-08-18, see docs/experiments/
    # eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md finding 1b.
    x_exit_raw, y_exit, frame_exit, exit_diag = liveatr._build_exit_dataset_entry_label_live_atr_barrier(
        frames["train_df"], frames["s_train_label"],
        candidate_idx=candidate_idx, risk_margin=None, risk_leverage=None,
        fee=fee, slip=slip, cost_mult=float(args.cost_mult),
        atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars),
        giveback_min=float(args.giveback_min),
    )
    build_elapsed = time.time() - t0
    print(f"  rows={exit_diag['rows']} used_candidates={exit_diag['used_candidates']} positive_rate={exit_diag['positive_rate']:.4f} "
          f"reason_counts={exit_diag['continued_exit_reasons']} elapsed={build_elapsed:.1f}s", flush=True)
    exit_diag["build_elapsed_sec"] = build_elapsed

    print("stage=retrain_exit_head", flush=True)
    t0 = time.time()
    retrain_info = _retrain_exit_head_only(
        x_exit_raw, y_exit, frame_exit, seed=int(args.seed), epochs=int(args.epochs), device=device, out_dir=out_dir,
        parent_bundle=args.parent_bundle, hard_regime_filter=bool(args.hard_regime_filter),
    )
    print(f"  retrain elapsed={time.time() - t0:.1f}s", flush=True)

    report = {
        "model_id": MODEL_ID, "parent_bundle": str(args.parent_bundle), "hard_regime_filter": bool(args.hard_regime_filter),
        "giveback_min": float(args.giveback_min),
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

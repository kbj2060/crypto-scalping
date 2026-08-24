#!/usr/bin/env python3
"""RESEARCH ONLY -- retrain the h48qual/zig075 exit_head on a NEW label source.

Question: the frozen live exit heads are trained by
`train_eval_omega4_3head_parent72_loose_entry_quality_20260620._build_exit_dataset_entry_label_terminal_giveback`,
whose positive label is "within `terminal_window` bars of THIS ZIGZAG SEGMENT'S OWN END".
Verified directly from that function + both live bundles' report.json: only 732-813 independent
zigzag segments exist in the training window, and 2,179/2,182 (99.86%) of positive rows come from
the `terminal_window_exit` branch -- i.e. "a zigzag pivot is imminent", not "closing now is good
P&L". This is why `research_eth_omega461_exit_head_retrain_eval_20260721.py` found that sweeping
`giveback_min` over 0.45..0.85 produced byte-identical VAL/OOS results: the giveback branch almost
never fires.

This script does NOT add a new hazard/rescue classifier on top of the frozen SLTP lifecycle (that
whole axis is RETIRED_DO_NOT_SHADOW_FOR_PROMOTION per docs/experiments/eth_omega461_exit_learning_20260724.md
and out of scope here). It keeps the exit_head's own architecture/inputs and retrains ONLY its
weights (`_fit_exit_head_only`, encoder/direction/quality frozen) on a new label SOURCE: instead of
each candidate's own zigzag-segment end, the "near-exit" anchor is that candidate's own h48_conservative
(48-bar ATR triple-barrier) resolution bar, sourced from the Odyssey sub-project's dense per-bar
`tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/train_triple_barrier_labels.csv`
(built by scripts/build_omega1_2_triple_barrier_labels_20260619.py). Every zigzag_action in (1,2)
bar (not just the first bar of a run) is its own entry candidate, which is the main lever for fixing
the independent-window scarcity documented above.

New function: `_build_exit_dataset_entry_label_h48cons_barrier`. Does NOT modify
`_build_exit_dataset_entry_label_terminal_giveback` or any other existing function.

fresh_forward_bar_by_bar=true (VAL replay is a single causal forward pass via the existing,
already-certified `research_eth_omega461_exit_sweep_20260721.replay_exit_variant`).
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. Training itself uses only the pre-2025-10-01 TRAIN split.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint. VAL only -- this script never loads or scores OOS data.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as pricemove_retrain  # noqa: E402
import eth_live_promotion_seed_robustness_prefix_snapshot_20260819 as omega4  # noqa: E402  -- redirected from the dirty (uncommitted 2026-08-18 exit_head bugfix) working-tree file to the git-HEAD snapshot, see that file's own docstring for why
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

MODEL_ID = "eth_omega461_exit_head_h48cons_relabel_20260813"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BARRIER_TAG = "h48_conservative"
TB_TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/train_triple_barrier_labels.csv"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"

# Reference numbers for the CURRENTLY DEPLOYED (frozen) exit head recipe, read directly from both
# live bundles' report.json (`exit_label.diag`, identical for h48qual and zig075 because both share
# the same direction_label_dir and the exit dataset only depends on zigzag_action + base features,
# not on the quality-head label source). Hardcoded here only as the stage-1 checkpoint reference,
# not re-derived, so the comparison is against the exact numbers already cited in
# docs/experiments/eth_omega461_exit_learning_20260724.md.
ORIGINAL_RECIPE_REFERENCE = {
    "label_mode": "entry_label_terminal_giveback_every_in_position_bar",
    "rows": 30000,
    "used_segments": 732,
    "positive_count": 2182,
    "positive_rate": 0.07273333333333333,
    "continued_exit_reasons": {"hold": 27818, "mfe_giveback_exit": 3, "terminal_window_exit": 2179},
    "note": "max_exit_samples=30000 truncated the deployed run before the full ~813-segment train window was exhausted (see stage-1 diag 'reference_full_window_segment_count').",
}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _merge_h48cons_barrier_cols(frame: pd.DataFrame, tb_frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp",
        f"tb_long_bars_{BARRIER_TAG}",
        f"tb_short_bars_{BARRIER_TAG}",
        f"tb_long_reason_{BARRIER_TAG}",
        f"tb_short_reason_{BARRIER_TAG}",
    ]
    missing = sorted(set(cols) - set(tb_frame.columns))
    if missing:
        raise RuntimeError(f"triple-barrier frame missing columns: {missing}")
    merged = frame[["timestamp"]].merge(tb_frame[cols], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != len(frame):
        raise RuntimeError("h48_conservative barrier merge changed row count")
    return merged


def _reference_full_window_segment_count(frame: pd.DataFrame) -> int:
    """Same maximal-run segment definition as `_build_exit_dataset_entry_label_terminal_giveback`,
    computed over the FULL (uncapped) train frame -- diagnostic only, for the stage-1 checkpoint
    comparison (the deployed 732 count was truncated by max_exit_samples=30000)."""
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    n = len(action)
    last_i = n - 2
    segs = 0
    i = 0
    while i < last_i:
        a = int(action[i])
        if a not in (1, 2):
            i += 1
            continue
        while i < last_i and int(action[i]) == a:
            i += 1
        segs += 1
    return int(segs)


def _build_exit_dataset_entry_label_h48cons_barrier(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    tb_frame: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    terminal_window: int = 3,
    adverse_unreal: float = -0.010,
    min_mfe_for_giveback: float = 0.006,
    giveback_min: float = 0.65,
    max_candidates: int = 0,
    max_rows: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """New exit-head label source. Every zigzag_action in (1,2) bar `i` is its own entry candidate
    (side from action[i], fill via the same `omega._try_execution` convention as the original
    function: entry_i = i+1). Instead of anchoring "near exit" to the zigzag segment's own end,
    it anchors to THIS candidate's own h48_conservative barrier resolution bar
    (entry_i + tb_{side}_bars_h48_conservative - 1, i.e. the bar where that candidate's causal
    48-bar ATR triple-barrier trade would have hit TP, hit SL, or timed out). Position-feature
    construction (`exit_head._position_feature_row`) and the adverse-unrealized / mfe-giveback
    branches are unchanged from `_build_exit_dataset_entry_label_terminal_giveback` -- only the
    "how do I find the end of this trade" anchor and the candidate density change, isolating the
    label-source variable. Rows whose barrier-resolution bar cannot be exactly matched inside
    `frame` (frame gap, or the resolution would fall past the frame's own tail) are skipped and
    counted, never silently clamped.
    """
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"h48cons-barrier exit dataset missing columns: {missing}")
    if len(frame) != len(state):
        raise RuntimeError("h48cons-barrier exit frame/state length mismatch")
    tb_aligned = _merge_h48cons_barrier_cols(frame, tb_frame)

    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    ts = pd.to_datetime(frame["timestamp"], errors="raise").to_numpy()
    tb_long_bars = pd.to_numeric(tb_aligned[f"tb_long_bars_{BARRIER_TAG}"], errors="coerce").to_numpy(dtype=np.float64)
    tb_short_bars = pd.to_numeric(tb_aligned[f"tb_short_bars_{BARRIER_TAG}"], errors="coerce").to_numpy(dtype=np.float64)
    tb_long_reason = tb_aligned[f"tb_long_reason_{BARRIER_TAG}"].to_numpy()
    tb_short_reason = tb_aligned[f"tb_short_reason_{BARRIER_TAG}"].to_numpy()
    route_cols = list(hard.ROUTE_COLS)
    route_values = frame[route_cols].to_numpy(dtype=np.float64)

    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    leverage = float(omega.BASE_TEMPLATE["leverage"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    tw = max(int(terminal_window), 1)
    bar_delta = np.timedelta64(5, "m")

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    tb_barrier_reason_counts: dict[str, int] = {}
    used_candidates = 0
    skipped_candidates = 0
    skip_reasons: dict[str, int] = {}
    positive_count = 0
    last_i = max(len(frame) - 2, 0)
    stop = False

    for i in range(0, last_i):
        if stop:
            break
        side_action = int(action[i])
        if side_action not in (1, 2):
            continue
        side = 1 if side_action == 1 else -1
        bars_val = float(tb_long_bars[i]) if side > 0 else float(tb_short_bars[i])
        tb_reason = tb_long_reason[i] if side > 0 else tb_short_reason[i]
        if not np.isfinite(bars_val) or bars_val < 1.0:
            skipped_candidates += 1
            skip_reasons["no_h48cons_barrier_match"] = skip_reasons.get("no_h48cons_barrier_match", 0) + 1
            continue
        bars_int = int(round(bars_val))
        entry_i = min(i + 1, len(frame) - 1)
        barrier_end_i = entry_i + bars_int - 1
        if barrier_end_i >= len(frame):
            skipped_candidates += 1
            skip_reasons["barrier_beyond_frame_tail"] = skip_reasons.get("barrier_beyond_frame_tail", 0) + 1
            continue
        expected_end_ts = ts[i] + bar_delta * bars_int
        if ts[barrier_end_i] != expected_end_ts:
            skipped_candidates += 1
            skip_reasons["frame_gap_before_barrier_end"] = skip_reasons.get("frame_gap_before_barrier_end", 0) + 1
            continue
        filled, entry_price, _entry_fee, _route = omega._try_execution(
            arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff,
        )
        if not filled:
            skipped_candidates += 1
            skip_reasons["entry_not_filled"] = skip_reasons.get("entry_not_filled", 0) + 1
            continue

        entry_state = state.iloc[int(i)]
        mfe = 0.0
        mae = 0.0
        candidate_rows = 0
        for row_i in range(entry_i, barrier_end_i + 1):
            px = float(arrays["close"][int(row_i)])
            raw = (
                (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1.0e-12)
                if side > 0
                else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1.0e-12)
            )
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            giveback = (mfe - unreal) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
            bars_to_barrier_end = int(barrier_end_i) - int(row_i)
            terminal = bars_to_barrier_end < tw
            adverse = unreal <= float(adverse_unreal)
            gave_back = mfe >= float(min_mfe_for_giveback) and giveback >= float(giveback_min) and unreal > 0.0
            if terminal:
                label = 1
                reason = "near_barrier_resolution_exit"
            elif adverse:
                label = 1
                reason = "adverse_unreal_exit"
            elif gave_back:
                label = 1
                reason = "mfe_giveback_exit"
            else:
                label = 0
                reason = "hold"
            row = exit_head._position_feature_row(
                state, entry_state, row_i=int(row_i), side=side, entry_price=float(entry_price),
                entry_i=int(entry_i), notional=notional, leverage=leverage, take_profit=take_profit,
                stop_loss=stop_loss, mfe=mfe, mae=mae, unreal=unreal,
            )
            rows.append(row)
            labels.append(label)
            positive_count += int(label)
            candidate_rows += 1
            frame_rows.append(
                {
                    "timestamp": ts[row_i],
                    **{route_cols[j]: float(route_values[row_i, j]) for j in range(len(route_cols))},
                    "exit_path_candidate_signal_i": int(i),
                    "exit_path_entry_i": int(entry_i),
                    "exit_path_barrier_end_i": int(barrier_end_i),
                    "exit_path_side": int(side),
                    "exit_path_hold_bars": int(max(int(row_i) - int(entry_i), 0)),
                    "exit_h48cons_label": int(label),
                    "exit_h48cons_reason": reason,
                    "exit_path_mfe": float(mfe),
                    "exit_path_mae": float(mae),
                    "exit_path_unrealized": float(unreal),
                    "exit_path_giveback": float(giveback),
                    "exit_path_bars_to_barrier_end": int(bars_to_barrier_end),
                    "exit_tb_barrier_reason": str(tb_reason),
                }
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if max_rows > 0 and len(rows) >= int(max_rows):
                stop = True
                break
        if candidate_rows > 0:
            used_candidates += 1
            tb_barrier_reason_counts[str(tb_reason)] = tb_barrier_reason_counts.get(str(tb_reason), 0) + 1
        else:
            skipped_candidates += 1
            skip_reasons["empty_candidate_window"] = skip_reasons.get("empty_candidate_window", 0) + 1
        if stop:
            break
        if max_candidates > 0 and used_candidates >= int(max_candidates):
            break

    if not rows:
        raise RuntimeError("empty h48cons-barrier Exit Head dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, f, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(positive_count),
        "negative_count": int(len(y) - positive_count),
        "continued_exit_reasons": reason_counts,
        "tb_barrier_reason_counts_at_used_candidates": tb_barrier_reason_counts,
        "used_candidates": int(used_candidates),
        "skipped_candidates": int(skipped_candidates),
        "skip_reasons": skip_reasons,
        "risk_template": {"notional": notional, "leverage": leverage, "take_profit": take_profit, "stop_loss": stop_loss},
        "label_mode": "entry_label_h48cons_barrier_every_action_bar",
        "barrier_tag": BARRIER_TAG,
        "terminal_window": int(tw),
        "adverse_unreal": float(adverse_unreal),
        "min_mfe_for_giveback": float(min_mfe_for_giveback),
        "giveback_min": float(giveback_min),
    }


def _retrain_component_exit_head(
    component: str,
    x_exit_raw: pd.DataFrame,
    y_exit: np.ndarray,
    frame_exit: pd.DataFrame,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    baseline_bundle_path = Path(sweep.COMPONENTS[component]["bundle"])
    bundle = torch.load(baseline_bundle_path, map_location=device, weights_only=False)
    baseline_models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        model_path = out_dir / component / "models" / f"{expert}_3head_tabm_exit_h48cons.pt"
        payload = pricemove_retrain._fit_exit_head_only(
            baseline_models[expert], x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(seed), epochs=int(epochs), device=device, model_path=model_path,
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(model_path),
            "exit_epochs_ran": int(payload["exit_epochs_ran"]),
            "best_exit_validation_loss": float(payload["best_exit_validation_loss"]),
        }

    bundle_path = out_dir / component / "true_3head_tabm_bundle.pt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_id": MODEL_ID},
        bundle_path,
    )
    return {"baseline_bundle": str(baseline_bundle_path), "new_bundle": str(bundle_path), "summaries": summaries}


def _evaluate_val(component: str, new_bundle_path: Path) -> dict[str, Any]:
    cfg = dict(sweep.COMPONENTS[component])
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_pred = sweep.EXT_PRED_DIR / component / f"validation_predictions_{cfg['q_tag']}.csv"

    baseline_prepped = sweep.prep_component(component, cfg, val_frame, val_pred, oof=True)
    m_baseline, _ledger_baseline = sweep.replay_exit_variant(
        baseline_prepped["frame"], baseline_prepped["x"], baseline_prepped["dec"], baseline_prepped["loaded"],
        risk_margin_fraction=baseline_prepped["margin"], risk_leverage=baseline_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=baseline_prepped["fee"], slip=baseline_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=baseline_prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )

    cfg_new = dict(cfg)
    cfg_new["bundle"] = new_bundle_path
    new_prepped = sweep.prep_component(component, cfg_new, val_frame, val_pred, oof=True)
    m_new, _ledger_new = sweep.replay_exit_variant(
        new_prepped["frame"], new_prepped["x"], new_prepped["dec"], new_prepped["loaded"],
        risk_margin_fraction=new_prepped["margin"], risk_leverage=new_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=new_prepped["fee"], slip=new_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=new_prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    return {"baseline": m_baseline, "h48cons_relabel": m_new}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dataset_only", "full"], default="full")
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = no cap (all zigzag_action bars)")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no cap on total bar-rows")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cost-mult", type=float, default=3.0, help="matches original training recipe's default (eval harness itself always uses its own COST_MULT=1.0, unchanged)")
    ap.add_argument("--seed", type=int, default=260813)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = torch.device("cpu")
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=prepare_frames", flush=True)
    t0 = time.time()
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    print(f"  train_df rows={len(frames['train_df'])} elapsed={time.time() - t0:.1f}s", flush=True)

    print("stage=load_h48_conservative_barrier_labels", flush=True)
    tb_frame = pd.read_csv(
        TB_TRAIN_CSV,
        usecols=[
            "timestamp",
            f"tb_long_bars_{BARRIER_TAG}", f"tb_short_bars_{BARRIER_TAG}",
            f"tb_long_reason_{BARRIER_TAG}", f"tb_short_reason_{BARRIER_TAG}",
        ],
    )
    tb_frame["timestamp"] = pd.to_datetime(tb_frame["timestamp"])

    print("stage=build_h48cons_barrier_exit_dataset", flush=True)
    t0 = time.time()
    x_exit_raw, y_exit, frame_exit, exit_diag = _build_exit_dataset_entry_label_h48cons_barrier(
        frames["train_df"], frames["s_train_label"], tb_frame,
        fee=fee, slip=slip, cost_mult=float(args.cost_mult),
        max_candidates=int(args.max_candidates), max_rows=int(args.max_rows),
    )
    build_elapsed = time.time() - t0
    print(f"  rows={exit_diag['rows']} used_candidates={exit_diag['used_candidates']} elapsed={build_elapsed:.1f}s", flush=True)

    reference_full_segments = _reference_full_window_segment_count(frames["train_df"])
    checkpoint = {
        "stage": "1_label_density_diversity_checkpoint",
        "original_recipe_reference": ORIGINAL_RECIPE_REFERENCE,
        "original_recipe_reference_full_uncapped_train_window_segment_count": reference_full_segments,
        "new_recipe": exit_diag,
        "candidate_density_ratio_vs_original_used_segments": float(exit_diag["used_candidates"]) / float(ORIGINAL_RECIPE_REFERENCE["used_segments"]),
        "candidate_density_ratio_vs_full_window_segments": float(exit_diag["used_candidates"]) / float(reference_full_segments),
        "build_elapsed_sec": build_elapsed,
        "build_args": {"max_candidates": int(args.max_candidates), "max_rows": int(args.max_rows), "cost_mult": float(args.cost_mult)},
    }
    (out_dir / "stage1_checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=_json_default), flush=True)

    if str(args.stage) == "dataset_only":
        print("stage=done (dataset_only)", flush=True)
        return 0

    results: dict[str, Any] = {"checkpoint": checkpoint, "components": {}}
    for component in ("h48qual", "zig075"):
        print(f"stage=retrain_exit_head component={component}", flush=True)
        t0 = time.time()
        retrain_info = _retrain_component_exit_head(
            component, x_exit_raw, y_exit, frame_exit,
            seed=int(args.seed), epochs=int(args.epochs), device=device, out_dir=out_dir,
        )
        print(f"  {component} retrain elapsed={time.time() - t0:.1f}s", flush=True)

        print(f"stage=evaluate_val component={component}", flush=True)
        val_metrics = _evaluate_val(component, Path(retrain_info["new_bundle"]))
        print(json.dumps({"component": component, "val": val_metrics}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
        results["components"][component] = {"retrain": retrain_info, "val_metrics": val_metrics}

    report = {
        "model_id": MODEL_ID,
        "design": (
            "h48qual/zig075 exit_head retrained (encoder/direction/quality frozen) on a new label "
            "source: near-exit anchor = each candidate's own h48_conservative triple-barrier "
            "resolution bar instead of its zigzag segment's own end, with every zigzag_action bar "
            "(not just segment starts) as an independent entry candidate."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "exit_threshold_held_fixed_at": sweep.BASELINE_EXIT_THRESHOLD,
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_opened": False,
        **results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

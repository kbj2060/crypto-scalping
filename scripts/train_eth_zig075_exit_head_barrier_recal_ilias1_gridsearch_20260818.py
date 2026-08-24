#!/usr/bin/env python3
"""One-factor-at-a-time local sensitivity grid search around the already-tested center point
(adverse_unreal=-0.020, min_mfe_for_giveback=0.015, giveback_min=0.45 -- see
[[eth_zig075_exit_head_barrier_recal_20260818]]'s 2026-08-18 후속 -- REJECTED, same monotonic
PnL-decline pattern as the pos_tp/pos_sl-buggy original) for zig075's barrier-recal exit_head on
Ilias 1's bug-fixed encoder. Asks whether moving any SINGLE one of the 3 dense-labeling parameters
away from that rejected center changes the qualitative verdict, before committing to a full
factorial grid.

Key efficiency insight (from reading research_eth_omega461_exit_head_liveatr_relabel_20260813.py's
_build_exit_dataset_entry_label_live_atr_barrier, lines 482-593): the EXPENSIVE part of dataset-
building (the per-candidate intrabar TP/SL barrier walk, ~430-600s) is completely independent of
adverse_unreal/min_mfe_for_giveback/giveback_min -- those 3 params only gate a cheap per-row
comparison against already-computed mfe/unreal/giveback/bars_to_barrier_end, all persisted in the
returned frame_exit's exit_path_* columns. So this script builds the dataset ONCE, relabels for
each grid point via a vectorized reimplementation of that exact per-row branching (validated
byte-for-byte against the loop's own reference output before trusting it for the grid), and only
pays the ~1320s retrain-3-experts cost per point -- not another ~600s dataset rebuild per point.

Grid (terminal_window=3 held fixed throughout, matching the center point's own isolation
principle):
  g1: giveback_min=0.30 (center 0.45)      g2: giveback_min=0.60
  g3: adverse_unreal=-0.012 (center -0.020, 30% of the 4.0% SL floor)   g4: adverse_unreal=-0.028 (70%)
  g5: min_mfe_for_giveback=0.010 (center 0.015, 13% of the 7.5% TP floor)  g6: min_mfe_for_giveback=0.025 (33%)
The center itself is NOT retrained here -- it already exists at
tmp/causal_regen_20260516/eth_zig075_exit_head_barrier_recal_20260818_ilias1_encoder/.

fresh_forward_bar_by_bar=true (dataset build + retrain mechanics unchanged from the original
script, only reused). No stored trade ledger used as input.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818 as canon  # noqa: E402  (side effect: canonical TRAIN_CSV/EVAL_CSV/REGIME3_*)
import train_eth_zig075_exit_head_barrier_recal_20260818 as barrier_recal  # noqa: E402

assert barrier_recal.liveatr.omega4.omega is canon.omega, "module-cache sharing assumption broken"

liveatr = barrier_recal.liveatr

ILIAS1_ZIG075_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt"
ILIAS1_ZIG075_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl"
OUT_ROOT = ROOT / "tmp/causal_regen_20260516/eth_zig075_exit_head_barrier_recal_gridsearch_20260818"

# Same override as train_eth_zig075_exit_head_barrier_recal_ilias1_20260818.py -- _risk_sizing_for_
# component("zig075", ...) hardcodes reading h48cons.sweep.COMPONENTS["zig075"], so point it at
# Ilias 1's own bundle/sidecar (module-attribute override, shared module source untouched).
_sweep = liveatr.h48cons.sweep
_sweep.COMPONENTS["zig075"] = {
    **_sweep.COMPONENTS["zig075"],
    "bundle": ILIAS1_ZIG075_BUNDLE, "sidecar_pkl": ILIAS1_ZIG075_SIDECAR,
    "q_tag": "q080", "quality_threshold": 0.80,
}

# retrain_exit_head prints nothing per-expert by default (see feedback_always_log_and_monitor_
# epoch_metrics) -- wrap (not reimplement) the unmodified original per-expert fit with start/end
# timing + the val_loss/epochs_ran it already returns. Same pattern as the single-point script.
_orig_fit_exit_head_only = barrier_recal.pricemove_retrain._fit_exit_head_only


def _fit_exit_head_only_logged(baseline_payload, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device, model_path, **kwargs):
    expert_name = liveatr.hard.EXPERT_NAMES[int(expert_idx)]
    t0 = time.time()
    payload = _orig_fit_exit_head_only(
        baseline_payload, x_exit, y_exit, exit_route_frame,
        expert_idx=expert_idx, seed=seed, epochs=epochs, device=device, model_path=model_path, **kwargs,
    )
    print(f"    expert={expert_name} epochs_ran={payload['exit_epochs_ran']} "
          f"best_val_loss={payload['best_exit_validation_loss']:.5f} elapsed={time.time() - t0:.1f}s", flush=True)
    return payload


barrier_recal.pricemove_retrain._fit_exit_head_only = _fit_exit_head_only_logged

CENTER = {"adverse_unreal": -0.020, "min_mfe_for_giveback": 0.015, "giveback_min": 0.45}
TERMINAL_WINDOW = 3
SEED = 101
EPOCHS = 8
MAX_CANDIDATES = 1500
MAX_HORIZON_BARS = 6000
COST_MULT = 3.0

GRID = [
    {"name": "g1_giveback030", **{**CENTER, "giveback_min": 0.30}},
    {"name": "g2_giveback060", **{**CENTER, "giveback_min": 0.60}},
    {"name": "g3_adverse012", **{**CENTER, "adverse_unreal": -0.012}},
    {"name": "g4_adverse028", **{**CENTER, "adverse_unreal": -0.028}},
    {"name": "g5_mfe010", **{**CENTER, "min_mfe_for_giveback": 0.010}},
    {"name": "g6_mfe025", **{**CENTER, "min_mfe_for_giveback": 0.025}},
]


def _relabel(frame_exit: pd.DataFrame, *, terminal_window: int, adverse_unreal: float,
             min_mfe_for_giveback: float, giveback_min: float) -> tuple[np.ndarray, dict[str, int]]:
    """Vectorized reimplementation of _build_exit_dataset_entry_label_live_atr_barrier's per-row
    label/reason branching (that function's lines ~537-553: terminal > adverse > gave_back > hold
    precedence), operating on the exit_path_* diagnostic columns that function already persists
    per row. Validated against the reference loop's own output in main() before use."""
    mfe = frame_exit["exit_path_mfe"].to_numpy(dtype=np.float64)
    unreal = frame_exit["exit_path_unrealized"].to_numpy(dtype=np.float64)
    giveback = frame_exit["exit_path_giveback"].to_numpy(dtype=np.float64)
    bars_to_end = frame_exit["exit_path_bars_to_barrier_end"].to_numpy(dtype=np.int64)

    terminal = bars_to_end < int(terminal_window)
    adverse = unreal <= float(adverse_unreal)
    gave_back = (mfe >= float(min_mfe_for_giveback)) & (giveback >= float(giveback_min)) & (unreal > 0.0)

    label = np.where(terminal, 1, np.where(adverse, 1, np.where(gave_back, 1, 0))).astype(np.int64)
    reason = np.where(terminal, "near_barrier_resolution_exit",
              np.where(adverse, "adverse_unreal_exit",
              np.where(gave_back, "mfe_giveback_exit", "hold")))
    reason_counts = {str(r): int((reason == r).sum()) for r in np.unique(reason)}
    return label, reason_counts


def main() -> int:
    liveatr._seed_everything(SEED)
    device = torch.device("cpu")

    print("stage=prepare_frames", flush=True)
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
    tc = liveatr._fast_timescale_checkpoint(frames["train_df"], atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=MAX_HORIZON_BARS)
    gate_pass = bool(
        tc["long_bars_stats"].get("median", 0.0) >= liveatr.TIMESCALE_GATE_MIN_MEDIAN_BARS
        and tc["short_bars_stats"].get("median", 0.0) >= liveatr.TIMESCALE_GATE_MIN_MEDIAN_BARS
    )
    print(f"  gate_pass={gate_pass} elapsed={time.time() - t0:.1f}s", flush=True)
    if not gate_pass:
        print("stage=ABORT gate_pass=False", flush=True)
        return 1

    rng = np.random.default_rng(SEED)
    valid_idx = np.asarray(tc["valid_candidate_idx"], dtype=np.int64)
    n_sample = min(MAX_CANDIDATES, len(valid_idx))
    candidate_idx = np.sort(rng.choice(valid_idx, size=n_sample, replace=False))

    print("stage=risk_sizing component=zig075", flush=True)
    risk_margin, risk_leverage = liveatr._risk_sizing_for_component("zig075", frames["train_df"], seed=SEED)

    print(f"stage=build_exit_dataset_ONCE candidates={len(candidate_idx)} center={CENTER}", flush=True)
    t0 = time.time()
    x_exit_raw, y_center, frame_exit, exit_diag = liveatr._build_exit_dataset_entry_label_live_atr_barrier(
        frames["train_df"], frames["s_train_label"],
        candidate_idx=candidate_idx, risk_margin=risk_margin, risk_leverage=risk_leverage,
        fee=fee, slip=slip, cost_mult=COST_MULT,
        atr_cfg=liveatr.LIVE_ATR_CFG, max_horizon_bars=MAX_HORIZON_BARS,
        terminal_window=TERMINAL_WINDOW, **CENTER,
    )
    print(f"  build done rows={exit_diag['rows']} positive_rate={exit_diag['positive_rate']:.4f} "
          f"reason_counts={exit_diag['continued_exit_reasons']} elapsed={time.time() - t0:.1f}s", flush=True)

    y_check, reason_check = _relabel(frame_exit, terminal_window=TERMINAL_WINDOW, **CENTER)
    ref_reason_counts = {str(k): int(v) for k, v in exit_diag["continued_exit_reasons"].items()}
    if not np.array_equal(y_check, y_center) or reason_check != ref_reason_counts:
        print("stage=ABORT vectorized relabel does not match reference loop output for the center point", flush=True)
        print(f"  reference reason_counts={ref_reason_counts}", flush=True)
        print(f"  vectorized reason_counts={reason_check}", flush=True)
        return 1
    print("  relabel self-check PASSED (byte-identical to reference loop for the center point)", flush=True)

    results = []
    for point in GRID:
        name = point["name"]
        params = {k: v for k, v in point.items() if k != "name"}
        print(f"stage=grid_point name={name} params={params}", flush=True)
        y_exit, reason_counts = _relabel(frame_exit, terminal_window=TERMINAL_WINDOW, **params)
        positive_rate = float(np.mean(y_exit))
        print(f"  relabeled positive_rate={positive_rate:.4f} reason_counts={reason_counts}", flush=True)

        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        retrain_info = barrier_recal._retrain_exit_head_only(
            x_exit_raw, y_exit, frame_exit, seed=SEED, epochs=EPOCHS, device=device, out_dir=out_dir,
            parent_bundle=ILIAS1_ZIG075_BUNDLE, unfreeze_encoder=False,
        )
        retrain_elapsed = time.time() - t0
        print(f"  retrain done elapsed={retrain_elapsed:.1f}s", flush=True)

        report = {
            "model_id": f"eth_zig075_exit_head_barrier_recal_gridsearch_20260818_{name}",
            "parent_bundle": str(ILIAS1_ZIG075_BUNDLE), "grid_point": name, **params,
            "terminal_window": TERMINAL_WINDOW, "seed": SEED,
            "positive_rate": positive_rate, "reason_counts": reason_counts,
            "retrain": retrain_info, "retrain_elapsed_sec": retrain_elapsed,
            "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=liveatr._json_default) + "\n", encoding="utf-8"
        )
        results.append({"name": name, **params, "positive_rate": positive_rate, "bundle": str(out_dir / "true_3head_tabm_bundle.pt")})
        print(f"  report={out_dir / 'report.json'}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "grid_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"grid_summary={OUT_ROOT / 'grid_summary.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ensemble-average the 5 SWA-trained seeds' predictions for ETH's final candidate (h48qual and
zig075, on the live-matching 2024+2025 tape, correct per-component config matching
scripts/run_pinned102_2024tape_controls_20260727.sh) and compute the resulting VAL/OOS
performance -- the final deliverable for this session's SWA+ensemble seed-variance investigation.

Reuses the exact same decision/metrics pipeline the training script itself uses
(cat_dq._prediction_output -> parent._to_decisions -> asset omega._metrics), substituting the
5-seed-averaged direction/quality probability matrices for any single seed's output.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [260620, 260728, 260729, 260730, 260731]
DIR_TEMPLATE = "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_final_swa_{component}_seed{seed}"
Q_TAG = {"h48qual": "q050", "zig075": "q075"}


def load_avg_proba(component: str, split: str, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = Q_TAG[component]
    dir_stack, qual_stack, ts_ref = [], [], None
    for seed in SEEDS:
        path = ROOT / pred_dir(component, seed) / f"{split}_predictions_{tag}.csv"
        df = pd.read_csv(path)
        ts = df["timestamp"].to_numpy()
        if ts_ref is None:
            ts_ref = ts
        else:
            assert (ts == ts_ref).all(), f"timestamp mismatch seed {seed} split {split}"
        dir_stack.append(df[[f"{prefix}dir_p_cash", f"{prefix}dir_p_long", f"{prefix}dir_p_short"]].to_numpy(dtype=np.float64))
        qual_stack.append(df[[f"{prefix}quality_p_cash", f"{prefix}quality_p_long", f"{prefix}quality_p_short"]].to_numpy(dtype=np.float64))
    return ts_ref, np.mean(dir_stack, axis=0), np.mean(qual_stack, axis=0)


def pred_dir(component: str, seed: int) -> str:
    return DIR_TEMPLATE.format(component=component, seed=seed)


def run_component(component: str) -> dict:
    wrapper = importlib.import_module("train_eval_omega4_3head_parent72_pinned102_2024tape_20260727")
    parent_script = wrapper.parent_script
    parent = parent_script.parent
    omega = parent_script.omega

    wrapper.pinned._install_pin(component)
    omega._load_omega_frames = wrapper._load_omega_frames_2024tape

    if component == "h48qual":
        direction_label_dir = Path(f"{ROOT}/tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531")
        quality_mode = "quality_label_action"
        quality_label_dir = Path(f"{ROOT}/tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps")
        threshold = 0.50
    else:
        direction_label_dir = Path(f"{ROOT}/tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531")
        quality_mode = "same_as_direction"
        quality_label_dir = None
        threshold = 0.75

    frames = parent_script._prepare_frames(
        disable_tp_sl=False, direction_label_dir=direction_label_dir, quality_mode=quality_mode,
        quality_label_dir=quality_label_dir, quality_min_edge=0.0010, quality_max_mae=0.0100,
        quality_min_mfe_mae=1.20, quality_max_hold_bars=288,
    )
    val_raw, oos_raw = frames["val_raw"], frames["oos_raw"]
    fee, slip = omega._load_fee_slip()

    out = {}
    for split, raw, prefix, oof in [
        ("validation", val_raw, "omega1_regime3_expertdq_oof_", True),
        ("oos", oos_raw, "omega1_regime3_expertdq_", False),
    ]:
        ts_ref, avg_dir, avg_qual = load_avg_proba(component, split, prefix)
        assert len(avg_dir) == len(raw), f"{component} {split}: pred rows {len(avg_dir)} != raw rows {len(raw)}"
        pred_out = parent._prediction_output(raw, avg_dir, avg_qual, threshold=threshold, prefix=prefix.rstrip("_"))
        dec = parent._to_decisions(pred_out, oof=oof)
        m = omega._metrics(raw, dec, fee=fee, slip=slip, cost_mult=3.0)
        out[split] = {"pnl": m["pnl"], "mdd": m["mdd"], "trades": m["trades"], "wr": m["wr"]}
    return out


if __name__ == "__main__":
    component = sys.argv[1]
    result = run_component(component)
    print(json.dumps(result, indent=2, default=str))

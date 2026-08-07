"""Lower-frequency retune of build_sol_dp_trajectory_labels_20260715.py.

The first pass (ETH-inherited DP params: MIN_ENTRY_EDGE=0.00008, HOLD_PENALTY=0.000002,
MAX_AGE=96) produced labels with median hold ~3-4 bars -- far shorter than SOL's existing
zigzag labels (median wave_bars=61, p90=162, from zigzag_action_labels_2025.csv). At that
frequency the DP "optimal" trades were mostly unforecastable 5m micro-noise: the downstream
CatBoost parent (train_eval_omega4_3head_catboost_parent_sol_dp_labels_20260715.py) never
reached the 0.70 quality-head confidence threshold and produced zero trades in VAL/OOS.

Swept MIN_ENTRY_EDGE/MAX_AGE/HOLD_PENALTY (see scratchpad sweep) and picked
MIN_ENTRY_EDGE=0.003, MAX_AGE=600, HOLD_PENALTY=0.00001, which yields median hold 52 bars
(vs zigzag's 61) at the cost of far fewer trades (~109/year vs zigzag's ~1000+/year) -- the DP
oracle is far more selective about which moves are large/persistent enough to be worth it once
forced to hold this long, which is the intended lower-frequency-only-real-signal effect.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_omega1_2_1_dp_trajectory_labels_20260620 as dp  # noqa: E402

MODEL_ID = "sol_omega1_2_1_dp_trajectory_daytrade_lowfreq_20260715"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLITS = ROOT / "data/splits/year_oos"

dp.MODEL_ID = MODEL_ID
dp.OUT_DIR = OUT_DIR
dp.MAX_AGE = 600
dp.MIN_ENTRY_EDGE = 0.003
dp.HOLD_PENALTY = 0.00001


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year, split_name in ((2024, "train_2024"), (2025, "train_2025"), (2026, "oos_2026")):
        path = SPLITS / f"sol_features_{year}.csv"
        frame = dp._read_frame(path)
        labels, action_path, diag = dp._build_dp_labels(frame, split_name)
        entry_out = OUT_DIR / f"sol_{split_name}_dp_trajectory_labels.csv"
        path_out = OUT_DIR / f"sol_{split_name}_dp_action_path_labels.csv"
        labels.to_csv(entry_out, index=False)
        action_path.to_csv(path_out, index=False)
        print(f"{split_name}: {len(labels)} rows, side_counts={diag['side_counts']} -> {entry_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

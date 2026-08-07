"""Convert the SOL DP oracle trajectory labels (build_sol_dp_trajectory_labels_20260715.py
output: label_side_id in {-1 SHORT, 0 CASH, 1 LONG}) into the zigzag_action_labels_{year}.csv
schema expected by train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py's
_read_labels() (timestamp, zigzag_action; zigzag_action in {0 CASH, 1 LONG, 2 SHORT}), so the
DP labels are a drop-in replacement for the SOL v1 zigzag labels via --direction-label-dir.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega1_2_1_dp_trajectory_daytrade_20260715"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_dp_trajectory_action_labels_20260715"

SIDE_ID_TO_ZIGZAG_ACTION = {0: 0, 1: 1, -1: 2}  # CASH=0, LONG=1, SHORT=2


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, year in (("train_2025", 2025), ("oos_2026", 2026)):
        src = DP_DIR / f"sol_{split_name}_dp_trajectory_labels.csv"
        df = pd.read_csv(src, parse_dates=["timestamp"])
        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "zigzag_action": df["label_side_id"].map(SIDE_ID_TO_ZIGZAG_ACTION).astype("int64"),
        })
        invalid = out["zigzag_action"].isna().sum()
        if invalid:
            raise RuntimeError(f"{src}: {invalid} unmapped label_side_id values")
        dest = OUT_DIR / f"zigzag_action_labels_{year}.csv"
        out.to_csv(dest, index=False)
        print(f"{year}: {len(out)} rows, counts={out['zigzag_action'].value_counts().sort_index().to_dict()} -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

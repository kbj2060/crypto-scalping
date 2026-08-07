"""Same conversion as convert_sol_dp_labels_to_zigzag_schema_20260715.py, pointed at the
lower-frequency DP labels from build_sol_dp_trajectory_labels_lowfreq_20260715.py.

Fixed 2026-07-20: the original version read `label_side_id` from `*_dp_trajectory_labels.csv`,
which marks only the single entry bar of each trade (run-length 1 everywhere). zigzag_action is
expected to hold its value for every bar of a position (run-length = holding period, like the
original zigzag builder), so every segment downstream was 1 bar long and got skipped by
`_build_exit_dataset_entry_label_terminal_giveback`'s `end_i < entry_i` check, producing an
empty exit dataset. The `*_dp_action_path_labels.csv` file's `state` column is the correct
bar-by-bar trajectory (FLAT/LONG/SHORT, median non-flat run length 52 bars -- matches this
label set's own median-hold-52-bars design note) and is used here instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega1_2_1_dp_trajectory_daytrade_lowfreq_20260715"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_dp_trajectory_action_labels_lowfreq_20260715"

STATE_TO_ZIGZAG_ACTION = {"FLAT": 0, "LONG": 1, "SHORT": 2}  # CASH=0, LONG=1, SHORT=2


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, year in (("train_2025", 2025), ("oos_2026", 2026)):
        src = DP_DIR / f"sol_{split_name}_dp_action_path_labels.csv"
        df = pd.read_csv(src, parse_dates=["timestamp"])
        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "zigzag_action": df["state"].map(STATE_TO_ZIGZAG_ACTION).astype("int64"),
        })
        invalid = out["zigzag_action"].isna().sum()
        if invalid:
            raise RuntimeError(f"{src}: {invalid} unmapped state values")
        dest = OUT_DIR / f"zigzag_action_labels_{year}.csv"
        out.to_csv(dest, index=False)
        print(f"{year}: {len(out)} rows, counts={out['zigzag_action'].value_counts().sort_index().to_dict()} -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

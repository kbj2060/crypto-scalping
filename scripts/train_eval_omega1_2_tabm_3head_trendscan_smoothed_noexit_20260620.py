#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_omega1_direction_head_direction_only_20260602 as direction_base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402


MODEL_ID = "omega1_2_true_3head_tabm_trendscan_smoothed_20260620"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_smoothed_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _add_smoothed_trendscan_labels(year: int) -> pd.DataFrame:
    path = LABEL_DIR / f"wave3_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "wave3_action"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    out = (
        labels[["timestamp", "wave3_action"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .rename(columns={"wave3_action": "zigzag_action"})
        .reset_index(drop=True)
    )
    y = pd.to_numeric(out["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"{path} invalid smoothed trend-scanning classes: {invalid}")
    return out


def main() -> int:
    direction_base._add_labels = _add_smoothed_trendscan_labels
    parent.MODEL_ID = MODEL_ID
    parent.OUT_DIR = OUT_DIR
    sys.argv = [
        sys.argv[0],
        "--epochs",
        "4",
        "--max-train-rows",
        "30000",
        "--max-exit-samples",
        "12000",
        "--quality-threshold",
        "0.45",
        "--thresholds",
        "",
        "--out-suffix",
        "smoke_e4_train30k_exit12k",
        "--device",
        "cpu",
    ]
    return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())

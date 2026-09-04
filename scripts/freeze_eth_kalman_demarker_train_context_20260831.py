#!/usr/bin/env python3
"""Freezes the TRAIN-only (ts < VAL_START) slice of each signal's FIRES csv into the
tabpfn_train_context_frozen_<name>_20260831.csv format live_evidence_signal_metalabel_20260829.py
expects (timestamp + feature columns + a column named exactly "hit" -- the FIRES csv itself names
it "hit_plain", this repo's own convention for the demarker/kalman lineage specifically, renamed
here only for the live-serving contract). No other prior signal's freeze script exists in this
repo (each was evidently done as a small uncommitted one-off) -- this one is kept since it may be
useful again if either signal's config changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_kalman_demarker_gridscreen_20260831 import VAL_START  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = ROOT / "data/labels/eth_5m_kalman_demarker_metalabel_20260831"

SIGNAL_FILES = {
    "demarker_extreme": (DATA_DIR / "eth_5m_demarker_extreme_metalabel_features_H8_GAP12_K0.7.csv", FEATURE_COLUMNS + ["dem"]),
    "kalman_deviation_meanrev": (DATA_DIR / "eth_5m_kalman_deviation_meanrev_metalabel_features_H12_GAP12_K2.5.csv", FEATURE_COLUMNS + ["kalman_dev_z"]),
}


def main() -> None:
    for name, (src_path, feature_cols) in SIGNAL_FILES.items():
        fires = pd.read_csv(src_path, parse_dates=["timestamp"])
        train = fires.loc[fires["timestamp"] < VAL_START].copy()
        train = train.rename(columns={"hit_plain": "hit"})
        keep_cols = ["timestamp"] + feature_cols + ["hit"]
        train = train[keep_cols]
        out_path = DATA_DIR / f"tabpfn_train_context_frozen_{name}_20260831.csv"
        train.to_csv(out_path, index=False)
        print(f"{name}: {len(train)} train rows (pos={int(train['hit'].sum())}) -> {out_path}")


if __name__ == "__main__":
    main()

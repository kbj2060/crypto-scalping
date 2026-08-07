#!/usr/bin/env python3
"""Remove short-lived BTC H48 label flicker with causal hysteresis.

A new state must persist for ``confirm_bars`` consecutive raw labels and the
current output state must have lasted at least ``min_dwell_bars``. The filter
uses only the current and previous raw labels; it adds no look-ahead beyond the
underlying offline H48 training target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"
)
DEFAULT_OUTPUT = (
    ROOT
    / "tmp/causal_regen_20260516/btc_h48_conservative_causal_hysteresis_c6_d12_20260715"
)


def causal_hysteresis(actions: np.ndarray, *, confirm_bars: int, min_dwell_bars: int) -> np.ndarray:
    raw = np.asarray(actions, dtype=np.int64)
    if len(raw) == 0:
        return raw.copy()
    invalid = sorted(set(np.unique(raw).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"invalid action classes: {invalid}")
    if confirm_bars < 1 or min_dwell_bars < 1:
        raise ValueError("confirm_bars and min_dwell_bars must be positive")

    out = np.empty_like(raw)
    state = int(raw[0])
    candidate = state
    candidate_count = 0
    dwell = int(min_dwell_bars)
    for i, action_value in enumerate(raw):
        action = int(action_value)
        if action == state:
            candidate = state
            candidate_count = 0
        else:
            if action == candidate:
                candidate_count += 1
            else:
                candidate = action
                candidate_count = 1
            if dwell >= int(min_dwell_bars) and candidate_count >= int(confirm_bars):
                state = candidate
                dwell = 0
                candidate_count = 0
        out[i] = state
        dwell += 1
    return out


def _flips(actions: np.ndarray) -> int:
    arr = np.asarray(actions, dtype=np.int64)
    return int(np.sum(arr[1:] != arr[:-1])) if len(arr) > 1 else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-bars", type=int, default=6)
    parser.add_argument("--min-dwell-bars", type=int, default=12)
    args = parser.parse_args()

    source_paths = sorted(args.input_dir.glob("zigzag_action_labels_*.csv"))
    if not source_paths:
        raise FileNotFoundError(f"no zigzag_action_labels_*.csv files in {args.input_dir}")
    frames = []
    for path in source_paths:
        year = int(path.stem.rsplit("_", 1)[-1])
        frame = pd.read_csv(path, usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        frame["source_year_file"] = int(year)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if combined["timestamp"].duplicated().any():
        raise RuntimeError("H48 source contains duplicate timestamps across year files")

    raw = pd.to_numeric(combined["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    smooth = causal_hysteresis(
        raw,
        confirm_bars=int(args.confirm_bars),
        min_dwell_bars=int(args.min_dwell_bars),
    )
    combined["raw_h48_action"] = raw
    combined["zigzag_action"] = smooth

    args.output_dir.mkdir(parents=True, exist_ok=True)
    years = {}
    calendar_years = sorted(int(x) for x in combined["timestamp"].dt.year.unique())
    for year in calendar_years:
        part = combined.loc[combined["timestamp"].dt.year == year, [
            "timestamp", "zigzag_action", "raw_h48_action"
        ]].reset_index(drop=True)
        if part.empty:
            raise RuntimeError(f"no output rows for calendar year {year}")
        path = args.output_dir / f"zigzag_action_labels_{year}.csv"
        part.to_csv(path, index=False)
        raw_part = part["raw_h48_action"].to_numpy(dtype=np.int64)
        smooth_part = part["zigzag_action"].to_numpy(dtype=np.int64)
        raw_flips = _flips(raw_part)
        smooth_flips = _flips(smooth_part)
        years[str(year)] = {
            "rows": int(len(part)),
            "raw_flips": raw_flips,
            "smoothed_flips": smooth_flips,
            "flip_reduction_ratio": float(1.0 - smooth_flips / max(raw_flips, 1)),
            "smoothed_class_counts": {
                str(k): int(v)
                for k, v in part["zigzag_action"].value_counts().sort_index().items()
            },
            "path": str(path),
        }

    manifest = {
        "label_family": "btc_h48_conservative_causal_hysteresis",
        "source_dir": str(args.input_dir),
        "confirm_bars": int(args.confirm_bars),
        "confirmation_minutes": int(args.confirm_bars) * 5,
        "min_dwell_bars": int(args.min_dwell_bars),
        "min_dwell_minutes": int(args.min_dwell_bars) * 5,
        "causal_filter": True,
        "added_future_rows": 0,
        "years": years,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "years": years}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build lower-flip SOL Zig labels from the frozen SOL Zig label artifacts.

This is an offline label transform only. It removes active label runs shorter than four hours
(48 five-minute bars), then bridges same-side runs separated by at most one hour (12 bars) of
CASH. It never changes a LONG label to SHORT or vice versa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_hysteresis_labels_20260719"
MIN_ACTIVE_BARS = 48
MAX_SAME_SIDE_CASH_GAP_BARS = 12


def _runs(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(actions[1:] != actions[:-1]) + 1]
    ends = np.r_[starts[1:], len(actions)]
    return starts, ends, actions[starts]


def _smooth(actions: np.ndarray) -> np.ndarray:
    out = actions.astype(np.int64, copy=True)
    starts, ends, values = _runs(out)
    for start, end, value in zip(starts, ends, values):
        if value != 0 and end - start < MIN_ACTIVE_BARS:
            out[start:end] = 0

    starts, ends, values = _runs(out)
    for idx in range(1, len(values) - 1):
        same_active_side = values[idx - 1] == values[idx + 1] and values[idx - 1] != 0
        short_cash_gap = values[idx] == 0 and ends[idx] - starts[idx] <= MAX_SAME_SIDE_CASH_GAP_BARS
        if same_active_side and short_cash_gap:
            out[starts[idx]:ends[idx]] = values[idx - 1]
    return out


def _diag(actions: np.ndarray) -> dict[str, Any]:
    starts, ends, values = _runs(actions)
    active_lengths = ends[values != 0] - starts[values != 0]
    return {
        "rows": int(len(actions)),
        "counts": {str(value): int(np.sum(actions == value)) for value in (0, 1, 2)},
        "active_ratio": float(np.mean(actions != 0)),
        "entries": int(np.sum((actions[1:] != 0) & (actions[1:] != actions[:-1]))),
        "transitions": int(np.sum(actions[1:] != actions[:-1])),
        "direct_long_short_flips": int(np.sum(
            ((actions[:-1] == 1) & (actions[1:] == 2))
            | ((actions[:-1] == 2) & (actions[1:] == 1))
        )),
        "active_run_median_bars": float(np.median(active_lengths)) if len(active_lengths) else 0.0,
        "active_run_p10_bars": float(np.quantile(active_lengths, 0.10)) if len(active_lengths) else 0.0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "contract": {
            "source": str(SOURCE_DIR),
            "bar_minutes": 5,
            "min_active_bars": MIN_ACTIVE_BARS,
            "min_active_minutes": MIN_ACTIVE_BARS * 5,
            "max_same_side_cash_gap_bars": MAX_SAME_SIDE_CASH_GAP_BARS,
            "max_same_side_cash_gap_minutes": MAX_SAME_SIDE_CASH_GAP_BARS * 5,
            "short_active_runs_become": "CASH",
            "short_cash_gap_bridge_requires": "same non-CASH side on both sides",
            "long_short_relabeling": False,
            "offline_training_label_only": True,
        },
        "splits": {},
    }
    for year in (2024, 2025, 2026):
        source = SOURCE_DIR / f"zigzag_action_labels_{year}.csv"
        frame = pd.read_csv(source, parse_dates=["timestamp"])
        actions = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
        if not np.isin(actions, [0, 1, 2]).all():
            raise RuntimeError(f"{source}: unexpected label id")
        smoothed = _smooth(actions)
        output = pd.DataFrame({"timestamp": frame["timestamp"], "zigzag_action": smoothed})
        output.to_csv(OUT_DIR / f"zigzag_action_labels_{year}.csv", index=False)
        audit["splits"][str(year)] = {
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "before": _diag(actions),
            "after": _diag(smoothed),
            "changed_rows": int(np.sum(actions != smoothed)),
        }
        print(year, json.dumps(audit["splits"][str(year)]), flush=True)

    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR / 'audit.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

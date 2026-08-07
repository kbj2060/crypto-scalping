#!/usr/bin/env python3
"""Sigma3 1h dataset, continuous version: concatenates 2024/2025/2026 5m source files BEFORE
resampling/feature computation, instead of processing each year independently
(build_1h_trendscan_dataset_20260705.py's per-year loop cold-starts every rolling window --
e.g. dist_sma50/rvol_48/vol_z_48 -- at each Jan 1, discarding real prior-year history that is
available and would otherwise warm them up).

Rationale for doing this now: the plan is to retrain including 2024 as real training rows (not
just leaked-through-rolling-context warmup), via purged walk-forward folds over 2024-01..2025-08,
holdout VAL 2025-09..12, OOS 2026-01..03 untouched. That requires 2025-01-01's rolling features
to be computed on continuous history, not cold-started, same as every other day in the series.

Causality unchanged from the original: resample("1h", label="left", closed="left"); features are
rolling/shift only; trend-scanning labels use forward windows (label construction only, standard
convention, excluded from the feature set downstream). Single continuous frame means trend-scan's
forward window can now cross what used to be a year boundary -- this is fine, it's still using
only data that becomes available causally after each bar, no different from within a single year.
"""

from __future__ import annotations

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

from build_1h_trendscan_dataset_extended_20260720 import (  # noqa: E402
    compute_features,
    resample_1h,
    trend_scan_fast,
)

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_continuous_20260801"
SRC_FILES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]

TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in SRC_FILES:
        src = pd.read_csv(path, low_memory=False)
        src["timestamp"] = pd.to_datetime(src["timestamp"])
        frames.append(src)
    full = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"concatenated 5m rows: {len(full)}, range {full['timestamp'].min()}..{full['timestamp'].max()}", flush=True)

    r = resample_1h(full)
    feats = compute_features(r)

    windows = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
    t_vals, opt_l, betas = trend_scan_fast(logc, windows)
    labels = np.zeros(len(feats), dtype=np.int64)
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats["ts_action"] = labels
    feats["ts_t_value"] = t_vals.astype(np.float32)
    feats["ts_opt_L"] = opt_l.astype(np.int16)

    out_path = OUT_DIR / "sigma3_1h_continuous.parquet"
    feats.to_parquet(out_path, index=False)

    # sanity: confirm no cold-start gap at the old year boundaries (2025-01-01, 2026-01-01) --
    # dist_sma50 (window 50) and rvol_48/vol_z_48 (window 48) should already be non-NaN there,
    # unlike the per-year build where they'd restart from scratch.
    check_points = ["2025-01-01 00:00:00", "2026-01-01 00:00:00"]
    checks = {}
    for cp in check_points:
        row = feats[feats["timestamp"] == pd.Timestamp(cp)]
        if len(row):
            checks[cp] = {
                "dist_sma50_isnan": bool(pd.isna(row["dist_sma50"].iloc[0])),
                "vol_z_48_isnan": bool(pd.isna(row["vol_z_48"].iloc[0])),
            }
    counts = np.bincount(labels, minlength=3).tolist()
    report = {
        "windows": windows.tolist(), "threshold": float(TS_THRESHOLD),
        "rows_1h": int(len(feats)),
        "range": [str(feats["timestamp"].min()), str(feats["timestamp"].max())],
        "label_counts_CASH_LONG_SHORT": counts,
        "label_ratios": [round(c / max(len(feats), 1), 3) for c in counts],
        "year_boundary_coldstart_check": checks,
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

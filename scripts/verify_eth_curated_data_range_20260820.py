"""Generic 5-min-grid integrity verifier for curated ETH CSV datasets.

Checks (same rigor as docs/eth_canonical_data_date_range_verification_20260820.md and the
memory note of the same name): row count vs expected 5-min grid, timestamp gaps (with sizes),
duplicate timestamps, per-column NaN counts, inf counts. Read-only -- never modifies its input.

Usage:
  python scripts/verify_eth_curated_data_range_20260820.py <csv_path> [--ts-col timestamp] [--freq 5min]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--ts-col", default="timestamp")
    ap.add_argument("--freq", default="5min")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path, low_memory=False)
    ts_col = args.ts_col
    df[ts_col] = pd.to_datetime(df[ts_col])

    n = len(df)
    tmin, tmax = df[ts_col].min(), df[ts_col].max()
    print(f"file: {args.csv_path}")
    print(f"rows: {n:,}")
    print(f"range: {tmin} .. {tmax}")
    print(f"columns: {len(df.columns)}")

    # duplicate timestamps
    dupe_mask = df[ts_col].duplicated(keep=False)
    n_dupe = int(df[ts_col].duplicated(keep="first").sum())
    print(f"duplicate timestamps (extra copies beyond first): {n_dupe}")
    if n_dupe:
        print("  sample duplicated timestamps:", df.loc[dupe_mask, ts_col].drop_duplicates().head(10).tolist())

    # sorted?
    is_sorted = df[ts_col].is_monotonic_increasing
    print(f"strictly sorted ascending: {is_sorted}")

    # grid completeness
    expected = pd.date_range(tmin, tmax, freq=args.freq)
    actual_set = set(df[ts_col])
    missing = sorted(set(expected) - actual_set)
    print(f"expected grid bars: {len(expected):,}")
    print(f"missing bars: {len(missing)}")
    if missing:
        # group consecutive missing timestamps into gap runs
        gaps = []
        run_start = missing[0]
        prev = missing[0]
        step = pd.Timedelta(args.freq)
        for t in missing[1:]:
            if t - prev == step:
                prev = t
                continue
            gaps.append((run_start, prev))
            run_start = t
            prev = t
        gaps.append((run_start, prev))
        print(f"gap runs: {len(gaps)}")
        for gs, ge in gaps:
            nbars = int((ge - gs) / step) + 1
            print(f"  gap: {gs} -> {ge}  ({nbars} bars)")

    # NaN audit
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    total_nan = int(nan_counts.sum())
    print(f"total NaN cells: {total_nan}")
    if len(nan_cols):
        print(f"columns with NaN ({len(nan_cols)}):")
        for col, cnt in nan_cols.items():
            pct = 100.0 * cnt / n
            print(f"  {col}: {cnt} ({pct:.4f}%)")

    # inf audit (numeric cols only)
    num_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(num_df).sum()
    inf_cols = inf_counts[inf_counts > 0].sort_values(ascending=False)
    total_inf = int(inf_counts.sum())
    print(f"total inf cells: {total_inf}")
    if len(inf_cols):
        for col, cnt in inf_cols.items():
            print(f"  {col}: {cnt}")

    print("\nSUMMARY:", "CLEAN" if (n_dupe == 0 and len(missing) == 0 and total_nan == 0 and total_inf == 0) else "ISSUES_FOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

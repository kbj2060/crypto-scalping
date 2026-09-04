#!/usr/bin/env python3
"""Truncate a features CSV to timestamp <= --cutoff, with a backup.

Why this exists: 2026-08-31 A4 pipeline run discovered that extending raw klines all the way to
"now" pushes the features canonical files' tails PAST what the daily metrics archive
(data.binance.vision, published only through the previous day -- see
scripts/download_eth_binance_metrics_archive_20260823.py's own END default) can support. The
2026-08-23 metrics-integrity fix scripts (fix_eth_canonical_2026_oi_futureleak_20260823.py /
fix_btcsol_metrics_vintage_20260823.py) replace raw OI/long-short-ratio columns via an exact-join
+ 9-hour-tolerance merge_asof against that reference; a tail that outruns the reference by more
than 9h leaves those rows with unfixable NaN, and both fix scripts correctly fail fast rather than
silently leaving (or worse, ffilling) a gap. Rather than loosening that tolerance (a deliberate,
already-validated design choice from the original fix scripts, not something to relax casually),
this script caps each asset's feature extension at the metrics reference's own actual coverage --
the correct fix for a real data-availability boundary, consistent with the project's existing
"data finality buffer" pattern (scripts/extend_regime3_wide24_sol_btc_20260721.py's docstring).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--cutoff", type=str, required=True, help="YYYY-MM-DD HH:MM:SS, inclusive upper bound")
    ap.add_argument("--backup-suffix", type=str, default=".bak_pre_metrics_safe_truncate_20260831")
    args = ap.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    df = pd.read_csv(args.path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n0 = len(df)
    max0 = df["timestamp"].max()
    if max0 <= cutoff:
        print(f"{args.path}: max timestamp {max0} already <= cutoff {cutoff}, nothing to truncate", flush=True)
        return 0

    kept = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    dropped = n0 - len(kept)

    backup = args.path.with_name(args.path.name + args.backup_suffix)
    if not backup.exists():
        shutil.copy2(args.path, backup)
        print(f"backed up to {backup}", flush=True)

    # Explicit string formatting before write: pandas' default datetime64->CSV serialization was
    # observed (2026-08-31) to render at least one row (the exact cutoff row, time=00:00:00) as a
    # bare "YYYY-MM-DD" with no time component while every other row got the full
    # "YYYY-MM-DD HH:MM:SS", breaking downstream pd.to_datetime(..., format=None) calls on mixed
    # formats. Force a single, unambiguous format for every row instead of trusting the default.
    kept["timestamp"] = kept["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    tmp = args.path.with_suffix(args.path.suffix + ".tmp_trunc")
    kept.to_csv(tmp, index=False)
    tmp.replace(args.path)
    print(f"{args.path}: truncated {n0} -> {len(kept)} rows (dropped {dropped} rows > {cutoff}), "
          f"new max timestamp {kept['timestamp'].max()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

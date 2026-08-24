"""Extend the canonical curated ETH feature file
(data/splits/year_oos/training_features_2026_rebuilt.csv, currently ends 2026-07-20) forward to
the latest date the raw sources actually support (target 2026-08-19), for the C-plan
anchored-walk-forward re-split retest (docs/model_contracts/
ilias_eth_human_direction_risk_management_contract_20260817.md "## 데이터 Split 재설계 제안").

Mechanism (reverse-engineered 2026-08-20 -- no single canonical rebuild script existed for this
exact target file; scripts/update_features.py is the underlying raw-download + FeatureEngineer
pipeline, confirmed by its docstring and by scripts/apply_regime3_*_extended_*.py's consumption of
training_features_2026_rebuilt.csv as the base to overlay regime3 columns onto). This script reuses
scripts/update_features.py's helper functions directly (ensure_metrics/ensure_funding/ensure_klines/
load_all_sources/build_features) rather than shelling out to it, for two reasons:
  1. update_features.py's own accumulator (data/training_features_5m.csv) is currently a stale,
     narrow scratch file (3,563 rows, 2026-07-20..08-01 only, last touched 2026-08-02) -- NOT a
     continuous 2024+ master. Overwriting/seeding it risks colliding with whatever left it in that
     state, and this repo runs many concurrent Claude sessions against the same working tree
     (project memory: concurrent_claude_sessions_shared_repo). This script never touches that file.
  2. update_features.py's own check_gaps()+load_all_sources() combo, when given a real gap window,
     feeds FeatureEngineer.process() ONLY the raw rows inside [gap_start, gap_end] -- so every
     recursive/rolling feature (GARCH(1,1) alpha+beta=0.95, OU halflife's 1440-bar/5-day AR(1)
     window, CVP lookback=200, most others <=288 bars) gets a cold start at gap_start. This script
     instead feeds it a 60-day PRE-GAP BUFFER (>10x the largest observed window, 1440 bars) so every
     rolling/recursive feature is fully warmed up before the first genuinely-new row, then discards
     the buffer rows before appending -- so the newly-appended rows are not systematically biased
     relative to how the rest of the file was computed.

Raw source downloads (klines/metrics/funding zips) still land in the normal shared binance_data/
cache via update_features.py's own ensure_* functions -- that cache is meant to be shared/growing
infra (same as scripts/extend_klines_20260713.py), unlike the FEATURES_CSV accumulator.

Usage:
  python scripts/extend_eth_curated_features_20260820.py [--end 2026-08-19] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_features import (  # noqa: E402
    ensure_metrics, ensure_funding, ensure_klines, load_all_sources, build_features,
)

TARGET = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
BUFFER_DAYS = 60  # >10x the largest rolling/recursive window in features/ (1440 bars = 5 days)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-08-19", help="requested end date (YYYY-MM-DD), inclusive")
    ap.add_argument("--dry-run", action="store_true", help="download+compute but do not overwrite TARGET")
    args = ap.parse_args()

    existing = pd.read_csv(TARGET, low_memory=False)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"]).astype("datetime64[us]")
    existing_max = existing["timestamp"].max()
    existing_min = existing["timestamp"].min()
    print(f"existing TARGET: rows={len(existing):,} range=({existing_min}, {existing_max})")

    gap_start = existing_max + timedelta(minutes=5)
    requested_end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(hours=23, minutes=55)
    buffer_start = (existing_max - timedelta(days=BUFFER_DAYS)).to_pydatetime()

    print(f"gap_start (first genuinely-new row)={gap_start}")
    print(f"buffer_start (warmup feed, discarded after compute)={buffer_start}")
    print(f"requested_end={requested_end}")

    print("\n[1/5] ensure_metrics/ensure_funding/ensure_klines (shared binance_data/ cache)...")
    ensure_metrics(buffer_start, requested_end)
    ensure_funding(buffer_start, requested_end)
    ensure_klines(buffer_start, requested_end)

    print("\n[2/5] load_all_sources over buffered window...")
    eth_df, btc_df, metrics_df, funding_df = load_all_sources(buffer_start, requested_end)
    print(f"  eth_df rows={len(eth_df):,} range=({eth_df['timestamp'].min()}, {eth_df['timestamp'].max()})")
    print(f"  btc_df rows={len(btc_df):,} range=({btc_df['timestamp'].min()}, {btc_df['timestamp'].max()})")

    print("\n[3/5] build_features (FeatureEngineer.process, full buffered range)...")
    new_features = build_features(eth_df, btc_df, metrics_df, funding_df)
    new_features["timestamp"] = pd.to_datetime(new_features["timestamp"]).astype("datetime64[us]")
    print(f"  computed rows={len(new_features):,} range=({new_features['timestamp'].min()}, {new_features['timestamp'].max()})")

    print("\n[4/5] slice to genuinely-new rows (timestamp > existing_max) and check schema parity...")
    fresh = new_features[new_features["timestamp"] > existing_max].copy().sort_values("timestamp").reset_index(drop=True)
    if fresh.empty:
        print("  No new rows produced (raw sources do not yet extend past existing_max). Nothing to do.")
        return 1
    achieved_end = fresh["timestamp"].max()
    print(f"  fresh new rows={len(fresh):,} range=({fresh['timestamp'].min()}, {achieved_end})")

    existing_cols = set(existing.columns)
    fresh_cols = set(fresh.columns)
    missing_in_fresh = existing_cols - fresh_cols
    extra_in_fresh = fresh_cols - existing_cols
    if missing_in_fresh:
        print(f"  !! SCHEMA MISMATCH missing_in_fresh={sorted(missing_in_fresh)} -- refusing (a real feature disappeared)")
        return 2
    if extra_in_fresh:
        # Current features/schema.py active-feature list has grown since existing was built
        # (e.g. session_europe_open/session_japan/session_japan_open/session_us_open --
        # deterministic calendar dummies, features/engineering.py:23-24,503-523). Out of scope
        # for a pure date-range extension (Shared Feature Contract: no new features here) --
        # drop to preserve exact schema parity with the frozen 142-col file being extended.
        print(f"  note: dropping {len(extra_in_fresh)} cols not in existing schema (schema grew since existing was built): {sorted(extra_in_fresh)}")
    fresh = fresh[existing.columns.tolist()]
    print(f"  schema parity OK: {len(existing.columns)} columns match exactly")

    n_nan_fresh = int(fresh.isna().sum().sum())
    n_inf_fresh = int(np.isinf(fresh.select_dtypes(include=[np.number])).sum().sum())
    print(f"  fresh rows NaN cells={n_nan_fresh} inf cells={n_inf_fresh}")
    if n_nan_fresh or n_inf_fresh:
        print("  !! fresh rows contain NaN/inf -- refusing to append a corrupt extension")
        return 3

    if args.dry_run:
        print("\n--dry-run: not writing TARGET. Done.")
        return 0

    print("\n[5/5] backup + write...")
    backup_path = TARGET.with_name(TARGET.name + ".bak_pre_extend_20260820")
    if backup_path.exists():
        print(f"  !! backup already exists, refusing to overwrite: {backup_path}")
        return 4
    existing.to_csv(backup_path, index=False)
    print(f"  backup written: {backup_path} ({len(existing):,} rows)")

    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = (combined.drop_duplicates(subset=["timestamp"], keep="first")
                         .sort_values("timestamp").reset_index(drop=True))
    combined.to_csv(TARGET, index=False)
    print(f"  wrote TARGET: {TARGET} rows={len(combined):,} range=({combined['timestamp'].min()}, {combined['timestamp'].max()})")
    print(f"\nDONE. achieved_end={achieved_end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

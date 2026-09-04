#!/usr/bin/env python3
"""Build ETH's raw frame (ETH primary + BTC cross) from currently-available local klines/
metrics/funding zips, matching build_sol_raw_frame_20260707.py / build_btc_raw_frame_20260708.py's
exact column contract and merge conventions (same required columns, same ffill-for-scattered-gaps
treatment, same coverage-trim-to-first-fully-covered-row, same bare `FeatureEngineer()` call with
no keep_only_active override). Then safely EXTENDS the canonical research file
data/splits/year_oos/training_features_2026_rebuilt.csv with an existing-first merge: existing rows
are preserved byte-identical (including the 2026-08-23 metrics-integrity fixes), and only genuinely
new tail rows (timestamp > current file's max) are appended from this fresh computation. Does NOT
touch training_features_2024.csv / training_features_2025.csv -- the 2026-08-23 audit found ETH's
2024/2025 files are deliberately frozen on an older features/elite.py formula vintage, with
unification explicitly deferred to a future full rebuild
(docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md).

Why this script exists (context): docs/pipeline_integrity_and_research_redesign_20260730.md
found there is no single canonical writer script for training_features_2026_rebuilt.csv ("dozens
of one-off build scripts are candidates"). scripts/update_features.py only ever writes
data/training_features_5m.csv (FEATURES_CSV, a much narrower/shorter live-bot-oriented file,
FeatureEngineer(keep_only_active=True)) and never touches the _rebuilt file at all -- confirmed by
direct code reading 2026-08-31. Written for the A4 cross-symbol-exposure-cap fresh confirmation
task (docs/eth_cross_symbol_exposure_cap_design_20260831.md), mirroring the SOL/BTC "full
FeatureEngineer recompute, not append-only" methodology the task calls for (avoids
ou_halflife/garch_vol_z rolling-window seeding errors), applied here only to produce the fresh
tail (the existing-first merge protects already-fixed history).

IMPORTANT KNOWN CAVEAT: the raw sum_open_interest_value/sum_toptrader_long_short_ratio/
count_long_short_ratio columns computed here for the new tail still carry the known Binance-
archive label-vintage risk (create_time used as-is here, no +5min correction -- see
docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md,
section 4: a systematic 1-bucket future-reference join is possible if the archive's current
convention is "bucket start"). This is corrected AFTERWARD by re-running (unmodified)
scripts/fix_eth_canonical_2026_oi_futureleak_20260823.py against a freshly-extended
data/TOTAL_ETHUSDT_metrics_2024_2026.csv reference (that script's fix window is
WIN_START..df["timestamp"].max(), i.e. dynamically "through file end", so re-running it after this
script naturally extends the correction to the new tail too). Do not skip that follow-up step.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

SYMBOL = "ETHUSDT"
KLINE_PATH = ROOT / f"binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
CROSS_KLINE_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
CANONICAL_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
BACKUP = CANONICAL_2026.with_suffix(".csv.bak_pre_extend_20260831")

PRIMARY_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
                     "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate"]
CROSS_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
REQUIRED = PRIMARY_RAW_COLS + CROSS_RAW_COLS[1:]


def load_metrics() -> pd.DataFrame:
    frames = []
    for p in sorted(METRICS_DIR.glob(f"{SYMBOL}-metrics-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["create_time"])
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]


def load_funding() -> pd.DataFrame:
    frames = []
    for p in sorted(FUNDING_DIR.glob(f"{SYMBOL}-fundingRate-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["calc_time"], unit="ms")
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "last_funding_rate"]]


def build_raw_frame() -> pd.DataFrame:
    kline = pd.read_csv(KLINE_PATH, low_memory=False)
    kline["timestamp"] = pd.to_datetime(kline["timestamp"])
    kline = kline.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"ETH klines: {len(kline)} rows {kline['timestamp'].iloc[0]}..{kline['timestamp'].iloc[-1]}", flush=True)

    metrics = load_metrics()
    print(f"ETH metrics: {len(metrics)} rows {metrics['timestamp'].iloc[0]}..{metrics['timestamp'].iloc[-1]}", flush=True)
    funding = load_funding()
    print(f"ETH funding: {len(funding)} rows {funding['timestamp'].iloc[0]}..{funding['timestamp'].iloc[-1]}", flush=True)

    btc = pd.read_csv(CROSS_KLINE_PATH, low_memory=False, usecols=["timestamp", "close", "volume", "quote_volume"])
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    btc = btc.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})

    frame = kline.merge(metrics, on="timestamp", how="left")
    frame = pd.merge_asof(frame.sort_values("timestamp"), funding.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    frame = frame.merge(btc[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")

    ffill_cols = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                  "close_btc", "volume_btc", "quote_volume_btc"]
    frame[ffill_cols] = frame[ffill_cols].ffill()

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise RuntimeError(f"missing required columns after merge: {missing}")

    coverage_ok = frame[REQUIRED].notna().all(axis=1)
    first_ok = coverage_ok.idxmax()
    frame = frame.iloc[first_ok:].reset_index(drop=True)
    na_counts = frame[REQUIRED].isna().sum()
    print("\nNA counts per required column after trimming to full-coverage region:")
    print(na_counts[na_counts > 0] if na_counts.any() else "  (none)")
    print(f"\nETH raw frame: {len(frame)} rows {frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}", flush=True)
    return frame


def main() -> int:
    frame = build_raw_frame()

    primary_df = frame[PRIMARY_RAW_COLS].copy()
    cross_df = frame[CROSS_RAW_COLS].copy()

    fe = FeatureEngineer()
    features = fe.process(primary_df, cross_df)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    print(f"\nETH engineered features (full recompute): {len(features)} rows, {len(features.columns)} columns", flush=True)
    print(f"range: {features['timestamp'].iloc[0]}..{features['timestamp'].iloc[-1]}", flush=True)

    if not CANONICAL_2026.exists():
        raise RuntimeError(f"expected existing canonical file at {CANONICAL_2026}, refusing to create from scratch here")

    existing = pd.read_csv(CANONICAL_2026, low_memory=False)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    existing_max = existing["timestamp"].max()
    print(f"\nexisting canonical: {len(existing)} rows, max timestamp {existing_max}", flush=True)

    new_tail = features[features["timestamp"] > existing_max].reset_index(drop=True)
    print(f"new tail rows beyond existing canonical max: {len(new_tail)}", flush=True)
    if new_tail.empty:
        print("nothing new to add -- canonical file already covers the full recomputed range", flush=True)
        return 0

    missing_in_new = set(existing.columns) - set(new_tail.columns)
    extra_in_new = set(new_tail.columns) - set(existing.columns)
    if missing_in_new:
        raise RuntimeError(f"new tail is missing columns present in existing canonical -- aborting: {sorted(missing_in_new)}")
    if extra_in_new:
        print(f"WARNING: new tail has extra columns not in existing canonical (dropped to match): {sorted(extra_in_new)}", flush=True)
    new_tail = new_tail[existing.columns.tolist()]

    nan_counts = new_tail.isna().sum()
    bad_cols = nan_counts[nan_counts > 0]
    if len(bad_cols):
        raise RuntimeError(f"new tail has NaN values (expected none this far past warmup) -- aborting: {bad_cols.to_dict()}")

    if not BACKUP.exists():
        shutil.copy2(CANONICAL_2026, BACKUP)
        print(f"backed up existing canonical to {BACKUP}", flush=True)

    combined = pd.concat([existing, new_tail], ignore_index=True)
    combined = (combined.drop_duplicates(subset=["timestamp"], keep="first")
                         .sort_values("timestamp").reset_index(drop=True))
    # Explicit string formatting before write -- pandas' default datetime64->CSV serialization
    # was observed (2026-08-31) to occasionally render an exact-midnight row as bare "YYYY-MM-DD"
    # with no time component while every other row got the full "YYYY-MM-DD HH:MM:SS", which then
    # breaks downstream pd.to_datetime(..., format=None) calls on the resulting mixed formats.
    combined["timestamp"] = pd.to_datetime(combined["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    tmp = CANONICAL_2026.with_suffix(".csv.tmp_write")
    combined.to_csv(tmp, index=False)
    tmp.replace(CANONICAL_2026)
    print(f"\nextended canonical: {len(combined)} rows, {combined['timestamp'].min()}..{combined['timestamp'].max()} -> {CANONICAL_2026}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Join the live microstructure_1m duckdb table (order-book-derived signals: OBI, taker-buy
ratio, whale/retail flow, spoofing score, shadow toxicity/absorption, etc.) onto the 1m ETH
feature set built by build_features_1m_20260716.py.

microstructure_1m only exists from 2026-05-03 (KST) onward -- this necessarily bounds any
microstructure-augmented model to that window (~70 days as of 2026-07-16), unlike the
pure-price 1m features which cover 2024-01-01 onward. Output keeps every row from the base
feature set; microstructure columns are NaN outside the overlap window (rows outside the
overlap should be dropped by the training script, not here, so this file stays a superset).

Output: data/training_features_1m_with_microstructure.csv
"""
import os
import sys

import duckdb
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
DUCKDB_PATH = os.path.join(_ROOT_DIR, 'data', 'live', 'microstructure.duckdb')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_with_microstructure.csv')

# Only the derived-signal columns -- drop connectivity/staleness bookkeeping columns
# (data_stale, *_connected, *_age_sec, warmup_30m_ready, schema_version) that describe the
# live collector's own health, not market state.
MICROSTRUCTURE_COLS = [
    'ts', 'obi', 'taker_buy_ratio', 'spoofing_score', 'nif_whale', 'nif_retail', 'eai',
    'oi_delta_pct', 'funding_rate', 'kelly_mult', 'signal_bias',
    'shadow_toxicity_score', 'shadow_toxicity_regime', 'shadow_queue_collapse',
    'shadow_absorption_score', 'shadow_queue_bias', 'shadow_regime_tag', 'shadow_regime_conf',
    'recent_trade_count_5m', 'recent_trade_notional_5m', 'recent_whale_count_5m',
]


def main():
    print("Loading 1m ETH feature base...")
    feat = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
    print(f"  {len(feat):,} rows, {feat['timestamp'].min()} -> {feat['timestamp'].max()}")

    print("Loading microstructure_1m from duckdb...")
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    cols_sql = ', '.join(MICROSTRUCTURE_COLS)
    micro = con.execute(f"SELECT {cols_sql} FROM microstructure_1m ORDER BY ts").fetchdf()
    con.close()
    # ts is KST tz-aware in the duckdb table -- convert to naive UTC to match the Binance-sourced
    # feature timestamps (Binance kline timestamps are UTC, naive in training_features_1m.csv).
    micro['timestamp'] = pd.to_datetime(micro['ts']).dt.tz_convert('UTC').dt.tz_localize(None)
    micro = micro.drop(columns=['ts']).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    print(f"  {len(micro):,} rows, {micro['timestamp'].min()} -> {micro['timestamp'].max()} (UTC)")

    print("Merging (backward asof, 2min tolerance, no look-ahead)...")
    merged = pd.merge_asof(
        feat.sort_values('timestamp'), micro,
        on='timestamp', direction='backward', tolerance=pd.Timedelta('2min'),
    )
    overlap_rows = merged['obi'].notna().sum()
    print(f"  Merged: {len(merged):,} rows total, {overlap_rows:,} rows have microstructure data "
          f"({overlap_rows / len(merged):.1%})")

    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}: {len(merged):,} rows, {len(merged.columns)} cols")


if __name__ == '__main__':
    main()

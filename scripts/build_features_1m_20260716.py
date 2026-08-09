"""One-shot 1m feature build for the new ETH scalping model line.

Unlike update_features.py (5m, incremental gap-fill), this does a single full build from the
freshly-downloaded full-history 1m klines (scripts/download_klines_1m_20260716.py). Reuses the
same FeatureEngineer / metrics / funding merge logic as the 5m pipeline, but with
candle_minutes=1. Output: data/training_features_1m.csv

NOTE: FeatureEngineer's rolling-window features (windows={'short':5,'medium':20,'long':288,...})
are bar-count based, not real-time based -- candle_minutes is stored but not threaded into any
window size (verified: no other reference in features/engineering.py). Running it on 1m bars
therefore compresses every window's real-time span by 5x versus the 5m pipeline (e.g. the
288-bar "long" window is 24h on 5m bars but ~4.8h on 1m bars). This is intentional here: this is
a dedicated scalping model, not a resampled clone of the 5m model, so faster-reacting windows are
the intended behavior, not a bug to correct.
"""
import os
import sys
import zipfile
import io

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from features.engineering import FeatureEngineer
from features.schema import prune_to_active_feature_keep

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
BINANCE_DIR = os.path.join(_ROOT_DIR, 'binance_data')
METRICS_DIR = os.path.join(BINANCE_DIR, 'metrics')
FUNDING_RATE_DIR = os.path.join(BINANCE_DIR, 'funding_rate')
ETH_1M_CSV = os.path.join(BINANCE_DIR, 'klines', 'ETHUSDT', 'ETHUSDT-1m-api.csv')
# BTC 1m history is thin/unnecessary for this build (per user direction) -- reuse the existing
# full-history 5m BTC klines as the cross-asset input. FeatureEngineer._merge_data() already
# does a backward merge_asof onto the ETH timeline internally, so passing a coarser-granularity
# BTC series just step-fills each 1m ETH row with the latest known 5m BTC bar (no look-ahead).
BTC_5M_CSV = os.path.join(BINANCE_DIR, 'klines', 'BTCUSDT', 'BTCUSDT-5m-api.csv')
METRICS_CSV = os.path.join(DATA_DIR, 'TOTAL_ETHUSDT_metrics.csv')
FEATURES_CSV = os.path.join(DATA_DIR, 'training_features_1m.csv')


def _load_zips_df(directory: str, ts_col_candidates: list[str], extra_drop=None) -> pd.DataFrame:
    dfs = []
    if not os.path.isdir(directory):
        return pd.DataFrame()
    for fname in sorted(f for f in os.listdir(directory) if f.endswith('.zip')):
        try:
            with zipfile.ZipFile(os.path.join(directory, fname)) as zf:
                raw = zf.read(zf.namelist()[0])
                df = pd.read_csv(io.BytesIO(raw))
                found_col = next((c for c in ts_col_candidates if c in df.columns), None)
                if found_col:
                    ct_num = pd.to_numeric(df[found_col], errors='coerce')
                    if ct_num.notna().mean() > 0.9:
                        df['timestamp'] = pd.to_datetime(ct_num, unit='ms', errors='coerce')
                    else:
                        df['timestamp'] = pd.to_datetime(df[found_col], errors='coerce')
                    df = df.dropna(subset=['timestamp'])
                    drop_cols = ([found_col] if found_col != 'timestamp' else []) + (extra_drop or [])
                    dfs.append(df.drop(columns=drop_cols, errors='ignore'))
        except Exception as e:
            print(f"  WARN zip read failed ({fname}): {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_klines(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


def main():
    print("Loading ETH 1m klines and BTC 5m klines (cross-asset input)...")
    eth_df = load_klines(ETH_1M_CSV)
    btc_df = load_klines(BTC_5M_CSV)[['timestamp', 'close', 'volume', 'quote_volume']]

    start_time = max(eth_df['timestamp'].min(), btc_df['timestamp'].min())
    end_time = min(eth_df['timestamp'].max(), btc_df['timestamp'].max())
    print(f"Common range: {start_time} -> {end_time}")
    eth_df = eth_df[(eth_df['timestamp'] >= start_time) & (eth_df['timestamp'] <= end_time)].copy()
    btc_df = btc_df[(btc_df['timestamp'] >= start_time) & (btc_df['timestamp'] <= end_time)].copy()
    print(f"  ETH: {len(eth_df):,} rows, BTC: {len(btc_df):,} rows")

    print("Loading metrics/funding (ETHUSDT, resolution-independent, reused from 5m pipeline)...")
    metrics_df = pd.DataFrame()
    if os.path.exists(METRICS_CSV):
        m = pd.read_csv(METRICS_CSV)
        m['timestamp'] = pd.to_datetime(m['create_time'], errors='coerce')
        metrics_df = m.dropna(subset=['timestamp']).drop(columns=['create_time', 'symbol'], errors='ignore')
    zip_met = _load_zips_df(METRICS_DIR, ts_col_candidates=['create_time'], extra_drop=['symbol'])
    metrics_df = pd.concat([metrics_df, zip_met], ignore_index=True) if not zip_met.empty else metrics_df
    metrics_df = metrics_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    metrics_df['timestamp'] = metrics_df['timestamp'].astype('datetime64[us]')

    bad_funding = [f for f in sorted(os.listdir(FUNDING_RATE_DIR)) if f.endswith('.zip') and 'ETHUSDT' not in f]
    if bad_funding:
        raise RuntimeError(f"Funding zip contract violation: expected ETHUSDT files only, got {bad_funding[:5]}")
    zip_fund = _load_zips_df(FUNDING_RATE_DIR, ts_col_candidates=['calcTime', 'calc_time'])
    if 'calcTime' in zip_fund.columns:
        zip_fund = zip_fund.rename(columns={'calcTime': 'timestamp'})
    if 'fundingRate' in zip_fund.columns:
        zip_fund = zip_fund.rename(columns={'fundingRate': 'last_funding_rate'})
    funding_df = zip_fund[['timestamp', 'last_funding_rate']].drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    funding_df['timestamp'] = funding_df['timestamp'].astype('datetime64[us]')
    print(f"  Metrics: {len(metrics_df):,} rows, Funding: {len(funding_df):,} rows")

    print("Merging metrics/funding onto ETH (backward asof, no look-ahead)...")
    overlap = [c for c in metrics_df.columns if c in eth_df.columns and c != 'timestamp']
    if overlap:
        eth_df = eth_df.drop(columns=overlap)
    eth_merged = pd.merge_asof(
        eth_df.sort_values('timestamp'), metrics_df.sort_values('timestamp'),
        on='timestamp', direction='backward', tolerance=pd.Timedelta('9h'),
    )
    eth_merged = pd.merge_asof(
        eth_merged.sort_values('timestamp'), funding_df.sort_values('timestamp'),
        on='timestamp', direction='backward',
    )
    print(f"  Merged: {len(eth_merged):,} rows")

    print("Building features (FeatureEngineer, candle_minutes=1)...")
    engineer = FeatureEngineer(candle_minutes=1, keep_only_active=True, include_entry_price=False)
    result = engineer.process(eth_merged, btc_df)
    print(f"  Features built: {len(result):,} rows, {len(result.columns)} cols")

    before = len(result)
    result = result.dropna()
    print(f"  Dropped {before - len(result):,} NaN rows -> {len(result):,} rows")

    result = prune_to_active_feature_keep(
        result, include_entry_price=False, extra_keep=["timestamp"],
    )
    result.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {FEATURES_CSV}: {len(result):,} rows, {len(result.columns)} cols, "
          f"{result['timestamp'].min()} -> {result['timestamp'].max()}")


if __name__ == '__main__':
    main()

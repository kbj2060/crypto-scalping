"""Causal rebuild of the 1m ETH feature set, fixing the P0 finding in
docs/model_contracts/eth_scalp_1m_20260717_audit_findings.md section 4: BTC 5m klines were
merged onto ETH 1m rows using the BTC bar's OPEN-time timestamp (Binance kline convention), but
`close`/`volume`/`quote_volume` for that bar aren't actually known until the bar CLOSES 5 minutes
later. `merge_asof(direction='backward')` therefore attached each BTC bar's close/volume to ETH
rows up to ~4 minutes before that data was really available -- a genuine semantic look-ahead
affecting the 15 BTC-derived features in ULTIMATE_FEATURE_COLS (btc_corr_60, btc_ret_*,
eth_btc_ret_spread_*, btc_lead_eth_follow_gap_3, etc.), present in every experiment built on
training_features_1m.csv this session. Independently corroborated by a separate DeepScalp-PnL v1
research line, which found the same leak (BTC-feature raw IC ~0.22-0.23, collapsed once removed).

Fix: shift the BTC dataframe's timestamp forward by 5 minutes (bar close) before the asof merge,
so a bar's close/volume only become joinable onto ETH rows from its actual availability time
onward. Everything else is unchanged from build_features_1m_20260716.py.

Output: data/training_features_1m_causal_btc.csv
"""
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from features.engineering import FeatureEngineer
from features.schema import prune_to_active_feature_keep
from build_features_1m_20260716 import (
    ETH_1M_CSV, BTC_5M_CSV, METRICS_CSV, METRICS_DIR, FUNDING_RATE_DIR,
    _load_zips_df, load_klines,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
OUT_CSV = os.path.join(DATA_DIR, 'training_features_1m_causal_btc.csv')

BTC_AVAILABILITY_LAG_MIN = 5  # BTC bar is 5min; its close/volume aren't known until it closes


def main():
    print("Loading ETH 1m klines and BTC 5m klines (cross-asset input)...")
    eth_df = load_klines(ETH_1M_CSV)
    btc_df = load_klines(BTC_5M_CSV)[['timestamp', 'close', 'volume', 'quote_volume']]

    print(f"Shifting BTC availability timestamp by +{BTC_AVAILABILITY_LAG_MIN}min "
          f"(bar-open -> bar-close, fixing the P0 semantic lookahead)...")
    btc_df = btc_df.copy()
    btc_df['timestamp'] = btc_df['timestamp'] + pd.Timedelta(minutes=BTC_AVAILABILITY_LAG_MIN)

    start_time = max(eth_df['timestamp'].min(), btc_df['timestamp'].min())
    end_time = min(eth_df['timestamp'].max(), btc_df['timestamp'].max())
    print(f"Common range (post-shift): {start_time} -> {end_time}")
    eth_df = eth_df[(eth_df['timestamp'] >= start_time) & (eth_df['timestamp'] <= end_time)].copy()
    btc_df = btc_df[(btc_df['timestamp'] >= start_time) & (btc_df['timestamp'] <= end_time)].copy()
    print(f"  ETH: {len(eth_df):,} rows, BTC: {len(btc_df):,} rows")

    print("Loading metrics/funding (ETHUSDT, resolution-independent)...")
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

    print("Building features (FeatureEngineer, candle_minutes=1, BTC availability-fixed)...")
    engineer = FeatureEngineer(candle_minutes=1, keep_only_active=True, include_entry_price=False)
    result = engineer.process(eth_merged, btc_df)
    print(f"  Features built: {len(result):,} rows, {len(result.columns)} cols")

    before = len(result)
    result = result.dropna()
    print(f"  Dropped {before - len(result):,} NaN rows -> {len(result):,} rows "
          f"(expect ~{BTC_AVAILABILITY_LAG_MIN} extra warmup-adjacent NaNs vs the uncorrected build)")

    result = prune_to_active_feature_keep(
        result, include_entry_price=False, extra_keep=["timestamp"],
    )
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}: {len(result):,} rows, {len(result.columns)} cols, "
          f"{result['timestamp'].min()} -> {result['timestamp'].max()}")


if __name__ == '__main__':
    main()

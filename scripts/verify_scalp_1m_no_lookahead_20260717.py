"""Empirical (not just code-review) no-lookahead test: recompute the full 1m feature pipeline
using data TRUNCATED to a cutoff timestamp deep in the OOS window, and compare the resulting
row's feature values against what's actually stored in data/training_features_1m.csv (which was
built from the FULL dataset). If the feature pipeline is genuinely causal, truncating everything
after the cutoff can't change the cutoff row's own feature values -- any column that differs is
proof that column used information from after the cutoff.

Mirrors build_features_1m_20260716.py's exact data loading / merge steps, just with all raw
inputs (ETH klines, BTC klines, metrics, funding) truncated to <= CUTOFF before running
FeatureEngineer.

Output: printed diff report + data/ensemble/reports/scalp_1m_lookahead_check_20260717.json
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
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
BTC_5M_CSV = os.path.join(BINANCE_DIR, 'klines', 'BTCUSDT', 'BTCUSDT-5m-api.csv')
METRICS_CSV = os.path.join(DATA_DIR, 'TOTAL_ETHUSDT_metrics.csv')
FULL_FEATURES_CSV = os.path.join(DATA_DIR, 'training_features_1m.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

# Two cutoffs deep in the OOS window this session's models were evaluated on -- if there's a leak
# tied to some periodic boundary (day/session/etc.) picking two different times of day reduces
# the chance of accidentally landing on a boundary that hides the bug.
CUTOFFS = ['2026-06-15 14:37:00', '2026-06-22 03:12:00']
CONTEXT_DAYS = 30  # how much trailing history to include before the cutoff (plenty for any rolling window used, max is 288 bars = ~4.8h)


def _load_zips_df(directory, ts_col_candidates, extra_drop=None):
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


def load_klines(path, upto_ts=None):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if upto_ts is not None:
        df = df[df['timestamp'] <= upto_ts]
    return df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


def build_features_truncated(cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    context_start = cutoff_ts - pd.Timedelta(days=CONTEXT_DAYS)

    eth_df = load_klines(ETH_1M_CSV, upto_ts=cutoff_ts)
    eth_df = eth_df[eth_df['timestamp'] >= context_start].reset_index(drop=True)
    btc_df = load_klines(BTC_5M_CSV, upto_ts=cutoff_ts)[['timestamp', 'close', 'volume', 'quote_volume']]
    btc_df = btc_df[btc_df['timestamp'] >= context_start].reset_index(drop=True)

    metrics_df = pd.DataFrame()
    if os.path.exists(METRICS_CSV):
        m = pd.read_csv(METRICS_CSV)
        m['timestamp'] = pd.to_datetime(m['create_time'], errors='coerce')
        metrics_df = m.dropna(subset=['timestamp']).drop(columns=['create_time', 'symbol'], errors='ignore')
    zip_met = _load_zips_df(METRICS_DIR, ts_col_candidates=['create_time'], extra_drop=['symbol'])
    metrics_df = pd.concat([metrics_df, zip_met], ignore_index=True) if not zip_met.empty else metrics_df
    metrics_df = metrics_df[metrics_df['timestamp'] <= cutoff_ts]
    metrics_df = metrics_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    metrics_df['timestamp'] = metrics_df['timestamp'].astype('datetime64[us]')

    zip_fund = _load_zips_df(FUNDING_RATE_DIR, ts_col_candidates=['calcTime', 'calc_time'])
    if 'calcTime' in zip_fund.columns:
        zip_fund = zip_fund.rename(columns={'calcTime': 'timestamp'})
    if 'fundingRate' in zip_fund.columns:
        zip_fund = zip_fund.rename(columns={'fundingRate': 'last_funding_rate'})
    funding_df = zip_fund[['timestamp', 'last_funding_rate']]
    funding_df = funding_df[funding_df['timestamp'] <= cutoff_ts]
    funding_df = funding_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    funding_df['timestamp'] = funding_df['timestamp'].astype('datetime64[us]')

    overlap = [c for c in metrics_df.columns if c in eth_df.columns and c != 'timestamp']
    if overlap:
        eth_df = eth_df.drop(columns=overlap)
    eth_merged = pd.merge_asof(eth_df.sort_values('timestamp'), metrics_df.sort_values('timestamp'),
                                on='timestamp', direction='backward', tolerance=pd.Timedelta('9h'))
    eth_merged = pd.merge_asof(eth_merged.sort_values('timestamp'), funding_df.sort_values('timestamp'),
                                on='timestamp', direction='backward')

    engineer = FeatureEngineer(candle_minutes=1, keep_only_active=True, include_entry_price=False)
    result = engineer.process(eth_merged, btc_df)
    result = prune_to_active_feature_keep(result, include_entry_price=False, include_m7_artifacts=True,
                                           extra_keep=["timestamp"])
    return result


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading the FULL stored feature file for comparison...")
    full_df = pd.read_csv(FULL_FEATURES_CSV, parse_dates=['timestamp'])

    all_reports = []
    for cutoff_str in CUTOFFS:
        cutoff_ts = pd.Timestamp(cutoff_str)
        print(f"\n{'=' * 70}\nCutoff: {cutoff_ts}\n{'=' * 70}")

        print("Rebuilding features from truncated raw data...")
        truncated = build_features_truncated(cutoff_ts)
        trunc_row = truncated[truncated['timestamp'] == cutoff_ts]
        full_row = full_df[full_df['timestamp'] == cutoff_ts]

        if trunc_row.empty:
            print(f"  !! cutoff row not found in truncated rebuild (dropped by NaN/warmup) -- inconclusive")
            all_reports.append({'cutoff': cutoff_str, 'status': 'inconclusive_row_missing_truncated'})
            continue
        if full_row.empty:
            print(f"  !! cutoff row not found in full stored file -- inconclusive")
            all_reports.append({'cutoff': cutoff_str, 'status': 'inconclusive_row_missing_full'})
            continue

        trunc_row = trunc_row.iloc[0]
        full_row = full_row.iloc[0]
        common_cols = [c for c in trunc_row.index if c in full_row.index and c != 'timestamp']

        mismatches = []
        for col in common_cols:
            a, b = trunc_row[col], full_row[col]
            try:
                af, bf = float(a), float(b)
                if not np.isclose(af, bf, rtol=1e-5, atol=1e-8, equal_nan=True):
                    mismatches.append({'column': col, 'truncated_value': af, 'full_value': bf,
                                        'abs_diff': abs(af - bf)})
            except (ValueError, TypeError):
                if a != b:
                    mismatches.append({'column': col, 'truncated_value': str(a), 'full_value': str(b)})

        print(f"  Compared {len(common_cols)} columns. Mismatches: {len(mismatches)}")
        for m in mismatches:
            print(f"    LOOKAHEAD SUSPECT: {m}")

        all_reports.append({
            'cutoff': cutoff_str, 'status': 'compared',
            'n_columns_compared': len(common_cols), 'n_mismatches': len(mismatches),
            'mismatches': mismatches,
        })

    verdict = 'PASS' if all(r.get('n_mismatches', 1) == 0 for r in all_reports if r.get('status') == 'compared') \
        and any(r.get('status') == 'compared' for r in all_reports) else 'NEEDS_REVIEW'
    print(f"\n{'=' * 70}\nOVERALL VERDICT: {verdict}\n{'=' * 70}")

    with open(os.path.join(REPORT_DIR, 'scalp_1m_lookahead_check_20260717.json'), 'w') as f:
        json.dump({'verdict': verdict, 'cutoff_reports': all_reports}, f, indent=2, default=str)
    print("Saved scalp_1m_lookahead_check_20260717.json")


if __name__ == '__main__':
    main()

"""Shared evaluation harness for the two alternative PRIMARY direction labels (levers B-3 DP
trajectory, B-4 trend-scanning), vs. the base triple-barrier label's HGB baseline
(scalp_1m_tune_maker_realistic_20260716.json, OOS +3.74%).

Both alternative labels only supply a LONG/SHORT/CASH direction target -- the classifier trains
to predict THAT target instead of the base triple-barrier's scalp_action, but the actual trade
P&L is still evaluated with the base label's ATR-scaled TP/SL + the same realistic maker-fill
simulation used everywhere else in this line of experiments, so the comparison isolates "which
target trains a better direction classifier" from "how a taken trade is priced," matching the
approach already used for the trend-scan/DP scripts' own docstrings.

Usage: python train_eval_scalp_1m_altlabel_20260716.py --source {dp,trendscan}

Output: data/ensemble/reports/scalp_1m_altlabel_{source}_20260716.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for, HGB_PARAMS
from simulate_maker_entry_scalp_1m_20260716 import (
    simulate_maker_fills, backtest_maker, predict_with_threshold, LABELS_CSV,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
FIXED_THRESHOLD = 0.55

SOURCES = {
    'dp': {
        'csv': os.path.join(DATA_DIR, 'training_features_1m_dp_labels.csv'),
        'action_col': 'dp_action',
        'valid_col': 'dp_has_full_horizon',
    },
    'trendscan': {
        'csv': os.path.join(DATA_DIR, 'training_features_1m_trendscan_labels.csv'),
        'action_col': 'trendscan_action',
        'valid_col': None,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, choices=list(SOURCES.keys()))
    args = ap.parse_args()
    cfg = SOURCES[args.source]

    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Loading data + base labels (for TP/SL sim) + {args.source} alt-labels (for classifier target)...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    base_labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(base_labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)

    alt = pd.read_csv(cfg['csv'], parse_dates=['timestamp'])
    keep_cols = ['timestamp', cfg['action_col']] + ([cfg['valid_col']] if cfg['valid_col'] else [])
    df = df.merge(alt[keep_cols], on='timestamp', how='inner')
    if cfg['valid_col']:
        df = df[df[cfg['valid_col']]].reset_index(drop=True)
    print(f"  {len(df):,} rows")
    dist = df[cfg['action_col']].value_counts()
    print(f"  Alt-label distribution: {dist.to_dict()}")

    print("Simulating maker-entry fills (long + short, from base label TP/SL)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    train = df[df['timestamp'] <= TRAIN_END]
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    print(f"Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")

    X_train = train[feat_cols].fillna(0.0)
    y_train = train[cfg['action_col']]
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(X_train, y_train)

    result = {'source': args.source, 'action_col': cfg['action_col'], 'fixed_threshold': FIXED_THRESHOLD}
    for name, split_df in [('val', val), ('oos', oos)]:
        X = split_df[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X, FIXED_THRESHOLD)
        idx = split_df.index
        long_sim_s, short_sim_s = long_sim.loc[idx].reset_index(drop=True), short_sim.loc[idx].reset_index(drop=True)
        bt = backtest_maker(pred, long_sim_s, short_sim_s)
        print(f"  [{name}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        result[name] = bt

    result['baseline_for_comparison'] = {
        'report': 'scalp_1m_tune_maker_realistic_20260716.json',
        'label': 'base triple-barrier (scalp_action)',
        'oos_total_pnl_pct': 3.7390646402123644,
    }
    result['compliance'] = {
        'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
        'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
    }
    result['note'] = (f'Classifier trained to predict {cfg["action_col"]} instead of the base '
                       f'triple-barrier scalp_action; trade P&L still evaluated via the base '
                       f'label\'s ATR-scaled TP/SL + realistic maker-fill simulation, unchanged.')
    out_path = os.path.join(REPORT_DIR, f'scalp_1m_altlabel_{args.source}_20260716.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved {os.path.basename(out_path)}")


if __name__ == '__main__':
    main()

"""The decisive test: with the BTC availability-timestamp lookahead fixed
(training_features_1m_causal_btc.csv), does the baseline model's edge survive? Same architecture,
same base triple-barrier label, same Experiment A split, same threshold=0.55, same maker-fill
simulator as the original (still-imperfect, per the audit's other P1 findings) baseline -- only
the BTC feature causality changes, isolating exactly this one variable.

Output: data/ensemble/reports/scalp_1m_causal_btc_retest_20260717.json
"""
import json
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import feature_cols_for, HGB_PARAMS
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, predict_with_threshold, LABELS_CSV

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')
CAUSAL_BTC_CSV = os.path.join(DATA_DIR, 'training_features_1m_causal_btc.csv')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
THRESHOLD = 0.55

BTC_DERIVED_COLS = [
    'btc_corr_60', 'eth_btc_ratio_change', 'btc_ret_1', 'btc_ret_3', 'btc_ret_6', 'btc_ret_12',
    'btc_ret_z_48', 'eth_btc_ret_spread_12', 'eth_btc_ret_spread_48', 'eth_btc_beta_residual_z',
    'btc_lead_eth_follow_gap_3', 'btc_breakout_eth_lag_dir', 'btc_volume_impulse_z',
    'btc_eth_volume_rank_spread', 'btc_impulse_x_eth_beta',
]


def run_one(df, feat_cols, name):
    train = df[df['timestamp'] <= TRAIN_END]
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    print(f"\n--- {name}: {len(feat_cols)} features ---")
    print(f"  Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")

    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    clf = fit_model(train, feat_cols)
    result = {'name': name, 'n_features': len(feat_cols)}
    for split_name, split_df in [('val', val), ('oos', oos)]:
        X = split_df[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X, THRESHOLD)
        idx = split_df.index
        long_sim_s, short_sim_s = long_sim.loc[idx].reset_index(drop=True), short_sim.loc[idx].reset_index(drop=True)
        bt = backtest_maker(pred, long_sim_s, short_sim_s)
        print(f"  [{split_name}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        result[split_name] = bt
    return result


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading BTC-availability-FIXED features + existing (ETH-OHLC-only) labels...")
    df = pd.read_csv(CAUSAL_BTC_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    all_feat_cols = feature_cols_for(df, [])
    fixed_btc_feat_cols = all_feat_cols  # BTC cols are already availability-fixed in this CSV
    no_btc_feat_cols = [c for c in all_feat_cols if c not in BTC_DERIVED_COLS]

    results = []
    results.append(run_one(df, fixed_btc_feat_cols, 'A_causal_btc_fixed_with_btc_features'))
    results.append(run_one(df, no_btc_feat_cols, 'B_no_btc_features_at_all'))

    out = {
        'baseline_for_comparison': {
            'report': 'scalp_1m_tune_maker_realistic_20260716.json (original, BTC-lookahead-contaminated)',
            'oos_total_pnl_pct': 3.7390646402123644, 'oos_hit_rate': 0.7559133126934985,
        },
        'results': results,
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_causal_btc_retest_20260717.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved scalp_1m_causal_btc_retest_20260717.json")


if __name__ == '__main__':
    main()

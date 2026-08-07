"""New direction after the BTC-lookahead finding: BTC features are gone, but the confidence
threshold (0.55) was picked on the OLD (BTC-contaminated) model's confidence distribution. A
pure-ETH-only model may have a systematically different confidence distribution -- re-sweep the
threshold from scratch on val, using the causal (already availability-fixed) feature build so this
stays clean, and confirm the val-selected threshold on OOS with the same realistic maker-fill sim
used throughout this line.

Output: data/ensemble/reports/scalp_1m_pure_eth_threshold_sweep_20260717.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import feature_cols_for
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, predict_with_threshold, LABELS_CSV
from retest_scalp_1m_causal_btc_20260717 import CAUSAL_BTC_CSV, BTC_DERIVED_COLS, TRAIN_END, VAL_END, OOS_END

REPORT_DIR = os.path.join(_ROOT_DIR, 'data', 'ensemble', 'reports')

THRESHOLDS = [0.34, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading BTC-availability-fixed features, dropping all BTC-derived columns (pure ETH)...")
    df = pd.read_csv(CAUSAL_BTC_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)

    all_feat_cols = feature_cols_for(df, [])
    feat_cols = [c for c in all_feat_cols if c not in BTC_DERIVED_COLS]
    print(f"  {len(df):,} rows, {len(feat_cols)} pure-ETH features")

    train = df[df['timestamp'] <= TRAIN_END]
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    print(f"  Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")

    print("Simulating maker-entry fills (long + short, vectorized)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    clf = fit_model(train, feat_cols)

    val_idx, oos_idx = val.index, oos.index
    long_sim_val, short_sim_val = long_sim.loc[val_idx].reset_index(drop=True), short_sim.loc[val_idx].reset_index(drop=True)
    long_sim_oos, short_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True), short_sim.loc[oos_idx].reset_index(drop=True)

    X_val = val[feat_cols].fillna(0.0)
    print("\nSweeping confidence threshold on val (pure-ETH model, realistic maker-fill PnL)...")
    sweep = []
    for thr in THRESHOLDS:
        pred_val = predict_with_threshold(clf, X_val, thr)
        bt = backtest_maker(pred_val, long_sim_val, short_sim_val)
        sweep.append({'threshold': thr, **bt})
        print(f"  [val thr={thr}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

    viable = [s for s in sweep if s['n_filled'] and s['n_filled'] >= 20]
    best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else None

    result = {
        'experiment': 'pure_eth_threshold_sweep_causal',
        'n_features': len(feat_cols),
        'val_sweep': sweep,
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
    }

    if best is None:
        print("\nNo threshold produced >=20 filled val trades -- pure-ETH model has no viable "
              "confident-signal regime at any threshold tested.")
        result['best_val_threshold'] = None
        result['oos_at_chosen_threshold'] = None
    else:
        best_thr = best['threshold']
        print(f"\n-> best val threshold: {best_thr} (total_pnl_pct={best['total_pnl_pct']})")
        X_oos = oos[feat_cols].fillna(0.0)
        pred_oos = predict_with_threshold(clf, X_oos, best_thr)
        bt_oos = backtest_maker(pred_oos, long_sim_oos, short_sim_oos)
        print(f"[oos @ thr={best_thr}] signals={bt_oos['n_signals']:,} filled={bt_oos['n_filled']:,} "
              f"fill_rate={bt_oos['fill_rate']} hit_rate={bt_oos['hit_rate']} total_pnl_pct={bt_oos['total_pnl_pct']}")
        result['best_val_threshold'] = best_thr
        result['oos_at_chosen_threshold'] = bt_oos

    with open(os.path.join(REPORT_DIR, 'scalp_1m_pure_eth_threshold_sweep_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_pure_eth_threshold_sweep_20260717.json")


if __name__ == '__main__':
    main()

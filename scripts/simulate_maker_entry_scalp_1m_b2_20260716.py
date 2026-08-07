"""Same confidence-threshold + realistic maker-fill combination as
simulate_maker_entry_scalp_1m_20260716.py, applied to Experiment B2 (price + microstructure_1m
features, bounded to the 2026-05-03->07-12 overlap window) instead of Experiment A. Answers: does
the winning combo (which flipped A from -14.7% to +3.7% OOS) also work on the microstructure
model, or was B2's earlier lag behind A (see tune_scalp_1m_levers_20260716.py's CONF/MAKER
results, where B2 stayed marginally negative under both individual levers) too structural to fix?

Output: data/ensemble/reports/scalp_1m_tune_maker_realistic_b2_20260716.json
"""
import json
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import MICRO_CSV, MICROSTRUCTURE_COLS, feature_cols_for, split_by_date
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import (
    simulate_maker_fills, backtest_maker, predict_with_threshold, LABELS_CSV, CONF_THRESHOLDS,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + labels (Experiment B2: microstructure window, price+microstructure)...")
    df = pd.read_csv(MICRO_CSV, parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= '2026-05-03') & (df['timestamp'] <= '2026-07-12')].reset_index(drop=True)
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    print("Simulating maker-entry fills (long + short, vectorized)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, MICROSTRUCTURE_COLS)
    train, val, oos = split_by_date(df, '2026-06-20', '2026-06-30', '2026-07-12')
    print(f"Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")
    clf = fit_model(train, feat_cols)

    val_idx, oos_idx = val.index, oos.index
    long_sim_val, short_sim_val = long_sim.loc[val_idx].reset_index(drop=True), short_sim.loc[val_idx].reset_index(drop=True)
    long_sim_oos, short_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True), short_sim.loc[oos_idx].reset_index(drop=True)

    X_val = val[feat_cols].fillna(0.0)
    print("\nSweeping confidence threshold on val (realistic maker-fill PnL)...")
    sweep = []
    for thr in CONF_THRESHOLDS:
        pred_val = predict_with_threshold(clf, X_val, thr)
        bt = backtest_maker(pred_val, long_sim_val, short_sim_val)
        sweep.append({'threshold': thr, **bt})
        print(f"  [val thr={thr}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

    viable = [s for s in sweep if s['n_filled'] and s['n_filled'] >= 20]
    best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else sweep[0]
    best_thr = best['threshold']
    print(f"-> best val threshold: {best_thr}")

    X_oos = oos[feat_cols].fillna(0.0)
    pred_oos = predict_with_threshold(clf, X_oos, best_thr)
    bt_oos = backtest_maker(pred_oos, long_sim_oos, short_sim_oos)
    print(f"[oos @ thr={best_thr}] signals={bt_oos['n_signals']:,} filled={bt_oos['n_filled']:,} "
          f"fill_rate={bt_oos['fill_rate']} hit_rate={bt_oos['hit_rate']} total_pnl_pct={bt_oos['total_pnl_pct']}")

    result = {
        'experiment': 'B2_conf_plus_realistic_maker_fill',
        'val_sweep': sweep,
        'chosen_threshold': best_thr,
        'oos_at_chosen_threshold': bt_oos,
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Same realistic maker-fill model as the A-experiment version, applied to the '
                 'microstructure-window price+microstructure feature set. ~70-day-bounded window, '
                 '~12-day OOS -- thin sample, directional signal only.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_tune_maker_realistic_b2_20260716.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved scalp_1m_tune_maker_realistic_b2_20260716.json")


if __name__ == '__main__':
    main()

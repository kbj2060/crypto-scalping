"""Trains and evaluates the short-horizon (5min, matching the DP oracle's actual median hold
time) triple-barrier label, using the same realistic maker-fill methodology as the established
20min baseline (scalp_1m_tune_maker_realistic_20260716.json: OOS +3.74%, 8,075 filled trades,
192/day) -- only HORIZON/FILL_LOOKAHEAD/TP-SL scale change to match the shorter time budget:
FILL_LOOKAHEAD shrinks from 3min to 1min (waiting 3 of 5 total minutes for a limit fill would eat
60% of the trade's time budget), everything else (maker/taker fee split, threshold-sweep
methodology, val/OOS split) stays identical for a fair comparison.

Output: data/ensemble/reports/scalp_1m_short_horizon_20260717.json
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

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import backtest_maker, MAKER_FEE, TAKER_FEE

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_SHORT_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels_short.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

HORIZON = 5
OFFSET = 0.0001
FILL_LOOKAHEAD = 1  # shrunk from the 20min label's 3min -- don't burn most of a 5min trade waiting to fill
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
CONF_THRESHOLDS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _first_touch_from(high, low, tp_level, sl_level, direction, k_start_per_row, k_max, active):
    n = len(high)
    tp_hit_at = np.full(n, np.nan)
    sl_hit_at = np.full(n, np.nan)
    for k in range(1, k_max + 1):
        row_active = active & (k >= k_start_per_row)
        if not row_active.any():
            continue
        high_k = np.concatenate([high[k:], np.full(k, np.nan)])
        low_k = np.concatenate([low[k:], np.full(k, np.nan)])
        if direction == 'long':
            tp_cond = row_active & (high_k >= tp_level) & np.isnan(tp_hit_at)
            sl_cond = row_active & (low_k <= sl_level) & np.isnan(sl_hit_at)
        else:
            tp_cond = row_active & (low_k <= tp_level) & np.isnan(tp_hit_at)
            sl_cond = row_active & (high_k >= sl_level) & np.isnan(sl_hit_at)
        tp_hit_at[tp_cond] = k
        sl_hit_at[sl_cond] = k
    return tp_hit_at, sl_hit_at


def simulate_maker_fills_short(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    n = len(df)
    open_ = df['open'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    tp_move = df['scalp_tp_move'].to_numpy()
    sl_move = df['scalp_sl_move'].to_numpy()

    entry_open = np.concatenate([open_[1:], [np.nan]])
    if direction == 'long':
        limit_price = entry_open * (1 - OFFSET)
    else:
        limit_price = entry_open * (1 + OFFSET)

    fill_offset = np.full(n, np.nan)
    for f in range(1, FILL_LOOKAHEAD + 1):
        high_f = np.concatenate([high[f:], np.full(f, np.nan)])
        low_f = np.concatenate([low[f:], np.full(f, np.nan)])
        if direction == 'long':
            cond = (low_f <= limit_price) & np.isnan(fill_offset)
        else:
            cond = (high_f >= limit_price) & np.isnan(fill_offset)
        fill_offset[cond] = f

    filled = ~np.isnan(fill_offset)
    entry_price = np.where(filled, limit_price, np.nan)
    if direction == 'long':
        tp_level = entry_price * (1 + tp_move)
        sl_level = entry_price * (1 - sl_move)
    else:
        tp_level = entry_price * (1 - tp_move)
        sl_level = entry_price * (1 + sl_move)

    k_start = np.where(filled, fill_offset, HORIZON + 1).astype(float)
    tp_hit_at, sl_hit_at = _first_touch_from(high, low, tp_level, sl_level, direction, k_start, HORIZON, filled)

    outcome_move = np.full(n, np.nan)
    both = ~np.isnan(tp_hit_at) & ~np.isnan(sl_hit_at)
    tp_only = ~np.isnan(tp_hit_at) & np.isnan(sl_hit_at)
    sl_only = np.isnan(tp_hit_at) & ~np.isnan(sl_hit_at)
    tp_first = both & (tp_hit_at < sl_hit_at)
    sl_first_or_tie = both & (tp_hit_at >= sl_hit_at)
    outcome_move[tp_only | tp_first] = tp_move[tp_only | tp_first]
    outcome_move[sl_only | sl_first_or_tie] = -sl_move[sl_only | sl_first_or_tie]
    filled_no_touch = filled & np.isnan(tp_hit_at) & np.isnan(sl_hit_at)
    outcome_move[filled_no_touch] = 0.0

    return pd.DataFrame({'filled': filled, 'fill_offset': fill_offset, 'realized_move': outcome_move}, index=df.index)


def predict_with_threshold(clf, X, threshold):
    proba = clf.predict_proba(X)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred = classes[max_idx].copy()
    return np.where(max_proba >= threshold, pred, 'CASH')


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + short-horizon labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_SHORT_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    print("Simulating maker-entry fills (short-horizon fill model)...")
    long_sim = simulate_maker_fills_short(df, 'long')
    short_sim = simulate_maker_fills_short(df, 'short')

    feat_cols = feature_cols_for(df, [])
    train = df[df['timestamp'] <= TRAIN_END]
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    n_days = (oos['timestamp'].max() - oos['timestamp'].min()).total_seconds() / 86400
    print(f"Train={len(train):,} Val={len(val):,} OOS={len(oos):,} ({n_days:.1f} days)")

    clf = fit_model(train, feat_cols)

    X_val = val[feat_cols].fillna(0.0)
    val_idx = val.index
    long_sim_val, short_sim_val = long_sim.loc[val_idx].reset_index(drop=True), short_sim.loc[val_idx].reset_index(drop=True)
    print("\nSweeping confidence threshold on val...")
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
    oos_idx = oos.index
    long_sim_oos, short_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True), short_sim.loc[oos_idx].reset_index(drop=True)
    pred_oos = predict_with_threshold(clf, X_oos, best_thr)
    bt_oos = backtest_maker(pred_oos, long_sim_oos, short_sim_oos)
    trades_per_day = (bt_oos['n_filled'] or 0) / n_days
    print(f"\n[oos @ thr={best_thr}] signals={bt_oos['n_signals']:,} filled={bt_oos['n_filled']:,} "
          f"({trades_per_day:.1f}/day) fill_rate={bt_oos['fill_rate']} hit_rate={bt_oos['hit_rate']} "
          f"total_pnl_pct={bt_oos['total_pnl_pct']}")

    result = {
        'label': 'short_horizon_5min',
        'horizon_bars': HORIZON, 'fill_lookahead_bars': FILL_LOOKAHEAD,
        'val_sweep': sweep,
        'chosen_threshold': best_thr,
        'oos_at_chosen_threshold': {**bt_oos, 'trades_per_day': trades_per_day},
        'baseline_for_comparison': {
            'report': 'scalp_1m_tune_maker_realistic_20260716.json',
            'horizon_bars': 20, 'oos_total_pnl_pct': 3.7390646402123644,
            'oos_trades_per_day': 8075 / 42.0,
        },
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_short_horizon_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_short_horizon_20260717.json")


if __name__ == '__main__':
    main()

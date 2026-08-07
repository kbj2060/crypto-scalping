"""Evaluate the new trailing-stop label (build_scalp_1m_trailing_labels_20260717.py) on pure-ETH
features (BTC-derived columns excluded, causal/availability-fixed feature build). Same HGB
architecture as the rest of this line; PnL = realized trailing-stop move for the predicted
direction minus round-trip taker fee (entry via market order -- no maker-fill sim yet, this is a
first-pass viability check, not a final promotion-grade backtest).

Output: data/ensemble/reports/scalp_1m_trailing_pure_eth_20260717.json
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
from simulate_maker_entry_scalp_1m_20260716 import predict_with_threshold
from retest_scalp_1m_causal_btc_20260717 import CAUSAL_BTC_CSV, BTC_DERIVED_COLS, TRAIN_END, VAL_END, OOS_END

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_trailing_labels.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

ROUND_TRIP_TAKER_FEE = 0.0009
THRESHOLDS = [0.34, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70]


def backtest(pred_action: np.ndarray, long_move: np.ndarray, short_move: np.ndarray) -> dict:
    is_long = pred_action == 'LONG'
    is_short = pred_action == 'SHORT'
    n_trades = int(is_long.sum() + is_short.sum())
    if n_trades == 0:
        return {'n_trades': 0, 'hit_rate': None, 'avg_pnl_pct': None, 'total_pnl_pct': None}
    move = np.full(len(pred_action), np.nan)
    move[is_long] = long_move[is_long]
    move[is_short] = short_move[is_short]
    realized = move[is_long | is_short]
    net = realized - ROUND_TRIP_TAKER_FEE
    return {
        'n_trades': n_trades,
        'hit_rate': float((net > 0).mean()),
        'avg_pnl_pct': float(net.mean()),
        'total_pnl_pct': float(net.sum()),
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading BTC-availability-fixed features (pure ETH, BTC cols dropped) + trailing labels...")
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
    print(f"  Train label dist:\n{train['scalp_action'].value_counts()}")

    clf = fit_model(train, feat_cols)

    X_val = val[feat_cols].fillna(0.0)
    val_long_move = val['scalp_long_move'].to_numpy()
    val_short_move = val['scalp_short_move'].to_numpy()

    print("\nSweeping confidence threshold on val...")
    sweep = []
    for thr in THRESHOLDS:
        pred_val = predict_with_threshold(clf, X_val, thr)
        bt = backtest(pred_val, val_long_move, val_short_move)
        sweep.append({'threshold': thr, **bt})
        print(f"  [val thr={thr}] trades={bt['n_trades']:,} hit_rate={bt['hit_rate']} "
              f"avg_pnl_pct={bt['avg_pnl_pct']} total_pnl_pct={bt['total_pnl_pct']}")

    viable = [s for s in sweep if s['n_trades'] and s['n_trades'] >= 20]
    best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else None

    result = {
        'experiment': 'trailing_stop_label_pure_eth',
        'label_config': {'K_MAX': 60, 'SL_ATR_MULT': 1.0, 'TRAIL_ATR_MULT': 0.6,
                          'MIN_PROFITABLE_MOVE': 0.0015},
        'n_features': len(feat_cols),
        'val_sweep': sweep,
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': 'Market-order entry (no maker-fill sim yet) -- first-pass viability check only.',
    }

    if best is None:
        print("\nNo threshold produced >=20 val trades.")
        result['best_val_threshold'] = None
    else:
        best_thr = best['threshold']
        print(f"\n-> best val threshold: {best_thr} (total_pnl_pct={best['total_pnl_pct']})")
        X_oos = oos[feat_cols].fillna(0.0)
        oos_long_move = oos['scalp_long_move'].to_numpy()
        oos_short_move = oos['scalp_short_move'].to_numpy()
        pred_oos = predict_with_threshold(clf, X_oos, best_thr)
        bt_oos = backtest(pred_oos, oos_long_move, oos_short_move)
        print(f"[oos @ thr={best_thr}] trades={bt_oos['n_trades']:,} hit_rate={bt_oos['hit_rate']} "
              f"avg_pnl_pct={bt_oos['avg_pnl_pct']} total_pnl_pct={bt_oos['total_pnl_pct']}")
        result['best_val_threshold'] = best_thr
        result['oos_at_chosen_threshold'] = bt_oos

    with open(os.path.join(REPORT_DIR, 'scalp_1m_trailing_pure_eth_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_trailing_pure_eth_20260717.json")


if __name__ == '__main__':
    main()

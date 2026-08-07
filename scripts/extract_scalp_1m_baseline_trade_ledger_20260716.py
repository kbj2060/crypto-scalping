"""Reconstructs the per-trade ledger for the established baseline
(scalp_1m_tune_maker_realistic_20260716.json: Experiment A, fixed confidence threshold=0.55,
realistic maker-fill simulation) on the OOS window (2026-06-01 -> 2026-07-12), since the
aggregate backtest_maker() function only returns summary stats, not per-trade records. Same
model/threshold/fill-sim code path, just also capturing per-row entry timestamp, side, filled,
and net PnL for charting/inspection.

Output: data/ensemble/reports/scalp_1m_baseline_trade_ledger_oos_20260716.csv
"""
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
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, LABELS_CSV, ROUND_TRIP_FEE

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')
TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
THRESHOLD = 0.55


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)

    print("Simulating maker-entry fills...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    train = df[df['timestamp'] <= TRAIN_END]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    print(f"Train={len(train):,} OOS={len(oos):,}")

    clf = fit_model(train, feat_cols)
    X_oos = oos[feat_cols].fillna(0.0)
    proba = clf.predict_proba(X_oos)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred = classes[max_idx]
    pred = np.where(max_proba >= THRESHOLD, pred, 'CASH')

    idx = oos.index
    long_sim_oos = long_sim.loc[idx].reset_index(drop=True)
    short_sim_oos = short_sim.loc[idx].reset_index(drop=True)

    oos_r = oos.reset_index(drop=True)
    is_long = pred == 'LONG'
    is_short = pred == 'SHORT'
    is_signal = is_long | is_short

    ledger = oos_r.loc[is_signal, ['timestamp', 'close']].copy()
    ledger['side'] = pred[is_signal]
    ledger['filled'] = np.where(is_long[is_signal], long_sim_oos.loc[is_signal, 'filled'],
                                 short_sim_oos.loc[is_signal, 'filled'])
    ledger['realized_move'] = np.where(is_long[is_signal], long_sim_oos.loc[is_signal, 'realized_move'],
                                        short_sim_oos.loc[is_signal, 'realized_move'])
    ledger['net_pnl_pct'] = np.where(ledger['filled'], ledger['realized_move'] - ROUND_TRIP_FEE, np.nan)
    ledger = ledger.sort_values('timestamp').reset_index(drop=True)
    ledger['cum_pnl_pct'] = ledger['net_pnl_pct'].fillna(0.0).cumsum()

    out_path = os.path.join(REPORT_DIR, 'scalp_1m_baseline_trade_ledger_oos_20260716.csv')
    ledger.to_csv(out_path, index=False)

    n_signals = len(ledger)
    n_filled = int(ledger['filled'].sum())
    n_days = (oos['timestamp'].max() - oos['timestamp'].min()).total_seconds() / 86400
    print(f"\nSaved {out_path}")
    print(f"  n_signals={n_signals:,} n_filled={n_filled:,} over {n_days:.1f} days")
    print(f"  signals/day={n_signals / n_days:.1f}  filled_trades/day={n_filled / n_days:.1f}")
    print(f"  final cum PnL={ledger['cum_pnl_pct'].iloc[-1]:.3f}%")

    # median gap between filled trades
    filled_ts = ledger.loc[ledger['filled'], 'timestamp']
    gaps_min = filled_ts.diff().dt.total_seconds().dropna() / 60
    print(f"  median gap between filled trades: {gaps_min.median():.1f} min, "
          f"mean: {gaps_min.mean():.1f} min")


if __name__ == '__main__':
    main()

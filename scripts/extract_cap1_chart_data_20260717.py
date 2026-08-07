"""Extracts full-resolution equity curve + trade sequence for cap=1 (single position, 100%
notional) on the OOS window, for charting -- shows exactly why the compounding math looks
explosive: full-resolution equity curve (every settlement, not downsampled), per-trade records,
and daily trade counts.

Output: data/ensemble/reports/scalp_1m_cap1_chart_data_20260717.json
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
from simulate_maker_entry_scalp_1m_20260716 import LABELS_CSV, ROUND_TRIP_FEE
from simulate_portfolio_capped_scalp_1m_20260717 import simulate_maker_fills_with_exit, predict_with_threshold

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
THRESHOLD = 0.55
CAP = 1


def main():
    print("Loading data + labels, training baseline primary model...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)

    print("Simulating maker-entry fills with exit timing...")
    long_sim = simulate_maker_fills_with_exit(df, 'long')
    short_sim = simulate_maker_fills_with_exit(df, 'short')

    feat_cols = feature_cols_for(df, [])
    train = df[df['timestamp'] <= TRAIN_END]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)].reset_index(drop=False).rename(columns={'index': 'orig_idx'})
    print(f"Train={len(train):,} OOS={len(oos):,}")

    clf = fit_model(train, feat_cols)
    X_oos = oos[feat_cols].fillna(0.0)
    pred = predict_with_threshold(clf, X_oos, THRESHOLD)

    oos_idx = oos['orig_idx'].to_numpy()
    long_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True)
    short_sim_oos = short_sim.loc[oos_idx].reset_index(drop=True)

    # single-position (cap=1) event-driven compounding sim, keeping every trade record
    ts = oos['timestamp'].to_numpy()
    is_long = pred == 'LONG'
    is_short = pred == 'SHORT'
    signal_idx = np.flatnonzero(is_long | is_short)

    equity = 1.0
    open_exit_time = None
    trades = []  # dict per accepted trade
    equity_points = [{'t': str(oos['timestamp'].iloc[0]), 'equity': 1.0}]

    for i in signal_idx:
        entry_t = ts[i]
        if open_exit_time is not None and entry_t < open_exit_time:
            continue  # slot busy, reject
        sim = long_sim_oos if is_long[i] else short_sim_oos
        if not bool(sim.iloc[i]['filled']):
            continue
        exit_offset_min = float(sim.iloc[i]['exit_offset'])
        exit_t = entry_t + np.timedelta64(int(exit_offset_min), 'm')
        realized_move = float(sim.iloc[i]['realized_move'])
        net = realized_move - ROUND_TRIP_FEE
        notional = equity  # cap=1: full equity every trade
        equity = equity + notional * net
        open_exit_time = exit_t
        trades.append({
            't_entry': str(pd.Timestamp(entry_t)), 't_exit': str(pd.Timestamp(exit_t)),
            'side': 'LONG' if is_long[i] else 'SHORT', 'net_return': net,
            'equity_after': equity, 'close': float(oos['close'].iloc[i]),
        })
        equity_points.append({'t': str(pd.Timestamp(exit_t)), 'equity': equity})

    print(f"Trades accepted: {len(trades):,}")
    print(f"Final equity: {equity:.4f} ({(equity - 1) * 100:.1f}% return)")

    eq_vals = [p['equity'] for p in equity_points]
    peak = eq_vals[0]
    max_dd = 0.0
    dd_series = []
    for p in equity_points:
        peak = max(peak, p['equity'])
        dd = (peak - p['equity']) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        dd_series.append({'t': p['t'], 'drawdown_pct': dd * 100})
    print(f"Max drawdown: {max_dd * 100:.2f}%")

    # daily trade counts
    trades_df = pd.DataFrame(trades)
    trades_df['date'] = pd.to_datetime(trades_df['t_entry']).dt.date.astype(str)
    daily_counts = trades_df.groupby('date').size()
    full_range = pd.date_range(oos['timestamp'].min().date(), oos['timestamp'].max().date(), freq='D').astype(str)
    daily_counts = daily_counts.reindex(full_range, fill_value=0)

    out = {
        'summary': {
            'n_trades': len(trades), 'final_equity': equity,
            'total_return_pct': (equity - 1) * 100, 'max_drawdown_pct': max_dd * 100,
            'hit_rate': float(np.mean([t['net_return'] > 0 for t in trades])) if trades else None,
        },
        'equity_curve': equity_points,  # FULL resolution, every settlement
        'drawdown_curve': dd_series,
        'daily_trade_counts': [{'date': d, 'count': int(c)} for d, c in daily_counts.items()],
        'trades_sample': trades[:2000],  # cap size for the artifact; still plenty for a chart
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_cap1_chart_data_20260717.json'), 'w') as f:
        json.dump(out, f, default=str)
    print(f"Saved scalp_1m_cap1_chart_data_20260717.json ({len(equity_points):,} equity points)")


if __name__ == '__main__':
    main()

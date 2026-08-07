"""Rebuilds the cap=1 / cap=5 equity-curve chart data using only the 7 genuinely clean,
independent walk-forward folds (2025-07-01 -> 2026-05-15, ~10.5 months) instead of the original
single OOS window (2026-06-01->07-12), which was found to overlap the val/OOS window used to
originally select the fixed confidence threshold. Each fold's own model is trained ONLY on data
up to that fold's own train_end and evaluated ONLY on its own held-out test window; folds are
then stitched together in chronological order into one continuous equity curve, so the resulting
chart represents ~10.5 months a real bot would never have seen used for tuning anything.

Output: data/ensemble/reports/scalp_1m_clean7fold_chart_data_20260717.json
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

FIXED_THRESHOLD = 0.55
CAPS = [1, 5]

# The 7 genuinely clean folds (fold 8 excluded -- see stress_test_scalp_1m_block_bootstrap_20260717.py)
FOLDS = [
    ('2025-06-30', '2025-07-01', '2025-08-15'),
    ('2025-08-15', '2025-08-16', '2025-09-30'),
    ('2025-09-30', '2025-10-01', '2025-11-15'),
    ('2025-11-15', '2025-11-16', '2026-01-01'),
    ('2026-01-01', '2026-01-02', '2026-02-15'),
    ('2026-02-15', '2026-02-16', '2026-04-01'),
    ('2026-04-01', '2026-04-02', '2026-05-15'),
]


def run_cap_across_folds(fold_data, long_sim, short_sim, cap):
    equity = 1.0
    equity_points = []
    open_exit_time = None
    n_trades = 0
    n_wins = 0

    for fold_i, test, pred in fold_data:
        ts = test['timestamp'].to_numpy()
        is_long = pred == 'LONG'
        is_short = pred == 'SHORT'
        signal_idx = np.flatnonzero(is_long | is_short)
        test_idx = test['orig_idx'].to_numpy()
        long_sim_test = long_sim.loc[test_idx].reset_index(drop=True)
        short_sim_test = short_sim.loc[test_idx].reset_index(drop=True)

        # a position open at the very end of one fold's window is settled before the next fold
        # starts (folds don't overlap and a fresh model begins each fold, so we don't carry an
        # open position across the fold boundary)
        open_exit_time = None

        for j in signal_idx:
            entry_t = ts[j]
            if open_exit_time is not None and entry_t < open_exit_time:
                continue
            sim = long_sim_test if is_long[j] else short_sim_test
            if not bool(sim.iloc[j]['filled']):
                continue
            exit_offset_min = float(sim.iloc[j]['exit_offset'])
            exit_t = entry_t + np.timedelta64(int(exit_offset_min), 'm')
            net = float(sim.iloc[j]['realized_move']) - ROUND_TRIP_FEE
            notional = equity / cap
            equity = equity + notional * net
            open_exit_time = exit_t
            n_trades += 1
            n_wins += 1 if net > 0 else 0
            equity_points.append({'t': str(pd.Timestamp(exit_t)), 'equity': equity, 'fold': fold_i})

    peak = 1.0
    max_dd = 0.0
    dd_curve = []
    for p in equity_points:
        peak = max(peak, p['equity'])
        dd = (peak - p['equity']) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        dd_curve.append({'t': p['t'], 'drawdown_pct': dd * 100})

    return {
        'cap': cap, 'n_trades': n_trades, 'hit_rate': n_wins / n_trades if n_trades else None,
        'final_equity': equity, 'total_return_pct': (equity - 1) * 100, 'max_drawdown_pct': max_dd * 100,
        'equity_curve': equity_points, 'drawdown_curve': dd_curve,
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)

    print("Simulating maker-entry fills with exit timing...")
    long_sim = simulate_maker_fills_with_exit(df, 'long')
    short_sim = simulate_maker_fills_with_exit(df, 'short')
    feat_cols = feature_cols_for(df, [])

    print("Fitting all 7 clean folds...")
    fold_data = []
    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        train = df[df['timestamp'] <= train_end]
        test = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].reset_index(drop=False).rename(columns={'index': 'orig_idx'})
        if len(test) == 0 or len(train) < 50_000:
            continue
        print(f"  fitting fold {i}: train<= {train_end}, test [{test_start}, {test_end}]...")
        clf = fit_model(train, feat_cols)
        X_test = test[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X_test, FIXED_THRESHOLD)
        fold_data.append((i, test, pred))

    results = {}
    for cap in CAPS:
        print(f"\nRunning cap={cap} across all 7 clean folds (chronologically stitched)...")
        res = run_cap_across_folds(fold_data, long_sim, short_sim, cap)
        print(f"  n_trades={res['n_trades']:,} hit_rate={res['hit_rate']} "
              f"final_equity={res['final_equity']:.4f} return={res['total_return_pct']:.1f}% "
              f"max_dd={res['max_drawdown_pct']:.2f}%")
        # daily trade counts for the bar chart
        eq_df = pd.DataFrame(res['equity_curve'])
        eq_df['date'] = pd.to_datetime(eq_df['t']).dt.date.astype(str)
        daily_counts = eq_df.groupby('date').size()
        full_range = pd.date_range(FOLDS[0][1], FOLDS[-1][2], freq='D').astype(str)
        daily_counts = daily_counts.reindex(full_range, fill_value=0)
        res['daily_trade_counts'] = [{'date': d, 'count': int(c)} for d, c in daily_counts.items()]
        results[str(cap)] = res

    out_path = os.path.join(REPORT_DIR, 'scalp_1m_clean7fold_chart_data_20260717.json')
    with open(out_path, 'w') as f:
        json.dump({'fold_boundaries': FOLDS, 'results_by_cap': results}, f, default=str)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()

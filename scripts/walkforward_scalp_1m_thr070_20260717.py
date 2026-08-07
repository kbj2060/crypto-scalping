"""Walk-forward re-validation of the threshold=0.70 frequency-reduction lever
(reduce_scalp_1m_trade_frequency_20260717.py: single-window OOS 79.6 trades/day, +2.92% vs the
threshold=0.55 baseline's 192.3 trades/day, +3.74%) -- same 8 expanding-window folds and same
realistic maker-fill simulation as the threshold=0.55 walk-forward
(scalp_1m_walkforward_conf_maker_20260716.json: 8/8 positive, mean +3.51%/fold), only the fixed
threshold changes, to check whether the frequency/return tradeoff holds up over 13 months or was
specific to the single OOS window it was picked on.

Output: data/ensemble/reports/scalp_1m_walkforward_thr070_20260717.json
"""
import json
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import (
    simulate_maker_fills, backtest_maker, predict_with_threshold, LABELS_CSV,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

FIXED_THRESHOLD = 0.70

FOLDS = [
    ('2025-06-30', '2025-07-01', '2025-08-15'),
    ('2025-08-15', '2025-08-16', '2025-09-30'),
    ('2025-09-30', '2025-10-01', '2025-11-15'),
    ('2025-11-15', '2025-11-16', '2026-01-01'),
    ('2026-01-01', '2026-01-02', '2026-02-15'),
    ('2026-02-15', '2026-02-16', '2026-04-01'),
    ('2026-04-01', '2026-04-02', '2026-05-15'),
    ('2026-05-15', '2026-05-16', '2026-07-12'),
]


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("Simulating maker-entry fills (long + short, once for the whole dataset)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    fold_results = []
    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        train = df[df['timestamp'] <= train_end]
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
        test = df[test_mask]
        if len(test) == 0 or len(train) < 50_000:
            print(f"Fold {i}: skipped (train={len(train):,}, test={len(test):,})")
            continue
        n_days = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400
        print(f"\nFold {i}: train<= {train_end} (n={len(train):,}), test [{test_start}, {test_end}] "
              f"(n={len(test):,}, {n_days:.1f} days)")

        clf = fit_model(train, feat_cols)
        X_test = test[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X_test, FIXED_THRESHOLD)

        test_idx = test.index
        long_sim_test = long_sim.loc[test_idx].reset_index(drop=True)
        short_sim_test = short_sim.loc[test_idx].reset_index(drop=True)
        bt = backtest_maker(pred, long_sim_test, short_sim_test)
        trades_per_day = (bt['n_filled'] or 0) / max(n_days, 1e-6)
        print(f"  signals={bt['n_signals']:,} filled={bt['n_filled']:,} ({trades_per_day:.1f}/day) "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

        fold_results.append({
            'fold': i, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end,
            'n_train': len(train), 'n_test': len(test), 'n_days': n_days,
            'trades_per_day': trades_per_day, **bt,
        })

    pnls = [f['total_pnl_pct'] for f in fold_results if f['total_pnl_pct'] is not None]
    tpds = [f['trades_per_day'] for f in fold_results]
    n_positive = sum(1 for p in pnls if p > 0)
    summary = {
        'fixed_threshold': FIXED_THRESHOLD,
        'n_folds': len(fold_results),
        'n_folds_positive': n_positive,
        'pct_folds_positive': n_positive / len(pnls) if pnls else None,
        'mean_pnl_pct': float(sum(pnls) / len(pnls)) if pnls else None,
        'min_pnl_pct': float(min(pnls)) if pnls else None,
        'max_pnl_pct': float(max(pnls)) if pnls else None,
        'mean_trades_per_day': float(sum(tpds) / len(tpds)) if tpds else None,
        'max_trades_per_day': float(max(tpds)) if tpds else None,
    }
    print(f"\n{'=' * 60}\nSummary: {summary}\n{'=' * 60}")

    result = {
        'folds': fold_results, 'summary': summary,
        'baseline_for_comparison': {
            'report': 'scalp_1m_walkforward_conf_maker_20260716.json',
            'threshold': 0.55, 'n_folds_positive': 8, 'mean_pnl_pct': 3.508987048019917,
        },
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Fixed threshold=0.70 (chosen once on the original single-window val/OOS split) held '
                 'constant across all 8 folds to avoid per-fold multiple-comparison overfitting -- '
                 'tests whether the frequency/return tradeoff found on one window holds over 13 months.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_walkforward_thr070_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved scalp_1m_walkforward_thr070_20260717.json")


if __name__ == '__main__':
    main()

"""Walk-forward validation of the compounding, capital-capped portfolio backtest
(simulate_portfolio_capped_scalp_1m_20260717.py) at two position-sizing policies the user asked
about directly: cap=5 (20% notional per position, one of the single-window OOS options) and
cap=1 (true single-position -- no concurrency at all, the "does our model already trade one at a
time" question this came from -- it does NOT by default, cap=1 is a new, more conservative
constraint being tested here for the first time).

Same 8 expanding-window folds and fixed threshold=0.55 as every other walk-forward in this line
of experiments (scalp_1m_walkforward_conf_maker_20260716.json: unconstrained-capital baseline,
8/8 positive, mean +3.51%/fold). This version reports compounding portfolio return AND max
drawdown per fold, not just a summed-PnL figure.

Output: data/ensemble/reports/scalp_1m_walkforward_portfolio_capped_20260717.json
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
from simulate_maker_entry_scalp_1m_20260716 import LABELS_CSV
from simulate_portfolio_capped_scalp_1m_20260717 import (
    simulate_maker_fills_with_exit, predict_with_threshold, portfolio_backtest,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

FIXED_THRESHOLD = 0.55
CAPS = [1, 5]

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

    print("Simulating maker-entry fills with exit timing (long + short, once for the whole dataset)...")
    long_sim = simulate_maker_fills_with_exit(df, 'long')
    short_sim = simulate_maker_fills_with_exit(df, 'short')

    feat_cols = feature_cols_for(df, [])
    results_by_cap = {cap: [] for cap in CAPS}

    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        train = df[df['timestamp'] <= train_end]
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
        test = df[test_mask].reset_index(drop=False).rename(columns={'index': 'orig_idx'})
        if len(test) == 0 or len(train) < 50_000:
            print(f"Fold {i}: skipped")
            continue
        n_days = (test['timestamp'].max() - test['timestamp'].min()).total_seconds() / 86400
        print(f"\nFold {i}: train<= {train_end}, test [{test_start}, {test_end}] (n={len(test):,}, {n_days:.1f} days)")

        clf = fit_model(train, feat_cols)
        X_test = test[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X_test, FIXED_THRESHOLD)

        test_idx = test['orig_idx'].to_numpy()
        long_sim_test = long_sim.loc[test_idx].reset_index(drop=True)
        short_sim_test = short_sim.loc[test_idx].reset_index(drop=True)

        for cap in CAPS:
            res = portfolio_backtest(test['timestamp'], pred, long_sim_test, short_sim_test, cap)
            res.pop('equity_curve_sample', None)  # drop per-fold curve from the printed/stored summary, keep it light
            res.update({'fold': i, 'train_end': train_end, 'test_start': test_start,
                        'test_end': test_end, 'n_days': n_days})
            results_by_cap[cap].append(res)
            print(f"  cap={cap}: accepted={res['n_accepted']:,}/{res['n_signals']:,} "
                  f"hit_rate={res['hit_rate']} return={res['portfolio_return_pct']:.2f}% "
                  f"max_dd={res['max_drawdown_pct']:.2f}%")

    summaries = {}
    for cap, fold_results in results_by_cap.items():
        rets = [f['portfolio_return_pct'] for f in fold_results]
        dds = [f['max_drawdown_pct'] for f in fold_results]
        n_pos = sum(1 for r in rets if r > 0)
        summaries[cap] = {
            'n_folds': len(fold_results), 'n_folds_positive': n_pos,
            'pct_folds_positive': n_pos / len(rets) if rets else None,
            'mean_return_pct': sum(rets) / len(rets) if rets else None,
            'min_return_pct': min(rets) if rets else None, 'max_return_pct': max(rets) if rets else None,
            'mean_max_drawdown_pct': sum(dds) / len(dds) if dds else None,
            'worst_max_drawdown_pct': max(dds) if dds else None,
        }
        print(f"\ncap={cap} summary: {summaries[cap]}")

    result = {
        'fixed_threshold': FIXED_THRESHOLD,
        'folds_by_cap': {str(k): v for k, v in results_by_cap.items()},
        'summary_by_cap': {str(k): v for k, v in summaries.items()},
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Compounding position-cap portfolio backtest (equity/cap sizing per new position, '
                 'settled on exit); cap=1 = single position, no concurrency; cap=5 = up to 5 concurrent '
                 'positions at 20% equity each at entry time.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_walkforward_portfolio_capped_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_walkforward_portfolio_capped_20260717.json")


if __name__ == '__main__':
    main()

"""Lever B-1 test: does concurrency-based sample weighting + purged training beat the existing
8/8-positive baseline (scripts/walkforward_scalp_1m_conf_maker_20260716.py, mean +3.51%/fold)?

Two changes vs. the baseline walk-forward, both from Advances in Financial Machine Learning ch.4
(purging/embargo) and ch.4.5 (sample uniqueness), applied via
build_scalp_1m_tb_labels_weighted_20260716.py's scalp_uniqueness_weight /
scalp_resolution_offset columns:

  1. PURGE: training rows within HORIZON=20 minutes of each fold's train_end are dropped -- their
     triple-barrier label's forward window extends past train_end into the test period, so
     including them would leak test-period price action into the fitted model.
  2. WEIGHT: HistGradientBoostingClassifier.fit(..., sample_weight=scalp_uniqueness_weight) so
     overlapping/redundant labels (labels that resolved slowly and therefore overlap many
     neighbors) count less than fast-resolving, more independent ones -- corrects the
     overconfidence risk from ~95%-overlapping 20-bar windows on every 1-minute bar.

Same fixed threshold (0.55), same realistic maker-fill simulation, same 8 folds as the baseline,
so the comparison isolates exactly these two changes.

Output: data/ensemble/reports/scalp_1m_walkforward_weighted_purged_20260716.json
"""
import json
import os
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for, HGB_PARAMS
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, predict_with_threshold

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_WEIGHTED_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels_weighted.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

FIXED_THRESHOLD = 0.55
HORIZON_MINUTES = 20  # must match build_scalp_1m_tb_labels_weighted_20260716.py's HORIZON

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


def fit_model_weighted(train: pd.DataFrame, feat_cols: list[str]) -> HistGradientBoostingClassifier:
    X_train = train[feat_cols].fillna(0.0)
    y_train = train['scalp_action']
    w_train = train['scalp_uniqueness_weight']
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(X_train, y_train, sample_weight=w_train)
    return clf


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + weighted labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_WEIGHTED_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("Simulating maker-entry fills (long + short, once for the whole dataset)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    purge_delta = pd.Timedelta(minutes=HORIZON_MINUTES)
    fold_results = []
    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        purge_cutoff = pd.Timestamp(train_end) - purge_delta
        train = df[df['timestamp'] <= purge_cutoff]
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
        test = df[test_mask]
        if len(test) == 0 or len(train) < 50_000:
            print(f"Fold {i}: skipped (train={len(train):,}, test={len(test):,})")
            continue
        n_purged = len(df[(df['timestamp'] > purge_cutoff) & (df['timestamp'] <= train_end)])
        print(f"\nFold {i}: train<={purge_cutoff} (n={len(train):,}, purged {n_purged:,} rows), "
              f"test [{test_start}, {test_end}] (n={len(test):,})")

        clf = fit_model_weighted(train, feat_cols)
        X_test = test[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X_test, FIXED_THRESHOLD)

        test_idx = test.index
        long_sim_test = long_sim.loc[test_idx].reset_index(drop=True)
        short_sim_test = short_sim.loc[test_idx].reset_index(drop=True)
        bt = backtest_maker(pred, long_sim_test, short_sim_test)
        print(f"  signals={bt['n_signals']:,} filled={bt['n_filled']:,} fill_rate={bt['fill_rate']} "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

        fold_results.append({
            'fold': i, 'train_end': train_end, 'purge_cutoff': str(purge_cutoff),
            'n_purged': n_purged, 'test_start': test_start, 'test_end': test_end,
            'n_train': len(train), 'n_test': len(test), **bt,
        })

    pnls = [f['total_pnl_pct'] for f in fold_results if f['total_pnl_pct'] is not None]
    n_positive = sum(1 for p in pnls if p > 0)
    summary = {
        'fixed_threshold': FIXED_THRESHOLD,
        'n_folds': len(fold_results),
        'n_folds_positive': n_positive,
        'pct_folds_positive': n_positive / len(pnls) if pnls else None,
        'mean_pnl_pct': float(sum(pnls) / len(pnls)) if pnls else None,
        'min_pnl_pct': float(min(pnls)) if pnls else None,
        'max_pnl_pct': float(max(pnls)) if pnls else None,
    }
    print(f"\n{'=' * 60}\nSummary: {summary}\n{'=' * 60}")
    print("Baseline (unweighted, unpurged) for comparison: 8/8 positive, mean +3.51%, "
          "range [+1.44%, +5.66%] (scalp_1m_walkforward_conf_maker_20260716.json)")

    result = {
        'folds': fold_results, 'summary': summary,
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Same 8 folds/threshold/maker-fill-sim as scalp_1m_walkforward_conf_maker_20260716.json, '
                 'isolating the effect of purging the last 20min of each train window + weighting '
                 'by concurrency-based average uniqueness (Lopez de Prado AFML ch.4/4.5).'),
        'baseline_for_comparison': {
            'report': 'scalp_1m_walkforward_conf_maker_20260716.json',
            'n_folds_positive': 8, 'mean_pnl_pct': 3.508987048019917,
        },
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_walkforward_weighted_purged_20260716.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved scalp_1m_walkforward_weighted_purged_20260716.json")


if __name__ == '__main__':
    main()

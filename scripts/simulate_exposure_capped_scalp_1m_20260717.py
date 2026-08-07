"""Replaces the slot-count `cap` model (simulate_portfolio_capped_scalp_1m_20260717.py) with two
explicit, independent parameters, per user feedback that "cap" conflated two different things and
that more slots isn't necessarily safer once trades are correlated:

  PER_TRADE_PCT        -- size of each individual new position, as a fraction of current equity.
  MAX_TOTAL_EXPOSURE_PCT -- hard ceiling on the SUM of all currently-open positions' notional
                            (as a fraction of equity). A new signal is rejected outright if
                            accepting it would push total open exposure over this ceiling,
                            regardless of how many separate positions that involves.

This directly bounds worst-case single-event loss: even in the worst case (every open position
loses 100% of its notional simultaneously, e.g. all long into a crash), the maximum possible loss
is exactly MAX_TOTAL_EXPOSURE_PCT of equity -- a clean, provable number, not an indirect
consequence of "cap x per-slot-size" that could still let 100% of equity be at risk if enough
correlated slots fire at once (as cap=5 x 20%/slot could).

Tested on the same 7 clean, threshold-selection-independent walk-forward folds
(2025-07-01 -> 2026-05-15) used throughout this line of experiments.

Output: data/ensemble/reports/scalp_1m_exposure_capped_20260717.json
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

# (per_trade_pct, max_total_exposure_pct) -- includes a strict single-position-at-20% policy
# (the cleanest read of "cap=1 but not 100%"), plus a couple of split-across-slots variants at
# the same total ceiling for direct comparison.
CONFIGS = [
    (0.50, 0.50),   # single position, 50% -- user-directed live-candidate sizing (separate dedicated account)
    (0.20, 0.20),   # single position, 20% -- the direct fix for "cap=1 is too aggressive"
    (0.05, 0.20),   # up to 4 concurrent at 5% each, same 20% total ceiling
    (0.02, 0.20),   # up to 10 concurrent at 2% each, same 20% total ceiling
    (0.05, 0.10),   # more conservative: 10% total ceiling
    (0.01, 0.05),   # very conservative: 5% total ceiling
]

# The 7 genuinely clean folds (fold 8 excluded -- overlapped the threshold-selection window)
FOLDS = [
    ('2025-06-30', '2025-07-01', '2025-08-15'),
    ('2025-08-15', '2025-08-16', '2025-09-30'),
    ('2025-09-30', '2025-10-01', '2025-11-15'),
    ('2025-11-15', '2025-11-16', '2026-01-01'),
    ('2026-01-01', '2026-01-02', '2026-02-15'),
    ('2026-02-15', '2026-02-16', '2026-04-01'),
    ('2026-04-01', '2026-04-02', '2026-05-15'),
]


def run_exposure_capped(fold_data, long_sim, short_sim, per_trade_pct, max_total_exposure_pct):
    equity = 1.0
    open_positions = []  # list of [exit_time, notional]
    n_signals = 0
    n_accepted = 0
    n_rejected_exposure = 0
    n_unfilled = 0
    n_wins = 0
    equity_points = []

    for fold_i, test, pred in fold_data:
        ts = test['timestamp'].to_numpy()
        is_long = pred == 'LONG'
        is_short = pred == 'SHORT'
        signal_idx = np.flatnonzero(is_long | is_short)
        test_idx = test['orig_idx'].to_numpy()
        long_sim_test = long_sim.loc[test_idx].reset_index(drop=True)
        short_sim_test = short_sim.loc[test_idx].reset_index(drop=True)
        open_positions = []  # fresh model per fold; don't carry positions across fold boundaries

        for j in signal_idx:
            n_signals += 1
            entry_t = ts[j]
            open_positions = [p for p in open_positions if p[0] > entry_t]
            current_exposure = sum(p[1] for p in open_positions)
            if current_exposure + per_trade_pct > max_total_exposure_pct + 1e-9:
                n_rejected_exposure += 1
                continue
            sim = long_sim_test if is_long[j] else short_sim_test
            if not bool(sim.iloc[j]['filled']):
                n_unfilled += 1
                continue
            exit_offset_min = float(sim.iloc[j]['exit_offset'])
            exit_t = entry_t + np.timedelta64(int(exit_offset_min), 'm')
            net = float(sim.iloc[j]['realized_move']) - ROUND_TRIP_FEE
            notional = per_trade_pct * equity
            equity = equity + notional * net
            open_positions.append([exit_t, per_trade_pct])
            n_accepted += 1
            n_wins += 1 if net > 0 else 0
            equity_points.append(equity)

    peak = 1.0
    max_dd = 0.0
    for e in equity_points:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak if peak > 0 else 0.0)

    return {
        'per_trade_pct': per_trade_pct, 'max_total_exposure_pct': max_total_exposure_pct,
        'n_signals': n_signals, 'n_accepted': n_accepted,
        'n_rejected_exposure': n_rejected_exposure, 'n_unfilled': n_unfilled,
        'hit_rate': n_wins / n_accepted if n_accepted else None,
        'final_equity': equity, 'total_return_pct': (equity - 1) * 100,
        'max_drawdown_pct': max_dd * 100,
        'theoretical_worst_single_event_loss_pct': max_total_exposure_pct * 100,
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
        print(f"  fitting fold {i}...")
        clf = fit_model(train, feat_cols)
        X_test = test[feat_cols].fillna(0.0)
        pred = predict_with_threshold(clf, X_test, FIXED_THRESHOLD)
        fold_data.append((i, test, pred))

    results = []
    print("\n--- Exposure-capped sizing grid (chronologically stitched across 7 clean folds) ---")
    for per_trade_pct, max_total_exposure_pct in CONFIGS:
        res = run_exposure_capped(fold_data, long_sim, short_sim, per_trade_pct, max_total_exposure_pct)
        results.append(res)
        print(f"  per_trade={per_trade_pct:.0%} max_exposure={max_total_exposure_pct:.0%}: "
              f"accepted={res['n_accepted']:,}/{res['n_signals']:,} hit_rate={res['hit_rate']} "
              f"return={res['total_return_pct']:.1f}% max_dd={res['max_drawdown_pct']:.2f}% "
              f"worst_theoretical_single_event={res['theoretical_worst_single_event_loss_pct']:.0f}%")

    with open(os.path.join(REPORT_DIR, 'scalp_1m_exposure_capped_20260717.json'), 'w') as f:
        json.dump({'threshold': FIXED_THRESHOLD, 'results': results}, f, indent=2, default=str)
    print("\nSaved scalp_1m_exposure_capped_20260717.json")


if __name__ == '__main__':
    main()

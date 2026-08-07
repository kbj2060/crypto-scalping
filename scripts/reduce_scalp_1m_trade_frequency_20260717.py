"""Reduces the baseline's trade frequency (192 filled trades/day median, 3min median gap between
entries -- while holding for 20min, meaning dozens of overlapping concurrent positions), tested
on the same Experiment A OOS window (2026-06-01 -> 2026-07-12) and the same realistic maker-fill
simulation as scalp_1m_tune_maker_realistic_20260716.json (baseline: OOS +3.74%, 8,075 filled
trades, threshold=0.55).

Two independent levers, swept separately and combined:
  1. HIGHER CONFIDENCE THRESHOLD -- the existing lever, just swept further into higher values
     than the val-optimal 0.55 (which was chosen to maximize total PnL, not to control volume).
  2. COOLDOWN -- a hard rule with no retraining: after taking a trade, ignore any new signal for
     N minutes. This directly targets the actual problem (median 3min gap while holding up to
     20min) rather than indirectly reducing volume by raising the bar for what counts as a
     signal. Applied post-hoc, sequentially in time order, to the SAME underlying signal stream
     the baseline threshold produces.

NOT implemented here: a max-concurrent-open-positions cap (a more realistic representation of
"how many slots does the bot actually have"). That needs per-trade exit timing, which
simulate_maker_fills doesn't currently expose (only fill status + realized outcome) -- would
require extending that function; noted as a natural next step, not done in this pass.

Output: data/ensemble/reports/scalp_1m_frequency_reduction_20260717.json
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
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, LABELS_CSV

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
COOLDOWNS_MIN = [0, 5, 10, 20, 30, 60]


def apply_cooldown(timestamps: pd.Series, side: np.ndarray, cooldown_min: int) -> np.ndarray:
    """Sequential scan: keep a signal only if >= cooldown_min minutes have passed since the last
    KEPT signal (regardless of side -- this caps overall entry rate, not per-direction)."""
    if cooldown_min <= 0:
        return side.copy()
    kept = np.array(['CASH'] * len(side), dtype=object)
    last_kept_ts = None
    ts_vals = timestamps.to_numpy()
    for i in range(len(side)):
        if side[i] == 'CASH':
            continue
        if last_kept_ts is None or (ts_vals[i] - last_kept_ts) >= np.timedelta64(cooldown_min, 'm'):
            kept[i] = side[i]
            last_kept_ts = ts_vals[i]
    return kept


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + labels, training baseline primary model...")
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
    n_days = (oos['timestamp'].max() - oos['timestamp'].min()).total_seconds() / 86400
    print(f"Train={len(train):,} OOS={len(oos):,} ({n_days:.1f} days)")

    clf = fit_model(train, feat_cols)
    X_oos = oos[feat_cols].fillna(0.0)
    proba = clf.predict_proba(X_oos)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    raw_pred = classes[max_idx]

    # IMPORTANT: capture .index BEFORE reset_index -- long_sim/short_sim are indexed by absolute
    # row position in the full `df`, not by position within the OOS slice. Resetting first (as an
    # earlier version of this script did) silently pulls maker-fill results for the wrong rows
    # (the start of the full 2024-2026 dataset instead of the actual OOS window).
    idx = oos.index
    long_sim_oos = long_sim.loc[idx].reset_index(drop=True)
    short_sim_oos = short_sim.loc[idx].reset_index(drop=True)
    oos = oos.reset_index(drop=True)

    results = []

    print("\n--- Lever 1: threshold alone (no cooldown) ---")
    for thr in THRESHOLDS:
        pred = np.where(max_proba >= thr, raw_pred, 'CASH')
        bt = backtest_maker(pred, long_sim_oos, short_sim_oos)
        trades_per_day = (bt['n_filled'] or 0) / n_days
        print(f"  thr={thr}: filled={bt['n_filled']:,} ({trades_per_day:.1f}/day) "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        results.append({'lever': 'threshold_only', 'threshold': thr, 'cooldown_min': 0,
                         'trades_per_day': trades_per_day, **bt})

    print("\n--- Lever 2: cooldown alone (threshold=0.55 baseline signal stream) ---")
    base_pred = np.where(max_proba >= 0.55, raw_pred, 'CASH')
    for cd in COOLDOWNS_MIN:
        pred = apply_cooldown(oos['timestamp'], base_pred, cd)
        bt = backtest_maker(pred, long_sim_oos, short_sim_oos)
        trades_per_day = (bt['n_filled'] or 0) / n_days
        print(f"  cooldown={cd}min: filled={bt['n_filled']:,} ({trades_per_day:.1f}/day) "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        results.append({'lever': 'cooldown_only', 'threshold': 0.55, 'cooldown_min': cd,
                         'trades_per_day': trades_per_day, **bt})

    print("\n--- Lever 3: threshold + cooldown combined (a few practical operating points) ---")
    combos = [(0.65, 20), (0.65, 60), (0.75, 20), (0.75, 60), (0.85, 20)]
    for thr, cd in combos:
        pred = np.where(max_proba >= thr, raw_pred, 'CASH')
        pred = apply_cooldown(oos['timestamp'], pred, cd)
        bt = backtest_maker(pred, long_sim_oos, short_sim_oos)
        trades_per_day = (bt['n_filled'] or 0) / n_days
        print(f"  thr={thr} + cooldown={cd}min: filled={bt['n_filled']:,} ({trades_per_day:.1f}/day) "
              f"hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")
        results.append({'lever': 'threshold_plus_cooldown', 'threshold': thr, 'cooldown_min': cd,
                         'trades_per_day': trades_per_day, **bt})

    result = {
        'oos_window': {'start': VAL_END, 'end': OOS_END, 'n_days': n_days},
        'baseline_for_comparison': {
            'report': 'scalp_1m_tune_maker_realistic_20260716.json',
            'threshold': 0.55, 'cooldown_min': 0,
            'trades_per_day': 8075 / n_days, 'oos_total_pnl_pct': 3.7390646402123644,
        },
        'scenarios': results,
        'not_implemented_note': ('Max-concurrent-open-positions cap would need per-trade exit '
                                  'timing (simulate_maker_fills only returns fill status + '
                                  'realized outcome, not exit offset) -- noted as a natural next '
                                  'step, not built in this pass.'),
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_frequency_reduction_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_frequency_reduction_20260717.json")


if __name__ == '__main__':
    main()

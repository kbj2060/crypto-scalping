"""Correlation-aware, tail-risk-aware validation, replacing the single-historical-path drawdown
figures from walkforward_scalp_1m_portfolio_capped_20260717.py (which only shows what happened
on the ONE path 2025-07->2026-07 actually took) with a block-bootstrap distribution of possible
outcomes.

Key insight enabling a clean bootstrap: with cap-based sizing (notional = equity/cap at entry),
each trade's multiplicative effect on equity is SCALE-INVARIANT: (1 + net_return/cap), regardless
of the absolute equity level. So a calendar day's cumulative effect on equity is just the PRODUCT
of that day's trade multipliers, computed once from the already-correctly-decided (cap-enforced,
chronological) accept/reject sequence -- no need to re-simulate accept/reject logic per bootstrap
draw. This lets us resample whole DAYS (preserving within-day trade correlation exactly, since a
day's multiplier is one atomic block) to build a real distribution of possible year-outcomes,
rather than trusting the single historical sequence's drawdown.

Method:
  1. Refit each of the same 8 expanding-window folds used throughout this line of experiments,
     get threshold=0.55 signals, run the SAME cap-constrained event-driven accept/reject logic as
     walkforward_scalp_1m_portfolio_capped_20260717.py for cap in {1, 5}.
  2. For every calendar day in the combined 8-fold OOS coverage (~376 days total, 2025-07->
     2026-07), compute that day's equity multiplier (product of that day's accepted trades'
     (1+net/cap) factors; 1.0 for days with zero trades).
  3. Block bootstrap: for BLOCK_SIZE in {1, 5} days, draw BOOTSTRAP_DAYS/BLOCK_SIZE contiguous
     blocks with replacement from the actual daily-multiplier sequence (contiguous blocks
     preserve local/multi-day correlation; single-day blocks preserve only within-day
     correlation), concatenate into a synthetic year-length path, compute total return + max
     drawdown for that path. Repeat N_SIMULATIONS times.
  4. Report the resulting distribution: mean/median return, 5th/1st percentile (VaR-style "bad
     scenario"), worst simulated drawdown, and P(any drawdown > 20%) / P(ruin, equity<0.5) across
     simulations -- real probabilistic risk metrics instead of one historical path's numbers.

Honest limitation stated, not hidden: day multipliers are drawn i.i.d. (single-day blocks) or
from fixed historical 5-day sequences (multi-day blocks) -- this captures within-block
correlation but not full cross-block regime persistence (e.g. a genuinely unprecedented volatility
regime spanning months, worse than anything in the 2024-2026 sample, still isn't represented;
bootstrap can only resample what's IN the historical sample, not extrapolate beyond it).

Output: data/ensemble/reports/scalp_1m_block_bootstrap_20260717.json
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
BLOCK_SIZES = [1, 5]
N_SIMULATIONS = 5000

FOLDS = [
    ('2025-06-30', '2025-07-01', '2025-08-15'),
    ('2025-08-15', '2025-08-16', '2025-09-30'),
    ('2025-09-30', '2025-10-01', '2025-11-15'),
    ('2025-11-15', '2025-11-16', '2026-01-01'),
    ('2026-01-01', '2026-01-02', '2026-02-15'),
    ('2026-02-15', '2026-02-16', '2026-04-01'),
    ('2026-04-01', '2026-04-02', '2026-05-15'),
    # Fold 8 (2026-05-16 -> 2026-07-12) REMOVED (2026-07-17): its test window overlaps the
    # val/OOS window (2026-04-30..2026-05-31 / 2026-05-31..2026-07-12) originally used to SELECT
    # the fixed confidence threshold (0.55, later 0.70) -- reusing that same window as a
    # "validation" fold isn't a genuinely independent test of the threshold policy. Folds 1-7
    # never touched threshold selection and remain clean.
]


def collect_fold_signals(df, feat_cols):
    """Fits each of the 8 folds ONCE (shared across every cap value -- cap only affects the
    post-hoc position-sizing/accept-reject replay, not training) and returns per-fold
    (test_df, pred) pairs."""
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
    return fold_data


def collect_daily_multipliers(fold_data, long_sim, short_sim, cap):
    """Runs the cap-constrained event-driven sim across all 8 folds (reusing pre-computed
    predictions) and returns one long chronological list of (date, day_multiplier) covering every
    calendar day in the combined OOS coverage."""
    daily = []
    for i, test, pred in fold_data:
        ts = test['timestamp'].to_numpy()
        is_long = pred == 'LONG'
        is_short = pred == 'SHORT'
        signal_idx = np.flatnonzero(is_long | is_short)
        test_idx = test['orig_idx'].to_numpy()
        long_sim_test = long_sim.loc[test_idx].reset_index(drop=True)
        short_sim_test = short_sim.loc[test_idx].reset_index(drop=True)

        open_exit_time = None
        day_factors = {}  # date -> list of (1+net/cap) factors, in chronological order
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
            open_exit_time = exit_t
            date_key = str(pd.Timestamp(entry_t).date())
            day_factors.setdefault(date_key, []).append(1.0 + net / cap)

        full_range = pd.date_range(test['timestamp'].min().date(), test['timestamp'].max().date(), freq='D')
        for d in full_range:
            dk = str(d.date())
            factors = day_factors.get(dk, [])
            mult = 1.0
            for f in factors:
                mult *= f
            daily.append({'date': dk, 'multiplier': mult, 'n_trades': len(factors)})
        print(f"  fold {i} (cap={cap}): {len(day_factors)} active days, "
              f"{sum(len(v) for v in day_factors.values())} trades")
    return daily


def block_bootstrap(daily_multipliers: list[float], block_size: int, n_days: int, n_sims: int, rng):
    n = len(daily_multipliers)
    arr = np.array(daily_multipliers)
    n_blocks_needed = int(np.ceil(n_days / block_size))
    results_return = np.zeros(n_sims)
    results_maxdd = np.zeros(n_sims)
    for s in range(n_sims):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks_needed)
        path = np.concatenate([arr[st:st + block_size] for st in starts])[:n_days]
        equity_path = np.cumprod(path)
        results_return[s] = equity_path[-1] - 1.0
        peak = np.maximum.accumulate(np.concatenate([[1.0], equity_path]))
        eq_with_start = np.concatenate([[1.0], equity_path])
        dd = (peak - eq_with_start) / peak
        results_maxdd[s] = dd.max()
    return results_return, results_maxdd


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    rng = np.random.default_rng(20260717)
    print("Loading data + labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)

    print("Simulating maker-entry fills with exit timing...")
    long_sim = simulate_maker_fills_with_exit(df, 'long')
    short_sim = simulate_maker_fills_with_exit(df, 'short')
    feat_cols = feature_cols_for(df, [])

    print("\nFitting all 8 folds once (shared across cap values)...")
    fold_data = collect_fold_signals(df, feat_cols)

    all_results = {}
    for cap in CAPS:
        print(f"\n{'=' * 70}\nCollecting daily multipliers across all 8 folds, cap={cap}\n{'=' * 70}")
        daily = collect_daily_multipliers(fold_data, long_sim, short_sim, cap)
        mults = [d['multiplier'] for d in daily]
        n_days = len(mults)
        worst_day = min(daily, key=lambda d: d['multiplier'])
        best_day = max(daily, key=lambda d: d['multiplier'])
        print(f"Pool: {n_days} days, worst day={worst_day}, best day={best_day}")

        cap_result = {
            'n_days_pool': n_days,
            'worst_single_day': worst_day, 'best_single_day': best_day,
            'actual_realized_return_pct': (np.prod(mults) - 1) * 100,
            'daily_multipliers': daily,  # full raw list, kept for downstream charting
            'block_bootstrap': {},
        }
        for block_size in BLOCK_SIZES:
            rets, dds = block_bootstrap(mults, block_size, n_days, N_SIMULATIONS, rng)
            summary = {
                'block_size_days': block_size, 'n_simulations': N_SIMULATIONS,
                'mean_return_pct': float(np.mean(rets) * 100),
                'median_return_pct': float(np.median(rets) * 100),
                'p5_return_pct': float(np.percentile(rets, 5) * 100),
                'p1_return_pct': float(np.percentile(rets, 1) * 100),
                'worst_return_pct': float(np.min(rets) * 100),
                'mean_max_drawdown_pct': float(np.mean(dds) * 100),
                'p95_max_drawdown_pct': float(np.percentile(dds, 95) * 100),
                'p99_max_drawdown_pct': float(np.percentile(dds, 99) * 100),
                'worst_max_drawdown_pct': float(np.max(dds) * 100),
                'prob_drawdown_gt_20pct': float(np.mean(dds > 0.20)),
                'prob_drawdown_gt_50pct': float(np.mean(dds > 0.50)),
                'prob_ruin_equity_below_half': float(np.mean(rets < -0.50)),
            }
            cap_result['block_bootstrap'][str(block_size)] = summary
            print(f"  block_size={block_size}d: mean={summary['mean_return_pct']:.1f}% "
                  f"p5={summary['p5_return_pct']:.1f}% p1={summary['p1_return_pct']:.1f}% "
                  f"worst={summary['worst_return_pct']:.1f}% | "
                  f"mean_maxdd={summary['mean_max_drawdown_pct']:.1f}% "
                  f"p99_maxdd={summary['p99_max_drawdown_pct']:.1f}% "
                  f"P(dd>20%)={summary['prob_drawdown_gt_20pct']:.1%} "
                  f"P(dd>50%)={summary['prob_drawdown_gt_50pct']:.1%}")

        all_results[str(cap)] = cap_result

    result = {
        'threshold': FIXED_THRESHOLD, 'n_simulations': N_SIMULATIONS,
        'results_by_cap': all_results,
        'limitation_note': ('Day multipliers drawn i.i.d. (block=1) or from fixed historical 5-day '
                             'sequences (block=5) -- captures within-block trade correlation but not '
                             'full cross-block regime persistence; cannot represent a genuinely '
                             'unprecedented regime worse than anything in the 2024-2026 sample.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_block_bootstrap_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_block_bootstrap_20260717.json")


if __name__ == '__main__':
    main()

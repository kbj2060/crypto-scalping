"""Fixes the capital-allocation gap found 2026-07-17: every PnL number reported so far in this
line of experiments (backtest_maker's total_pnl_pct) is a SUM of every individual trade's return,
which implicitly assumes each trade gets its own independent 100% notional -- i.e. unlimited
capital. With up to 4.9 average / 17 max concurrent open positions (3min entry gap, 20min hold),
that assumption is false; a real account has to split a single capital pool across whatever's
open at once. A rough concurrency-divide estimate put the realistic OOS return at ~+0.76% (vs the
headline +3.74%) -- this script replaces that estimate with an actual event-driven portfolio
simulation: a hard cap on concurrent open positions (CAP), each getting equal notional
(1/CAP of capital), new signals rejected once the cap is full (a real capacity constraint, not
just an approximation).

Requires knowing each trade's actual EXIT time, which the existing simulate_maker_fills()
(simulate_maker_entry_scalp_1m_20260716.py) doesn't expose (only fill status + realized move) --
this script's simulate_maker_fills_with_exit() is that function extended to also return the
resolution offset (bars from fill to first-touch), reusing the identical fill/TP/SL logic.

Tests CAP in {3, 5, 10, 20} at both threshold=0.55 (max-PnL policy) and threshold=0.70
(frequency-reduced policy) on the single OOS window first; the winning CAP is then walk-forward
validated across the same 8 folds used throughout this line of experiments.

Output: data/ensemble/reports/scalp_1m_portfolio_capped_20260717.json
"""
import heapq
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
from simulate_maker_entry_scalp_1m_20260716 import (
    LABELS_CSV, HORIZON, OFFSET, FILL_LOOKAHEAD, ROUND_TRIP_FEE,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'

CAPS = [3, 5, 10, 20, 50, 100]
THRESHOLDS = [0.55, 0.70]

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


def simulate_maker_fills_with_exit(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Same fill/TP-SL mechanics as simulate_maker_fills(), extended to also return the exit
    offset (bars from ENTRY, i.e. row i, to the actual resolution) needed for position-duration
    tracking."""
    n = len(df)
    open_ = df['open'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    tp_move = df['scalp_tp_move'].to_numpy()
    sl_move = df['scalp_sl_move'].to_numpy()

    entry_open = np.concatenate([open_[1:], [np.nan]])
    limit_price = entry_open * (1 - OFFSET) if direction == 'long' else entry_open * (1 + OFFSET)

    fill_offset = np.full(n, np.nan)
    for f in range(1, FILL_LOOKAHEAD + 1):
        high_f = np.concatenate([high[f:], np.full(f, np.nan)])
        low_f = np.concatenate([low[f:], np.full(f, np.nan)])
        cond = ((low_f <= limit_price) if direction == 'long' else (high_f >= limit_price)) & np.isnan(fill_offset)
        fill_offset[cond] = f

    filled = ~np.isnan(fill_offset)
    entry_price = np.where(filled, limit_price, np.nan)
    if direction == 'long':
        tp_level, sl_level = entry_price * (1 + tp_move), entry_price * (1 - sl_move)
    else:
        tp_level, sl_level = entry_price * (1 - tp_move), entry_price * (1 + sl_move)

    k_start = np.where(filled, fill_offset, HORIZON + 1).astype(float)
    tp_hit_at = np.full(n, np.nan)
    sl_hit_at = np.full(n, np.nan)
    for k in range(1, HORIZON + 1):
        active = filled & (k >= k_start)
        if not active.any():
            continue
        high_k = np.concatenate([high[k:], np.full(k, np.nan)])
        low_k = np.concatenate([low[k:], np.full(k, np.nan)])
        if direction == 'long':
            tp_cond = active & (high_k >= tp_level) & np.isnan(tp_hit_at)
            sl_cond = active & (low_k <= sl_level) & np.isnan(sl_hit_at)
        else:
            tp_cond = active & (low_k <= tp_level) & np.isnan(tp_hit_at)
            sl_cond = active & (high_k >= sl_level) & np.isnan(sl_hit_at)
        tp_hit_at[tp_cond] = k
        sl_hit_at[sl_cond] = k

    outcome_move = np.full(n, np.nan)
    exit_offset = np.full(n, np.nan)  # bars from entry (row i) to resolution
    both = ~np.isnan(tp_hit_at) & ~np.isnan(sl_hit_at)
    tp_only = ~np.isnan(tp_hit_at) & np.isnan(sl_hit_at)
    sl_only = np.isnan(tp_hit_at) & ~np.isnan(sl_hit_at)
    tp_first = both & (tp_hit_at < sl_hit_at)
    sl_first_or_tie = both & (tp_hit_at >= sl_hit_at)
    outcome_move[tp_only | tp_first] = tp_move[tp_only | tp_first]
    outcome_move[sl_only | sl_first_or_tie] = -sl_move[sl_only | sl_first_or_tie]
    exit_offset[tp_only | tp_first] = tp_hit_at[tp_only | tp_first]
    exit_offset[sl_only | sl_first_or_tie] = sl_hit_at[sl_only | sl_first_or_tie]
    filled_no_touch = filled & np.isnan(tp_hit_at) & np.isnan(sl_hit_at)
    outcome_move[filled_no_touch] = 0.0
    exit_offset[filled_no_touch] = HORIZON

    return pd.DataFrame({'filled': filled, 'exit_offset': exit_offset, 'realized_move': outcome_move}, index=df.index)


def predict_with_threshold(clf, X, threshold):
    proba = clf.predict_proba(X)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred = classes[max_idx].copy()
    return np.where(max_proba >= threshold, pred, 'CASH')


def portfolio_backtest(timestamps: pd.Series, pred: np.ndarray, long_sim: pd.DataFrame,
                        short_sim: pd.DataFrame, cap: int, start_equity: float = 1.0) -> dict:
    """Event-driven, capital-capped, COMPOUNDING: at most `cap` positions open at once. Each new
    accepted position is sized at (current equity / cap) -- using equity AT THE MOMENT of entry,
    which only changes when a position actually settles (closes), so gains/losses genuinely
    compound over the 42-day run instead of being pinned to a fixed original-capital fraction.
    Signals are rejected outright once `cap` slots are already open (real capacity limit, not an
    approximation)."""
    ts = timestamps.to_numpy()
    is_long = pred == 'LONG'
    is_short = pred == 'SHORT'
    signal_idx = np.flatnonzero(is_long | is_short)

    open_heap = []  # (exit_time, notional, net_return) -- settled into equity when popped
    equity = start_equity
    accepted = 0
    rejected_cap = 0
    unfilled = 0
    accepted_returns = []
    equity_curve = []  # (timestamp, equity) sampled at each settlement

    def settle_up_to(t):
        nonlocal equity
        while open_heap and open_heap[0][0] <= t:
            _, notional, net = heapq.heappop(open_heap)
            equity += notional * net
            equity_curve.append((str(t), equity))

    for i in signal_idx:
        entry_t = ts[i]
        settle_up_to(entry_t)
        if len(open_heap) >= cap:
            rejected_cap += 1
            continue
        sim = long_sim if is_long[i] else short_sim
        if not bool(sim.iloc[i]['filled']):
            unfilled += 1
            continue
        exit_offset_min = float(sim.iloc[i]['exit_offset'])
        exit_t = entry_t + np.timedelta64(int(exit_offset_min), 'm')
        realized_move = float(sim.iloc[i]['realized_move'])
        net = realized_move - ROUND_TRIP_FEE
        notional = equity / cap
        heapq.heappush(open_heap, (exit_t, notional, net))
        accepted += 1
        accepted_returns.append(net)

    # settle anything still open at the very end
    settle_up_to(ts[-1] + np.timedelta64(HORIZON + 5, 'm'))

    n_signals = len(signal_idx)
    hit_rate = float(np.mean([r > 0 for r in accepted_returns])) if accepted_returns else None

    # max drawdown computed on the FULL settlement-event equity curve (not downsampled -- a
    # sparse sample can smooth over and hide most of the actual peak-to-trough moves)
    peak = start_equity
    max_dd = 0.0
    for _, v in equity_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak if peak > 0 else 0.0)

    return {
        'cap': cap, 'n_signals': n_signals, 'n_accepted': accepted,
        'n_rejected_cap_full': rejected_cap, 'n_unfilled': unfilled,
        'hit_rate': hit_rate, 'final_equity': equity,
        'portfolio_return_pct': (equity - start_equity) * 100,
        'max_drawdown_pct': max_dd * 100,
        'n_settlement_events': len(equity_curve),
        'equity_curve_sample': equity_curve[::max(1, len(equity_curve) // 200)],
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
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
    proba = clf.predict_proba(X_oos)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    raw_pred = classes[max_idx]

    oos_idx = oos['orig_idx'].to_numpy()
    long_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True)
    short_sim_oos = short_sim.loc[oos_idx].reset_index(drop=True)

    print("\n--- Single-window OOS: threshold x cap grid ---")
    grid = []
    for thr in THRESHOLDS:
        pred = np.where(max_proba >= thr, raw_pred, 'CASH')
        for cap in CAPS:
            res = portfolio_backtest(oos['timestamp'], pred, long_sim_oos, short_sim_oos, cap)
            res['threshold'] = thr
            grid.append(res)
            print(f"  thr={thr} cap={cap}: accepted={res['n_accepted']:,}/{res['n_signals']:,} "
                  f"(rejected_full={res['n_rejected_cap_full']:,}) hit_rate={res['hit_rate']} "
                  f"return={res['portfolio_return_pct']:.3f}% max_dd={res['max_drawdown_pct']:.2f}% "
                  f"final_equity={res['final_equity']:.4f}")

    result_single = {'oos_grid': grid}
    with open(os.path.join(REPORT_DIR, 'scalp_1m_portfolio_capped_20260717.json'), 'w') as f:
        json.dump(result_single, f, indent=2, default=str)
    print("\nSaved scalp_1m_portfolio_capped_20260717.json (single-window grid)")


if __name__ == '__main__':
    main()

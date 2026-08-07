"""Combines lever 2 (confidence-threshold filtering) with a REALISTIC maker-entry fill
simulation, replacing lever 3's naive fee-number substitution
(tune_scalp_1m_levers_20260716.py's MAKER result, which just recomputed the same market-order
predictions at a lower fee and openly flagged that this ignores fill risk / adverse selection).

Fill model (entry only -- exit stays a guaranteed taker fill, since you need certainty to honor
TP/SL risk management):
  - On a predicted LONG signal at bar i, place a resting limit buy at open[i+1]*(1-OFFSET)
    (OFFSET=1bp, a modest passive improvement over the entry-bar open). Mirror for SHORT (limit
    sell at open[i+1]*(1+OFFSET)).
  - The order is live for FILL_LOOKAHEAD=3 bars (i+1..i+3). It fills the first bar whose
    low (LONG) / high (SHORT) crosses the limit price; if never crossed within the window, the
    order is cancelled -- a "miss" (no trade, no cost, but also no chance at the signal's edge).
  - If filled at bar i+f, the position's TP/SL levels are computed from the ACTUAL fill price
    (the limit level), not the original open -- and the remaining triple-barrier scan only has
    HORIZON-f bars left (the total holding budget stays anchored to the original i+1+HORIZON
    boundary, it doesn't reset just because entry was delayed a few minutes). This is what
    captures adverse selection: fills happen precisely when price already moved toward your
    limit, i.e. partway toward the stop side of a momentum-continuation setup, and the
    lost time reduces how much of the horizon is left for the trade to work.
  - Round-trip fee: maker (0.02%) on entry (you supplied liquidity) + taker (0.045%) on exit
    (market order to lock in TP/SL) = 0.065% round trip, vs. 0.09% taker/taker in the base run.

Confidence threshold is swept on val using this realistic PnL (not reusing lever 2's
taker-market threshold, since the optimal cutoff may differ under a different fill/cost model),
then applied unchanged to OOS.

Output: data/ensemble/reports/scalp_1m_tune_maker_realistic_20260716.json
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

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for, split_by_date
from tune_scalp_1m_levers_20260716 import fit_model

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

HORIZON = 20
OFFSET = 0.0001  # 1bp passive limit placement below/above entry-bar open
FILL_LOOKAHEAD = 3  # bars the resting limit order stays live before cancel
MAKER_FEE = 0.0002
TAKER_FEE = 0.00045
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE  # maker entry + taker exit
CONF_THRESHOLDS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60]


def _first_touch_from(high: np.ndarray, low: np.ndarray, tp_level: np.ndarray, sl_level: np.ndarray,
                       direction: str, k_start_per_row: np.ndarray, k_max: int, active: np.ndarray):
    """Like the label script's first-touch scan, but each row's scan only starts at its own
    k_start_per_row (the fill offset) instead of uniformly at k=1, and only rows in `active` are
    considered (others left as NaN)."""
    n = len(high)
    tp_hit_at = np.full(n, np.nan)
    sl_hit_at = np.full(n, np.nan)
    for k in range(1, k_max + 1):
        row_active = active & (k >= k_start_per_row)
        if not row_active.any():
            continue
        high_k = np.concatenate([high[k:], np.full(k, np.nan)])
        low_k = np.concatenate([low[k:], np.full(k, np.nan)])
        if direction == 'long':
            tp_cond = row_active & (high_k >= tp_level) & np.isnan(tp_hit_at)
            sl_cond = row_active & (low_k <= sl_level) & np.isnan(sl_hit_at)
        else:
            tp_cond = row_active & (low_k <= tp_level) & np.isnan(tp_hit_at)
            sl_cond = row_active & (high_k >= sl_level) & np.isnan(sl_hit_at)
        tp_hit_at[tp_cond] = k
        sl_hit_at[sl_cond] = k
    return tp_hit_at, sl_hit_at


def simulate_maker_fills(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Returns per-row: filled (bool), fill_offset, realized_pnl_move (TP move, -SL move, or NaN
    if never filled), for the given direction, using the maker-entry fill model."""
    n = len(df)
    open_ = df['open'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    tp_move = df['scalp_tp_move'].to_numpy()
    sl_move = df['scalp_sl_move'].to_numpy()

    entry_open = np.concatenate([open_[1:], [np.nan]])  # open[i+1]
    if direction == 'long':
        limit_price = entry_open * (1 - OFFSET)
    else:
        limit_price = entry_open * (1 + OFFSET)

    fill_offset = np.full(n, np.nan)
    for f in range(1, FILL_LOOKAHEAD + 1):
        high_f = np.concatenate([high[f:], np.full(f, np.nan)])
        low_f = np.concatenate([low[f:], np.full(f, np.nan)])
        if direction == 'long':
            cond = (low_f <= limit_price) & np.isnan(fill_offset)
        else:
            cond = (high_f >= limit_price) & np.isnan(fill_offset)
        fill_offset[cond] = f

    filled = ~np.isnan(fill_offset)
    entry_price = np.where(filled, limit_price, np.nan)
    if direction == 'long':
        tp_level = entry_price * (1 + tp_move)
        sl_level = entry_price * (1 - sl_move)
    else:
        tp_level = entry_price * (1 - tp_move)
        sl_level = entry_price * (1 + sl_move)

    k_start = np.where(filled, fill_offset, HORIZON + 1).astype(float)
    tp_hit_at, sl_hit_at = _first_touch_from(high, low, tp_level, sl_level, direction, k_start,
                                              HORIZON, filled)

    outcome_move = np.full(n, np.nan)
    both = ~np.isnan(tp_hit_at) & ~np.isnan(sl_hit_at)
    tp_only = ~np.isnan(tp_hit_at) & np.isnan(sl_hit_at)
    sl_only = np.isnan(tp_hit_at) & ~np.isnan(sl_hit_at)
    tp_first = both & (tp_hit_at < sl_hit_at)
    sl_first_or_tie = both & (tp_hit_at >= sl_hit_at)

    outcome_move[tp_only | tp_first] = tp_move[tp_only | tp_first]
    outcome_move[sl_only | sl_first_or_tie] = -sl_move[sl_only | sl_first_or_tie]
    # filled but never hit TP or SL within remaining horizon -> flat at horizon end (rare given
    # SL/TP bounds are tight relative to 1m ATR); treat as a wash (0 move) rather than guessing.
    filled_no_touch = filled & np.isnan(tp_hit_at) & np.isnan(sl_hit_at)
    outcome_move[filled_no_touch] = 0.0

    return pd.DataFrame({
        'filled': filled,
        'fill_offset': fill_offset,
        'realized_move': outcome_move,
    }, index=df.index)


def backtest_maker(pred_action: np.ndarray, long_sim: pd.DataFrame, short_sim: pd.DataFrame) -> dict:
    is_long = pred_action == 'LONG'
    is_short = pred_action == 'SHORT'
    n_signals = int(is_long.sum() + is_short.sum())
    if n_signals == 0:
        return {'n_signals': 0, 'n_filled': 0, 'fill_rate': None, 'hit_rate': None,
                'avg_pnl_pct': None, 'total_pnl_pct': None}

    filled_mask = np.zeros(len(pred_action), dtype=bool)
    move = np.full(len(pred_action), np.nan)
    filled_mask[is_long] = long_sim['filled'].to_numpy()[is_long]
    move[is_long] = long_sim['realized_move'].to_numpy()[is_long]
    filled_mask[is_short] = short_sim['filled'].to_numpy()[is_short]
    move[is_short] = short_sim['realized_move'].to_numpy()[is_short]

    n_filled = int(filled_mask.sum())
    if n_filled == 0:
        return {'n_signals': n_signals, 'n_filled': 0, 'fill_rate': 0.0, 'hit_rate': None,
                'avg_pnl_pct': None, 'total_pnl_pct': None}

    realized = move[filled_mask]
    net_pnl = realized - ROUND_TRIP_FEE
    hit = realized > 0
    return {
        'n_signals': n_signals,
        'n_filled': n_filled,
        'fill_rate': n_filled / n_signals,
        'hit_rate': float(hit.mean()),
        'avg_pnl_pct': float(net_pnl.mean()),
        'total_pnl_pct': float(net_pnl.sum()),
    }


def predict_with_threshold(clf, X: pd.DataFrame, threshold: float) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred = classes[max_idx].copy()
    return np.where(max_proba >= threshold, pred, 'CASH')


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + labels (Experiment A: full-history, price-only)...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    print("Simulating maker-entry fills (long + short, vectorized)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')
    overall_fill_rate_long = long_sim['filled'].mean()
    overall_fill_rate_short = short_sim['filled'].mean()
    print(f"  Baseline fill rate (all rows, not just predicted signals): "
          f"long={overall_fill_rate_long:.1%} short={overall_fill_rate_short:.1%}")

    feat_cols = feature_cols_for(df, [])
    train, val, oos = split_by_date(df, '2026-04-30', '2026-05-31', '2026-07-12')
    print(f"Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")
    clf = fit_model(train, feat_cols)

    val_idx = val.index
    oos_idx = oos.index
    long_sim_val, short_sim_val = long_sim.loc[val_idx].reset_index(drop=True), short_sim.loc[val_idx].reset_index(drop=True)
    long_sim_oos, short_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True), short_sim.loc[oos_idx].reset_index(drop=True)

    X_val = val[feat_cols].fillna(0.0)
    print("\nSweeping confidence threshold on val (realistic maker-fill PnL)...")
    sweep = []
    for thr in CONF_THRESHOLDS:
        pred_val = predict_with_threshold(clf, X_val, thr)
        bt = backtest_maker(pred_val, long_sim_val, short_sim_val)
        sweep.append({'threshold': thr, **bt})
        print(f"  [val thr={thr}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

    viable = [s for s in sweep if s['n_filled'] and s['n_filled'] >= 20]
    best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else sweep[0]
    best_thr = best['threshold']
    print(f"-> best val threshold: {best_thr}")

    X_oos = oos[feat_cols].fillna(0.0)
    pred_oos = predict_with_threshold(clf, X_oos, best_thr)
    bt_oos = backtest_maker(pred_oos, long_sim_oos, short_sim_oos)
    print(f"[oos @ thr={best_thr}] signals={bt_oos['n_signals']:,} filled={bt_oos['n_filled']:,} "
          f"fill_rate={bt_oos['fill_rate']} hit_rate={bt_oos['hit_rate']} total_pnl_pct={bt_oos['total_pnl_pct']}")

    result = {
        'experiment': 'A_conf_plus_realistic_maker_fill',
        'fill_model': {
            'offset_bp': OFFSET * 10000, 'fill_lookahead_bars': FILL_LOOKAHEAD,
            'maker_fee': MAKER_FEE, 'taker_fee_exit': TAKER_FEE, 'round_trip_fee': ROUND_TRIP_FEE,
            'baseline_fill_rate_long': float(overall_fill_rate_long),
            'baseline_fill_rate_short': float(overall_fill_rate_short),
        },
        'val_sweep': sweep,
        'chosen_threshold': best_thr,
        'oos_at_chosen_threshold': bt_oos,
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Realistic maker-entry simulation: limit order at 1bp passive offset, 3-bar fill '
                 'window, cancels (no trade) if unfilled. TP/SL computed from actual fill price, '
                 'remaining horizon budget reduced by fill delay. Exit assumed taker (guaranteed). '
                 'This replaces tune_scalp_1m_levers_20260716.py\'s naive maker-fee substitution, '
                 'which ignored fill risk / adverse selection entirely.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_tune_maker_realistic_20260716.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved scalp_1m_tune_maker_realistic_20260716.json")


if __name__ == '__main__':
    main()

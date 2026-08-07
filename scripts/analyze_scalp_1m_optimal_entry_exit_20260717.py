"""What does a genuinely optimal 1m scalp entry/exit actually look like? Extends the DP oracle
recursion (build_scalp_1m_dp_labels_20260716.py, which only kept the entry decision p_flat) to
also recover the REALIZED exit timing per trade -- the DP value function already encodes the
optimal exit policy (p_long/p_short: HOLD vs EXIT at each age), this script keeps those tables
and forward-simulates each oracle entry to find when it actually exits and what it earns.

Three analyses:
  1. FEATURE PROFILE -- for a curated set of interpretable features, compare their distribution
     at oracle LONG/SHORT entry bars vs. all bars, to characterize what a genuinely good entry
     looks like (not what the HGB classifier learned to associate with one -- the DP oracle has
     no model, it's the cost-aware truth given perfect foresight).
  2. HOLD-TIME PROFILE -- histogram of realized exit offset (bars) for oracle trades, split by
     LONG/SHORT -- answers "how long should an optimal scalp actually be held."
  3. OVERLAP WITH OUR MODEL -- within the OOS window, what fraction of oracle-optimal entry bars
     does the deployed HGB model (threshold=0.55) actually fire on (coverage), and what fraction
     of the model's fired signals land on bars the oracle also considered worth entering
     (precision-vs-oracle) -- distinguishes "the model is finding real opportunities" from "the
     model is trading on noise the oracle wouldn't have touched."

Output: data/ensemble/reports/scalp_1m_optimal_entry_exit_analysis_20260717.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for
from tune_scalp_1m_levers_20260716 import fit_model
from simulate_maker_entry_scalp_1m_20260716 import LABELS_CSV

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

MAX_AGE = 60
NOTIONAL = 1.0
ENTRY_COST = 0.0002
EXIT_COST = 0.00045
HOLD_PENALTY = 0.0000005
MIN_ENTRY_EDGE = 0.00005

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
MODEL_THRESHOLD = 0.55

PROFILE_FEATURES = [
    'chop_index', 'hurst_48', 'garch_vol_z', 'rsi', 'bb_width_z', 'atr_pct_rank_288',
    'taker_acceleration', 'whale_retail_ratio', 'cvd_slope_12', 'compression_score',
    'mtf_trend_1h', 'trade_intensity', 'net_taker_ratio', 'volatility_z', 'ofi_acceleration',
]


@njit
def _dp_recursion_full(next_ret: np.ndarray, max_age: int, notional: float, entry_cost: float,
                        exit_cost: float, hold_penalty: float, min_entry_edge: float):
    n = len(next_ret)
    v_flat = np.zeros(n + 1, dtype=np.float64)
    v_long = np.zeros((n + 1, max_age + 2), dtype=np.float64)
    v_short = np.zeros((n + 1, max_age + 2), dtype=np.float64)
    p_flat = np.zeros(n, dtype=np.int8)
    p_long = np.zeros((n, max_age + 1), dtype=np.int8)
    p_short = np.zeros((n, max_age + 1), dtype=np.int8)

    for i in range(n - 2, -1, -1):
        ret = next_ret[i] * notional
        cash_v = v_flat[i + 1]
        enter_long = -entry_cost + ret - hold_penalty + v_long[i + 1, 1]
        enter_short = -entry_cost - ret - hold_penalty + v_short[i + 1, 1]
        best = 0
        best_v = cash_v
        if enter_long > best_v:
            best = 1
            best_v = enter_long
        if enter_short > best_v:
            best = 2
            best_v = enter_short
        if best != 0 and (best_v - cash_v) < min_entry_edge:
            best = 0
            best_v = cash_v
        p_flat[i] = best
        v_flat[i] = best_v
        for age in range(max_age, 0, -1):
            exit_v = -exit_cost + v_flat[i + 1]
            if age >= max_age:
                v_long[i, age] = exit_v
                v_short[i, age] = exit_v
                p_long[i, age] = 1
                p_short[i, age] = 1
                continue
            hold_long = ret - hold_penalty + v_long[i + 1, age + 1]
            hold_short = -ret - hold_penalty + v_short[i + 1, age + 1]
            if exit_v >= hold_long:
                v_long[i, age] = exit_v
                p_long[i, age] = 1
            else:
                v_long[i, age] = hold_long
            if exit_v >= hold_short:
                v_short[i, age] = exit_v
                p_short[i, age] = 1
            else:
                v_short[i, age] = hold_short

    return p_flat, p_long, p_short


@njit
def _simulate_exits(close: np.ndarray, p_flat: np.ndarray, p_long: np.ndarray, p_short: np.ndarray,
                     max_age: int):
    n = len(close)
    exit_offset = np.full(n, -1, dtype=np.int32)
    realized_ret = np.zeros(n, dtype=np.float64)
    for i in range(n - max_age - 2):
        side = p_flat[i]
        if side == 0:
            continue
        entry_i = i + 1
        if entry_i >= n:
            continue
        entry_px = close[entry_i]
        age = 1
        j = entry_i
        while j < n - 1 and age <= max_age:
            act = p_long[j, age] if side == 1 else p_short[j, age]
            if act == 1:
                break
            j += 1
            age += 1
        exit_px = close[j]
        exit_offset[i] = j - entry_i
        realized_ret[i] = (exit_px - entry_px) / entry_px if side == 1 else (entry_px - exit_px) / entry_px
    return exit_offset, realized_ret


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading 1m ETH data...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    n = len(df)
    close = df['close'].to_numpy(dtype=np.float64)
    next_ret = np.zeros(n, dtype=np.float64)
    next_ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    print("Running full DP recursion (keeping exit policy tables this time)...")
    p_flat, p_long, p_short = _dp_recursion_full(next_ret, MAX_AGE, NOTIONAL, ENTRY_COST, EXIT_COST,
                                                  HOLD_PENALTY, MIN_ENTRY_EDGE)

    print("Forward-simulating realized exits for every oracle entry...")
    exit_offset, realized_ret = _simulate_exits(close, p_flat, p_long, p_short, MAX_AGE)

    action = pd.Series(p_flat).map({0: 'CASH', 1: 'LONG', 2: 'SHORT'})
    df['dp_action'] = action
    df['dp_exit_offset'] = exit_offset
    df['dp_realized_ret'] = realized_ret
    has_full_horizon = np.arange(n) < (n - MAX_AGE - 2)
    df['dp_has_full_horizon'] = has_full_horizon

    trades = df[has_full_horizon & (df['dp_action'] != 'CASH')]
    print(f"\nOracle trades: {len(trades):,} ({len(trades) / has_full_horizon.sum():.1%} of usable bars)")

    # ---------------- Analysis 1: feature profile ----------------
    print("\n--- Analysis 1: feature profile at oracle entries vs all bars ---")
    all_bars = df[has_full_horizon]
    profile = {}
    for feat in PROFILE_FEATURES:
        if feat not in df.columns:
            continue
        base_mean, base_std = all_bars[feat].mean(), all_bars[feat].std()
        long_mean = trades.loc[trades['dp_action'] == 'LONG', feat].mean()
        short_mean = trades.loc[trades['dp_action'] == 'SHORT', feat].mean()
        long_z = (long_mean - base_mean) / base_std if base_std > 0 else 0.0
        short_z = (short_mean - base_mean) / base_std if base_std > 0 else 0.0
        profile[feat] = {
            'all_mean': float(base_mean), 'long_mean': float(long_mean), 'short_mean': float(short_mean),
            'long_z': float(long_z), 'short_z': float(short_z),
        }
        print(f"  {feat:22s} all={base_mean:8.4f}  LONG={long_mean:8.4f} (z={long_z:+.2f})  "
              f"SHORT={short_mean:8.4f} (z={short_z:+.2f})")

    # ---------------- Analysis 2: hold-time profile ----------------
    print("\n--- Analysis 2: realized hold-time distribution ---")
    hold_stats = {}
    for side in ['LONG', 'SHORT']:
        side_trades = trades[trades['dp_action'] == side]
        q = side_trades['dp_exit_offset'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        hold_stats[side] = {str(k): float(v) for k, v in q.items()}
        print(f"  {side}: median={q[0.5]:.0f} bars, p25={q[0.25]:.0f}, p75={q[0.75]:.0f}, "
              f"p90={q[0.9]:.0f}, p99={q[0.99]:.0f}  (n={len(side_trades):,})")
    ret_stats = {}
    for side in ['LONG', 'SHORT']:
        side_trades = trades[trades['dp_action'] == side]
        ret_stats[side] = {
            'mean_ret': float(side_trades['dp_realized_ret'].mean()),
            'median_ret': float(side_trades['dp_realized_ret'].median()),
        }
        print(f"  {side} realized return: mean={ret_stats[side]['mean_ret']:.4%} "
              f"median={ret_stats[side]['median_ret']:.4%}")

    # ---------------- Analysis 3: overlap with our deployed HGB model (OOS window) ----------------
    print("\n--- Analysis 3: overlap with deployed HGB model (OOS window) ---")
    labels = pd.read_csv(LABELS_CSV, parse_dates=['timestamp'])
    df_hgb = df.merge(labels[['timestamp', 'scalp_action', 'scalp_has_full_horizon']], on='timestamp', how='left')
    df_hgb = df_hgb[df_hgb['scalp_has_full_horizon'].fillna(False)].reset_index(drop=True)
    feat_cols = feature_cols_for(df_hgb, [])
    train = df_hgb[df_hgb['timestamp'] <= TRAIN_END]
    oos = df_hgb[(df_hgb['timestamp'] > VAL_END) & (df_hgb['timestamp'] <= OOS_END)].reset_index(drop=True)

    clf = fit_model(train, feat_cols)
    X_oos = oos[feat_cols].fillna(0.0)
    proba = clf.predict_proba(X_oos)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    hgb_pred = np.where(max_proba >= MODEL_THRESHOLD, classes[max_idx], 'CASH')
    oos['hgb_pred'] = hgb_pred

    oracle_entry = oos['dp_action'] != 'CASH'
    model_entry = oos['hgb_pred'] != 'CASH'
    n_oracle = int(oracle_entry.sum())
    n_model = int(model_entry.sum())
    n_both_same_side = int(((oracle_entry) & (model_entry) & (oos['dp_action'] == oos['hgb_pred'])).sum())
    n_both_any_side = int(((oracle_entry) & (model_entry)).sum())
    coverage = n_both_any_side / n_oracle if n_oracle else 0.0
    precision_vs_oracle = n_both_any_side / n_model if n_model else 0.0
    side_agreement_given_both = n_both_same_side / n_both_any_side if n_both_any_side else 0.0

    print(f"  OOS bars: {len(oos):,}")
    print(f"  Oracle-optimal entry bars: {n_oracle:,} ({n_oracle/len(oos):.1%})")
    print(f"  HGB model fired-signal bars (thr={MODEL_THRESHOLD}): {n_model:,} ({n_model/len(oos):.1%})")
    print(f"  Coverage (oracle bars the model ALSO fires on, any side): {coverage:.1%}")
    print(f"  Precision-vs-oracle (model signals landing on an oracle-entry bar): {precision_vs_oracle:.1%}")
    print(f"  Side agreement when both fire on the same bar: {side_agreement_given_both:.1%}")

    result = {
        'oracle_trade_count': int(len(trades)),
        'oracle_trade_rate': float(len(trades) / has_full_horizon.sum()),
        'feature_profile': profile,
        'hold_time_quantiles_bars': hold_stats,
        'realized_return_stats': ret_stats,
        'overlap_with_hgb_model_oos': {
            'n_oos_bars': int(len(oos)),
            'n_oracle_entry_bars': n_oracle,
            'n_model_signal_bars': n_model,
            'coverage_oracle_bars_model_also_fires': coverage,
            'precision_model_signals_on_oracle_bars': precision_vs_oracle,
            'side_agreement_given_both_fire': side_agreement_given_both,
        },
        'dp_params': {'MAX_AGE': MAX_AGE, 'ENTRY_COST': ENTRY_COST, 'EXIT_COST': EXIT_COST,
                       'HOLD_PENALTY': HOLD_PENALTY, 'MIN_ENTRY_EDGE': MIN_ENTRY_EDGE},
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_optimal_entry_exit_analysis_20260717.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_optimal_entry_exit_analysis_20260717.json")

    # also save the full per-bar oracle trade table for downstream charting
    trades_out = trades[['timestamp', 'close', 'dp_action', 'dp_exit_offset', 'dp_realized_ret']].copy()
    trades_out.to_csv(os.path.join(DATA_DIR, 'training_features_1m_dp_trades_with_exits.csv'), index=False)
    print(f"Saved oracle trade table: {len(trades_out):,} rows")


if __name__ == '__main__':
    main()

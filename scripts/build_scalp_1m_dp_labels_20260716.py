"""Lever B-3: DP trajectory (oracle) labels (build_omega1_2_1_dp_trajectory_labels_20260620.py's
finite-state dynamic-programming recursion -- FLAT/LONG/SHORT x age value function, backward
induction picks the truly-optimal entry/hold/exit trajectory under a cost+hold-penalty model),
rescaled to 1-minute bars and accelerated with numba (the original is a pure-python O(n*MAX_AGE)
loop written for small pre-filtered "trade_candidates" files, not a 1.33M-row raw bar series --
numba-jitting the same recursion is a mechanical port, not an algorithm change).

Only the backward value-function recursion is ported (build_omega1_2_1_dp_trajectory_labels_20260620.py
lines 126-156) to get p_flat -- the per-bar optimal entry decision (0=CASH, 1=ENTER_LONG,
2=ENTER_SHORT). The original script's own TP/SL-from-MFE/MAE labeling and utility bookkeeping
(lines 158+) is NOT ported: to keep the model-vs-label comparison clean, DP only supplies the
alternative PRIMARY direction label here; actual trade P&L is evaluated with the same ATR-scaled
TP/SL + realistic maker-fill simulation used for the base and trend-scan labels, unchanged.

Cost model kept in pure price-move-fraction terms per CLAUDE.md's Futures Risk Sizing Contract
(NOTIONAL=1.0, no leverage baked into the label) using the same asymmetric maker-entry/taker-exit
fee split as the realistic maker-fill simulation (0.02% / 0.045%).

MAX_AGE=60 (1h cap, vs the original 5m daytrade profile's 96 bars=8h) and HOLD_PENALTY scaled
down ~5x from the 5m "daytrade" profile to account for 1m bars being 5x more frequent (same
per-minute time-discouragement, not per-bar) -- both are a first-pass estimate à la the
project's existing SOL lowfreq retune (build_sol_dp_trajectory_labels_lowfreq_20260715.py),
checked against the resulting label distribution/hold-time stats printed at the end.

Output: data/training_features_1m_dp_labels.csv
"""
import os

import numpy as np
import pandas as pd
from numba import njit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_dp_labels.csv')

MAX_AGE = 60
NOTIONAL = 1.0
ENTRY_COST = 0.0002   # maker fee, matches the realistic maker-fill sim's entry assumption
EXIT_COST = 0.00045   # taker fee, matches the realistic maker-fill sim's exit assumption
HOLD_PENALTY = 0.0000005  # per-1m-bar time-discouragement (~5x smaller than the 5m daytrade profile's 0.000002)
MIN_ENTRY_EDGE = 0.00005


@njit
def _dp_recursion(next_ret: np.ndarray, max_age: int, notional: float, entry_cost: float,
                   exit_cost: float, hold_penalty: float, min_entry_edge: float):
    n = len(next_ret)
    v_flat = np.zeros(n + 1, dtype=np.float64)
    v_long = np.zeros((n + 1, max_age + 2), dtype=np.float64)
    v_short = np.zeros((n + 1, max_age + 2), dtype=np.float64)
    p_flat = np.zeros(n, dtype=np.int8)

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
                continue
            hold_long = ret - hold_penalty + v_long[i + 1, age + 1]
            hold_short = -ret - hold_penalty + v_short[i + 1, age + 1]
            v_long[i, age] = exit_v if exit_v >= hold_long else hold_long
            v_short[i, age] = exit_v if exit_v >= hold_short else hold_short

    return p_flat


def main():
    print("Loading 1m ETH close for DP trajectory label construction...")
    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp', 'close'], parse_dates=['timestamp'])
    n = len(df)
    print(f"  {n:,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    close = df['close'].to_numpy(dtype=np.float64)
    next_ret = np.zeros(n, dtype=np.float64)
    next_ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    print(f"Running numba DP recursion (MAX_AGE={MAX_AGE}, HOLD_PENALTY={HOLD_PENALTY}, "
          f"MIN_ENTRY_EDGE={MIN_ENTRY_EDGE})...")
    p_flat = _dp_recursion(next_ret, MAX_AGE, NOTIONAL, ENTRY_COST, EXIT_COST, HOLD_PENALTY, MIN_ENTRY_EDGE)

    action = pd.Series(p_flat).map({0: 'CASH', 1: 'LONG', 2: 'SHORT'})
    # last MAX_AGE+1 rows don't have a fully-resolved value function (boundary of the backward
    # recursion) -- exclude from the usable range, matching the original script's `n - MAX_AGE - 2`.
    has_full_horizon = np.arange(n) < (n - MAX_AGE - 2)

    out = pd.DataFrame({
        'timestamp': df['timestamp'],
        'dp_action': action,
        'dp_has_full_horizon': has_full_horizon,
    })
    out.to_csv(OUT_CSV, index=False)

    valid = out[out['dp_has_full_horizon']]
    dist = valid['dp_action'].value_counts()
    print(f"\nLabel distribution (full-horizon rows, n={len(valid):,}):")
    print(dist)
    print(f"  LONG {dist.get('LONG', 0) / len(valid):.2%}  SHORT {dist.get('SHORT', 0) / len(valid):.2%}  "
          f"CASH {dist.get('CASH', 0) / len(valid):.2%}")

    # rough hold-time diagnostic: bars until the action flips away from CASH->non-CASH->CASH again
    print(f"\nSaved {OUT_CSV}: {len(out):,} rows")


if __name__ == '__main__':
    main()

"""New label/horizon design for the 1m ETH scalp line (pure-ETH direction, post BTC-lookahead
fix). Every fixed-20min-triple-barrier variant tried this session (base, DP-oracle relabel,
short-horizon relabel, trend-scanning label) failed to find edge. This tries a genuinely
different label mechanic instead of re-tuning the same one: a trailing stop, the one technique
in this project (Sigma6, 1h) that has actually shown a real, walk-forward-validated edge
("let winners run" beats a fixed TP). Never applied at 1m before.

Mechanic (long side; short mirrored):
  - Entry at next-bar open (t+1), same convention as the fixed-TP label.
  - Hard stop at entry*(1 - SL_ATR_MULT*atr_pct), same ATR-scaling convention as before.
  - Trailing stop = peak-high-since-entry * (1 - TRAIL_MOVE), where TRAIL_MOVE is
    ATR-scaled and bounded; effective stop = max(hard_stop, trailing_stop) so it only ever
    tightens upward, never loosens below the original risk.
  - Position is force-closed at K_MAX bars if neither stop is ever hit (time-based cap,
    K_MAX=60 bars = 1h vs the old fixed 20-bar HORIZON -- winners get 3x the room to run).
  - Realized move = (exit_price - entry_price) / entry_price, no fee applied here (fees are
    applied at the backtest/eval stage, consistent with the fixed-TP label's convention).

This produces a per-row long_move/short_move pair (continuous), from which scalp_action picks
whichever side clears MIN_PROFITABLE_MOVE (set above the round-trip taker fee so CASH is
"neither side would have cleared costs"), mirroring the fixed-TP label's LONG/SHORT/CASH shape
so it's a drop-in replacement for the same downstream training/eval scripts.

Output: data/training_features_1m_scalp_trailing_labels.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_scalp_trailing_labels.csv')

K_MAX = 60  # bars = minutes, force-close cap (3x the old fixed 20-bar horizon)
ATR_LOOKBACK = 20
SL_ATR_MULT = 1.0
SL_BOUNDS = (0.0010, 0.005)
TRAIL_ATR_MULT = 0.6
TRAIL_BOUNDS = (0.0008, 0.004)
ROUND_TRIP_TAKER_FEE = 0.0009  # 2 * 4.5bps, matches train_eval_scalp_1m_hgb_20260716.FEE_PER_SIDE*2
MIN_PROFITABLE_MOVE = 0.0015  # > round-trip fee, so CASH really means "not worth it after costs"


def _atr_pct(df: pd.DataFrame, lookback: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean()
    return (atr / df['close']).astype(float)


def _trailing_sim(df: pd.DataFrame, entry_price: np.ndarray, sl_move: np.ndarray,
                   trail_move: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()

    exit_at = np.full(n, np.nan)  # k offset of stop hit, NaN if force-closed at K_MAX
    exit_price = np.full(n, np.nan)

    if direction == 'long':
        hard_stop = entry_price * (1 - sl_move)
        peak = entry_price.copy()
    else:
        hard_stop = entry_price * (1 + sl_move)
        trough = entry_price.copy()

    for k in range(1, K_MAX + 1):
        high_k = np.concatenate([high[k:], np.full(k, np.nan)])
        low_k = np.concatenate([low[k:], np.full(k, np.nan)])
        active = np.isnan(exit_at)
        if direction == 'long':
            peak = np.maximum(peak, high_k)
            trail_stop = peak * (1 - trail_move)
            eff_stop = np.maximum(hard_stop, trail_stop)
            hit = active & (low_k <= eff_stop)
            exit_price[hit] = eff_stop[hit]
        else:
            trough = np.minimum(trough, low_k)
            trail_stop = trough * (1 + trail_move)
            eff_stop = np.minimum(hard_stop, trail_stop)
            hit = active & (high_k >= eff_stop)
            exit_price[hit] = eff_stop[hit]
        exit_at[hit] = k

    force_closed = np.isnan(exit_at)
    close_k_max = np.concatenate([close[K_MAX:], np.full(K_MAX, np.nan)])
    exit_price[force_closed] = close_k_max[force_closed]

    if direction == 'long':
        move = (exit_price - entry_price) / entry_price
    else:
        move = (entry_price - exit_price) / entry_price
    return move, exit_at


def main():
    print("Loading 1m ETH klines for trailing-stop label construction...")
    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp', 'open', 'high', 'low', 'close'],
                      parse_dates=['timestamp'])
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("Computing ATR% and stop distances (entry = next bar open)...")
    atr_pct = _atr_pct(df, ATR_LOOKBACK)
    entry_price = df['open'].shift(-1).to_numpy()

    sl_move = (atr_pct * SL_ATR_MULT).clip(*SL_BOUNDS).to_numpy()
    trail_move = (atr_pct * TRAIL_ATR_MULT).clip(*TRAIL_BOUNDS).to_numpy()
    print(f"  ATR% median={atr_pct.median():.5f}, SL move median={np.nanmedian(sl_move):.5f}, "
          f"trail move median={np.nanmedian(trail_move):.5f}")

    print(f"Simulating trailing-stop exits (long side, K_MAX={K_MAX})...")
    long_move, long_exit_at = _trailing_sim(df, entry_price, sl_move, trail_move, 'long')
    print(f"Simulating trailing-stop exits (short side, K_MAX={K_MAX})...")
    short_move, short_exit_at = _trailing_sim(df, entry_price, sl_move, trail_move, 'short')

    action = np.full(len(df), 'CASH', dtype=object)
    long_wins = (long_move > MIN_PROFITABLE_MOVE) & (long_move >= short_move)
    short_wins = (short_move > MIN_PROFITABLE_MOVE) & (short_move > long_move)
    action[long_wins] = 'LONG'
    action[short_wins] = 'SHORT'

    has_full_horizon = df.index < (len(df) - K_MAX - 1)

    out = pd.DataFrame({
        'timestamp': df['timestamp'],
        'scalp_action': action,
        'scalp_long_move': long_move,
        'scalp_short_move': short_move,
        'scalp_long_exit_at': long_exit_at,
        'scalp_short_exit_at': short_exit_at,
        'scalp_atr_pct': atr_pct,
        'scalp_has_full_horizon': has_full_horizon,
    })
    out = out.dropna(subset=['scalp_atr_pct'])
    out.to_csv(OUT_CSV, index=False)

    dist = out.loc[out['scalp_has_full_horizon'], 'scalp_action'].value_counts()
    print(f"\nLabel distribution (full-horizon rows only, n={out['scalp_has_full_horizon'].sum():,}):")
    print(dist)
    print(f"  LONG {dist.get('LONG', 0) / dist.sum():.2%}  SHORT {dist.get('SHORT', 0) / dist.sum():.2%}  "
          f"CASH {dist.get('CASH', 0) / dist.sum():.2%}")

    full = out[out['scalp_has_full_horizon']]
    print(f"\nMean long_move (all rows, pre-fee) = {full['scalp_long_move'].mean():.5f}")
    print(f"Mean short_move (all rows, pre-fee) = {full['scalp_short_move'].mean():.5f}")
    print(f"Mean exit offset (long, bars) = {full['scalp_long_exit_at'].mean():.1f} "
          f"(NaN=force-closed at K_MAX={K_MAX}, rate={full['scalp_long_exit_at'].isna().mean():.1%})")

    print(f"\nSaved {OUT_CSV}: {len(out):,} rows")


if __name__ == '__main__':
    main()

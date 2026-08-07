"""Triple-barrier action labels for the new ETH 1m scalping model.

Rescaled version of the existing multi-horizon triple-barrier convention
(build_omega1_2_1_multihorizon_tb_labels_20260620.py, HORIZONS=[12,24,48,96,192] bars @ 5m =
1h-16h) for genuine 1m scalping: a single 20-bar (20-minute) horizon with ATR-scaled TP/SL an
order of magnitude smaller than the 5m-scale bounds, matching 1m bars' much smaller typical
range.

Entry is next-bar open (t+1), avoiding use of the entry bar's own close. For each side (LONG and
SHORT independently), first-touch resolution is computed vectorized over the HORIZON offsets
(no python row loop -- this dataset is 1.3M rows, ~40x larger than what the reference script's
python loop was written for). Same-bar tie -> SL wins, matching the existing convention.

Final 3-class action label: LONG only if the long-side barrier resolves TP-first AND the
short-side barrier does NOT also resolve TP-first; mirror for SHORT; everything else (including
ambiguous cases where both sides would have hit TP, or neither hits TP) is CASH.

Labels are price-move fractions per the Futures Risk Sizing Contract (CLAUDE.md) -- no
leverage/notional applied here; that happens downstream at the backtest/sizing stage.

Output: data/training_features_1m_scalp_labels.csv (timestamp + label columns only, joins back
onto the feature CSVs by timestamp).
"""
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_scalp_labels.csv')

HORIZON = 20  # bars = minutes
ATR_LOOKBACK = 20  # bars
TP_ATR_MULT = 1.2
SL_ATR_MULT = 1.0
TP_BOUNDS = (0.0015, 0.006)
SL_BOUNDS = (0.0010, 0.005)


def _atr_pct(df: pd.DataFrame, lookback: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean()
    return (atr / df['close']).astype(float)


def _first_touch_offsets(df: pd.DataFrame, tp_level: pd.Series, sl_level: pd.Series,
                          direction: str) -> tuple[pd.Series, pd.Series]:
    """Vectorized first-touch scan: for each row, the smallest k in [1, HORIZON] at which the
    high/low HORIZON bars ahead first crosses tp_level / sl_level. NaN if never touched.
    direction='long': TP is an upward cross (high >= tp_level), SL is downward (low <= sl_level).
    direction='short': mirrored.
    """
    n = len(df)
    tp_hit_at = np.full(n, np.nan)
    sl_hit_at = np.full(n, np.nan)
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    for k in range(1, HORIZON + 1):
        high_k = np.concatenate([high[k:], np.full(k, np.nan)])
        low_k = np.concatenate([low[k:], np.full(k, np.nan)])
        if direction == 'long':
            tp_cond = (high_k >= tp_level.to_numpy()) & np.isnan(tp_hit_at)
            sl_cond = (low_k <= sl_level.to_numpy()) & np.isnan(sl_hit_at)
        else:
            tp_cond = (low_k <= tp_level.to_numpy()) & np.isnan(tp_hit_at)
            sl_cond = (high_k >= sl_level.to_numpy()) & np.isnan(sl_hit_at)
        tp_hit_at[tp_cond] = k
        sl_hit_at[sl_cond] = k
    return pd.Series(tp_hit_at, index=df.index), pd.Series(sl_hit_at, index=df.index)


def _resolve_outcome(tp_hit_at: pd.Series, sl_hit_at: pd.Series) -> pd.Series:
    """'TP' if TP strictly first, 'SL' if SL first or tied (SL wins ties), else 'NONE'."""
    out = pd.Series('NONE', index=tp_hit_at.index)
    both = tp_hit_at.notna() & sl_hit_at.notna()
    out[both & (tp_hit_at < sl_hit_at)] = 'TP'
    out[both & (tp_hit_at >= sl_hit_at)] = 'SL'  # tie -> SL wins
    out[tp_hit_at.notna() & sl_hit_at.isna()] = 'TP'
    out[sl_hit_at.notna() & tp_hit_at.isna()] = 'SL'
    return out


def main():
    print("Loading 1m ETH klines/close for label construction...")
    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp', 'open', 'high', 'low', 'close'],
                      parse_dates=['timestamp'])
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("Computing ATR% and TP/SL levels (entry = next bar open)...")
    atr_pct = _atr_pct(df, ATR_LOOKBACK)
    entry_price = df['open'].shift(-1)

    tp_move = (atr_pct * TP_ATR_MULT).clip(*TP_BOUNDS)
    sl_move = (atr_pct * SL_ATR_MULT).clip(*SL_BOUNDS)

    long_tp = entry_price * (1 + tp_move)
    long_sl = entry_price * (1 - sl_move)
    short_tp = entry_price * (1 - tp_move)
    short_sl = entry_price * (1 + sl_move)

    print(f"  ATR% median={atr_pct.median():.5f}, TP move median={tp_move.median():.5f}, "
          f"SL move median={sl_move.median():.5f}")

    print("Scanning first-touch outcomes (long side)...")
    long_tp_at, long_sl_at = _first_touch_offsets(df, long_tp, long_sl, 'long')
    long_outcome = _resolve_outcome(long_tp_at, long_sl_at)

    print("Scanning first-touch outcomes (short side)...")
    short_tp_at, short_sl_at = _first_touch_offsets(df, short_tp, short_sl, 'short')
    short_outcome = _resolve_outcome(short_tp_at, short_sl_at)

    action = pd.Series('CASH', index=df.index)
    action[(long_outcome == 'TP') & (short_outcome != 'TP')] = 'LONG'
    action[(short_outcome == 'TP') & (long_outcome != 'TP')] = 'SHORT'

    # Rows without a full forward horizon (tail of the dataset) never get a first-touch result
    # on at least one side beyond HORIZON bars out -- both tp_hit_at/sl_hit_at stay NaN there,
    # which _resolve_outcome already maps to 'NONE'/CASH, so no separate tail-drop is needed;
    # but flag them explicitly so the training script can exclude ambiguous tail rows.
    has_full_horizon = df.index < (len(df) - HORIZON - 1)

    out = pd.DataFrame({
        'timestamp': df['timestamp'],
        'scalp_action': action,
        'scalp_tp_move': tp_move,
        'scalp_sl_move': sl_move,
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
    print(f"\nSaved {OUT_CSV}: {len(out):,} rows")


if __name__ == '__main__':
    main()

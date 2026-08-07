"""Short-horizon triple-barrier labels, rescaled per the DP oracle finding
(scalp_1m_optimal_entry_exit_analysis_20260717.json): the truly optimal scalp hold time is a
median of just 2 bars (p75=5, p90=9), an order of magnitude shorter than the base label's fixed
20-bar horizon. That base horizon was chosen a priori as "a genuine scalp hold"; this one is
chosen from what the data's own cost-aware oracle actually does.

HORIZON=5 (covers oracle's p75), ATR_LOOKBACK shortened to 10 bars to match the faster time
scale, TP/SL bounds shrunk to match the oracle's realized-return scale (mean 0.17%, median
0.12% over a 2-9 bar hold) rather than the base label's 20-bar-scale bounds.

Same first-touch mechanics and conventions as build_scalp_1m_tb_labels_20260716.py (entry = next
bar open, same-bar tie -> SL wins, LONG only if long-side resolves TP-first AND short-side
doesn't also).

Output: data/training_features_1m_scalp_labels_short.csv
"""
import os

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_scalp_labels_short.csv')

HORIZON = 5  # bars = minutes -- matches DP oracle's p75 hold time (2min median, 5min p75)
ATR_LOOKBACK = 10
TP_ATR_MULT = 1.0
SL_ATR_MULT = 0.8
TP_BOUNDS = (0.0008, 0.0030)
SL_BOUNDS = (0.0005, 0.0025)


def _atr_pct(df: pd.DataFrame, lookback: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=lookback).mean()
    return (atr / df['close']).astype(float)


def _first_touch_offsets(df: pd.DataFrame, tp_level: pd.Series, sl_level: pd.Series, direction: str):
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
    out = pd.Series('NONE', index=tp_hit_at.index)
    both = tp_hit_at.notna() & sl_hit_at.notna()
    out[both & (tp_hit_at < sl_hit_at)] = 'TP'
    out[both & (tp_hit_at >= sl_hit_at)] = 'SL'
    out[tp_hit_at.notna() & sl_hit_at.isna()] = 'TP'
    out[sl_hit_at.notna() & tp_hit_at.isna()] = 'SL'
    return out


def main():
    print("Loading 1m ETH klines/close for short-horizon label construction...")
    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp', 'open', 'high', 'low', 'close'],
                      parse_dates=['timestamp'])
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

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

    has_full_horizon = np.asarray(df.index < (len(df) - HORIZON - 1))

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

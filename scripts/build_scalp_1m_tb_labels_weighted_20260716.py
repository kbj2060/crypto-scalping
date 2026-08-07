"""Adds Lopez de Prado-style concurrency/uniqueness sample weights to the base triple-barrier
labels (build_scalp_1m_tb_labels_20260716.py), as lever B-1 of the user-approved label-first
research plan (2026-07-16).

Motivation: the base labels use a 20-bar forward window on every single 1-minute bar, so
consecutive labels overlap ~95% -- the classifier's effective sample size is far smaller than
1.3M rows, and it can become overconfident exactly where labels are most redundant. Standard fix
(Advances in Financial Machine Learning ch.4): weight each training sample by its *average
uniqueness* over its own lifespan, using the label's actual RESOLUTION time (first-touch bar),
not the full horizon -- a label that resolved in 2 bars overlaps far fewer neighbors than one
that took the full 20 bars.

Steps:
  1. Re-run the base label logic but keep the winning side's first-touch offset per row (dropped
     in the original script, which only kept the final action). CASH rows (no side hit TP first)
     get resolution_offset = HORIZON (the full window was "used up" without resolving).
  2. Build each row's lifespan [entry_i, entry_i + resolution_offset_i] as a sweep-line interval,
     compute concurrency (how many other labels' lifespans cover the same bar) via a diff-array,
     then average uniqueness = mean(1/concurrency) over each row's own lifespan.
  3. Weight = average uniqueness, rescaled to mean 1.0 over full-horizon rows (so it plugs into
     sklearn's sample_weight without changing the effective total weight magnitude).

Output: data/training_features_1m_scalp_labels_weighted.csv (all base label columns +
scalp_resolution_offset, scalp_uniqueness_weight).
"""
import os

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_scalp_labels_weighted.csv')

HORIZON = 20
ATR_LOOKBACK = 20
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


def _resolution_offset(tp_hit_at: pd.Series, sl_hit_at: pd.Series, outcome: pd.Series) -> np.ndarray:
    """The bar offset at which the outcome was actually determined (min of whichever fired), or
    HORIZON if neither fired (the full window was consumed without a first-touch event)."""
    both_min = pd.concat([tp_hit_at, sl_hit_at], axis=1).min(axis=1)
    offset = np.where(outcome == 'NONE', float(HORIZON), both_min.to_numpy())
    return offset


def _concurrency_weights(entry_idx: np.ndarray, resolution_offset: np.ndarray, n_rows: int) -> np.ndarray:
    """Sweep-line concurrency: label i covers absolute rows [entry_idx[i], entry_idx[i]+resolution_offset[i]].
    Returns per-row average uniqueness = mean(1/concurrency) over each label's own span."""
    span_end = entry_idx + resolution_offset.astype(int)
    # diff array over an index range padded past n_rows so span_end never overflows
    pad = HORIZON + 2
    diff = np.zeros(n_rows + pad, dtype=np.int64)
    np.add.at(diff, entry_idx, 1)
    np.add.at(diff, span_end + 1, -1)
    concurrency = np.cumsum(diff)[:n_rows + pad]
    concurrency = np.maximum(concurrency, 1)  # guard div-by-zero (shouldn't occur, each label covers itself)
    inv_conc = 1.0 / concurrency

    # average 1/concurrency over each row's own [entry_idx, span_end] span, vectorized over offsets
    max_span = int(resolution_offset.max()) + 1
    acc = np.zeros(len(entry_idx))
    cnt = np.zeros(len(entry_idx))
    for k in range(max_span):
        active = k <= resolution_offset
        if not active.any():
            continue
        pos = entry_idx[active] + k
        acc[active] += inv_conc[pos]
        cnt[active] += 1
    avg_uniqueness = acc / np.maximum(cnt, 1)
    return avg_uniqueness


def main():
    print("Loading 1m ETH klines/close for weighted label construction...")
    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp', 'open', 'high', 'low', 'close'],
                      parse_dates=['timestamp'])
    n = len(df)
    print(f"  {n:,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    atr_pct = _atr_pct(df, ATR_LOOKBACK)
    entry_price = df['open'].shift(-1)
    tp_move = (atr_pct * TP_ATR_MULT).clip(*TP_BOUNDS)
    sl_move = (atr_pct * SL_ATR_MULT).clip(*SL_BOUNDS)

    long_tp = entry_price * (1 + tp_move)
    long_sl = entry_price * (1 - sl_move)
    short_tp = entry_price * (1 - tp_move)
    short_sl = entry_price * (1 + sl_move)

    print("Scanning first-touch outcomes...")
    long_tp_at, long_sl_at = _first_touch_offsets(df, long_tp, long_sl, 'long')
    long_outcome = _resolve_outcome(long_tp_at, long_sl_at)
    short_tp_at, short_sl_at = _first_touch_offsets(df, short_tp, short_sl, 'short')
    short_outcome = _resolve_outcome(short_tp_at, short_sl_at)

    action = pd.Series('CASH', index=df.index)
    action[(long_outcome == 'TP') & (short_outcome != 'TP')] = 'LONG'
    action[(short_outcome == 'TP') & (long_outcome != 'TP')] = 'SHORT'

    # Resolution offset per row's FINAL action: the winning side's own outcome. For LONG, use the
    # long-side's resolution (long_tp_at if TP, else the min touch offset that produced 'TP'); we
    # already know the action is LONG only when long_outcome=='TP', so resolution = long_tp_at.
    # Mirror for SHORT. CASH rows use the full HORIZON (ambiguous / no resolving touch).
    resolution_offset = np.full(n, float(HORIZON))
    is_long = (action == 'LONG').to_numpy()
    is_short = (action == 'SHORT').to_numpy()
    resolution_offset[is_long] = long_tp_at.to_numpy()[is_long]
    resolution_offset[is_short] = short_tp_at.to_numpy()[is_short]
    resolution_offset = np.nan_to_num(resolution_offset, nan=float(HORIZON))
    resolution_offset = np.clip(resolution_offset, 1, HORIZON)

    has_full_horizon = np.asarray(df.index < (n - HORIZON - 1))
    entry_idx = (df.index + 1).to_numpy()  # absolute row of entry bar (i+1)

    print("Computing concurrency / average-uniqueness sample weights...")
    valid = has_full_horizon & atr_pct.notna().to_numpy()
    weights = np.full(n, np.nan)
    weights[valid] = _concurrency_weights(entry_idx[valid], resolution_offset[valid], n)
    # rescale to mean 1.0 over valid rows so it plugs into sample_weight without changing overall
    # loss magnitude vs. the unweighted baseline.
    mean_w = np.nanmean(weights)
    weights = weights / mean_w

    out = pd.DataFrame({
        'timestamp': df['timestamp'],
        'scalp_action': action,
        'scalp_tp_move': tp_move,
        'scalp_sl_move': sl_move,
        'scalp_atr_pct': atr_pct,
        'scalp_has_full_horizon': has_full_horizon,
        'scalp_resolution_offset': resolution_offset,
        'scalp_uniqueness_weight': weights,
    })
    out = out.dropna(subset=['scalp_atr_pct'])
    out.to_csv(OUT_CSV, index=False)

    valid_out = out[out['scalp_has_full_horizon']]
    print(f"\nWeight stats (full-horizon rows, n={len(valid_out):,}):")
    print(valid_out['scalp_uniqueness_weight'].describe())
    print(f"\nResolution offset stats:")
    print(valid_out['scalp_resolution_offset'].describe())
    print(f"\nSaved {OUT_CSV}: {len(out):,} rows")


if __name__ == '__main__':
    main()

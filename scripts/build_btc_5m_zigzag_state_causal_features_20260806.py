"""Zigzag as an INPUT FEATURE, not a prediction target -- per the diagnosis that a classifier
predicting zigzag's own hard label ("which wave am I in") doesn't capture the label's oracle
quality (73-75% win rate) because that's a retrospective/backfilled classification question, not
a forward-prediction one (see project-btc-oracle-label-selection-protocol-20260806 and
project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806). Mirrors the earlier successful pattern for
this project: swing_transition_prob injected as a feature into live h48qual improved its backtest
(project-btc-5m-swingtransition-h48qual-upgrade-20260806), rather than zigzag/pivot-transition
being used as a standalone prediction target (which failed the same way explored here).

Produces 4 CAUSAL (no future information, no backfill) real-time pivot-tracker features, computed
with the SAME corrected volatility basis as the triple-barrier TP/SL (12-bar cumulative-return
dispersion, 288-bar lookback) so the "reversal imminent" signal is on a consistent scale with the
trading structure itself:

- zz_trend_state: the CURRENTLY tracked trend direction at bar i (+1 up / -1 down / 0 undetermined)
  -- unlike the oracle zigzag label, this is never revised using future bars.
- zz_bars_since_pivot: bars elapsed since the last CONFIRMED pivot (log1p-scaled).
- zz_move_from_pivot_pct: % move from the last confirmed pivot's price to the current close.
- zz_dist_to_threshold: current retracement from the running extreme, as a fraction of the live
  reversal threshold (0 = at a fresh extreme, approaching 1 = a new pivot is about to confirm) --
  the causal analogue of "is a reversal imminent."
"""
from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_state_causal_features_20260806.parquet"

CUMRET_BARS = 12
VOL_LOOKBACK = 288
MIN_REVERSAL_PCT = 0.009
VOL_MULTIPLIER = 1.0

FEATURE_COLS = ["zz_trend_state", "zz_bars_since_pivot", "zz_move_from_pivot_pct", "zz_dist_to_threshold"]


@numba.njit(cache=True)
def _causal_zigzag_state(close: np.ndarray, threshold: np.ndarray):
    n = len(close)
    trend_state = np.zeros(n, dtype=np.float64)
    bars_since_pivot = np.zeros(n, dtype=np.float64)
    move_from_pivot_pct = np.zeros(n, dtype=np.float64)
    dist_to_threshold = np.zeros(n, dtype=np.float64)

    trend = 0
    low_idx, high_idx = 0, 0
    low_price, high_price = close[0], close[0]
    last_pivot_idx = 0
    last_pivot_price = close[0]

    for i in range(1, n):
        price = close[i]
        thr = threshold[i]

        if trend == 0:
            if price < low_price:
                low_idx, low_price = i, price
            if price > high_price:
                high_idx, high_price = i, price
            if high_price / max(low_price, 1e-12) - 1.0 >= thr:
                if low_idx < high_idx:
                    last_pivot_idx, last_pivot_price = low_idx, low_price
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    last_pivot_idx, last_pivot_price = high_idx, high_price
                    trend = -1
                    low_idx, low_price = i, price
            retracement = 0.0
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            drop = high_price / max(price, 1e-12) - 1.0
            if drop >= thr:
                last_pivot_idx, last_pivot_price = high_idx, high_price
                trend = -1
                low_idx, low_price = i, price
                retracement = 0.0
            else:
                retracement = drop
        else:
            if price < low_price:
                low_idx, low_price = i, price
            rise = price / max(low_price, 1e-12) - 1.0
            if rise >= thr:
                last_pivot_idx, last_pivot_price = low_idx, low_price
                trend = 1
                high_idx, high_price = i, price
                retracement = 0.0
            else:
                retracement = rise

        trend_state[i] = float(trend)
        bars_since_pivot[i] = np.log1p(float(i - last_pivot_idx))
        move_from_pivot_pct[i] = price / max(last_pivot_price, 1e-12) - 1.0
        dist_to_threshold[i] = retracement / max(thr, 1e-8)

    return trend_state, bars_since_pivot, move_from_pivot_pct, dist_to_threshold


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    threshold = np.where(np.isfinite(vol), np.maximum(MIN_REVERSAL_PCT, VOL_MULTIPLIER * vol), MIN_REVERSAL_PCT)

    trend_state, bars_since_pivot, move_from_pivot_pct, dist_to_threshold = _causal_zigzag_state(close, threshold)

    out = pd.DataFrame({
        "timestamp": panel["timestamp"],
        "zz_trend_state": trend_state.astype(np.float32),
        "zz_bars_since_pivot": bars_since_pivot.astype(np.float32),
        "zz_move_from_pivot_pct": move_from_pivot_pct.astype(np.float32),
        "zz_dist_to_threshold": dist_to_threshold.astype(np.float32),
    })
    # first VOL_LOOKBACK+CUMRET_BARS rows have no valid threshold yet -- leave as computed with the
    # MIN_REVERSAL_PCT floor fallback (matches how the rest of this session's scripts warm up)
    out.to_parquet(OUT_PATH, index=False)
    print(out[FEATURE_COLS].describe())
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lever B-4: trend-scanning labels (Lopez de Prado-style, the exact label family that produces
this project's best-generalizing model -- Sigma3/Sigma6, see
scripts/build_1h_trendscan_dataset_20260705.py's TS_WINDOWS=[3,6,12,24,36,48] HOURS,
THRESHOLD=2.5) rescaled to 1-minute bars, as an alternative PRIMARY direction label to the base
20-bar triple-barrier (build_scalp_1m_tb_labels_20260716.py).

Reuses the exact numba _trend_scan_fast kernel from build_trend_scanning_action_labels_20260531.py
unchanged -- only the window list and threshold are rescaled from hour-scale to minute-scale
(matching the base label's 20-minute scalp horizon: windows span 5-60 minutes instead of
3-48 hours).

Trend-scanning produces a DIRECTION label only (no TP/SL) -- to keep the model-vs-label
comparison clean, this label is used only to pick the classifier's TRAINING TARGET (does the
primary model learn to predict trend-scan's LONG/SHORT/CASH instead of triple-barrier's?); the
actual trade P&L evaluation still uses the base labels' ATR-scaled TP/SL + realistic maker-fill
simulation, unchanged, so the comparison isolates "which target trains a better direction
classifier" from "how is a trade priced once taken."

Output: data/training_features_1m_trendscan_labels.csv
"""
import os

import numpy as np
import pandas as pd
from numba import njit, prange

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FEATURES_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m.csv')
OUT_CSV = os.path.join(_ROOT_DIR, 'data', 'training_features_1m_trendscan_labels.csv')

TS_WINDOWS = [5, 10, 15, 20, 30, 45, 60]  # minutes -- analogue of the 1h version's 3-48h windows
# NOTE: 1m bars have much more serially-correlated noise than the 1h series this label family was
# designed for, which inflates short-window OLS trend t-stats (median |t|~9.4 on this dataset,
# vs. threshold=2.5 used for the 1h version) -- checked empirically against the resulting label
# distribution (threshold=2.0 gave <0.1% CASH, degenerate) and picked the ~80th percentile instead.
TS_THRESHOLD = 14.0


@njit(parallel=True)
def _trend_scan_fast(values: np.ndarray, windows: np.ndarray):
    """CAUSALITY FIX 2026-08-04: previously used values[t..t+L-1] (up to L-1 bars into the future
    relative to t) assigned directly to out_t[t]; now uses values[t-L+1..t] (ending at t), matching
    the fix applied to build_trend_scanning_action_labels_20260531.py's _trend_scan_fast (this file
    had pasted an independent copy, not an import, so needed its own fix)."""
    n = len(values)
    out_t = np.zeros(n, dtype=np.float64)
    out_l = np.full(n, -1, dtype=np.int32)
    out_beta = np.zeros(n, dtype=np.float64)
    for t in prange(n):
        best_t = 0.0
        best_l = -1
        best_beta = 0.0
        for wi in range(len(windows)):
            L = int(windows[wi])
            if L <= 2 or t - L + 1 < 0:
                continue
            start = t - L + 1
            mean_x = (L - 1) / 2.0
            var_x_sum = L * (L * L - 1.0) / 12.0
            mean_y = 0.0
            ok = True
            for k in range(L):
                v = values[start + k]
                if not np.isfinite(v):
                    ok = False
                    break
                mean_y += v
            if not ok:
                continue
            mean_y /= L
            cov_xy = 0.0
            for k in range(L):
                cov_xy += (k - mean_x) * (values[start + k] - mean_y)
            beta = cov_xy / var_x_sum
            alpha = mean_y - beta * mean_x
            rss = 0.0
            for k in range(L):
                residual = values[start + k] - (alpha + beta * k)
                rss += residual * residual
            if rss <= 1e-12:
                t_val = 0.0
            else:
                se_beta = np.sqrt(rss / (L - 2.0)) / np.sqrt(var_x_sum)
                if se_beta <= 1e-12:
                    t_val = 0.0
                else:
                    t_val = beta / se_beta
            if abs(t_val) > abs(best_t):
                best_t = t_val
                best_l = L
                best_beta = beta
        out_t[t] = best_t
        out_l[t] = best_l
        out_beta[t] = best_beta
    return out_t, out_l, out_beta


def main():
    print("Loading 1m ETH close for trend-scanning label construction...")
    df = pd.read_csv(FEATURES_CSV, usecols=['timestamp', 'close'], parse_dates=['timestamp'])
    print(f"  {len(df):,} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    prices = df['close'].to_numpy(dtype=np.float64)
    values = np.log(np.maximum(prices, 1e-12))
    windows = np.array(sorted(set(TS_WINDOWS)), dtype=np.int32)

    print(f"Running numba trend-scan (windows={TS_WINDOWS} min, threshold={TS_THRESHOLD})...")
    t_values, opt_l, betas = _trend_scan_fast(values, windows)

    labels = np.zeros(len(df), dtype=np.int8)
    labels[(np.abs(t_values) >= TS_THRESHOLD) & (betas > 0.0)] = 1
    labels[(np.abs(t_values) >= TS_THRESHOLD) & (betas < 0.0)] = 2
    action = pd.Series(labels).map({0: 'CASH', 1: 'LONG', 2: 'SHORT'})

    out = pd.DataFrame({
        'timestamp': df['timestamp'],
        'trendscan_action': action,
        'ts_t_value': t_values.astype(np.float32),
        'ts_opt_L': opt_l.astype(np.int16),
        'ts_beta': betas.astype(np.float32),
    })
    out.to_csv(OUT_CSV, index=False)

    dist = out['trendscan_action'].value_counts()
    print(f"\nLabel distribution (n={len(out):,}):")
    print(dist)
    print(f"  LONG {dist.get('LONG', 0) / len(out):.2%}  SHORT {dist.get('SHORT', 0) / len(out):.2%}  "
          f"CASH {dist.get('CASH', 0) / len(out):.2%}")
    print(f"\nSaved {OUT_CSV}: {len(out):,} rows")


if __name__ == '__main__':
    main()

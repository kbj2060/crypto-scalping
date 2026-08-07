"""Second oracle-label candidate, per [[project-btc-oracle-label-selection-protocol-20260806]]:
trend-scanning (Lopez de Prado style) direction, oracle-validated through the SAME corrected TP/SL
mechanics as the triple-barrier label before any modeling effort is spent on it.

For every bar, fit forward OLS regressions of log(close) on bar-index over several candidate
horizons, pick the horizon with the largest |t-stat| (strongest, most significant forward trend),
and label direction = sign(slope) if |t-stat| clears a significance floor, else CASH. This is a
genuinely different oracle notion than triple-barrier's first-touch barrier race -- it rewards
sustained directional drift over a self-selected horizon, not "which side wins a fixed race."

This label's own construction does NOT guarantee it wins the triple-barrier TP/SL race (unlike the
triple-barrier label, which is self-consistent with the simulator by definition) -- that's exactly
what makes trading it through the corrected TP/SL simulator (build_btc_5m_tripbarrier_..._20260806
.py's CUMRET_BARS=12/VOL_LOOKBACK=288/TP_MULT=2.5/SL_MULT=1.2/HORIZON_BARS=288) a meaningful oracle
check per protocol step 3, not a tautology.
"""
from __future__ import annotations

import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_trendscan_oracle_labels_20260806.parquet"

HORIZONS = np.array([24, 48, 96, 144, 288], dtype=np.int64)  # 2h,4h,8h,12h,24h
# NOTE: naive OLS t-stats on autocorrelated price paths are massively inflated (median |t| ~19,
# far above the textbook |t|>=2 significance floor -- serial correlation violates the iid-residual
# assumption and shrinks the standard error). Using an empirical quantile threshold instead of a
# textbook significance cutoff, calibrated to match the triple-barrier label's ~38% CASH share for
# a comparable-difficulty class balance.
TSTAT_FLOOR = 16.5


@numba.njit(cache=True)
def _trend_scan(log_close: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(log_close)
    n_h = len(horizons)
    best_tstat = np.zeros(n, dtype=np.float64)
    best_slope = np.zeros(n, dtype=np.float64)
    for i in range(n):
        best_abs_t = 0.0
        for hi in range(n_h):
            h = horizons[hi]
            if i + h >= n:
                continue
            y = log_close[i : i + h + 1]
            m = h + 1
            x_mean = (m - 1) / 2.0
            y_mean = 0.0
            for k in range(m):
                y_mean += y[k]
            y_mean /= m
            sxx = 0.0
            sxy = 0.0
            for k in range(m):
                dx = k - x_mean
                sxx += dx * dx
                sxy += dx * (y[k] - y_mean)
            if sxx <= 0.0:
                continue
            slope = sxy / sxx
            resid_ss = 0.0
            for k in range(m):
                pred = y_mean + slope * (k - x_mean)
                e = y[k] - pred
                resid_ss += e * e
            dof = m - 2
            if dof <= 0:
                continue
            sigma2 = resid_ss / dof
            se_slope = np.sqrt(sigma2 / sxx) if sxx > 0 else 0.0
            if se_slope <= 0.0:
                continue
            tstat = slope / se_slope
            if abs(tstat) > best_abs_t:
                best_abs_t = abs(tstat)
                best_tstat[i] = tstat
                best_slope[i] = slope
    return best_tstat, best_slope


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_close = np.log(np.clip(close, 1e-9, None))

    tstat, slope = _trend_scan(log_close, HORIZONS)

    label = np.zeros(len(close), dtype=np.int8)  # 0=CASH,1=LONG,2=SHORT
    label[(tstat >= TSTAT_FLOOR)] = 1
    label[(tstat <= -TSTAT_FLOOR)] = 2

    out = pd.DataFrame({
        "timestamp": panel["timestamp"],
        "trendscan_action": label,
        "trendscan_tstat": tstat.astype(np.float32),
        "trendscan_slope": slope.astype(np.float32),
    })
    out.to_parquet(OUT_PATH, index=False)

    counts = pd.Series(label).value_counts(normalize=True).sort_index()
    summary = {
        "rows": int(len(label)),
        "ratios": {"CASH": float(counts.get(0, 0)), "LONG": float(counts.get(1, 0)), "SHORT": float(counts.get(2, 0))},
        "horizons": HORIZONS.tolist(),
        "tstat_floor": TSTAT_FLOOR,
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

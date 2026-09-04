#!/usr/bin/env python3
"""Phase1 label-design diagnostics (docs/homer/README.md "재사용 방법론 템플릿" section 2's
checklist) for the 2 Homer candidate-pool signals confirmed to proceed: kalman_deviation_meanrev
and DeMarker extreme (2026-08-31 narrowing decision -- see memory
eth_autocorr_regime_gate_kalman_demarker_20260831). Reuses compute_demarker (research_eth_
demarker_evidence_signal_lift_check_20260831.py) and kalman_level_and_velocity/rolling_zscore
(research_eth_candidate_pool_raw_lift_check_20260831.py) verbatim -- no re-derivation.

Checklist items covered (numbering matches the template):
  1. sign-only forward hit-rate horizon sensitivity (15m/30m/1h/2h/4h)
  2. MFE-in-ATR-units magnitude distribution (is a typical favorable move big or noise-sized?)
  3. fire-bar vs true-local-extreme lag (+-2h/24-bar window argmax/argmin -- the exact window this
     template specifies, "±2h 넓은 창 argmax/argmin")
  4. consecutive-fire clustering (gap distribution -> informs the cluster-anchor gap choice)
Item 5 (persistence) is deliberately NOT attempted -- the template explicitly warns v5/taker and
V_REBOUND both regressed from adding a persistence gate, so this stays touch-only from the start.
Item 6 (20-example visual verification) is a separate follow-up script once this one's findings
settle a draft HORIZON/K.

Source: binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv (2023-12-31..present, gap-free) -- the
template's canonical klines-only source for label design, NOT data/eth_5m_1year.csv (which is a
frozen VAL/OOS-only snapshot used for the raw-lift scorecard scripts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402

SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
START = pd.Timestamp("2024-01-01")
LAG_WINDOW = 24          # +-2h, this template's standard phase1 window
CLUSTER_GAP_MERGE = 3    # starting point, matches taker/short_term_return_z's initial choice
ATR_N = 14               # matches live_evidence_signal_dashboard_20260823.py's ATR_N
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}


def compute_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ATR_N) -> pd.Series:
    prev_close = close.shift(1)
    prev_close.iloc[0] = close.iloc[0]
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean() / close.clip(lower=1e-12)


def cluster_dedup(idx: np.ndarray, extremeness: np.ndarray, most_negative: bool, gap: int = CLUSTER_GAP_MERGE) -> np.ndarray:
    """Copied verbatim from render_eth_5m_taker_delta_climax_metalabel_examples_20260829.py."""
    order = np.argsort(idx)
    idx_sorted, ex_sorted = idx[order], extremeness[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "ex": ex_sorted})
    keep = df.loc[df.groupby("cluster")["ex"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["ex"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def sign_hit_rates(close: np.ndarray, idx: np.ndarray, side: str) -> list[dict]:
    rows = []
    for h_name, H in HORIZONS.items():
        valid = idx[idx < len(close) - H]
        if side == "bottom":
            hit = close[valid + H] > close[valid]
        else:
            hit = close[valid + H] < close[valid]
        rows.append({"horizon": h_name, "bars": H, "n": len(valid), "sign_hit_rate": float(hit.mean())})
    return rows


def mfe_atr_units(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_pct: np.ndarray,
                   idx: np.ndarray, side: str, H: int) -> np.ndarray:
    valid = idx[idx < len(close) - H]
    entry = close[valid]
    a = atr_pct[valid]
    if side == "bottom":
        fut_ext = np.array([high[i + 1:i + H + 1].max() for i in valid])
        move = (fut_ext - entry) / entry
    else:
        fut_ext = np.array([low[i + 1:i + H + 1].min() for i in valid])
        move = (entry - fut_ext) / entry
    return move / np.clip(a, 1e-9, None)


def extreme_lag(high: np.ndarray, low: np.ndarray, idx: np.ndarray, side: str, window: int = LAG_WINDOW) -> np.ndarray:
    n = len(high)
    lags = np.empty(len(idx), dtype=int)
    for k, i in enumerate(idx):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        if side == "bottom":
            local_pos = lo + int(np.argmin(low[lo:hi]))
        else:
            local_pos = lo + int(np.argmax(high[lo:hi]))
        lags[k] = local_pos - i
    return lags


def report_side(name: str, side: str, idx_raw: np.ndarray, extremeness: np.ndarray,
                 high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_pct: np.ndarray) -> None:
    print(f"\n----- {name} / {side} -----")
    print(f"raw fires: {len(idx_raw)}")

    idx = cluster_dedup(idx_raw, extremeness[idx_raw], most_negative=(side == "bottom"))
    print(f"cluster-anchored ({CLUSTER_GAP_MERGE}-bar gap): {len(idx)} fires")

    gaps = np.diff(np.sort(idx))
    print(f"post-anchor consecutive-fire gap (bars): median={np.median(gaps):.1f}, "
          f"%within3={np.mean(gaps <= 3) * 100:.1f}%, %within6={np.mean(gaps <= 6) * 100:.1f}%, "
          f"%within12={np.mean(gaps <= 12) * 100:.1f}%")

    print("sign-only forward hit-rate by horizon:")
    for row in sign_hit_rates(close, idx, side):
        print(f"  {row['horizon']:>4}: n={row['n']:6d}  sign_hit_rate={row['sign_hit_rate'] * 100:5.1f}%")

    print("MFE (favorable move / ATR%) distribution:")
    for h_name in ("1h", "2h", "4h"):
        H = HORIZONS[h_name]
        m = mfe_atr_units(high, low, close, atr_pct, idx, side, H)
        print(f"  {h_name:>3}: median={np.median(m):.2f}x  mean={np.mean(m):.2f}x  "
              f"p25={np.percentile(m, 25):.2f}x  p75={np.percentile(m, 75):.2f}x  "
              f"%>=1.5x={np.mean(m >= 1.5) * 100:.1f}%  %>=2.5x={np.mean(m >= 2.5) * 100:.1f}%")

    lags = extreme_lag(high, low, idx, side)
    at_tol = 2  # +-2 bars (10min) counts as "AT" the fire bar
    n_before = int((lags < -at_tol).sum())
    n_at = int((np.abs(lags) <= at_tol).sum())
    n_after = int((lags > at_tol).sum())
    total = len(lags)
    print(f"fire-bar vs true local extreme lag (+-{LAG_WINDOW}-bar window, +bars=extreme comes AFTER fire): "
          f"median={np.median(lags):+.1f}  BEFORE={n_before / total * 100:.1f}%  "
          f"AT(+-{at_tol})={n_at / total * 100:.1f}%  AFTER={n_after / total * 100:.1f}%")


def main() -> None:
    klines = pd.read_csv(SOURCE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    high, low, close = klines["high"], klines["low"], klines["close"]
    atr_pct = compute_atr_pct(high, low, close).to_numpy()
    ts = klines["timestamp"].to_numpy()
    start_mask = ts >= np.datetime64(START)
    n = len(klines)

    dem = compute_demarker(high, low)
    dem_arr = dem.fillna(0.5).to_numpy()  # neutral fill only for indexing safety, extreme thresholds unaffected
    dem_top_idx = np.flatnonzero((dem >= 0.90).fillna(False).to_numpy() & start_mask)
    dem_top_idx = dem_top_idx[dem_top_idx < n - LAG_WINDOW]
    dem_bottom_idx = np.flatnonzero((dem <= 0.10).fillna(False).to_numpy() & start_mask)
    dem_bottom_idx = dem_bottom_idx[dem_bottom_idx < n - LAG_WINDOW]

    levels, _ = kalman_level_and_velocity(close.to_numpy())
    kalman_dev = pd.Series((close.to_numpy() - levels) / levels, index=close.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    kalman_z_arr = kalman_dev_z.fillna(0.0).to_numpy()
    kalman_top_idx = np.flatnonzero((kalman_dev_z >= 2.0).fillna(False).to_numpy() & start_mask)
    kalman_top_idx = kalman_top_idx[kalman_top_idx < n - LAG_WINDOW]
    kalman_bottom_idx = np.flatnonzero((kalman_dev_z <= -2.0).fillna(False).to_numpy() & start_mask)
    kalman_bottom_idx = kalman_bottom_idx[kalman_bottom_idx < n - LAG_WINDOW]

    high_a, low_a, close_a = high.to_numpy(), low.to_numpy(), close.to_numpy()

    print(f"Data: {SOURCE.name}, {n} bars, {klines['timestamp'].iloc[0]} .. {klines['timestamp'].iloc[-1]}")
    print(f"Fires counted from {START.date()} onward (bars before kept only for indicator warmup)")

    report_side("demarker_extreme", "top", dem_top_idx, dem_arr, high_a, low_a, close_a, atr_pct)
    report_side("demarker_extreme", "bottom", dem_bottom_idx, dem_arr, high_a, low_a, close_a, atr_pct)
    report_side("kalman_deviation_meanrev", "top", kalman_top_idx, kalman_z_arr, high_a, low_a, close_a, atr_pct)
    report_side("kalman_deviation_meanrev", "bottom", kalman_bottom_idx, kalman_z_arr, high_a, low_a, close_a, atr_pct)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild of build_1h_trendscan_dataset_btc_20260706.py (Sigma9's BTC leg dataset), fixing the
07-06 build's mistaken assumption that BTC has no funding/OI/top-trader data. Verified 2026-08-01
(project-btc-funding-oi-data-quality-verified-20260801.md): binance_data/funding_rate_other/
BTCUSDT-fundingRate-*.zip (31 months, 2024-01..2026-07, downloaded 2026-07-08 by
download_metrics_funding_btc_20260708.py) and binance_data/metrics/BTCUSDT-metrics-*.zip (932 daily
files) ARE already merged into data/splits/year_oos/btc_features_{2024,2025,2026}.csv with real,
non-null, correctly-varying values (874 unique last_funding_rate values in 2025 alone, cross-checked
against the raw zip and confirmed exact) -- this data was simply never wired into Sigma9's OWN
dataset builder, which sourced raw klines directly instead of the already-engineered feature file.

Only change from the original: source is data/splits/year_oos/btc_features_{2024,2025,2026}.csv
(has last_funding_rate/sum_open_interest_value/sum_toptrader_long_short_ratio/taker_buy_base
columns) instead of binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv (OHLCV only). resample_1h() and
compute_features() are imported UNCHANGED from build_1h_trendscan_dataset_20260705.py (ETH's
original) -- they already auto-detect and use these columns when present (see their `if col in
r.columns` branches), so this produces BTC's full feature set (funding/funding_z_48/funding_roc_6/
oi_change_6/oi_z_48/toptrader_z_48 added on top of the previous 28), matching ETH's feature count
modulo the close_btc/eth_btc_spread columns (BTC has no analogous third-asset cross-reference in
this pipeline, that gap is real, not a data-availability issue).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# resample_1h/compute_features are copied VERBATIM from build_1h_trendscan_dataset_20260705.py
# (ETH's original) rather than imported, because that module imports
# build_trend_scanning_action_labels_20260531 at module level, which imports numba -- and numba
# 0.61.2 in this venv refuses to import against the installed numpy 2.3.5. Importing would pull in
# that broken transitive dependency even though these two functions don't use numba themselves. See
# _trend_scan_numpy below for the same numba-avoidance reasoning applied to the trend-scan step.

def resample_1h(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f.set_index("timestamp").sort_index()
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum", "taker_buy_base": "sum",
        "close_btc": "last",
    }
    agg = {k: v for k, v in agg.items() if k in f.columns}
    last_cols = [c for c in ("last_funding_rate", "sum_open_interest_value",
                             "sum_toptrader_long_short_ratio", "count_long_short_ratio", "volume_btc")
                 if c in f.columns]
    r = f.resample("1h", label="left", closed="left").agg(agg)
    for c in last_cols:
        r[c] = f[c].resample("1h", label="left", closed="left").last()
    r = r.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return r


def _zscore(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(10, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(10, win // 4)).std().replace(0, np.nan)
    return ((s - m) / sd).clip(-8, 8)


def _rsi(close: pd.Series, win: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / win, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / win, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50) / 100.0


def compute_features(r: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": r["timestamp"]})
    close = r["close"].astype(float)
    high = r["high"].astype(float)
    low = r["low"].astype(float)
    open_ = r["open"].astype(float)
    logc = np.log(close.clip(lower=1e-9))
    logret = logc.diff()

    for h in (1, 2, 3, 6, 12, 24):
        out[f"logret_{h}"] = (logc - logc.shift(h)).clip(-1, 1)
    for w in (6, 12, 24, 48):
        out[f"rvol_{w}"] = logret.rolling(w, min_periods=max(4, w // 4)).std().clip(0, 1)
    prev_close = close.shift(1)
    tr = np.maximum.reduce([high - low, (high - prev_close).abs(), (low - prev_close).abs()])
    atr = pd.Series(tr, index=r.index).rolling(14, min_periods=4).mean()
    out["atr_pct"] = (atr / close.clip(lower=1e-9)).clip(0, 1)
    out["rsi_14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = ((macd - macd_sig) / close.clip(lower=1e-9)).clip(-0.1, 0.1)
    sma20 = close.rolling(20, min_periods=5).mean()
    sd20 = close.rolling(20, min_periods=5).std()
    out["bb_width"] = (4 * sd20 / sma20.clip(lower=1e-9)).clip(0, 1)
    out["bb_pos"] = ((close - sma20) / (2 * sd20).replace(0, np.nan)).clip(-3, 3)
    if "close_btc" in r.columns:
        logb = np.log(r["close_btc"].astype(float).clip(lower=1e-9))
        for h in (1, 6, 24):
            out[f"btc_logret_{h}"] = (logb - logb.shift(h)).clip(-1, 1)
        out["eth_btc_spread_6"] = (out["btc_logret_6"] - out["logret_6"]).clip(-1, 1)
    if "last_funding_rate" in r.columns:
        fr = pd.to_numeric(r["last_funding_rate"], errors="coerce").fillna(0.0)
        out["funding"] = fr.clip(-0.01, 0.01)
        out["funding_z_48"] = _zscore(fr, 48)
        out["funding_roc_6"] = (fr - fr.shift(6)).clip(-0.01, 0.01)
    if "sum_open_interest_value" in r.columns:
        oi = pd.to_numeric(r["sum_open_interest_value"], errors="coerce")
        out["oi_change_6"] = (oi.pct_change(6)).clip(-1, 1).fillna(0.0)
        out["oi_z_48"] = _zscore(oi.pct_change().fillna(0.0), 48)
    if "sum_toptrader_long_short_ratio" in r.columns:
        out["toptrader_z_48"] = _zscore(pd.to_numeric(r["sum_toptrader_long_short_ratio"], errors="coerce").fillna(0.0), 48)
    vol = pd.to_numeric(r["volume"], errors="coerce").fillna(0.0)
    out["vol_z_48"] = _zscore(np.log1p(vol), 48)
    if "taker_buy_base" in r.columns:
        tb = pd.to_numeric(r["taker_buy_base"], errors="coerce").fillna(0.0)
        out["taker_imb"] = ((2 * tb - vol) / vol.replace(0, np.nan)).clip(-1, 1).fillna(0.0)
    rng = (high - low).replace(0, np.nan)
    out["body_ratio"] = ((close - open_) / rng).clip(-1, 1).fillna(0.0)
    out["upper_wick"] = ((high - np.maximum(open_, close)) / rng).clip(0, 1).fillna(0.0)
    out["lower_wick"] = ((np.minimum(open_, close) - low) / rng).clip(0, 1).fillna(0.0)
    out["skew_24"] = logret.rolling(24, min_periods=8).skew().clip(-5, 5).fillna(0.0)
    out["kurt_24"] = logret.rolling(24, min_periods=8).kurt().clip(-5, 10).fillna(0.0)
    out["dist_sma50"] = ((close - close.rolling(50, min_periods=10).mean()) / close.clip(lower=1e-9)).clip(-0.5, 0.5)
    out["hurst_proxy"] = (out["logret_6"].abs() / (out["rvol_6"] * np.sqrt(6) + 1e-9)).clip(0, 3)
    ts = pd.to_datetime(r["timestamp"])
    hod = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hod / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    dow = ts.dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["open"] = open_.values
    out["high"] = high.values
    out["low"] = low.values
    out["close"] = close.values
    return out


def _trend_scan_numpy(values: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy reimplementation of build_trend_scanning_action_labels_20260531._trend_scan_fast
    (numba-jitted original) -- numba 0.61.2 in this venv refuses to import against numpy 2.3.5
    (ImportError: Numba needs NumPy 2.2 or less), and downgrading numpy repo-wide would risk the
    live trading venv, so this avoids the numba import path entirely. Same algorithm (per-window
    OLS trend t-statistic via sliding_window_view instead of an explicit python/numba loop), same
    tie-break (first/smallest window wins on an exact |t| tie, matching the original's strict `>`
    update rule since windows are iterated in ascending sorted order both places).

    CAUSALITY FIX 2026-08-04: sliding_window_view(values, L)[r] = values[r:r+L], so the ORIGINAL
    code's `out_t[:n_valid] = t_val` assigned that window's result to index r -- i.e. row r used
    bars r..r+L-1, up to L-1 bars INTO THE FUTURE relative to r. Confirmed empirically (exact
    match to a forward-window recomputation, mismatch to the causal backward window) on both BTC
    and ETH saved outputs. Fix: assign window[r]'s result to index r+L-1 (the window's own last/
    most-recent bar), so out_t[t] only ever uses values up to and including t."""
    n = len(values)
    out_t = np.zeros(n, dtype=np.float64)
    out_l = np.full(n, -1, dtype=np.int32)
    out_beta = np.zeros(n, dtype=np.float64)
    finite = np.isfinite(values)
    for L in sorted(int(w) for w in windows if int(w) > 2):
        n_valid = n - L + 1
        if n_valid <= 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(values, L)[:n_valid]
        ok = np.lib.stride_tricks.sliding_window_view(finite, L)[:n_valid].all(axis=1)
        mean_x = (L - 1) / 2.0
        var_x_sum = L * (L * L - 1.0) / 12.0
        k_centered = np.arange(L, dtype=np.float64) - mean_x
        mean_y = win.mean(axis=1)
        cov_xy = win @ k_centered
        beta = cov_xy / var_x_sum
        alpha = mean_y - beta * mean_x
        pred = alpha[:, None] + beta[:, None] * np.arange(L, dtype=np.float64)[None, :]
        rss = np.square(win - pred).sum(axis=1)
        se_beta = np.sqrt(np.maximum(rss, 0.0) / (L - 2.0)) / np.sqrt(var_x_sum)
        t_val = np.where((rss > 1e-12) & (se_beta > 1e-12), beta / np.where(se_beta > 1e-12, se_beta, 1.0), 0.0)
        t_val = np.where(ok, t_val, 0.0)
        dest = np.arange(L - 1, L - 1 + n_valid)
        improve = np.abs(t_val) > np.abs(out_t[dest])
        out_t[dest] = np.where(improve, t_val, out_t[dest])
        out_l[dest] = np.where(improve, L, out_l[dest])
        out_beta[dest] = np.where(improve, beta, out_beta[dest])
    return out_t, out_l, out_beta

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_full_20260801"
SOURCES = [ROOT / f"data/splits/year_oos/btc_features_{y}.csv" for y in (2024, 2025, 2026)]
TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5
YEAR_RANGES = {
    2024: ("2024-01-01", "2024-12-31 23:59:59"),
    2025: ("2025-01-01", "2025-12-31 23:59:59"),
    2026: ("2026-01-01", "2026-12-31 23:59:59"),
}
RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base",
            "last_funding_rate", "sum_open_interest_value", "sum_toptrader_long_short_ratio"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = pd.concat([pd.read_csv(p, usecols=RAW_COLS) for p in SOURCES], ignore_index=True)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    r_full = resample_1h(src)
    feats_full = compute_features(r_full)
    added_cols = [c for c in ("funding", "funding_z_48", "funding_roc_6", "oi_change_6", "oi_z_48",
                               "toptrader_z_48") if c in feats_full.columns]
    print(f"feature columns added vs OHLCV-only Sigma9: {added_cols} (total feature count={len(feats_full.columns)-1})")

    logc_full = np.log(np.maximum(feats_full["close"].to_numpy(dtype=np.float64), 1e-12))
    win = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    t_vals, opt_l, betas = _trend_scan_numpy(logc_full, win)
    labels = np.zeros(len(feats_full), dtype=np.int64)
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats_full["ts_action"] = labels
    feats_full["ts_t_value"] = t_vals.astype(np.float32)
    feats_full["ts_opt_L"] = opt_l.astype(np.int16)

    summary = {}
    ts = feats_full["timestamp"]
    for year, (start, end) in YEAR_RANGES.items():
        mask = (ts >= start) & (ts <= end)
        sub = feats_full.loc[mask].reset_index(drop=True)
        if len(sub) == 0:
            continue
        out_path = OUT_DIR / f"sigma9_btc_1h_full_{year}.parquet"
        sub.to_parquet(out_path, index=False)
        counts = np.bincount(sub["ts_action"].to_numpy(), minlength=3)
        summary[str(year)] = {
            "rows_1h": int(len(sub)),
            "range": [str(sub["timestamp"].iloc[0]), str(sub["timestamp"].iloc[-1])],
            "label_counts_CASH_LONG_SHORT": counts.tolist(),
            "label_ratios": (counts / max(counts.sum(), 1)).round(3).tolist(),
            "n_features": int(len(feats_full.columns) - 1 - 3),  # minus timestamp, minus 3 label cols
        }
        print(f"wrote {out_path} ({len(sub)} rows)")

    report = {"windows": TS_WINDOWS, "threshold": TS_THRESHOLD, "added_feature_cols": added_cols,
              "n_features_total": int(len(feats_full.columns) - 1 - 3), "summary": summary}
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

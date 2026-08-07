#!/usr/bin/env python3
"""Sigma3 dataset: resample the clean 5m training features to 1-HOUR bars, compute a focused set
of stationary features, and label with TREND-SCANNING (Lopez de Prado) instead of zigzag.

Motivation (from docs/model_contracts/sigma2_seq_zigzag_20260705_contract.md conclusion): across
5 approaches on the 5-minute feature universe, the signal is consistently "cost1-positive,
cost3-negative" -- the edge is smaller than 3x transaction costs at 5-minute frequency. The
single highest-leverage change is LOWER FREQUENCY: at 1h, a typical trade spans a 2-4% move, so
a 0.42% (cost3) round-trip is a much smaller fraction than at 5m (~0.6% moves). Web research
(2026-07): AEDL is SOTA but too complex/fragile to reimplement; trend-scanning is a robust,
well-founded "better than zigzag" label that adapts its horizon per bar via the max-|t-value|
forward linear-trend fit, and naturally filters to statistically-significant moves (t >=
threshold) -- directly targeting the cost problem by only labeling strong trends.

Order-book/execution duckdb data was inspected (data/live/microstructure.duckdb): it only spans
2026-05-03..07-05 (~2 months, live-only) and does NOT exist for 2024-2025, so it cannot be a
training feature for a historical backtest. Noted for future live use; not used here.

Causality: 1h bar timestamped at hour H covers [H, H+1); features use only data up to and
including that bar; trend-scanning labels use forward windows (offline label construction only,
standard convention). Backtest enters at next-bar open (price at H+1), same as the 5m pipeline.
"""

from __future__ import annotations

import argparse
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

from build_trend_scanning_action_labels_20260531 import _trend_scan_fast  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_20260705"
SRC_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}

# 1h trend-scanning: forward windows 3h..48h (2 days). threshold=2.5 keeps only trends whose
# forward linear-fit t-stat is significant, filtering chop to CASH.
TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5


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
    return (100 - 100 / (1 + rs)).fillna(50) / 100.0  # scaled 0..1


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
    # ATR%
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
    # Bollinger width
    sma20 = close.rolling(20, min_periods=5).mean()
    sd20 = close.rolling(20, min_periods=5).std()
    out["bb_width"] = (4 * sd20 / sma20.clip(lower=1e-9)).clip(0, 1)
    out["bb_pos"] = ((close - sma20) / (2 * sd20).replace(0, np.nan)).clip(-3, 3)
    # BTC
    if "close_btc" in r.columns:
        logb = np.log(r["close_btc"].astype(float).clip(lower=1e-9))
        for h in (1, 6, 24):
            out[f"btc_logret_{h}"] = (logb - logb.shift(h)).clip(-1, 1)
        out["eth_btc_spread_6"] = (out["btc_logret_6"] - out["logret_6"]).clip(-1, 1)
    # funding / OI
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
    # volume
    vol = pd.to_numeric(r["volume"], errors="coerce").fillna(0.0)
    out["vol_z_48"] = _zscore(np.log1p(vol), 48)
    if "taker_buy_base" in r.columns:
        tb = pd.to_numeric(r["taker_buy_base"], errors="coerce").fillna(0.0)
        out["taker_imb"] = ((2 * tb - vol) / vol.replace(0, np.nan)).clip(-1, 1).fillna(0.0)
    # candle shape
    rng = (high - low).replace(0, np.nan)
    out["body_ratio"] = ((close - open_) / rng).clip(-1, 1).fillna(0.0)
    out["upper_wick"] = ((high - np.maximum(open_, close)) / rng).clip(0, 1).fillna(0.0)
    out["lower_wick"] = ((np.minimum(open_, close) - low) / rng).clip(0, 1).fillna(0.0)
    # rolling higher moments
    out["skew_24"] = logret.rolling(24, min_periods=8).skew().clip(-5, 5).fillna(0.0)
    out["kurt_24"] = logret.rolling(24, min_periods=8).kurt().clip(-5, 10).fillna(0.0)
    # trend/mean-reversion
    out["dist_sma50"] = ((close - close.rolling(50, min_periods=10).mean()) / close.clip(lower=1e-9)).clip(-0.5, 0.5)
    out["hurst_proxy"] = (out["logret_6"].abs() / (out["rvol_6"] * np.sqrt(6) + 1e-9)).clip(0, 3)
    # calendar
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=",".join(str(w) for w in TS_WINDOWS))
    ap.add_argument("--threshold", type=float, default=TS_THRESHOLD)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = np.array(sorted({int(w) for w in args.windows.split(",")}), dtype=np.int32)
    summary = {}
    for year, path in SRC_FILES.items():
        src = pd.read_csv(path, low_memory=False)
        src["timestamp"] = pd.to_datetime(src["timestamp"])
        r = resample_1h(src)
        feats = compute_features(r)
        # trend-scanning labels on 1h log-close
        logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
        t_vals, opt_l, betas = _trend_scan_fast(logc, windows)
        labels = np.zeros(len(feats), dtype=np.int64)
        thr = float(args.threshold)
        labels[(np.abs(t_vals) >= thr) & (betas > 0)] = 1
        labels[(np.abs(t_vals) >= thr) & (betas < 0)] = 2
        feats["ts_action"] = labels
        feats["ts_t_value"] = t_vals.astype(np.float32)
        feats["ts_opt_L"] = opt_l.astype(np.int16)
        out_path = OUT_DIR / f"sigma3_1h_{year}.parquet"
        feats.to_parquet(out_path, index=False)
        counts = np.bincount(labels, minlength=3).tolist()
        summary[str(year)] = {
            "rows_1h": int(len(feats)),
            "range": [str(feats["timestamp"].min()), str(feats["timestamp"].max())],
            "label_counts_CASH_LONG_SHORT": counts,
            "label_ratios": [round(c / max(len(feats), 1), 3) for c in counts],
            "n_features": int(len([c for c in feats.columns if c not in ("timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L")])),
        }
        print(f"{year}: {summary[str(year)]}", flush=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps({"windows": windows.tolist(), "threshold": float(args.threshold), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-run of build_1h_trendscan_dataset_20260705.py against the now-extended (through 2026-07-20)
training_features_2026_rebuilt.csv, avoiding the numba import that build_1h_trendscan_dataset_20260705.py
needs (numba requires numpy<2.3, this venv has numpy 2.3.5 -- same known conflict documented in
build_1h_trendscan_dataset_sol_20260715.py, the SOL port). resample_1h/compute_features are
identical to the ETH original; trend_scan_fast is the same vectorized OLS-identity reimplementation
already used and verified for the SOL port, just applied back to ETH here. Output path/filenames
match the original script exactly, so this is a drop-in re-run, not a new dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_20260705"
SRC_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}

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


def trend_scan_fast(values: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized numpy reimplementation of the (now-fixed) numba _trend_scan_fast kernel.

    CAUSALITY FIX 2026-08-04: the previous pandas-rolling version computed each window's rolling
    sum (naturally ending at each position, e.g. sum_y_full[i] = sum(values[i-L+1:i+1])) but then
    explicitly SHIFTED it backward via `sum_y[:n-L+1] = sum_y_full[L-1:]`, which re-assigns the
    window ending at L-1 to output index 0 -- i.e. index j receives the sum of values[j:j+L],
    reading up to L-1 bars INTO THE FUTURE relative to j. The `s1` term had the same forward-shift
    bug (`shifted[:n-k] = values[k:]`). Confirmed via direct empirical recomputation against
    stored ts_t_value on saved BTC/ETH outputs. Rewritten below using sliding_window_view with an
    explicit end-of-window destination index, verified causal the same way (see
    scripts/build_btc_1h_trendscan_causal_fix_20260804.py)."""
    n = len(values)
    out_t = np.zeros(n, dtype=np.float64)
    out_l = np.full(n, -1, dtype=np.int32)
    out_beta = np.zeros(n, dtype=np.float64)
    finite = np.isfinite(values)
    for L in sorted(int(w) for w in windows if int(w) > 2):
        n_valid = n - L + 1
        if n_valid <= 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(values, L)[:n_valid]  # win[j] = values[j:j+L]
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
        # win[j] = values[j:j+L] ends at index j+L-1 -- assign to that index, not j, so out_t[t]
        # only ever uses values up to and including t.
        dest = np.arange(L - 1, L - 1 + n_valid)
        improve = np.abs(t_val) > np.abs(out_t[dest])
        out_t[dest] = np.where(improve, t_val, out_t[dest])
        out_l[dest] = np.where(improve, L, out_l[dest])
        out_beta[dest] = np.where(improve, beta, out_beta[dest])
    return out_t, out_l, out_beta


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    summary = {}
    for year, path in SRC_FILES.items():
        src = pd.read_csv(path, low_memory=False)
        src["timestamp"] = pd.to_datetime(src["timestamp"])
        r = resample_1h(src)
        feats = compute_features(r)
        logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
        t_vals, opt_l, betas = trend_scan_fast(logc, windows)
        labels = np.zeros(len(feats), dtype=np.int64)
        labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
        labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
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
        }
        print(f"{year}: {summary[str(year)]}", flush=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps({"windows": windows.tolist(), "threshold": float(TS_THRESHOLD), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

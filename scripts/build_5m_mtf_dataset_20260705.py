#!/usr/bin/env python3
"""Sigma4 dataset: 5-MINUTE decision cadence with 1-HOUR trend context as reference features.

Per user (2026-07-05): (a) cost3 (3x fee stress) is too strict -- evaluate on cost1 (realistic
1x fees) as the real bar, report cost3 only as context; (b) make trading decisions every 5m,
using 1h only as reference.

Design: multi-timeframe. The model trades on 5m bars (5m barriers, 5m cadence) but sees the
higher-timeframe trend via BACKWARD-LOOKING 1h context features. Sigma3 proved the trend signal
is real and direction-transferable OOS; here we keep the 5m granularity (which had the strongest
cost1 in-sample historically) and add 1h context as a filter/aligner.

CAUSALITY (critical): the 1h trend-scanning LABEL uses forward windows and is NOT usable as a
feature. Only backward-looking 1h stats (momentum, RSI, realized vol, past-bar OLS slope) are
used, and they are merged causally: the 1h bar timestamped H covers [H,H+1) and only "completes"
at H+1:00, so its context is attached to 5m bars at/after H+1:00 (merge_asof on completion time).

Labels: 5m trend-scanning (windows 12..96 bars = 1h..8h forward), threshold 2.5.
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

from build_1h_trendscan_dataset_20260705 import compute_features, resample_1h  # noqa: E402
from build_trend_scanning_action_labels_20260531 import _trend_scan_fast  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma4_5m_mtf_20260705"
SRC_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
TS_WINDOWS_5M = [12, 24, 48, 96]  # 1h..8h forward at 5m
TS_THRESHOLD = 2.5


def _ols_slope_backward(logc: pd.Series, win: int) -> pd.Series:
    """Rolling OLS slope over the PAST `win` bars (backward, causal). t-normalized by dividing
    by rolling std so it's a stationary trend-strength proxy."""
    x = np.arange(win)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(vals: np.ndarray) -> float:
        y = vals
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    sl = logc.rolling(win, min_periods=win).apply(slope, raw=True)
    return sl


def build_1h_context(src: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking 1h context features, keyed by their COMPLETION time (H+1:00) for causal
    merge onto 5m."""
    r = resample_1h(src)
    r["timestamp"] = pd.to_datetime(r["timestamp"])
    logc = np.log(r["close"].astype(float).clip(lower=1e-9))
    ctx = pd.DataFrame({"avail_ts": r["timestamp"] + pd.Timedelta("1h")})
    for h in (1, 3, 6, 12):
        ctx[f"h1_ret_{h}"] = (logc - logc.shift(h)).clip(-1, 1).values
    ctx["h1_rvol_12"] = logc.diff().rolling(12, min_periods=4).std().clip(0, 1).values
    d = logc.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ctx["h1_rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).values / 100.0
    sl12 = _ols_slope_backward(logc, 12)
    ctx["h1_slope_12"] = (sl12 / (logc.diff().rolling(12, min_periods=4).std() + 1e-9)).clip(-8, 8).values
    ctx["h1_slope_sign"] = np.sign(ctx["h1_slope_12"]).fillna(0.0)
    ctx = ctx.dropna(subset=["avail_ts"]).reset_index(drop=True)
    return ctx


def build_year(year: int) -> pd.DataFrame:
    src = pd.read_csv(SRC_FILES[year], low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    # 5m features (compute_features is bar-count based -> works at 5m; returns over 1..24 bars = 5m..2h)
    feats = compute_features(src)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    # 1h context, causal as-of merge on completion time
    ctx = build_1h_context(src)
    feats = pd.merge_asof(feats.sort_values("timestamp"), ctx.sort_values("avail_ts"),
                          left_on="timestamp", right_on="avail_ts", direction="backward")
    feats = feats.drop(columns=["avail_ts"])
    # 5m trend-scanning labels
    logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
    win = np.array(sorted(TS_WINDOWS_5M), dtype=np.int32)
    t_vals, opt_l, betas = _trend_scan_fast(logc, win)
    labels = np.zeros(len(feats), dtype=np.int64)
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats["ts_action"] = labels
    feats["ts_t_value"] = t_vals.astype(np.float32)
    feats["ts_opt_L"] = opt_l.astype(np.int16)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ = {}
    for year in (2024, 2025, 2026):
        feats = build_year(year)
        h1_cols = [c for c in feats.columns if c.startswith("h1_")]
        # drop warmup rows where 1h context is NaN (first ~2h of the earliest data only)
        before = len(feats)
        feats = feats.dropna(subset=h1_cols).reset_index(drop=True)
        out_path = OUT_DIR / f"sigma4_5m_{year}.parquet"
        feats.to_parquet(out_path, index=False)
        counts = np.bincount(feats["ts_action"].to_numpy(), minlength=3).tolist()
        summ[str(year)] = {"rows": int(len(feats)), "dropped_warmup": int(before - len(feats)),
                           "range": [str(feats["timestamp"].min()), str(feats["timestamp"].max())],
                           "h1_context_features": h1_cols, "label_CLS": counts,
                           "n_features": int(len([c for c in feats.columns if c not in ("timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L")]))}
        print(f"{year}: rows={len(feats)} feats={summ[str(year)]['n_features']} h1ctx={len(h1_cols)} labelCLS={counts}", flush=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

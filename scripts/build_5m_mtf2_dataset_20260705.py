#!/usr/bin/env python3
"""Sigma7 dataset: 5-MINUTE decision cadence with BOTH 1h and 6h trend context (multi-timeframe),
per user (2026-07-05): decide on 5m, use 1h + 6h as auxiliary/reference.

Combines everything the session learned: 5m granularity (user's requirement) + multi-timeframe
context + (in the runner) the regime filter + let-winners-run execution that rescued 1h
trend-following (Sigma6). All higher-TF context is BACKWARD-LOOKING and merged causally on the
higher bar's COMPLETION time (H+1h / H+6h), so no lookahead. Labels: 5m trend-scanning.
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

from build_1h_trendscan_dataset_20260705 import compute_features  # noqa: E402
from build_trend_scanning_action_labels_20260531 import _trend_scan_fast  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma7_5m_mtf2_20260705"
SRC_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
TS_WINDOWS_5M = [12, 24, 48, 96]
TS_THRESHOLD = 2.5


def _resample(src: pd.DataFrame, freq: str) -> pd.DataFrame:
    f = src.copy().set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in f.columns}
    r = f.resample(freq, label="left", closed="left").agg(agg)
    return r.dropna(subset=["close"]).reset_index()


def _tf_context(src: pd.DataFrame, freq: str, offset: pd.Timedelta, prefix: str) -> pd.DataFrame:
    r = _resample(src, freq)
    logc = np.log(r["close"].astype(float).clip(lower=1e-9))
    ctx = pd.DataFrame({"avail_ts": pd.to_datetime(r["timestamp"]) + offset})
    for h in (1, 3, 6, 12):
        ctx[f"{prefix}_ret_{h}"] = (logc - logc.shift(h)).clip(-1, 1).values
    d = logc.diff()
    ctx[f"{prefix}_rvol_12"] = d.rolling(12, min_periods=4).std().clip(0, 1).values
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ctx[f"{prefix}_rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).values / 100.0
    # backward OLS slope over past 12 bars, vol-normalized
    x = np.arange(12); xm = x.mean(); den = ((x - xm) ** 2).sum()
    sl = logc.rolling(12, min_periods=12).apply(lambda y: float(((x - xm) * (y - y.mean())).sum() / den), raw=True)
    ctx[f"{prefix}_slope_12"] = (sl / (d.rolling(12, min_periods=4).std() + 1e-9)).clip(-8, 8).values
    return ctx.dropna(subset=["avail_ts"]).reset_index(drop=True)


def build_year(year: int) -> pd.DataFrame:
    src = pd.read_csv(SRC_FILES[year], low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    feats = compute_features(src)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    for freq, off, pfx in (("1h", pd.Timedelta("1h"), "h1"), ("6h", pd.Timedelta("6h"), "h6")):
        ctx = _tf_context(src, freq, off, pfx)
        feats = pd.merge_asof(feats.sort_values("timestamp"), ctx.sort_values("avail_ts"),
                              left_on="timestamp", right_on="avail_ts", direction="backward").drop(columns=["avail_ts"])
    logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
    win = np.array(sorted(TS_WINDOWS_5M), dtype=np.int32)
    t_vals, opt_l, betas = _trend_scan_fast(logc, win)
    lab = np.zeros(len(feats), dtype=np.int64)
    lab[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    lab[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats["ts_action"] = lab
    feats["ts_t_value"] = t_vals.astype(np.float32)
    return feats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ = {}
    for year in (2024, 2025, 2026):
        feats = build_year(year)
        ctx_cols = [c for c in feats.columns if c.startswith(("h1_", "h6_"))]
        before = len(feats)
        feats = feats.dropna(subset=ctx_cols).reset_index(drop=True)
        feats.to_parquet(OUT_DIR / f"sigma7_5m_{year}.parquet", index=False)
        summ[str(year)] = {"rows": int(len(feats)), "dropped": int(before - len(feats)),
                           "ctx_cols": len(ctx_cols),
                           "n_features": int(len([c for c in feats.columns if c not in ("timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L")])),
                           "labelCLS": np.bincount(feats["ts_action"].to_numpy(), minlength=3).tolist()}
        print(f"{year}: rows={len(feats)} feats={summ[str(year)]['n_features']} ctx={len(ctx_cols)} labelCLS={summ[str(year)]['labelCLS']}", flush=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

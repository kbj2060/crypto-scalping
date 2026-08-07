#!/usr/bin/env python3
"""Sigma8 dataset: the working 1h Sigma3 feature set PLUS the 4 features the 2026-07-05 feature
audit found genuinely add value (dist_lo_atr rank 3, ret_skew_48 rank 9, vol_expansion, eff_ratio),
computed at 1h. Dead/redundant features from the audit are simply not added (Sigma3's base 38 are
already a curated set; trees are robust to the few weak ones). Labels: 1h trend-scanning (same as
Sigma3). Purpose: does the enriched feature set improve the Sigma6 regime-trend result on
VALIDATION (OOS 2026-03..06 is heavily peeked now, so validation improvement is the real test).
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

from build_1h_trendscan_dataset_20260705 import compute_features, resample_1h, SRC_FILES  # noqa: E402
from build_trend_scanning_action_labels_20260531 import _trend_scan_fast  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma8_1h_enriched_20260705"
TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5


def add_audit_winners(feats: pd.DataFrame) -> list[str]:
    close = feats["close"].astype(float)
    high = feats["high"].astype(float)
    low = feats["low"].astype(float)
    logc = np.log(close.clip(lower=1e-9))
    ret = logc.diff()
    absret = ret.abs()
    prev = close.shift(1)
    tr = np.maximum.reduce([high - low, (high - prev).abs(), (low - prev).abs()])
    atr = pd.Series(tr, index=feats.index).rolling(24, min_periods=6).mean()
    # dist to rolling 48-bar (2-day) low in ATR units -- the rank-3 audit winner
    ll = low.rolling(48, min_periods=12).min()
    feats["dist_lo_atr"] = ((close - ll) / (atr + 1e-9)).clip(0, 30).fillna(0.0)
    hh = high.rolling(48, min_periods=12).max()
    feats["dist_hi_atr"] = ((close - hh) / (atr + 1e-9)).clip(-30, 0).fillna(0.0)
    # rolling return skew (rank-9 winner)
    feats["ret_skew_48"] = ret.rolling(48, min_periods=12).skew().clip(-5, 5).fillna(0.0)
    # short/long realized-vol expansion
    rvol12 = ret.rolling(12, min_periods=4).std()
    rvol96 = ret.rolling(96, min_periods=24).std()
    feats["vol_expansion"] = (rvol12 / rvol96.replace(0, np.nan)).clip(0, 5).fillna(1.0)
    # Kaufman efficiency ratio (12 bars)
    net = (logc - logc.shift(12)).abs()
    path = absret.rolling(12, min_periods=4).sum()
    feats["eff_ratio_12"] = (net / path.replace(0, np.nan)).clip(0, 1).fillna(0.0)
    return ["dist_lo_atr", "dist_hi_atr", "ret_skew_48", "vol_expansion", "eff_ratio_12"]


def build_year(year: int) -> pd.DataFrame:
    src = pd.read_csv(SRC_FILES[year], low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    r = resample_1h(src)
    feats = compute_features(r)
    new_cols = add_audit_winners(feats)
    logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
    win = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    t_vals, opt_l, betas = _trend_scan_fast(logc, win)
    lab = np.zeros(len(feats), dtype=np.int64)
    lab[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    lab[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats["ts_action"] = lab
    feats["ts_t_value"] = t_vals.astype(np.float32)
    feats.attrs["new_cols"] = new_cols
    return feats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ = {}
    for year in (2024, 2025, 2026):
        feats = build_year(year)
        feats.to_parquet(OUT_DIR / f"sigma8_1h_{year}.parquet", index=False)
        nf = int(len([c for c in feats.columns if c not in ("timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L")]))
        summ[str(year)] = {"rows": int(len(feats)), "n_features": nf, "label_CLS": np.bincount(feats["ts_action"].to_numpy(), minlength=3).tolist()}
        print(f"{year}: rows={len(feats)} features={nf} (+5 audit-winner cols) labelCLS={summ[str(year)]['label_CLS']}", flush=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

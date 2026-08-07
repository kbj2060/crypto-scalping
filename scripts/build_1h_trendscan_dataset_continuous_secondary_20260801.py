#!/usr/bin/env python3
"""Sigma3 1h dataset, continuous + secondary features: same continuous-warmup fix as
build_1h_trendscan_dataset_continuous_20260801.py, plus the 125 already-engineered "2차 가공"
columns from the source 5m feature file (whale_retail_ratio, cvd_*, ou_halflife, sig_trend_health,
crowding_pressure, etc. -- verified rule-based/rolling, no pre-fit model component, see
features/elite.py's VolatilityModelEngine/NewEliteSignalEngine and features/high_order_state.py).

These are 5m-native indicators; instead of redesigning each formula's window for 1h (large,
error-prone effort), each 1h bin samples the LAST 5m value within the bin -- the same convention
already used for last_funding_rate/sum_open_interest_value in resample_1h(). This is causal (the
value at bin close is exactly what would be known at decision time) even though the indicator's
own internal lookback window stays 5m-scaled.
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

from build_1h_trendscan_dataset_extended_20260720 import (  # noqa: E402
    compute_features,
    resample_1h,
    trend_scan_fast,
)

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_continuous_secondary_20260801"
SRC_FILES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]

TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5

BASE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
    "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate",
    "close_btc", "volume_btc", "quote_volume_btc",
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in SRC_FILES:
        src = pd.read_csv(path, low_memory=False)
        src["timestamp"] = pd.to_datetime(src["timestamp"])
        frames.append(src)
    full = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    secondary_cols = [c for c in full.columns if c not in BASE_COLS]
    print(f"concatenated 5m rows: {len(full)}, secondary cols: {len(secondary_cols)}", flush=True)

    r = resample_1h(full)
    feats = compute_features(r)

    # sample each 1h bin's LAST 5m value for the secondary columns (causal: known at bin close)
    f = full.set_index("timestamp").sort_index()
    sec_last = f[secondary_cols].resample("1h", label="left", closed="left").last().reset_index()
    sec_last = sec_last.rename(columns={c: f"sec_{c}" for c in secondary_cols})
    feats = feats.merge(sec_last, on="timestamp", how="left")
    for c in [f"sec_{c}" for c in secondary_cols]:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0).clip(-1e6, 1e6)

    windows = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    logc = np.log(np.maximum(feats["close"].to_numpy(dtype=np.float64), 1e-12))
    t_vals, opt_l, betas = trend_scan_fast(logc, windows)
    labels = np.zeros(len(feats), dtype=np.int64)
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats["ts_action"] = labels
    feats["ts_t_value"] = t_vals.astype(np.float32)
    feats["ts_opt_L"] = opt_l.astype(np.int16)

    out_path = OUT_DIR / "sigma3_1h_continuous_secondary.parquet"
    feats.to_parquet(out_path, index=False)

    n_feature_cols = len([c for c in feats.columns if c not in
                           ("timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L")])
    counts = np.bincount(labels, minlength=3).tolist()
    report = {
        "windows": windows.tolist(), "threshold": float(TS_THRESHOLD),
        "rows_1h": int(len(feats)), "n_feature_cols": n_feature_cols,
        "n_base_features": 38, "n_secondary_features": len(secondary_cols),
        "range": [str(feats["timestamp"].min()), str(feats["timestamp"].max())],
        "label_counts_CASH_LONG_SHORT": counts,
        "label_ratios": [round(c / max(len(feats), 1), 3) for c in counts],
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

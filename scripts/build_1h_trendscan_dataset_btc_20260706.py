#!/usr/bin/env python3
"""Sigma9 BTC leg: same 1h resample + feature set + trend-scanning labels as Sigma3
(build_1h_trendscan_dataset_20260705.py), but sourced from the raw BTCUSDT 5m klines instead of
the curated ETH training_features CSVs.

Data-availability constraint (checked 2026-07-06): only BTCUSDT and ETHUSDT have local kline data
(binance_data/klines/), and funding_rate/metrics (OI, top-trader ratio) only exist for ETHUSDT.
So this is OHLCV-only for BTC -- compute_features() skips the funding/OI/close_btc branches
naturally since those columns are absent. No lookahead: same causal 1h resample (label=left,
closed=left) and trend-scan labels (forward-window offline label construction, standard
convention, enter at next-bar open in the backtest).
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

from build_1h_trendscan_dataset_20260705 import compute_features, resample_1h  # noqa: E402
from build_trend_scanning_action_labels_20260531 import _trend_scan_fast  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_20260706"
SRC = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
TS_WINDOWS = [3, 6, 12, 24, 36, 48]
TS_THRESHOLD = 2.5
YEAR_RANGES = {
    2024: ("2024-01-01", "2024-12-31 23:59:59"),
    2025: ("2025-01-01", "2025-12-31 23:59:59"),
    2026: ("2026-01-01", "2026-12-31 23:59:59"),
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(SRC, low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    r_full = resample_1h(src)
    feats_full = compute_features(r_full)
    logc_full = np.log(np.maximum(feats_full["close"].to_numpy(dtype=np.float64), 1e-12))
    win = np.array(sorted(TS_WINDOWS), dtype=np.int32)
    t_vals, opt_l, betas = _trend_scan_fast(logc_full, win)
    labels = np.zeros(len(feats_full), dtype=np.int64)
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas > 0)] = 1
    labels[(np.abs(t_vals) >= TS_THRESHOLD) & (betas < 0)] = 2
    feats_full["ts_action"] = labels
    feats_full["ts_t_value"] = t_vals.astype(np.float32)
    feats_full["ts_opt_L"] = opt_l.astype(np.int16)

    summary = {}
    for year, (lo, hi) in YEAR_RANGES.items():
        m = (feats_full["timestamp"] >= lo) & (feats_full["timestamp"] <= hi)
        feats = feats_full.loc[m].reset_index(drop=True)
        if feats.empty:
            continue
        out_path = OUT_DIR / f"sigma9_btc_1h_{year}.parquet"
        feats.to_parquet(out_path, index=False)
        counts = np.bincount(feats["ts_action"].to_numpy(), minlength=3).tolist()
        summary[str(year)] = {
            "rows_1h": int(len(feats)),
            "range": [str(feats["timestamp"].min()), str(feats["timestamp"].max())],
            "label_counts_CASH_LONG_SHORT": counts,
            "label_ratios": [round(c / max(len(feats), 1), 3) for c in counts],
            "n_features": int(len([c for c in feats.columns if c not in ("timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L")])),
        }
        print(f"{year}: {summary[str(year)]}", flush=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps({"windows": win.tolist(), "threshold": TS_THRESHOLD, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Standalone correlation check for the 5 new BTC metrics4 features (taker_vol_ratio_z,
count_toptrader_ratio_z, toptrader_count_size_divergence, sig_whale, sig_oi_divergence) against
forward return, per year, on the isolated feature file
data/splits/year_oos/btc_features_2024_2026_metrics4_20260802.csv (built by
build_btc_features_metrics4_20260802.py). Purely diagnostic/screening -- causal features (known at
bar t) vs forward log-return over horizon bars, no lookahead. Per-year split checks temporal
stability (this project has repeatedly found BTC "promising" results to be settlement-lag/lookahead
artifacts that decay or flip sign year to year -- e.g. project-btc-cross-exchange-funding-divergence-
failed-20260802, project-btc-features-2026-drift-root-cause-found-20260801).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_metrics4_20260802.csv"

NEW_COLS = ["taker_vol_ratio_z", "count_toptrader_ratio_z", "toptrader_count_size_divergence",
            "sig_whale", "sig_oi_divergence"]
HORIZONS = {"h6_30m": 6, "h48_4h": 48, "h288_1d": 288}


def main() -> int:
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["year"] = df["timestamp"].dt.year

    close = df["close"].astype(float)
    for hname, h in HORIZONS.items():
        df[f"fwd_ret_{hname}"] = np.log(close.shift(-h) / close)

    rows = []
    for col in NEW_COLS:
        for hname in HORIZONS:
            fwd_col = f"fwd_ret_{hname}"
            for year in (2024, 2025, 2026):
                sub = df[(df["year"] == year)][[col, fwd_col]].dropna()
                if len(sub) < 200:
                    continue
                x = sub[col].to_numpy(dtype=float)
                y = sub[fwd_col].to_numpy(dtype=float)
                if np.std(x) < 1e-12:
                    continue
                r, p = stats.pearsonr(x, y)
                n = len(x)
                t = r * np.sqrt((n - 2) / max(1e-12, (1 - r ** 2)))
                rows.append({"feature": col, "horizon": hname, "year": year, "n": n,
                             "pearson_r": r, "t_stat": t, "p_value": p})
            # full-sample (all years combined) too
            sub_all = df[[col, fwd_col]].dropna()
            if len(sub_all) >= 200:
                x = sub_all[col].to_numpy(dtype=float)
                y = sub_all[fwd_col].to_numpy(dtype=float)
                if np.std(x) > 1e-12:
                    r, p = stats.pearsonr(x, y)
                    n = len(x)
                    t = r * np.sqrt((n - 2) / max(1e-12, (1 - r ** 2)))
                    rows.append({"feature": col, "horizon": hname, "year": "ALL", "n": n,
                                 "pearson_r": r, "t_stat": t, "p_value": p})

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print(out.to_string(index=False))

    out_path = ROOT / "tmp/research_20260802/btc_metrics4_standalone_correlation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

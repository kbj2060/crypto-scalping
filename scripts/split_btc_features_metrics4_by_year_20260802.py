"""Split the metrics4-expanded BTC feature CSV into year-based files, isolated from the baseline
data/splits/year_oos/btc_features_{2024,2025,2026}.csv (which other work in this repo depends on
being unchanged). Mirrors split_btc_features_by_year_20260708.py / the adaptive_squeeze/
regime_docs42 isolated-split convention (e.g. data/splits/year_oos_adaptive_squeeze_btc_20260720/).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/splits/year_oos/btc_features_2024_2026_metrics4_20260802.csv"
OUT_DIR = ROOT / "data/splits/year_oos_metrics4_btc_20260802"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for year in (2024, 2025, 2026):
        seg = df[df["timestamp"].dt.year == year].reset_index(drop=True)
        if seg.empty:
            continue
        out = OUT_DIR / f"btc_features_{year}.csv"
        seg.to_csv(out, index=False)
        print(f"{year}: {len(seg)} rows {seg['timestamp'].iloc[0]}..{seg['timestamp'].iloc[-1]} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

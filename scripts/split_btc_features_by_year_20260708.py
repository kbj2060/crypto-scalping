"""Split the combined BTC feature CSV into year-based files. Mirrors
split_sol_features_by_year_20260707.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
OUT_DIR = ROOT / "data/splits/year_oos"


def main() -> int:
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

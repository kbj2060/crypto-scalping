"""One-off audit script: verify BTC last_funding_rate causal alignment against a pure
reference BTCUSDT funding series built directly from binance_data/funding_rate_other zips.
Mirrors the methodology of docs/audits/last_funding_rate_source_audit_20260528.md (the ETH audit).
Read-only / diagnostic. Does not modify any data.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
FUNDING_DIR_PRIMARY = ROOT / "binance_data/funding_rate"
SYMBOL = "BTCUSDT"


def build_reference(funding_dir: Path, symbol: str) -> pd.DataFrame:
    frames = []
    for p in sorted(funding_dir.glob(f"{symbol}-fundingRate-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["calc_time"], unit="ms")
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "last_funding_rate"]]


def compare(feature_path: Path, ref: pd.DataFrame, label: str):
    df = pd.read_csv(feature_path, usecols=["timestamp", "last_funding_rate"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["last_funding_rate"]).sort_values("timestamp").reset_index(drop=True)

    prev = pd.merge_asof(df[["timestamp"]], ref, on="timestamp", direction="backward")
    nxt = pd.merge_asof(df[["timestamp"]], ref, on="timestamp", direction="forward")

    feat_val = df["last_funding_rate"].to_numpy()
    prev_val = prev["last_funding_rate"].to_numpy()
    next_val = nxt["last_funding_rate"].to_numpy()

    valid = ~np.isnan(prev_val) & ~np.isnan(next_val)
    n = valid.sum()
    prev_match = np.isclose(feat_val[valid], prev_val[valid], atol=1e-9).mean() * 100
    next_match = np.isclose(feat_val[valid], next_val[valid], atol=1e-9).mean() * 100

    print(f"| `{label}` | {prev_match:.3f}% | {next_match:.3f}% | n={n} |")
    return prev_match, next_match, n


def main():
    ref_other = build_reference(FUNDING_DIR, SYMBOL)
    print(f"Reference BTCUSDT funding (funding_rate_other): {len(ref_other)} rows, "
          f"{ref_other['timestamp'].iloc[0]} .. {ref_other['timestamp'].iloc[-1]}\n")

    # Check if a primary-dir BTCUSDT variant also exists
    primary_files = sorted(FUNDING_DIR_PRIMARY.glob(f"{SYMBOL}-fundingRate-*.zip"))
    print(f"BTCUSDT zips in binance_data/funding_rate/ (primary dir): {len(primary_files)}")
    if primary_files:
        ref_primary = build_reference(FUNDING_DIR_PRIMARY, SYMBOL)
        merged = ref_other.merge(ref_primary, on="timestamp", how="outer", suffixes=("_other", "_primary"))
        diff = (merged["last_funding_rate_other"] - merged["last_funding_rate_primary"]).abs()
        print(f"Max abs diff between the two sources: {diff.max()}")
    print()

    # Check for any non-BTCUSDT BTC-funding files (e.g. BTCFIUSDT wrong-symbol substitution)
    import subprocess
    print("Searching for any wrong-symbol BTC funding variants under binance_data/ ...")
    hits = []
    for base in [ROOT / "binance_data"]:
        for p in base.rglob("*fundingRate*"):
            if "BTC" in p.name.upper() and not p.name.upper().startswith("BTCUSDT"):
                hits.append(str(p))
    print(f"  wrong-symbol BTC funding files found: {hits if hits else 'NONE'}\n")

    print("| File | Previous match % | Next match % | Verdict |")
    print("|---|---:|---:|---|")
    files = {
        "data/splits/year_oos/btc_features_2024.csv": "btc_features_2024.csv",
        "data/splits/year_oos/btc_features_2025.csv": "btc_features_2025.csv",
        "data/splits/year_oos/btc_features_2026.csv": "btc_features_2026.csv",
        "data/splits/year_oos/btc_features_2024_2026.csv": "btc_features_2024_2026.csv",
        "data/splits/year_oos/btc_raw_frame_2024_2026.csv": "btc_raw_frame_2024_2026.csv",
    }
    for rel, label in files.items():
        path = ROOT / rel
        if path.exists():
            compare(path, ref_other, label)
        else:
            print(f"| `{label}` | MISSING FILE | | |")

    # freshness check
    print("\nFreshness check:")
    print("Last funding zip covers through:", ref_other["timestamp"].iloc[-1])
    latest_zip = sorted(FUNDING_DIR.glob(f"{SYMBOL}-fundingRate-*.zip"))[-1]
    print("Latest zip file:", latest_zip.name)


if __name__ == "__main__":
    raise SystemExit(main())

"""BTC replication, step 2: merge BTC's raw 5m klines + daily OI/top-trader metrics + monthly
funding rate into a single raw input frame matching features/engineering.py::FeatureEngineer.
process()'s column contract. Mirrors build_sol_raw_frame_20260707.py.

Cross-asset secondary series: FeatureEngineer.process(eth_df, btc_df) is hardcoded to name its two
args/columns "eth"(primary)/"btc"(cross-reference) from its original ETH-primary design, but
functionally it's just primary-vs-cross-reference (SOL's pilot fed SOL as the primary arg and real
BTC as the cross arg). Since BTC is now the PRIMARY asset here, it cannot cross-reference itself --
ETH is used as the cross-reference instead, populated into the required close_btc/volume_btc/
quote_volume_btc-named columns (same naming SOL used, just a different actual source asset).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "BTCUSDT"
KLINE_PATH = ROOT / f"binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
CROSS_KLINE_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_2024_2026.csv"


def load_metrics() -> pd.DataFrame:
    frames = []
    for p in sorted(METRICS_DIR.glob(f"{SYMBOL}-metrics-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["create_time"])
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]


def load_funding() -> pd.DataFrame:
    frames = []
    for p in sorted(FUNDING_DIR.glob(f"{SYMBOL}-fundingRate-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["calc_time"], unit="ms")
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "last_funding_rate"]]


def main() -> int:
    kline = pd.read_csv(KLINE_PATH, low_memory=False)
    kline["timestamp"] = pd.to_datetime(kline["timestamp"])
    kline = kline.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"BTC klines: {len(kline)} rows {kline['timestamp'].iloc[0]}..{kline['timestamp'].iloc[-1]}", flush=True)

    metrics = load_metrics()
    print(f"BTC metrics: {len(metrics)} rows {metrics['timestamp'].iloc[0]}..{metrics['timestamp'].iloc[-1]}", flush=True)
    funding = load_funding()
    print(f"BTC funding: {len(funding)} rows {funding['timestamp'].iloc[0]}..{funding['timestamp'].iloc[-1]}", flush=True)

    cross = pd.read_csv(CROSS_KLINE_PATH, low_memory=False, usecols=["timestamp", "close", "volume", "quote_volume"])
    cross["timestamp"] = pd.to_datetime(cross["timestamp"])
    cross = cross.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    cross = cross.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})

    frame = kline.merge(metrics, on="timestamp", how="left")
    frame = pd.merge_asof(frame.sort_values("timestamp"), funding.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    frame = frame.merge(cross[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")

    ffill_cols = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                  "close_btc", "volume_btc", "quote_volume_btc"]
    frame[ffill_cols] = frame[ffill_cols].ffill()

    required = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate",
                "close_btc", "volume_btc", "quote_volume_btc"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"missing required columns after merge: {missing}")

    coverage_ok = frame[required].notna().all(axis=1)
    first_ok = coverage_ok.idxmax()
    frame = frame.iloc[first_ok:].reset_index(drop=True)
    na_counts = frame[required].isna().sum()
    print("\nNA counts per required column after trimming to full-coverage region:")
    print(na_counts[na_counts > 0] if na_counts.any() else "  (none)")

    frame.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(frame)} rows {frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

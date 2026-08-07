"""Stage 0.5 extension: cross-sectional funding-rate/OI/positioning features from the 60-symbol
panel's already-downloaded metrics (binance_data/metrics/*.zip, 5m-native) and funding
(binance_data/funding_rate_other/*.zip, 8h-native) zips, aligned causally to BTC's 5m grid.

Complements scripts/build_btc_panel_marketstate_features_20260804.py (price/volume-only), which
was tested first (cheapest) and found to have no incremental Stage-0.5 signal. This extension
checks the other data source named in the design doc (funding/OI cross-section) before making the
final GO/NO-GO call, since the raw zips are already downloaded (Stage 0) -- the only added cost is
parsing, not new data acquisition.

All features are as-of-timestamp causal: for a given 5m bar, only funding/metrics values whose
own timestamp is <= that bar are used (merge_asof direction='backward'), never a future value.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_panel_funding_oi_features_20260804.parquet"

OI_CHG_WINDOW = 288  # 24h


def load_symbol_metrics(sym: str) -> pd.DataFrame:
    files = sorted(METRICS_DIR.glob(f"{sym}-metrics-*.zip"))
    dfs = []
    for fp in files:
        z = zipfile.ZipFile(fp)
        with z.open(z.namelist()[0]) as f:
            dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame(columns=["create_time", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                                      "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    df = pd.concat(dfs, ignore_index=True)
    df["create_time"] = pd.to_datetime(df["create_time"])
    return df.drop_duplicates(subset=["create_time"]).sort_values("create_time")


def load_symbol_funding(sym: str) -> pd.DataFrame:
    files = sorted(FUNDING_DIR.glob(f"{sym}-fundingRate-*.zip"))
    dfs = []
    for fp in files:
        z = zipfile.ZipFile(fp)
        with z.open(z.namelist()[0]) as f:
            dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame(columns=["calc_time", "last_funding_rate"])
    df = pd.concat(dfs, ignore_index=True)
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms")
    return df.drop_duplicates(subset=["calc_time"]).sort_values("calc_time")


def align_asof(btc_ts: pd.DataFrame, src: pd.DataFrame, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    if src.empty:
        out = btc_ts.copy()
        for c in value_cols:
            out[c] = np.nan
        return out
    return pd.merge_asof(btc_ts, src[[time_col] + value_cols], left_on="timestamp", right_on=time_col,
                          direction="backward")[["timestamp"] + value_cols]


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]

    btc_ts = pd.read_parquet(BTC_FRAME_PATH, columns=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"BTC grid: {len(btc_ts)} bars", flush=True)

    oi_wide, toptrader_wide, funding_wide = {}, {}, {}
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        m = load_symbol_metrics(sym)
        aligned = align_asof(btc_ts, m, "create_time", ["sum_open_interest_value", "sum_toptrader_long_short_ratio"])
        oi_wide[sym] = aligned["sum_open_interest_value"]
        toptrader_wide[sym] = aligned["sum_toptrader_long_short_ratio"]

        f = load_symbol_funding(sym)
        aligned_f = align_asof(btc_ts, f, "calc_time", ["last_funding_rate"])
        funding_wide[sym] = aligned_f["last_funding_rate"]

        if i % 15 == 0:
            print(f"  loaded {i}/{len(symbols)} symbols in {time.time()-t0:.1f}s", flush=True)

    oi_df = pd.DataFrame(oi_wide)
    toptrader_df = pd.DataFrame(toptrader_wide)
    funding_df = pd.DataFrame(funding_wide)
    print(f"loaded all panel metrics/funding in {time.time()-t0:.1f}s", flush=True)

    out = pd.DataFrame({"timestamp": btc_ts["timestamp"]})

    # --- OI: cross-sectional percentile rank of BTC's trailing OI growth ---
    oi_chg = oi_df / oi_df.shift(OI_CHG_WINDOW) - 1.0
    oi_chg_rank = oi_chg.rank(axis=1, pct=True, na_option="keep")
    out["panel_oi_chg_pctrank_btc"] = oi_chg_rank["BTCUSDT"]
    out["panel_oi_chg_dispersion"] = oi_chg.std(axis=1, skipna=True)

    # --- positioning: cross-sectional level/percentile of top-trader long/short ratio (crowding) ---
    toptrader_rank = toptrader_df.rank(axis=1, pct=True, na_option="keep")
    out["panel_toptrader_pctrank_btc"] = toptrader_rank["BTCUSDT"]
    out["panel_toptrader_avg"] = toptrader_df.mean(axis=1, skipna=True)

    # --- funding: cross-sectional percentile rank + dispersion ---
    funding_rank = funding_df.rank(axis=1, pct=True, na_option="keep")
    out["panel_funding_pctrank_btc"] = funding_rank["BTCUSDT"]
    out["panel_funding_dispersion"] = funding_df.std(axis=1, skipna=True)
    out["panel_funding_median"] = funding_df.median(axis=1, skipna=True)

    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={out.shape}", flush=True)
    print(out.describe().T[["count", "mean", "std", "min", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

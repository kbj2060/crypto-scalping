#!/usr/bin/env python3
"""Fetch the deployed GBM2 trend/chop regime state for the full pilot window (not just the live
120-bar tail live_regime_gbm2_trend_chop_signal_20260827.py returns), reusing that script's exact
model/feature pipeline unmodified. Binance's /futures/data/* OI and L/S-ratio endpoints have a
known ~30-day retention regardless of pagination (per prior research in this repo) -- so coverage
will likely fall short of the full 41-day pilot window; this script reports exactly how far back it
actually gets rather than assuming.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_regime_wide24_signal_20260826 import SYMBOL, BTC_SYMBOL, _fetch_klines, _fetch_data_api, _fetch_funding  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eth_regime_gbm2_trend_chop_20260827 import _apply_hysteresis  # noqa: E402

MODEL_PATH = ROOT / "tmp/eth_regime_gbm2_trend_chop_20260827/model.joblib"
CLASSES2 = ["chop", "trend"]
OUT_PATH = ROOT / "data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828/regime_gbm2_history.csv"

START = pd.Timestamp("2026-07-18 12:00:00", tz="UTC")
END = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tz is None else pd.Timestamp.utcnow()


def _load_spliced_oi_ratio() -> pd.DataFrame:
    """/futures/data/openInterestHist etc. only retain ~1.7 days at 5m period regardless of
    requested startTime (empirically confirmed -- returns a 400 past ~30d, and even within that
    ignores startTime and just returns the newest ~500 points). Splice two verified local sources
    instead: the archive (covers 2024-01~2026-08-22, see reference_clean_data_locations memory) for
    the older portion, and this pilot's own server-pulled oi_lsratio_5m.csv (covers 2026-08-22~now)
    for the recent portion -- both are the same underlying Binance metrics, just column-renamed."""
    archive = pd.read_csv(ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv", parse_dates=["create_time"])
    archive = archive.rename(columns={"create_time": "timestamp"})  # naive, already UTC (matches _fetch_klines convention)
    archive = archive[["timestamp", "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]

    live = pd.read_csv(
        ROOT / "data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828/oi_lsratio_5m.csv",
        parse_dates=["ts"])
    live["ts"] = pd.to_datetime(live["ts"], utc=True).dt.tz_localize(None)  # drop tz to match archive/klines convention
    live = live.rename(columns={"ts": "timestamp", "top_pos_ls_ratio": "sum_toptrader_long_short_ratio",
                                 "global_ls_ratio": "count_long_short_ratio"})
    live = live[["timestamp", "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]

    spliced = pd.concat([archive[archive["timestamp"] < live["timestamp"].min()], live]).sort_values("timestamp")
    spliced = spliced.drop_duplicates("timestamp").reset_index(drop=True)
    print(f"  spliced OI/ratio: archive {archive['timestamp'].min()}~{archive['timestamp'].max()} "
          f"+ live {live['timestamp'].min()}~{live['timestamp'].max()} -> combined {spliced['timestamp'].min()}~{spliced['timestamp'].max()}, {len(spliced)} rows")
    return spliced


def main() -> None:
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    print("fetching eth/btc klines + funding (REST, full depth OK)...")
    eth_kline = _fetch_klines(SYMBOL, start_ms, end_ms)
    btc_kline = _fetch_klines(BTC_SYMBOL, start_ms, end_ms)
    print(f"  eth klines: {len(eth_kline)} rows, {eth_kline['timestamp'].min()} -> {eth_kline['timestamp'].max()}")
    funding = _fetch_funding(SYMBOL, start_ms, end_ms)
    print("loading OI/L-S ratio from spliced local sources (REST endpoint too shallow, see docstring)...")
    oi_ratio = _load_spliced_oi_ratio()

    raw = eth_kline.copy()
    raw = pd.merge_asof(raw.sort_values("timestamp"), oi_ratio.sort_values("timestamp"), on="timestamp", direction="backward")
    raw = pd.merge_asof(raw.sort_values("timestamp"), funding, on="timestamp", direction="backward")

    btc = btc_kline.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})
    raw = raw.merge(btc[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")
    before = len(raw)
    raw = raw.dropna(subset=["close_btc", "sum_open_interest_value", "last_funding_rate",
                              "sum_toptrader_long_short_ratio", "count_long_short_ratio"]).reset_index(drop=True)
    print(f"  after merge+dropna: {len(raw)}/{before} rows survive, range {raw['timestamp'].min()} -> {raw['timestamp'].max()}")

    eth_raw_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
    btc_raw_cols = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
    eth_df = raw[eth_raw_cols].copy()
    btc_df = raw[btc_raw_cols].copy()

    print("running FeatureEngineer + raw_state12 + GBM2 model (reused verbatim from the live script)...")
    from features.engineering import FeatureEngineer
    feats = FeatureEngineer().process(eth_df, btc_df)
    feats = _with_raw_state12(feats)

    payload = joblib.load(MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        print(f"  WARNING: {len(missing)} model feature cols missing from engineered frame, median-filled: {missing}")
    for c in missing:
        feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)

    out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
    out["trend_prob"] = proba[:, CLASSES2.index("trend")]
    out = out.dropna().reset_index(drop=True)
    hcfg = payload["hysteresis_config"]
    confirmed_codes = _apply_hysteresis(out["trend_prob"].to_numpy(), hcfg["k_bars"], hcfg["band"])
    out["confirmed_state"] = [CLASSES2[i] for i in confirmed_codes]

    print(f"final regime series: {len(out)} bars, {out['timestamp'].min()} -> {out['timestamp'].max()}")
    print(out["confirmed_state"].value_counts())
    out.to_csv(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

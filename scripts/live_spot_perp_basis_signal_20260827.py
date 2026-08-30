#!/usr/bin/env python3
"""Read-only live "베이시스 청산압박" (spot-perp basis liquidation-pressure) model indicator,
2026-08-27: replaces the 독성(toxicity) slot on the Snapshot tab's model-indicator panel.
shadow_toxicity_score was independently confirmed uninformative on BOTH axes this repo tests --
direction (eth_microstructure_1m_history_archive_and_whale_confirmation_rejected_20260823) and
volatility-framing (eth_model_indicator_volatility_framing_screen_20260825: IC~+0.004~+0.014,
z<2 both) -- user request to replace it with a signal that DID replicate.

*** RISK GAUGE, NOT A PRICE-DIRECTION CALL. *** This does not claim price goes up or down -- it
reports which side (long or short) basis predicts will face MORE forced-liquidation $ volume over
the next 1-4h. See research_eth_spot_perp_basis_volatility_liquidation_reframe_20260827.py: basis_z48
extreme positive (contango) -> forward short-liquidation volume up (z=+3.9~+4.4 at 1h/4h) and
forward long-liquidation volume down (z=-4.3~-5.7); extreme negative (backwardation) -> mirror.
EXPLORATORY caveat carried into this live copy: that finding is from ~1 month of data
(data/live/tail_risk.duckdb's documented-reliable window starts 2026-07-18), a single window, not
this repo's usual 3-split TRAIN/VAL/OOS replication -- surfaced to the user via
MODEL_INDICATOR_DETAIL.liq_pressure in app.js, not hidden.

2026-08-29 (user request): the live chip's color now maps this same short/long-pressure read onto
the dashboard's standard directional palette (good=green=short_pressure i.e. long-favoring,
bad=red=long_pressure i.e. short-favoring, neutral=gray) instead of a calm/caution/danger severity
gauge, matching the other directional model-indicator chips (see DIRECTIONAL_MODEL_CHIP_KEYS in
app.js). This is a display-convention change, not a new validated price-direction claim -- the
underlying finding is still the liquidation-volume-by-side one above, not a price backtest.

Formula: basis_raw = (perp_close - spot_close) / spot_close (fapi.binance.com perp vs
api.binance.com spot, ETHUSDT); basis_z48 = 48-bar (4h) rolling z-score of basis_raw -- verbatim
from research_eth_spot_perp_basis_direction_cheap_gate_20260820.py's _load_basis_frame(), just
computed live here instead of from historical CSVs. No spot-kline collector runs continuously yet
(the historical spot CSV was a one-time backfill, eth_fetch_spot_klines_20260820.py) -- this module
fetches its own small rolling window (FETCH_ROWS below) every cache cycle instead, same "fetch
fresh each time" pattern live_regime_wide24_signal_20260826.py already uses for its own klines."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "ETHUSDT"
PERP_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "5m"
FETCH_ROWS = 500      # ~41.7h of 5m bars -- comfortable buffer above the 48-bar(4h) z-score warmup
Z_WINDOW = 48          # 4h, matches basis_z48's original definition verbatim
HISTORY_BARS = 48      # 4h strip, matches the other model indicators' convention
WARN_TH = 1.0
# 2026-08-31: HYPEUSDT has no Binance SPOT listing (api.binance.com/api/v3/klines returns
# {"code":-1121,"msg":"Invalid symbol."}, confirmed live) -- only the fapi.binance.com perp exists.
# That's permanent (a market-listing fact, not a "not enough data yet" warmup gap), so skip the
# spot fetch/retry entirely for these symbols rather than burning ~14s of retries every cache
# cycle on a request that can never succeed.
NO_SPOT_MARKET_SYMBOLS = {"HYPEUSDT"}


def _fetch_klines(url: str, symbol: str, limit: int = FETCH_ROWS, max_retries: int = 3, timeout: float = 15.0) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params={"symbol": symbol, "interval": INTERVAL, "limit": limit}, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                raise ValueError("empty klines response")
            df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                             "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
            df["close"] = df["close"].astype(np.float64)
            df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
            df["close_time"] = df["close_time"].astype(np.int64)
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            now_ms = int(time.time() * 1000)
            if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
                df = df.iloc[:-1].reset_index(drop=True)  # drop still-forming bar
            return df[["timestamp", "close"]]
        except Exception as e:  # noqa: BLE001 -- retry on any fetch/parse failure
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch klines from {url}: {last_err}")


def _direction(z: float) -> str:
    if z >= WARN_TH:
        return "short_pressure"  # positive basis (contango) extreme -> forward short-liq volume up
    if z <= -WARN_TH:
        return "long_pressure"   # negative basis (backwardation) extreme -> forward long-liq volume up
    return "neutral"


# 2026-08-29 user request: this chip should read short/long/no-signal like the dashboard's other
# directional model-indicators (good=long-favoring/bad=short-favoring/neutral=no lean), not the
# calm/caution/danger risk-severity gauge it used before. short_pressure (more shorts predicted to
# get forcibly liquidated -> squeeze-driven buying -> bullish/long-favoring) maps to "good";
# long_pressure mirrors to "bad". WARN_TH alone still gates neutral vs directional via _direction().
_DIRECTION_TONE = {"short_pressure": "good", "long_pressure": "bad", "neutral": "neutral"}


def _tone(z: float) -> str:
    return _DIRECTION_TONE[_direction(z)]


def compute_basis_liquidation_signal(symbol: str = SYMBOL) -> dict[str, Any]:
    """Returns {"warmed_up", "error", "basis_z48", "direction", "tone", "tone_history",
    "bars_loaded", "latest_ts_utc"}. tone_history is oldest-to-newest (same convention as
    live_oi_delta_signal_20260824.py) so the Snapshot tab's activity strip survives a page refresh.
    Never raises -- degrades to warmed_up=False on any fetch/compute problem."""
    if symbol in NO_SPOT_MARKET_SYMBOLS:
        return _empty("no_spot_market")
    try:
        perp = _fetch_klines(PERP_KLINES_URL, symbol).rename(columns={"close": "perp_close"})
        spot = _fetch_klines(SPOT_KLINES_URL, symbol).rename(columns={"close": "spot_close"})
    except RuntimeError as e:
        return _empty(f"fetch_failed: {e}")

    df = perp.merge(spot, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    if len(df) < Z_WINDOW:
        return _empty("insufficient_overlap", bars_loaded=int(len(df)))

    df["basis_raw"] = (df["perp_close"] - df["spot_close"]) / df["spot_close"]
    roll = df["basis_raw"].rolling(Z_WINDOW)
    df["basis_z48"] = (df["basis_raw"] - roll.mean()) / roll.std()

    latest_z = df["basis_z48"].iloc[-1]
    if pd.isna(latest_z):
        return _empty("not_warmed_up_yet", bars_loaded=int(len(df)))

    z = float(latest_z)
    tone_series = df["basis_z48"].dropna().tail(HISTORY_BARS).apply(_tone)
    return {
        "warmed_up": True,
        "error": None,
        "basis_z48": z,
        "direction": _direction(z),
        "tone": _tone(z),
        "tone_history": tone_series.tolist(),
        "bars_loaded": int(len(df)),
        "latest_ts_utc": df["timestamp"].iloc[-1].isoformat(),
    }


def _empty(error: str, **extra) -> dict[str, Any]:
    out = {"warmed_up": False, "error": error, "basis_z48": None, "direction": None,
           "tone": "neutral", "tone_history": [], "bars_loaded": 0, "latest_ts_utc": None}
    out.update(extra)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(compute_basis_liquidation_signal(), indent=2, default=str))

"""Stage 1 (Rho1 panel design) step 1: build a common, symbol-agnostic causal feature set for
ALL 60 panel symbols (data/splits/panel_universe_symbols_20260804.json), so they can be pooled
into one training set for the panel backbone.

Deliberately restricted to per-symbol OHLCV + own funding/OI/positioning (no cross-symbol
features, no BTC-specific features like eth_btc_ret_spread) -- cross-symbol context is the
BACKBONE's job (cross-symbol attention over the pooled batch), not a hand-engineered input
column. This keeps every symbol's feature vector directly comparable, which a shared model
requires (unlike causalfix_final's BTC-only 114 cols, several of which reference ETH/BTC by
name and can't be computed symmetrically for e.g. DOGEUSDT).

All indicators are computed causally (rolling/shift, no centered windows, no future data).
Output: one parquet per symbol under data/panel/features/{SYMBOL}.parquet, long-format columns
identical across symbols so they concatenate directly into a pooled training frame.
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
KLINES_DIR = ROOT / "binance_data/klines"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
OUT_DIR = ROOT / "data/panel/features"

FEATURE_COLS = [
    "ret_1", "realized_vol_12", "realized_vol_48", "realized_vol_288",
    "rsi_14", "macd_hist", "bb_width_20", "atr_pct_14",
    "rvol_12", "rvol_48", "taker_buy_ratio",
    "hour_sin", "hour_cos",
    "funding_rate", "funding_roc_288",
    "oi_chg_288", "toptrader_ratio", "taker_long_short_vol_ratio",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close


def load_metrics(sym: str) -> pd.DataFrame:
    files = sorted(METRICS_DIR.glob(f"{sym}-metrics-*.zip"))
    dfs = []
    for fp in files:
        z = zipfile.ZipFile(fp)
        with z.open(z.namelist()[0]) as f:
            dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame(columns=["create_time", "sum_open_interest_value",
                                      "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    df = pd.concat(dfs, ignore_index=True)
    df["create_time"] = pd.to_datetime(df["create_time"])
    return df.drop_duplicates(subset=["create_time"]).sort_values("create_time")


def load_funding(sym: str) -> pd.DataFrame:
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


def build_symbol_features(sym: str) -> pd.DataFrame:
    kl = pd.read_csv(KLINES_DIR / sym / f"{sym}-5m-api.csv", low_memory=False)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"])
    kl = kl.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    close, high, low, volume = kl["close"], kl["high"], kl["low"], kl["volume"]
    quote_volume, taker_buy_quote = kl["quote_volume"], kl["taker_buy_quote"]

    out = pd.DataFrame({"timestamp": kl["timestamp"], "symbol": sym, "open": kl["open"],
                         "high": high, "low": low, "close": close})
    out["ret_1"] = np.log(close / close.shift(1))
    out["realized_vol_12"] = out["ret_1"].rolling(12).std()
    out["realized_vol_48"] = out["ret_1"].rolling(48).std()
    out["realized_vol_288"] = out["ret_1"].rolling(288).std()
    out["rsi_14"] = _rsi(close)
    out["macd_hist"] = _macd_hist(close) / close  # scale-free
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_width_20"] = (4 * std20) / sma20.replace(0, np.nan)
    out["atr_pct_14"] = _atr_pct(high, low, close)
    vol_ma12 = volume.rolling(48).mean()
    out["rvol_12"] = (volume.rolling(12).mean() / vol_ma12.replace(0, np.nan)).clip(0, 20)
    out["rvol_48"] = (volume / vol_ma12.replace(0, np.nan)).clip(0, 20)
    out["taker_buy_ratio"] = taker_buy_quote / quote_volume.replace(0, np.nan)
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    m = load_metrics(sym)
    if not m.empty:
        m_aligned = pd.merge_asof(out[["timestamp"]], m, left_on="timestamp", right_on="create_time",
                                   direction="backward")
        oi = m_aligned["sum_open_interest_value"]
        out["oi_chg_288"] = oi.pct_change(288).clip(-1.0, 5.0)
        # Both are positive ratios nominally centered near 1 -- log-compress and clip to guard
        # against rare corrupt/extreme metrics-feed values (seen as high as 8.9e7 on some
        # low-liquidity symbols, e.g. 1000SATSUSDT) that would otherwise blow up training.
        out["toptrader_ratio"] = np.log(m_aligned["sum_toptrader_long_short_ratio"].clip(lower=1e-3)).clip(-5, 5)
        out["taker_long_short_vol_ratio"] = np.log(
            m_aligned["sum_taker_long_short_vol_ratio"].clip(lower=1e-3)).clip(-5, 5)
    else:
        out["oi_chg_288"] = np.nan
        out["toptrader_ratio"] = np.nan
        out["taker_long_short_vol_ratio"] = np.nan

    f = load_funding(sym)
    if not f.empty:
        f_aligned = pd.merge_asof(out[["timestamp"]], f, left_on="timestamp", right_on="calc_time",
                                   direction="backward")
        out["funding_rate"] = f_aligned["last_funding_rate"]
        out["funding_roc_288"] = out["funding_rate"].diff(288)
    else:
        out["funding_rate"] = np.nan
        out["funding_roc_288"] = np.nan

    return out


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        feats = build_symbol_features(sym)
        feats.to_parquet(OUT_DIR / f"{sym}.parquet", index=False)
        n_nan = feats[FEATURE_COLS].isna().sum().sum()
        print(f"[{i}/{len(symbols)}] {sym:16s} rows={len(feats):>8d} nan_cells={n_nan:>7d} "
              f"({time.time()-t0:.1f}s elapsed)", flush=True)
    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

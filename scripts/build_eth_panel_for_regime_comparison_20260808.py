"""Build an ETH feature panel + TB trade-outcome labels with the IDENTICAL pipeline used for SOL
(scripts/build_sol_raw_frame_20260707.py -> build_sol_features_20260707.py ->
build_sol_5m_tripbarrier_tradeoutcome_labels_20260807.py), for the ETH-vs-SOL regime/feature
comparison (2026-08-08). Same window 2024-06-01.., same FeatureEngineer, same label constants.

Outputs:
  data/splits/year_oos/eth_features_2024_2026_analysis.csv
  data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from features.engineering import FeatureEngineer  # noqa: E402
from build_sol_5m_tripbarrier_tradeoutcome_labels_20260807 import (  # noqa: E402
    _triple_barrier_race, CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS, SOFT_TEMPERATURE,
)

SYMBOL = "ETHUSDT"
KLINE_PATH = ROOT / f"binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv"
BTC_KLINE_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
FEATURES_OUT = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
LABELS_OUT = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
START = pd.Timestamp("2024-06-01")

ETH_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_base", "taker_buy_quote",
                "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "last_funding_rate"]
BTC_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]


def load_zip_series(directory: Path, pattern: str, time_col: str, unit: str | None, cols: list[str]) -> pd.DataFrame:
    frames = []
    for p in sorted(directory.glob(pattern)):
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                frames.append(pd.read_csv(f))
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out[time_col], unit=unit) if unit else pd.to_datetime(out[time_col])
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp"] + cols]


def main() -> int:
    kline = pd.read_csv(KLINE_PATH, low_memory=False)
    kline["timestamp"] = pd.to_datetime(kline["timestamp"])
    kline = kline.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    kline = kline[kline["timestamp"] >= START].reset_index(drop=True)

    metrics = load_zip_series(METRICS_DIR, f"{SYMBOL}-metrics-*.zip", "create_time", None,
                              ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"])
    funding = load_zip_series(FUNDING_DIR, f"{SYMBOL}-fundingRate-*.zip", "calc_time", "ms", ["last_funding_rate"])

    btc = pd.read_csv(BTC_KLINE_PATH, usecols=["timestamp", "close", "volume", "quote_volume"], low_memory=False)
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc = btc.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    raw = kline[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                 "taker_buy_base", "taker_buy_quote"]].copy()
    raw = pd.merge_asof(raw, metrics, on="timestamp", direction="backward")
    raw = pd.merge_asof(raw, funding, on="timestamp", direction="backward")
    raw = raw.merge(btc, on="timestamp", how="left")
    for c in ("close_btc", "volume_btc", "quote_volume_btc"):
        raw[c] = raw[c].ffill()
    raw = raw.dropna(subset=["close_btc", "sum_open_interest_value", "last_funding_rate"]).reset_index(drop=True)
    print(f"ETH raw frame: {len(raw)} rows {raw['timestamp'].iloc[0]}..{raw['timestamp'].iloc[-1]}", flush=True)

    eth_df = raw[ETH_RAW_COLS].copy()
    btc_df = raw[BTC_RAW_COLS].copy()
    fe = FeatureEngineer()
    feats = fe.process(eth_df, btc_df)
    feats.to_csv(FEATURES_OUT, index=False)
    print(f"wrote {FEATURES_OUT}: {len(feats)} rows x {len(feats.columns)} cols", flush=True)

    panel = feats[["timestamp", "open", "high", "low", "close"]].copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    open_ = panel["open"].to_numpy(dtype=np.float64)
    high = panel["high"].to_numpy(dtype=np.float64)
    low = panel["low"].to_numpy(dtype=np.float64)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_move, sl_move = TP_MULT * vol, SL_MULT * vol
    label, long_score, short_score = _triple_barrier_race(open_, high, low, tp_move, sl_move, HORIZON_BARS)
    cash_score = np.zeros(len(label))
    logits = np.stack([cash_score, long_score, short_score], axis=1) / SOFT_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    soft = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    out = pd.DataFrame({
        "timestamp": panel["timestamp"], "trade_outcome_action": label,
        "trade_outcome_soft_cash": soft[:, 0], "trade_outcome_soft_long": soft[:, 1],
        "trade_outcome_soft_short": soft[:, 2], "tp_move": tp_move, "sl_move": sl_move,
    })
    out.to_parquet(LABELS_OUT, index=False)
    counts = pd.Series(label).value_counts().sort_index()
    print(f"wrote {LABELS_OUT}; label counts CASH/LONG/SHORT = "
          f"{int(counts.get(0, 0))}/{int(counts.get(1, 0))}/{int(counts.get(2, 0))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

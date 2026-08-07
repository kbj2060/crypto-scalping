"""Build the Kappa1 primitive-score dataset (Stage 1 of
docs/btc_kappa1_invariant_composite_policy_design_20260807.md).

5m cadence. Every column at bar t uses only information available at bar t's close:
  - OHLCV + ATR% from the pinned raw 5m frame
  - gmm_cluster_rank / gmm_confidence / if_score: cached causal series (2026-08-02 line)
  - evt_raw_score / evt_agreement / evt_gate_fired: causal event-gate series (2026-08-04
    prototype; its forward-looking event_label evaluation column is deliberately EXCLUDED)
  - flow_mean_5m / flow_mean_15m: mean 1m net_taker_ratio inside bar t / last 3 bars

Span: gate-series start (2025-01-01) .. 1m-data end (2026-07-12).
Output: data/splits/year_oos/btc_kappa1_primitives_5m_20260807.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_5M = ROOT / "data/splits/year_oos/btc_raw_frame_2024_2026.csv"
GMM = ROOT / "tmp/research_20260802/btc_gmm_volatility_signal_check/gmm_score_series_full.csv"
IFS = ROOT / "tmp/research_20260802/btc_isolation_forest_signal_check/if_score_series_full.csv"
GATE = ROOT / "tmp/research_20260804/btc_event_gate_prototype/gate_series.csv"
M1 = ROOT / "data/training_features_1m_causal_btc.csv"
OUT = ROOT / "data/splits/year_oos/btc_kappa1_primitives_5m_20260807.parquet"


def main() -> None:
    raw = pd.read_csv(RAW_5M, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    prev_close = raw["close"].shift()
    true_range = pd.concat([raw["high"] - raw["low"], (raw["high"] - prev_close).abs(),
                            (raw["low"] - prev_close).abs()], axis=1).max(axis=1)
    raw["atr_pct_14"] = true_range.rolling(14, min_periods=14).mean() / raw["close"]
    raw["atr_pct_96"] = true_range.rolling(96, min_periods=96).mean() / raw["close"]

    gmm = pd.read_csv(GMM, usecols=["timestamp", "gmm_cluster_rank", "gmm_confidence"])
    ifs = pd.read_csv(IFS, usecols=["timestamp", "if_score"])
    gate = pd.read_csv(GATE, usecols=["timestamp", "raw_score", "agreement", "gate_fired"])
    gate = gate.rename(columns={"raw_score": "evt_raw_score", "agreement": "evt_agreement",
                                "gate_fired": "evt_gate_fired"})
    gate["evt_gate_fired"] = gate["evt_gate_fired"].astype(int)
    for frame in (gmm, ifs, gate):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    m1 = pd.read_csv(M1, usecols=["timestamp", "net_taker_ratio"])
    m1["timestamp"] = pd.to_datetime(m1["timestamp"])
    m1 = m1.set_index("timestamp").sort_index()
    flow = m1["net_taker_ratio"].resample("5min", label="left", closed="left").mean()
    flow_frame = flow.rename("flow_mean_5m").reset_index()
    flow_frame["flow_mean_15m"] = flow_frame["flow_mean_5m"].rolling(3, min_periods=3).mean()

    out = raw.merge(gmm, on="timestamp", how="inner").merge(ifs, on="timestamp", how="inner")
    out = out.merge(gate, on="timestamp", how="inner").merge(flow_frame, on="timestamp", how="inner")
    out = out.dropna(subset=["atr_pct_96", "flow_mean_15m"]).reset_index(drop=True)
    assert out["timestamp"].is_monotonic_increasing and out["timestamp"].is_unique
    out.to_parquet(OUT, index=False)
    print(f"rows={len(out)} cols={len(out.columns)} span={out['timestamp'].iloc[0]} .. {out['timestamp'].iloc[-1]}")
    print(f"gate fired bars: {int(out['evt_gate_fired'].sum())}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

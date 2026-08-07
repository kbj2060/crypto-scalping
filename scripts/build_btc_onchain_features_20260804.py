"""New-axis Stage 0.5 (on-chain, per docs/btc_panel_crossasset_architecture_design_20260804.md's
"cheapest falsification first" principle, same pattern as build_btc_dvol_features_20260804.py):
build causal on-chain-derived features for BTC, aligned to the same 5m grid as causalfix_final,
BEFORE building any new model.

All features use merge_asof(direction="backward") against an explicit ``available_at`` timestamp.
CoinMetrics daily rows carry a per-field '-status-time' (when the value was finalized, ~1-3h after
UTC midnight); the conservative safe contract used here is the same one already used for DVOL --
make each day's value available one full period (1 day) after its own timestamp, which is always
later than the observed status-time.

Features:
- mvrv: raw CapMVRVCur level (market value / realized value -- classic on-chain valuation metric)
- mvrv_pctrank_90d: trailing 90-day percentile rank of MVRV (cheap or rich relative to own range)
- net_exchange_flow_pct_supply: (FlowInExNtv - FlowOutExNtv) / SplyExNtv -- net exchange inflow
  normalized by exchange-held supply (raw native-unit flow isn't comparable across the growing BTC
  supply on exchanges over multi-year history)
- sply_ex_roc_7d: 7-day rate of change of exchange-held supply (trend in exchange balances)
- active_addr_roc_7d: 7-day rate of change of active addresses (network activity momentum)
- active_addr_pctrank_90d: trailing 90-day percentile rank of active addresses
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ONCHAIN_PATH = ROOT / "data/onchain/coinmetrics/btc_onchain_daily.csv"
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_onchain_features_20260804.parquet"


def _rolling_pctrank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 4).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)


def main() -> int:
    btc_ts = pd.read_parquet(BTC_FRAME_PATH, columns=["timestamp"]).sort_values("timestamp")

    onchain = pd.read_csv(ONCHAIN_PATH)
    onchain["time"] = pd.to_datetime(onchain["time"]).dt.tz_localize(None)
    onchain = onchain.sort_values("time").reset_index(drop=True)

    onchain["mvrv"] = onchain["CapMVRVCur"]
    onchain["mvrv_pctrank_90d"] = _rolling_pctrank(onchain["mvrv"], 90)
    onchain["net_exchange_flow_pct_supply"] = (
        (onchain["FlowInExNtv"] - onchain["FlowOutExNtv"]) / onchain["SplyExNtv"]
    )
    onchain["sply_ex_roc_7d"] = onchain["SplyExNtv"].pct_change(7)
    onchain["active_addr_roc_7d"] = onchain["AdrActCnt"].pct_change(7)
    onchain["active_addr_pctrank_90d"] = _rolling_pctrank(onchain["AdrActCnt"], 90)

    feat_cols = [
        "mvrv", "mvrv_pctrank_90d", "net_exchange_flow_pct_supply",
        "sply_ex_roc_7d", "active_addr_roc_7d", "active_addr_pctrank_90d",
    ]
    onchain["available_at"] = (onchain["time"] + pd.Timedelta(days=1)).astype("datetime64[ns]")
    onchain_causal = onchain[["available_at"] + feat_cols].rename(columns={"available_at": "timestamp"})

    merged = pd.merge_asof(btc_ts, onchain_causal, on="timestamp", direction="backward")

    out = merged[["timestamp"] + feat_cols]
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(out.describe().T[["count", "mean", "std", "min", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

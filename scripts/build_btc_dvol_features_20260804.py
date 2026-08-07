"""New-axis Stage 0.5 (per docs/btc_panel_crossasset_architecture_design_20260804.md's "cheapest
falsification first" principle, now applied to the Deribit DVOL axis instead of the closed panel
axis): build causal DVOL-derived features for BTC, aligned to the same 5m grid as
causalfix_final, BEFORE building any new model.

All features use merge_asof(direction="backward") against an explicit ``available_at`` timestamp.
Deribit returns OHLC candles but does not document the candle timestamp as a close/publication
timestamp, so the safe contract makes an hourly candle available one full hour after its timestamp.

Features:
- dvol_btc, dvol_eth: raw level (annualized implied vol %, e.g. 45.0 = 45%)
- dvol_btc_eth_spread: cross-asset richness (BTC options priced relatively more/less than ETH's)
- dvol_btc_pctrank_720h: trailing 30-day (720h) percentile rank of BTC DVOL -- "is IV cheap or
  rich right now relative to its own recent range"
- dvol_btc_roc_24h, dvol_btc_roc_168h: 1-day / 1-week rate of change (vol-of-vol momentum)
- vol_risk_premium: DVOL (implied, annualized) minus BTC's own trailing realized vol
  (realized_vol_288, annualized the same way) -- the classic "options market expects more/less
  movement than has actually happened lately" signal from the options-pricing literature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DVOL_DIR = ROOT / "data/derivatives/deribit_dvol"
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
# realized_vol_288 isn't a causalfix_final column -- reuse the one already computed for the
# (now-closed) panel line in data/panel/features/BTCUSDT.parquet (build_panel_common_features_20260804.py),
# same definition (rolling std of 5m log returns, 288-bar/24h window), to avoid recomputing it.
BTC_PANEL_FEATURES_PATH = ROOT / "data/panel/features/BTCUSDT.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_dvol_features_20260804.parquet"

BARS_PER_YEAR = 288 * 365  # 5m bars/day * days/year, for annualizing realized_vol_288


def load_dvol(currency: str) -> pd.DataFrame:
    df = pd.read_csv(DVOL_DIR / f"{currency}_dvol_hourly.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    return (
        df[["available_at", "close"]]
        .rename(columns={"available_at": "timestamp", "close": f"dvol_{currency.lower()}"})
        .sort_values("timestamp")
    )


def _rolling_pctrank(s: pd.Series, window: int) -> pd.Series:
    """Fraction of the trailing `window` values <= the current value -- computed on the native
    (hourly) series where window sizes are small, rather than on the 5m-upsampled version where
    the same computation would be ~30x more rows for no extra information (DVOL is flat within
    each hour after the causal forward-fill)."""
    return s.rolling(window, min_periods=window // 4).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)


def main() -> int:
    btc_ts = pd.read_parquet(BTC_FRAME_PATH, columns=["timestamp"]).sort_values("timestamp")
    btc_vol = pd.read_parquet(BTC_PANEL_FEATURES_PATH, columns=["timestamp", "realized_vol_288"])
    btc_ts = btc_ts.merge(btc_vol, on="timestamp", how="left")

    dvol_btc = load_dvol("BTC")
    dvol_eth = load_dvol("ETH")

    # compute rate-of-change / percentile-rank on the NATIVE hourly series (720h/24h/168h windows
    # in hourly units) -- cheap -- then causally merge_asof the already-computed derived columns
    # onto the 5m grid, same as the raw level itself.
    dvol_btc = dvol_btc.copy()
    dvol_btc["dvol_btc_pctrank_720h"] = _rolling_pctrank(dvol_btc["dvol_btc"], 720)
    dvol_btc["dvol_btc_roc_24h"] = dvol_btc["dvol_btc"].pct_change(24)
    dvol_btc["dvol_btc_roc_168h"] = dvol_btc["dvol_btc"].pct_change(168)

    merged = pd.merge_asof(btc_ts, dvol_btc, on="timestamp", direction="backward")
    merged = pd.merge_asof(merged, dvol_eth, on="timestamp", direction="backward")

    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    out["dvol_btc"] = merged["dvol_btc"]
    out["dvol_eth"] = merged["dvol_eth"]
    out["dvol_btc_eth_spread"] = merged["dvol_btc"] - merged["dvol_eth"]
    out["dvol_btc_pctrank_720h"] = merged["dvol_btc_pctrank_720h"]
    out["dvol_btc_roc_24h"] = merged["dvol_btc_roc_24h"]
    out["dvol_btc_roc_168h"] = merged["dvol_btc_roc_168h"]

    realized_vol_annualized_pct = merged["realized_vol_288"] * np.sqrt(BARS_PER_YEAR) * 100
    out["vol_risk_premium"] = merged["dvol_btc"] - realized_vol_annualized_pct

    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(out.describe().T[["count", "mean", "std", "min", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

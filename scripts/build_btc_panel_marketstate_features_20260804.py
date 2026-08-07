"""Stage 0.5 (Rho1 panel design, docs/btc_panel_crossasset_architecture_design_20260804.md):
build ~12-15 causal cross-sectional market-state features from the 60-symbol panel
(data/splits/panel_universe_symbols_20260804.json), aligned to BTC's own 5m timestamp grid, to
be merged into the existing causalfix_final 114-col BTC frame for the cheap GO/NO-GO
falsification test (Stage 0.5) BEFORE any new model/backbone is built.

Scope note: this deliberately uses only price/volume from the panel klines (already
downloaded, no further I/O cost) -- NOT the metrics (OI/positioning) or funding zips. Those are
richer but require parsing tens of thousands of per-symbol daily/monthly zip files; per the
design doc's "cheapest reproducible falsification first" principle, we test whether price/volume
cross-sectional information alone has any incremental signal before paying that parsing cost.
If Stage 0.5 passes, funding/OI cross-section is a natural Stage 1 enhancement; if it doesn't,
that additional engineering is avoided entirely.

All features are computed causally: at row t, only klines up to and including bar t are used.
Symbols not yet listed at time t are excluded from that row's cross-sectional aggregates (their
column is NaN before their own onboard/listing date) -- this avoids a lookahead where a later-
listed altcoin's absence would otherwise bias breadth/dispersion low for the pre-listing period.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
KLINES_DIR = ROOT / "binance_data/klines"
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_panel_marketstate_features_20260804.parquet"

MA_WINDOWS = [96, 288]      # 8h, 24h breadth lookbacks
RET_WINDOWS = [12, 48]      # 1h, 4h dispersion lookbacks
CORR_WINDOW = 288           # 24h rolling correlation-to-index window
LEADLAG_LAGS = [1, 2, 3]    # bars (5/10/15 min) alt-leads-BTC check


def load_panel_closes_volumes(symbols: list[str], btc_ts: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = {}
    volumes = {}
    for i, sym in enumerate(symbols, 1):
        path = KLINES_DIR / sym / f"{sym}-5m-api.csv"
        df = pd.read_csv(path, usecols=["timestamp", "close", "quote_volume"], low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        df = df.reindex(btc_ts)  # align to BTC's own grid; NaN where symbol has no bar (not listed yet / gap)
        closes[sym] = df["close"]
        volumes[sym] = df["quote_volume"]
        if i % 15 == 0:
            print(f"  loaded {i}/{len(symbols)} symbols", flush=True)
    return pd.DataFrame(closes), pd.DataFrame(volumes)


def build_features(closes: pd.DataFrame, volumes: pd.DataFrame, btc_symbol: str = "BTCUSDT") -> pd.DataFrame:
    n_listed = closes.notna().sum(axis=1)  # causal: only counts symbols with a bar at/before this timestamp
    log_ret = np.log(closes / closes.shift(1))

    out = pd.DataFrame(index=closes.index)
    out["panel_n_listed"] = n_listed

    # --- breadth: fraction of currently-listed symbols trading above their own rolling MA ---
    for w in MA_WINDOWS:
        ma = closes.rolling(w, min_periods=max(5, w // 4)).mean()
        above = (closes > ma)
        above = above.where(closes.notna())  # keep NaN where not listed, so mean() ignores it
        out[f"panel_breadth_{w}"] = above.mean(axis=1, skipna=True)

    # --- dispersion: cross-sectional std of trailing N-bar returns ---
    for w in RET_WINDOWS:
        fwd_ret = closes / closes.shift(w) - 1.0
        out[f"panel_dispersion_{w}"] = fwd_ret.std(axis=1, skipna=True)
        out[f"panel_dispersion_{w}_median_abs"] = fwd_ret.abs().median(axis=1, skipna=True)

    # --- correlation regime: rolling correlation of each symbol's return to the equal-weight
    #     panel return, averaged across symbols -- proxy for "market mode" strength, cheaper
    #     than a full pairwise correlation matrix / eigen-decomposition ---
    eq_weight_ret = log_ret.mean(axis=1, skipna=True)
    corr_to_index = log_ret.rolling(CORR_WINDOW, min_periods=CORR_WINDOW // 4).corr(eq_weight_ret)
    out["panel_avg_corr_to_index_288"] = corr_to_index.mean(axis=1, skipna=True)

    # --- dominance: BTC's share of total panel quote volume, and its momentum ---
    total_vol = volumes.sum(axis=1, skipna=True)
    btc_dom = volumes[btc_symbol] / total_vol.replace(0, np.nan)
    out["panel_btc_dominance"] = btc_dom
    out["panel_btc_dominance_chg_288"] = btc_dom - btc_dom.shift(288)

    # --- lead-lag: correlation of BTC's return with the panel's (ex-BTC) average LAGGED
    #     return, at several lags -- tests whether alts move ahead of BTC ---
    alt_cols = [c for c in log_ret.columns if c != btc_symbol]
    alt_avg_ret = log_ret[alt_cols].mean(axis=1, skipna=True)
    btc_ret = log_ret[btc_symbol]
    for lag in LEADLAG_LAGS:
        alt_lagged = alt_avg_ret.shift(lag)
        out[f"panel_leadlag_altret_lag{lag}"] = alt_lagged
        out[f"panel_leadlag_corr_lag{lag}_48"] = btc_ret.rolling(48, min_periods=12).corr(alt_lagged)

    out["panel_btc_ret_1"] = btc_ret
    return out


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]

    btc_frame = pd.read_parquet(BTC_FRAME_PATH, columns=["timestamp"]).sort_values("timestamp")
    btc_ts = pd.DatetimeIndex(btc_frame["timestamp"])
    print(f"BTC grid: {len(btc_ts)} bars, {btc_ts.min()}..{btc_ts.max()}", flush=True)

    t0 = time.time()
    closes, volumes = load_panel_closes_volumes(symbols, btc_ts)
    print(f"loaded panel closes/volumes in {time.time()-t0:.1f}s, shape={closes.shape}", flush=True)

    t0 = time.time()
    feats = build_features(closes, volumes)
    print(f"built features in {time.time()-t0:.1f}s, shape={feats.shape}", flush=True)

    feats = feats.reset_index().rename(columns={"index": "timestamp"})
    feats.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}", flush=True)
    print(feats.describe().T[["count", "mean", "std", "min", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

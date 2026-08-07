"""Predictive-content analysis of the live ETH microstructure_1m duckdb table for a 1-minute
scalping model design (2026-07-18).

Causality contract (derived from microstructure_scanner.py _scan_loop):
  - A row labeled ts=T (KST, bar bucket) is INSERTed at wall-clock ~T+60..75s and its rolling
    windows include data up to that write moment (i.e., up to ~15s past the T+1min boundary).
  - Therefore at a decision made on the minute boundary D, the newest *safely* available row is
    ts = D - 2min (written by D-45s at the latest). Row ts = D-1min is written at D+0..15s and
    must NOT be used at decision D.
  - Live is fresher than this (the bot reads a 10s-refresh in-memory cache), so this backtest
    alignment is conservative, never optimistic.

Forward returns are computed off kline OPEN at D (the first executable price after the decision)
to close at D+h. Kline timestamps are bar-OPEN times (UTC) -- the bar starting at D opens at D.

Output: per-feature daily-block Spearman IC (mean, t-stat over daily ICs) at multiple horizons,
plus decile long-short spreads for the strongest features. Pure analysis -- no model, no fees.
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "data", "live", "microstructure.duckdb")
KLINES = os.path.join(_ROOT, "binance_data", "klines", "ETHUSDT", "ETHUSDT-1m-api.csv")
OUT_CSV = os.path.join(_ROOT, "data", "ensemble", "reports", "microstructure_edge_ic_20260718.csv")

HORIZONS = [1, 2, 3, 5, 10, 15, 30, 60]
AVAIL_SHIFT_MIN = 2  # micro row ts=T usable from decision D = T + 2min onward


def load_micro() -> pd.DataFrame:
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        """
        SELECT ts, obi, taker_buy_ratio, spoofing_score, nif_whale, nif_retail, eai,
               oi_delta_pct, funding_rate, shadow_toxicity_score, shadow_queue_collapse,
               shadow_absorption_score, shadow_queue_bias, shadow_regime_conf, shadow_regime_tag,
               recent_trade_count_5m, recent_trade_notional_5m, recent_whale_count_5m,
               data_stale, valid_taker_flow, valid_nif, warmup_30m_ready
        FROM microstructure_1m ORDER BY ts
        """
    ).fetchdf()
    con.close()
    df["ts_utc"] = pd.to_datetime(df["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.drop(columns=["ts"]).drop_duplicates(subset=["ts_utc"], keep="last")
    ok = (~df["data_stale"].astype(bool)) & df["valid_taker_flow"].astype(bool) & \
         df["valid_nif"].astype(bool) & df["warmup_30m_ready"].astype(bool)
    df = df[ok].drop(columns=["data_stale", "valid_taker_flow", "valid_nif", "warmup_30m_ready"])
    return df.set_index("ts_utc").sort_index()


def add_derived(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["tbr_dev"] = out["taker_buy_ratio"] - 0.5
    out["nif_spread"] = out["nif_whale"] - out["nif_retail"]
    for col in ["obi", "tbr_dev", "nif_whale", "nif_spread", "shadow_toxicity_score",
                "shadow_absorption_score", "eai", "oi_delta_pct"]:
        for w in [5, 15, 60]:
            out[f"{col}_m{w}"] = out[col].rolling(w, min_periods=max(2, w // 2)).mean()
        mu = out[col].rolling(240, min_periods=60).mean()
        sd = out[col].rolling(240, min_periods=60).std()
        out[f"{col}_z240"] = (out[col] - mu) / sd.replace(0.0, np.nan)
        out[f"{col}_d5"] = out[col] - out[col].shift(5)
    out["obi_x_tbr"] = out["obi"] * out["tbr_dev"]
    out["whale_notional_m15"] = out["recent_whale_count_5m"].rolling(15, min_periods=5).mean()
    out["queue_bias_m15"] = out["shadow_queue_bias"].rolling(15, min_periods=5).mean()
    return out


def main() -> None:
    micro = add_derived(load_micro())
    kl = pd.read_csv(KLINES, parse_dates=["timestamp"],
                     usecols=["timestamp", "open", "high", "low", "close", "volume"])
    kl = kl[kl["timestamp"] >= micro.index.min() - pd.Timedelta("1h")].set_index("timestamp").sort_index()
    print(f"micro rows (quality-filtered): {len(micro):,}  {micro.index.min()} -> {micro.index.max()}")
    print(f"klines rows in window: {len(kl):,}  -> {kl.index.max()}")

    # Decision grid: every minute D where both the shifted micro row and the entry bar exist.
    grid = pd.DataFrame(index=kl.index)
    grid["entry_open"] = kl["open"]
    for h in HORIZONS:
        grid[f"fwd_{h}"] = kl["close"].shift(-(h - 1)) / kl["open"] - 1.0
    micro_shifted = micro.copy()
    micro_shifted.index = micro_shifted.index + pd.Timedelta(minutes=AVAIL_SHIFT_MIN)
    df = grid.join(micro_shifted, how="inner")
    df = df.dropna(subset=[f"fwd_{h}" for h in HORIZONS])
    print(f"joined decision rows: {len(df):,}")

    feat_cols = [c for c in micro_shifted.columns if c != "shadow_regime_tag"]
    df["day"] = df.index.date

    rows = []
    for col in feat_cols:
        if df[col].std(skipna=True) == 0 or df[col].notna().sum() < 5000:
            continue
        for h in HORIZONS:
            daily = []
            for _, g in df.groupby("day"):
                x, y = g[col], g[f"fwd_{h}"]
                mask = x.notna() & y.notna()
                if mask.sum() < 300 or x[mask].nunique() < 10:
                    continue
                ic = spearmanr(x[mask], y[mask]).statistic
                if np.isfinite(ic):
                    daily.append(ic)
            if len(daily) < 20:
                continue
            daily = np.asarray(daily)
            t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
            rows.append({"feature": col, "horizon_min": h, "ic_mean": daily.mean(),
                         "ic_t": t, "days": len(daily), "pos_day_frac": (daily > 0).mean()})
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    print("\n=== features with |t| >= 3 at any horizon (sorted by |t|) ===")
    best = res.loc[res.groupby("feature")["ic_t"].apply(lambda s: s.abs().idxmax())]
    best = best[best["ic_t"].abs() >= 3.0].sort_values("ic_t", key=np.abs, ascending=False)
    print(best.to_string(index=False))

    print("\n=== decile long-short spread (bps) for top-8 features ===")
    for col in best["feature"].head(8):
        h = int(best.loc[best["feature"] == col, "horizon_min"].iloc[0])
        d = df[[col, f"fwd_{h}"]].dropna()
        q = pd.qcut(d[col], 10, labels=False, duplicates="drop")
        m = d.groupby(q)[f"fwd_{h}"].mean() * 1e4
        print(f"{col} (h={h}m): d0={m.iloc[0]:+.2f} d9={m.iloc[-1]:+.2f} "
              f"spread={m.iloc[-1] - m.iloc[0]:+.2f} bps  monotonic_rho={spearmanr(m.index, m.values).statistic:+.2f}")

    print(f"\nsaved: {OUT_CSV}")


if __name__ == "__main__":
    main()

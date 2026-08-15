#!/usr/bin/env python3
"""Causal-test harness for the Deribit GEX collector's stated next step (docs/experiments/
eth_trader_research_gex_infra_start_20260815.md "알려진 한계/다음 단계" #2): does dealer gamma
exposure SIGN predict SUBSEQUENT realized volatility regime (Karsan/SqueezeMetrics hypothesis:
GEX>0 -> dealers hedge against price, suppressing vol/favoring range; GEX<0 -> dealers hedge WITH
price, amplifying vol/favoring trend)?

STATUS AS OF 2026-08-15 (day 0 of collection, ~2h/4 snapshot-rounds in data/live/deribit_gex.duckdb):
this is infra prep, run once here purely to prove the join/metric pipeline is wired correctly
end-to-end -- NOT a result. n is far below any usable threshold; `main()` prints a loud warning and
refuses to state a verdict below MIN_USABLE_SNAPSHOTS. Re-run this unmodified once weeks of data have
accumulated (see [[deribit_gex_live_collector_started_20260815]] memory for the "how to apply" list).

Known caveat this harness must control for (found + confirmed by direct instrument-level diff,
2026-08-15, docs/experiments/eth_trader_research_gex_infra_start_20260815.md "수집 상태 점검"):
Deribit's standard daily option expiry (08:00 UTC) drops that day's expiring contracts out of the
live instrument universe between one hourly snapshot and the next, causing a real (not erroneous)
step-change in aggregate GEX independent of any dealer-positioning shift. flag_daily_expiry_
snapshots() marks the first snapshot at/after each day's 08:00 UTC boundary so it can be excluded or
modelled separately -- it must not be silently averaged in with genuine regime-driven GEX changes.

Design (pre-registered, mirrors the evidence_signal_quant_use_subproject discipline: always report a
FREE benchmark alongside the candidate signal, always report breakeven-relevant units, never a bare
correlation coefficient with no comparator):
  - primary signal: front_month_gex_usd sign/level (near-dated options drive the most active dealer
    hedging; total_gex_usd reported alongside for comparison, not as the primary claim)
  - target: forward realized volatility (stdev of 5m log returns) over 1h/4h/24h horizons from each
    snapshot's recorded_at_utc
  - free benchmark: TRAILING realized volatility over the same horizons, ending at the snapshot time
    (the obvious "vol clusters, so recent vol predicts near-future vol" explanation GEX would need to
    beat, matching the pattern that has absorbed every other evidence-signal candidate this session)
  - both spearmanr(gex, forward_vol) and spearmanr(trailing_vol, forward_vol) reported side by side;
    GEX is only informative to the extent its correlation exceeds trailing_vol's on genuinely new data

fresh_forward_bar_by_bar=true (forward vol windows only ever look at bars strictly after the snapshot
timestamp; trailing windows only ever look at bars strictly before it). trade_ledgers_used_as_input=false.
No promotion/signal claim is made by this script at any data volume below MIN_USABLE_SNAPSHOTS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
GEX_DB = ROOT / "data/live/deribit_gex.duckdb"
# NOTE: data/{eth,btc}_5m_1year.csv (used by every OTHER script in this repo) is a FROZEN historical
# file that ends 2026-02-17 -- ~6 months before GEX collection started (2026-08-15) and always
# falling further behind. There is no live-updating 5m price cache in data/live/ to reuse (checked:
# the duckdb files there are all strategy-specific shadow ledgers, not a generic OHLCV feed), so this
# harness fetches recent bars directly from Binance's public futures klines endpoint (no API key
# needed, matches core/binance_client.py's futures-first convention) instead. Every re-run pulls
# whatever range the current gex_summary table actually spans, so this needs no date edits later.
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = {"ETH": "ETHUSDT", "BTC": "BTCUSDT"}
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_gex_sign_vs_realized_vol_causal_20260815"

HORIZONS_BARS = {"1h": 12, "4h": 48, "24h": 288}  # 5m bars
MIN_USABLE_SNAPSHOTS = 100  # well below "weeks" but enough that spearmanr isn't pure noise-fitting


def log(msg: str) -> None:
    print(f"[gex_vol_causal] {msg}", flush=True)


def load_gex_summary() -> pd.DataFrame:
    con = duckdb.connect(str(GEX_DB), read_only=True)
    df = con.execute(
        "select recorded_at_utc, currency, spot_price, total_gex_usd, front_month_gex_usd, "
        "n_instruments, n_front_month from gex_summary order by recorded_at_utc, currency"
    ).df()
    con.close()
    df["recorded_at_utc"] = pd.to_datetime(df["recorded_at_utc"], utc=True)
    return df


def flag_daily_expiry_snapshots(gex: pd.DataFrame) -> pd.DataFrame:
    """Mark, per currency, the first snapshot at/after each UTC day's 08:00 expiry boundary. Uses
    n_front_month DROPPING versus the immediately preceding same-currency snapshot as the operational
    signature (confirmed 2026-08-15 to coincide exactly with expiring-instrument removal, not API
    flakiness) rather than a hardcoded clock check, so it still works if the collector's poll time
    drifts around the hour mark."""
    out = gex.sort_values(["currency", "recorded_at_utc"]).reset_index(drop=True)
    out["is_daily_expiry_snapshot"] = False
    for ccy, g in out.groupby("currency"):
        idx = g.index
        drop = g["n_front_month"].diff() < 0
        out.loc[idx, "is_daily_expiry_snapshot"] = drop.fillna(False).to_numpy()
    return out


def fetch_binance_5m_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Public futures klines, no API key. Paginates in <=1500-bar chunks (Binance's per-request cap)."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(BINANCE_FUTURES_KLINES_URL, params={
            "symbol": symbol, "interval": "5m", "startTime": cursor, "endTime": end_ms, "limit": 1500,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:  # safety: avoid an infinite loop if the API ever returns a stuck page
            break
        cursor = last_open + 1
        if len(batch) < 1500:
            break
    if not rows:
        raise RuntimeError(f"Binance futures klines returned 0 rows for {symbol} [{start_ms}, {end_ms}]")
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def load_price_frame(currency: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # pad well past both ends: 2 days before `start` covers the 24h trailing-vol window for the
    # earliest snapshot with margin; 1h after `end` covers the 1h forward-vol window for the latest.
    pad_start = start - pd.Timedelta(days=2)
    pad_end = end + pd.Timedelta(hours=1)
    df = fetch_binance_5m_klines(SYMBOLS[currency], int(pad_start.timestamp() * 1000), int(pad_end.timestamp() * 1000))
    df["log_ret"] = np.log(df["close"]).diff()
    return df


def _realized_vol(log_ret: np.ndarray, start_idx: int, n_bars: int, *, forward: bool) -> float:
    if forward:
        window = log_ret[start_idx + 1: start_idx + 1 + n_bars]
    else:
        window = log_ret[max(0, start_idx - n_bars): start_idx]
    window = window[np.isfinite(window)]
    if len(window) < max(3, n_bars // 3):  # require at least 1/3 of the bars present, else NaN not 0
        return float("nan")
    return float(np.std(window, ddof=1))


def attach_vol_features(gex: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    ts = price["timestamp"].to_numpy()
    log_ret = price["log_ret"].to_numpy(dtype=np.float64)
    idx = np.searchsorted(ts, gex["recorded_at_utc"].to_numpy())
    idx = np.clip(idx, 0, len(ts) - 1)
    gex = gex.copy()
    gex["_price_idx"] = idx
    gex["price_ts_matched"] = ts[idx]
    gex["match_gap_minutes"] = (gex["recorded_at_utc"].to_numpy() - ts[idx]).astype("timedelta64[s]").astype(float) / 60.0
    for label, n_bars in HORIZONS_BARS.items():
        gex[f"fwd_rv_{label}"] = [
            _realized_vol(log_ret, i, n_bars, forward=True) for i in gex["_price_idx"]
        ]
        gex[f"trail_rv_{label}"] = [
            _realized_vol(log_ret, i, n_bars, forward=False) for i in gex["_price_idx"]
        ]
    return gex.drop(columns=["_price_idx"])


def causal_test(gex_with_vol: pd.DataFrame) -> dict:
    result: dict = {}
    usable = gex_with_vol[~gex_with_vol["is_daily_expiry_snapshot"]]
    result["n_snapshots_total"] = int(len(gex_with_vol))
    result["n_snapshots_excl_expiry_transitions"] = int(len(usable))
    result["n_daily_expiry_snapshots_excluded"] = int(gex_with_vol["is_daily_expiry_snapshot"].sum())

    for label in HORIZONS_BARS:
        row = {}
        for gex_col in ("front_month_gex_usd", "total_gex_usd"):
            for series_name, series in (
                (gex_col, usable[gex_col]),
                (f"trailing_rv_{label}", usable[f"trail_rv_{label}"]),
            ):
                y = usable[f"fwd_rv_{label}"]
                m = series.notna() & y.notna()
                if m.sum() >= 3:
                    rho, p = spearmanr(series[m], y[m])
                else:
                    rho, p = float("nan"), float("nan")
                row[f"spearman({series_name}, fwd_rv_{label})"] = {"rho": rho, "p": p, "n": int(m.sum())}
        result[label] = row
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gex = load_gex_summary()
    gex = flag_daily_expiry_snapshots(gex)
    log(f"loaded {len(gex)} gex_summary rows, currencies={sorted(gex['currency'].unique())}, "
        f"{int(gex['is_daily_expiry_snapshot'].sum())} flagged as daily-expiry-transition snapshots")

    per_currency: dict[str, dict] = {}
    all_rows = []
    gex_start, gex_end = gex["recorded_at_utc"].min(), gex["recorded_at_utc"].max()
    for ccy in sorted(gex["currency"].unique()):
        if ccy not in SYMBOLS:
            log(f"no Binance symbol wired for {ccy}, skipping")
            continue
        price = load_price_frame(ccy, start=gex_start, end=gex_end)
        g = attach_vol_features(gex[gex["currency"] == ccy].reset_index(drop=True), price)
        all_rows.append(g)
        max_gap = g["match_gap_minutes"].abs().max()
        log(f"{ccy}: {len(g)} snapshots, max |snapshot-to-matched-bar| gap = {max_gap:.1f} min")
        per_currency[ccy] = causal_test(g)
        log(f"{ccy} causal test (n={per_currency[ccy]['n_snapshots_excl_expiry_transitions']} usable): "
            f"{json.dumps(per_currency[ccy]['1h'], default=str)}")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(OUT_DIR / "gex_with_vol_features.csv", index=False)

    n_usable_min = min((v["n_snapshots_excl_expiry_transitions"] for v in per_currency.values()), default=0)
    verdict_possible = n_usable_min >= MIN_USABLE_SNAPSHOTS
    if not verdict_possible:
        log(f"*** NOT ENOUGH DATA FOR A VERDICT *** (min usable snapshots across currencies = "
            f"{n_usable_min}, need >= {MIN_USABLE_SNAPSHOTS}). This run only validates that the "
            f"load/join/vol-computation/correlation pipeline executes correctly end-to-end. Re-run "
            f"this exact script, unmodified, once data/live/deribit_gex.duckdb has accumulated "
            f"weeks of hourly snapshots -- do not interpret the numbers above as a signal or its "
            f"absence at this sample size.")

    report = {
        "min_usable_snapshots_required": MIN_USABLE_SNAPSHOTS,
        "verdict_possible_at_current_data_volume": bool(verdict_possible),
        "horizons_bars": HORIZONS_BARS,
        "per_currency": per_currency,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {OUT_DIR / 'report.json'} and {OUT_DIR / 'gex_with_vol_features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

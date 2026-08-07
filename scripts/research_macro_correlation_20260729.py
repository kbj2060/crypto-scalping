"""
Lightweight first-pass research: does DXY / UST yield curve / VIX / SPX carry any
correlation or lead-lag predictive signal vs ETH/SOL/BTC daily returns?

This is a SCREENING pass only (per project convention, see CLAUDE.md + memory:
project-selection-stats-instrument-20260726). It reports raw correlations and
naive t-stats with NO multiple-testing correction. Nothing here should be treated
as promotion-grade evidence -- it only tells us whether it's worth building the
real DSR/PBO-gated pipeline (core/selection_stats.py, currently only on branch
claude/optimal-trading-formula-plan-2x7sml, not on main).

Data sources:
  - Macro: FRED public CSV endpoint (no API key needed), daily series:
      DTWEXBGS  Trade Weighted US Dollar Index, Broad (daily)
      DGS10     10-Year Treasury Constant Maturity Rate
      DGS2      2-Year Treasury Constant Maturity Rate
      VIXCLS    CBOE Volatility Index
      SP500     S&P 500 (FRED's daily close series)
  - Crypto: binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv, resampled to daily close.

Alignment: macro series are business-day (Mon-Fri, US holidays closed), crypto is
24/7. We resample crypto to the macro's calendar (as-of / forward-fill macro onto
crypto's daily close is the OTHER direction -- see note in align()) and test two
causally valid directions:
  (a) SAME-DAY: macro_change[t] vs crypto_return[t]   (contemporaneous, NOT tradeable live)
  (b) LAGGED:   macro_change[t-1] vs crypto_return[t]  (macro leads by 1 business day, tradeable)
Contemporaneous is reported for context only; only the lagged relationship would
ever be usable as a live feature (macro data has real-time or same-day-close lag
in practice too, so even "lagged 1 day" is optimistic for a live feed -- flagged).
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
KLINES = REPO / "binance_data" / "klines"

FRED_SERIES = {
    "DXY": "DTWEXBGS",
    "UST10Y": "DGS10",
    "UST2Y": "DGS2",
    "VIX": "VIXCLS",
    "SPX": "SP500",
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_fred(series_id: str) -> pd.Series:
    resp = requests.get(FRED_URL.format(sid=series_id), timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def load_crypto_daily(symbol: str) -> pd.Series:
    path = KLINES / symbol / f"{symbol}-5m-api.csv"
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    # daily close = last 5m bar of each UTC calendar day
    daily_close = df["close"].resample("1D").last().dropna()
    return daily_close


def build_macro_frame() -> pd.DataFrame:
    cols = {}
    for name, sid in FRED_SERIES.items():
        print(f"  fetching FRED:{sid} ({name}) ...", file=sys.stderr)
        cols[name] = fetch_fred(sid)
    macro = pd.DataFrame(cols)
    # yield curve slope, a common macro regime signal
    macro["UST_SLOPE_10Y2Y"] = macro["UST10Y"] - macro["UST2Y"]
    return macro


def daily_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change()


def significance(x: pd.Series, y: pd.Series):
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 30:
        return dict(n=n, r=np.nan, p=np.nan)
    r, p = stats.pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    return dict(n=n, r=r, p=p)


def main():
    print("=== Loading crypto daily closes ===", file=sys.stderr)
    crypto_daily = {}
    for symbol in ["ETHUSDT", "SOLUSDT", "BTCUSDT"]:
        try:
            crypto_daily[symbol] = load_crypto_daily(symbol)
        except FileNotFoundError:
            print(f"  skip {symbol}: no 5m kline file", file=sys.stderr)
    crypto_ret = {sym: daily_pct_change(s) for sym, s in crypto_daily.items()}

    print("=== Fetching macro data from FRED ===", file=sys.stderr)
    macro = build_macro_frame()
    macro_chg = macro.apply(daily_pct_change)
    macro_chg["UST_SLOPE_10Y2Y"] = macro["UST_SLOPE_10Y2Y"].diff()  # level diff, not pct

    n_tests = 0
    results = []
    for sym, ret in crypto_ret.items():
        ret.index = ret.index.tz_localize(None) if ret.index.tz else ret.index
        for macro_name in macro_chg.columns:
            m = macro_chg[macro_name].copy()
            m.index = m.index.tz_localize(None) if m.index.tz else m.index

            # (a) contemporaneous, same calendar day
            same = significance(m, ret)
            # (b) macro leads by 1 business day (causally usable direction)
            lag1 = significance(m.shift(1), ret)

            for label, res in [("same_day", same), ("macro_lag1->crypto", lag1)]:
                n_tests += 1
                results.append(dict(asset=sym, macro=macro_name, relation=label, **res))

    res_df = pd.DataFrame(results).sort_values("p")
    bonf_alpha = 0.05 / n_tests

    print(f"\n=== Results ({n_tests} tests run, naive; Bonferroni alpha={bonf_alpha:.5f}) ===\n")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(res_df.to_string(index=False))

    survivors = res_df[res_df["p"] < bonf_alpha]
    print(f"\n=== Survive Bonferroni-corrected p<{bonf_alpha:.5f}: {len(survivors)} / {n_tests} ===")
    if len(survivors):
        print(survivors.to_string(index=False))
    else:
        print("(none)")

    out_path = REPO / "data" / "research" / "macro_correlation_screen_20260729.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()

"""Build a daily macro + crypto panel from public GitHub-hosted mirrors (2026-10-27).

The session network blocks FRED/Yahoo/Binance directly, so every series comes from a
public GitHub repository that commits the raw observations. Provenance (repo + commit)
is written next to the panel so the numbers can be re-derived.

Sources (clone with `git clone --depth 1 https://github.com/<repo>` into --ext-dir):
  pk-claude/siberforge.xyz          data/fred/*.json   FRED series (hist + current year)
  Mario-SO/ohlcv                    data/*.csv         Yahoo daily OHLCV (BTC-USD, ETH-USD, ^GSPC, GC=F)
  coinmetrics/data  (raw download)  csv/{btc,eth,usdt,usdc,dai,busd}.csv  Coin Metrics community
  LEEYOUNGJIN-GIT/fred-macro-review data/macro/obs/*.csv KOSPI, STLFSI4, DEXKOUS, BAMLH0A0HYM2

Output: <out>/panel_daily.parquet (calendar-daily index, raw levels, no forward fill)
        <out>/provenance.json
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

FRED_SERIES = [
    # rates / policy
    "DGS2", "DGS10", "DFII10", "T10YIE", "T10Y2Y", "THREEFYTP10", "DFF", "SOFR",
    # credit / stress / conditions
    "BAA10Y", "BAMLH0A0HYM2", "NFCI", "ANFCI", "NFCIRISK", "NFCICREDIT", "NFCILEVERAGE", "VIXCLS",
    # dollar / fx
    "DTWEXBGS", "DEXJPUS", "DEXCHUS", "DEXUSEU",
    # commodities
    "DCOILWTICO", "DCOILBRENTEU", "PCOPPUSDM",
    # liquidity
    "WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "M2SL",
    # equity (cross-check)
    "SP500",
    # monthly macro
    "CPIAUCSL", "CPILFESL", "UNRATE", "INDPRO", "UMCSENT", "PAYEMS",
]


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "log", "-1", "--format=%H %cI"], text=True).strip()
    except Exception:
        return "n/a"


def load_fred(fred_dir: Path, sid: str) -> pd.Series:
    parts = []
    for suffix in (".hist.json", ".json"):
        f = fred_dir / f"{sid}{suffix}"
        if f.exists():
            d = json.loads(f.read_text())
            parts.append(pd.Series(d["v"], index=pd.to_datetime(d["d"]), dtype="float64"))
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.rename(sid)


def load_csv(path: Path, date_col: str, val_col: str, name: str) -> pd.Series:
    df = pd.read_csv(path, usecols=[date_col, val_col])
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values, index=pd.to_datetime(df[date_col]), name=name)
    return s.dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-dir", default="/tmp/claude-0/ext")
    ap.add_argument("--out", default="docs/experiments/macro_crypto_correlation_20261027")
    a = ap.parse_args()
    ext, out = Path(a.ext_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fred_dir = ext / "pk-claude_siberforge.xyz/data/fred"
    yahoo = ext / "Mario-SO_ohlcv/data"
    cm = ext / "cm"
    leey = ext / "LEEYOUNGJIN-GIT_fred-macro-review/data/macro/obs"

    cols = {sid: load_fred(fred_dir, sid) for sid in FRED_SERIES}
    cols["BTC"] = load_csv(yahoo / "btc.csv", "Date", "Close", "BTC")
    cols["ETH"] = load_csv(yahoo / "eth.csv", "Date", "Close", "ETH")
    cols["SPX"] = load_csv(yahoo / "sp500.csv", "Date", "Close", "SPX")
    cols["GOLD"] = load_csv(yahoo / "gold.csv", "Date", "Close", "GOLD")
    cols["BTC_CM"] = load_csv(cm / "btc.csv", "time", "PriceUSD", "BTC_CM")
    cols["ETH_CM"] = load_csv(cm / "eth.csv", "time", "PriceUSD", "ETH_CM")
    stable = []
    for coin in ("usdt", "usdc", "dai", "busd"):
        stable.append(load_csv(cm / f"{coin}.csv", "time", "SplyCur", coin.upper()))
    st = pd.concat(stable, axis=1)
    # sum only where USDT (dominant) is reported; others contribute 0 before launch
    cols["STABLE_SUPPLY"] = st.fillna(0.0).sum(axis=1).where(st["USDT"].notna())
    cols["KOSPI"] = load_csv(leey / "KOSPI.csv", "date", "value", "KOSPI")
    cols["STLFSI4"] = load_csv(leey / "STLFSI4.csv", "date", "value", "STLFSI4")
    cols["DEXKOUS"] = load_csv(leey / "DEXKOUS.csv", "date", "value", "DEXKOUS")
    # HY OAS: FRED only exposes the last ~3y; merge both mirrors
    hy2 = load_csv(leey / "BAMLH0A0HYM2.csv", "date", "value", "BAMLH0A0HYM2")
    cols["BAMLH0A0HYM2"] = pd.concat([cols["BAMLH0A0HYM2"], hy2]).groupby(level=0).last()

    for k, s in cols.items():
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        cols[k] = s[~s.index.duplicated(keep="last")].rename(k)
    panel = pd.concat(cols.values(), axis=1, sort=True)
    panel = panel.loc["2000-01-01":]

    # Net liquidity (USD mn): WALCL [mn] - WTREGEN [mn] - RRPONTSYD [bn]*1000. Weekly (Wed) grid.
    w = panel[["WALCL", "WTREGEN"]].dropna()
    rrp = panel["RRPONTSYD"].ffill(limit=5).reindex(w.index)
    panel["NETLIQ"] = (w["WALCL"] - w["WTREGEN"] - rrp.fillna(0.0) * 1000.0)

    panel.to_parquet(out / "panel_daily.parquet")

    # sanity: Yahoo vs Coin Metrics BTC returns (same UTC-day convention)
    chk = panel[["BTC", "BTC_CM"]].dropna()
    r = np.log(chk).diff().dropna()
    summary = {
        "rows": int(len(panel)),
        "coverage": {c: [str(panel[c].first_valid_index().date()), str(panel[c].last_valid_index().date()), int(panel[c].notna().sum())]
                     for c in panel.columns if panel[c].notna().any()},
        "btc_yahoo_vs_coinmetrics_daily_ret_corr": float(r.corr().iloc[0, 1]),
        "btc_yahoo_vs_coinmetrics_median_abs_level_gap_pct": float((chk["BTC"] / chk["BTC_CM"] - 1).abs().median() * 100),
        "sources": {
            "pk-claude/siberforge.xyz": git_head(ext / "pk-claude_siberforge.xyz"),
            "Mario-SO/ohlcv": git_head(ext / "Mario-SO_ohlcv"),
            "LEEYOUNGJIN-GIT/fred-macro-review": git_head(ext / "LEEYOUNGJIN-GIT_fred-macro-review"),
            "coinmetrics/data": "raw.githubusercontent.com/coinmetrics/data/master/csv (downloaded 2026-10-27)",
        },
        "units": {"WALCL": "USD mn", "WTREGEN": "USD mn", "RRPONTSYD": "USD bn", "NETLIQ": "USD mn", "WRESBAL": "USD mn"},
    }
    (out / "provenance.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"}, indent=1))
    for c, v in summary["coverage"].items():
        print(f"{c:14s} {v}")


if __name__ == "__main__":
    main()

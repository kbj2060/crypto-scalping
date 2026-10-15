"""중단기(3~14일) 방향 — 새 원천으로 처음부터 (2026-10-15).

앞 연구(research_trend_mid_long_20261015: 유동성 Z·펀딩·공포탐욕)의 피쳐·규칙은 쓰지 않는다.
새 원천: 코인베이스 프리미엄 · 김치 프리미엄 · 위키피디아 조회수 · 달력 · 상대 방향 표적.

  --fetch   → tmp/short_fresh_20261015/raw/
  --signal  최종 규칙(보류 2024~26 통과 근접: BTC 59·ETH 65%, 예측 28%)의 최근 값
            z = Δ7(7일 평균 ln(코인베이스 BTC-USD / 바이낸스 BTCUSDT)) ÷ 직전 365일 표준편차
            z ≥ 1 → 7일 상승, z ≤ −1 → 7일 하락, 사이 = 보류 (모든 코인에 BTC 값 적용)
  --selftest 인과성 점검
"""
import argparse, json, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_trend_mid_long_20261015 import get  # 재시도 다운로드만 재사용

OUT = Path(__file__).resolve().parents[1] / "tmp" / "short_fresh_20261015"
RAW = OUT / "raw"
CB = ["BTC", "ETH", "LTC", "BCH", "SOL", "ADA", "DOGE", "XRP", "LINK", "AVAX", "DOT"]
UB = ["BTC", "ETH", "XRP", "ADA", "DOGE", "SOL", "LINK", "ETC", "BCH", "TRX", "XLM", "DOT", "AVAX"]
WIKI = ["Bitcoin", "Ethereum", "Cryptocurrency", "Dogecoin", "Blockchain"]


def fetch_coinbase():
    for s in CB:
        if (RAW / f"cb_{s}.csv").exists():
            continue
        rows, t = [], datetime(2015, 1, 1, tzinfo=timezone.utc)
        end = datetime.now(timezone.utc)
        while t < end:
            t2 = min(t + timedelta(days=299), end)
            try:
                j = json.loads(get(f"https://api.exchange.coinbase.com/products/{s}-USD/candles?granularity=86400"
                                   f"&start={t:%Y-%m-%dT%H:%M:%SZ}&end={t2:%Y-%m-%dT%H:%M:%SZ}"))
            except Exception as e:  # 없는 상품
                print("cb", s, "FAIL", e); break
            rows += j if isinstance(j, list) else []
            t = t2
            time.sleep(0.25)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "vol"]).drop_duplicates("t")
        df["date"] = pd.to_datetime(df.t, unit="s")
        df.set_index("date").drop(columns="t").sort_index().to_csv(RAW / f"cb_{s}.csv")
        print("cb", s, len(df), df.date.min().date())


def fetch_upbit():
    for s in UB:
        if (RAW / f"ub_{s}.csv").exists():
            continue
        rows, to = [], ""
        while True:
            try:
                j = json.loads(get(f"https://api.upbit.com/v1/candles/days?market=KRW-{s}&count=200" + (f"&to={to}" if to else "")))
            except Exception as e:
                print("ub", s, "FAIL", e); break
            if not j:
                break
            rows += j
            to = j[-1]["candle_date_time_utc"] + "Z"
            time.sleep(0.15)
            if len(j) < 200:
                break
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = pd.DataFrame({"date": pd.to_datetime(df.candle_date_time_utc), "close_krw": df.trade_price,
                           "value_krw": df.candle_acc_trade_price}).drop_duplicates("date").set_index("date").sort_index()
        df.to_csv(RAW / f"ub_{s}.csv")
        print("ub", s, len(df), df.index.min().date())


def fetch_misc():
    end = datetime.now(timezone.utc).strftime("%Y%m%d")
    w = {}
    for a in WIKI:
        j = json.loads(get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{a}/daily/20150701/{end}"))
        w[a] = pd.Series({pd.Timestamp(x["timestamp"][:8]): x["views"] for x in j["items"]})
    pd.DataFrame(w).rename_axis("date").to_csv(RAW / "wiki.csv")
    open(RAW / "fred_DEXKOUS.csv", "wb").write(get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS"))
    print("wiki/krw ok")


def cb_premium_z(cb_close, bn_close):
    p = np.log(cb_close / bn_close).dropna()
    p7 = p.rolling(7, min_periods=5).mean()
    d7 = p7 - p7.shift(7)
    return d7 / d7.rolling(365, min_periods=180).std()


def _btc_closes():
    bn = pd.read_csv(OUT.parent / "trend_mid_long_20261015" / "raw" / "bn_BTC.csv", index_col=0, parse_dates=True).close
    return pd.read_csv(RAW / "cb_BTC.csv", index_col=0, parse_dates=True).close, bn


def selftest():
    cb, bn = _btc_closes()
    z = cb_premium_z(cb, bn)
    for t in ["2021-05-20", "2024-03-10", "2026-08-01"]:
        assert abs(cb_premium_z(cb.loc[:t], bn.loc[:t]).loc[t] - z.loc[t]) < 1e-12, t
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--signal", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.fetch:
        fetch_coinbase(); fetch_upbit(); fetch_misc()
    if a.signal:
        z = cb_premium_z(*_btc_closes()).dropna().tail(5)
        print(z.round(2).to_string(), "\n→", "상승" if z.iloc[-1] >= 1 else "하락" if z.iloc[-1] <= -1 else "보류")
    if a.selftest:
        selftest()

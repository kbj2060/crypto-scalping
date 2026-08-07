"""F4-C 1단계: 대체데이터 수집기 (일 1회, 무료/공개 소스만).

수집 항목:
  1. Fear & Greed Index (alternative.me, 무료 공개 API)
  2. Binance vs OKX 펀딩비 스프레드 (ETH/BTC/SOL perp)
  3. Binance 공지사항 (상장/상폐/점검 등, 공개 API)

data/research/altdata.duckdb에 append. 읽기전용 프로덕션 접근 없음, 신규 격리 DB.
WS-D의 D4 수집 건강 감시에 추후 편입 예정.
"""
from __future__ import annotations

import json
import time

import ccxt
import duckdb
import pandas as pd
import requests

DB_PATH = "data/research/altdata.duckdb"
ASSETS = {"ETHUSDT": "ETH/USDT:USDT", "BTCUSDT": "BTC/USDT:USDT", "SOLUSDT": "SOL/USDT:USDT"}


def connect_retry(path, retries=5, backoff=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(path)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


def ensure_tables(con):
    con.execute(
        """CREATE TABLE IF NOT EXISTS fear_greed_index (
            recorded_at_utc TIMESTAMPTZ, value INTEGER, classification VARCHAR
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS cross_exchange_funding_spread (
            recorded_at_utc TIMESTAMPTZ, asset VARCHAR,
            binance_funding_rate DOUBLE, okx_funding_rate DOUBLE, spread DOUBLE
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS binance_announcements (
            recorded_at_utc TIMESTAMPTZ, announcement_id VARCHAR, title VARCHAR,
            catalog_name VARCHAR, publish_time_utc TIMESTAMPTZ
        )"""
    )


def collect_fear_greed(con):
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=15)
    r.raise_for_status()
    data = r.json()["data"][0]
    now = pd.Timestamp.now(tz="UTC")
    con.execute(
        "INSERT INTO fear_greed_index VALUES (?,?,?)",
        [now, int(data["value"]), str(data["value_classification"])],
    )
    print("fear_greed:", data["value"], data["value_classification"])


def collect_funding_spread(con):
    binance = ccxt.binanceusdm()
    binance.load_markets()
    okx = ccxt.okx()
    now = pd.Timestamp.now(tz="UTC")
    for asset, okx_symbol in ASSETS.items():
        try:
            binance_symbol = f"{asset[:-4]}/USDT:USDT"
            b = binance.fetch_funding_rate(binance_symbol)
            o = okx.fetch_funding_rate(okx_symbol)
            b_rate = float(b.get("fundingRate") or 0.0)
            o_rate = float(o.get("fundingRate") or 0.0)
            con.execute(
                "INSERT INTO cross_exchange_funding_spread VALUES (?,?,?,?,?)",
                [now, asset, b_rate, o_rate, b_rate - o_rate],
            )
            print(f"funding_spread {asset}: binance={b_rate:.6f} okx={o_rate:.6f} spread={b_rate - o_rate:.6f}")
        except Exception as exc:
            print(f"funding_spread {asset} FAILED: {exc}")


def collect_binance_announcements(con):
    try:
        r = requests.get(
            "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query",
            params={"catalogId": "48", "pageNo": 1, "pageSize": 20},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        articles = r.json().get("data", {}).get("articles", [])
        now = pd.Timestamp.now(tz="UTC")
        n = 0
        for art in articles:
            aid = str(art.get("code") or art.get("id"))
            exists = con.execute(
                "SELECT COUNT(*) FROM binance_announcements WHERE announcement_id = ?", [aid]
            ).fetchone()[0]
            if exists:
                continue
            pub_ts = art.get("releaseDate") or art.get("publishDate")
            pub_dt = pd.to_datetime(pub_ts, unit="ms", utc=True) if pub_ts else None
            con.execute(
                "INSERT INTO binance_announcements VALUES (?,?,?,?,?)",
                [now, aid, str(art.get("title", "")), str(art.get("catalogName", "") or ""), pub_dt],
            )
            n += 1
        print(f"binance_announcements: {n} new")
    except Exception as exc:
        print(f"binance_announcements FAILED (non-fatal, endpoint may have changed): {exc}")


def main():
    con = connect_retry(DB_PATH)
    ensure_tables(con)
    try:
        collect_fear_greed(con)
    except Exception as exc:
        print("fear_greed FAILED:", exc)
    try:
        collect_funding_spread(con)
    except Exception as exc:
        print("funding_spread FAILED:", exc)
    try:
        collect_binance_announcements(con)
    except Exception as exc:
        print("announcements FAILED:", exc)
    con.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""**매크로 5분봉 수집기** — Yahoo FX·CME/ICE 선물·지수·채권 ETF 5분봉 (2026-10-28, 사용자 «통과 매크로 지표를 가능하면 5분봉으로 수집»).

왜: Macro Trend 판정표(L_p3_trend_verdict.csv)를 통과한 지표의 장중 원천. Yahoo 5분봉은 60일치만 남아 지금 쌓아야 한다.
  10-28 실측(range=60d): FX(=X)는 지연 0 · 하루 288봉(CNY=X 는 역내 시간만 200) · 이력 08-05~. CME·ICE 선물은 ~10분 지연
  (ZQ 13분, ^VIX 15분) · 이력 08-19~. ETF·^GSPC 는 미국 정규장만 78봉. 판정표 지표와 같은 것이 아니라 **대체 원천**이다:
  DXY 는 광의 달러지수(DTWEXBGS)보다 유로 비중이 크고, HYG·LQD·TIP ÷ IEF 는 HY·Baa 스프레드·기대인플레의 근사, ZQ 는 실제
  연방기금금리가 아니라 기대치다 -- 쓰기 전에 일간 변화를 FRED 원본과 대조한다. ZT·ZN(2·10년 국채선물)은 국채선물 수집기가 1분봉으로 받는다.
무엇: hot = data/hot/macro_5m.sqlite(WAL)
  yahoo_5m(ts_sec = 5분봉 시작, symbol = 아래 SYMS 키, OHLCV) · polls(ts_ms, symbol, status, bars) = 호출 한 번(주말엔 봉이 안 늘어
  감시기는 polls 의 ok 를 본다). 봉 하나는 ts_sec + 300 + 원천 지연 뒤에야 알 수 있다 -- 피쳐로 쓸 때 그 시각 규약을 지킨다.
  cron 5분마다. 종목마다 마지막 저장 봉이 4일 안이면 range=5d, 아니면(첫 실행 = 백필) 60d 를 받아 덮어쓴다(INSERT OR REPLACE) --
  서버가 며칠 꺼져도 다음 실행이 메운다. 🔴Yahoo 가 주는 **마지막 봉은 버린다**(진행 중일 수 있다 -- 선물 진행 봉은 시각이 5분
  경계라 시각만으로는 못 가른다). 다음 실행이 완결된 값으로 받는다. 경계가 아닌 시각·종가 없는 봉도 버린다.
  seal_to_lake 가 lake/yahoo/macro_5m/coin={키}/date= 로 봉인, hot 은 8일. 주문·바이낸스 호출 없음.
  🔴수집기만 8일 넘게 실패하면(서버는 살아 봉인은 돎) 장애 첫날이 반쯤 찬 채 봉인되고, 복구 뒤 60d 백필이 hot 을 채워도
    RESEAL_DAYS 밖이라 lake 는 안 고쳐진다 -- 그래서 감시기(polls ok)가 그 전에 울려야 한다. 고치려면 그 날짜 parquet 를 지우고 봉인.
  🔴로컬 PC 와 서버가 같은 공인 IP -- 로컬에서 돌리면 서버 국채선물 수집기(매분 Yahoo)와 429 를 나눠 맞는다. 서버에서만.
  python scripts/live_macro_5m_collector_20261028.py              # cron(서버)
  python scripts/live_macro_5m_collector_20261028.py --selftest   # 네트워크 없이
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_store import ROOT, rw_connect, sqlite_init  # noqa: E402

DB = Path(os.environ.get("MACRO_5M_DB", ROOT / "data/hot/macro_5m.sqlite"))
# 저장 키(lake coin= 경로에 «=»·«^» 가 안 들어가게) -> Yahoo 티커. 주석 = 판정표의 어느 지표 대신인가.
SYMS = {
    "USDJPY": "JPY=X", "USDKRW": "KRW=X", "EURUSD": "EURUSD=X",   # USD/JPY · 원/달러 · EUR/USD (지연 0)
    "USDCNH": "CNH=X", "USDCNY": "CNY=X",                          # USD/CNY (역외 24시간 · 역내)
    "DXY": "DX-Y.NYB",                                             # 달러지수(광의 지수 대신 ICE DXY)
    "ES": "ES=F", "SPX": "^GSPC",                                  # S&P500 (선물 · 현물 지수 정규장)
    "6J": "6J=F", "6E": "6E=F",                                    # CME 엔·유로 선물(FX 교차 확인용)
    "ZQ": "ZQ=F",                                                  # 연방기금(30일 선물 = 기대치)
    "VIX": "^VIX",
    "HYG": "HYG", "LQD": "LQD", "IEF": "IEF", "TIP": "TIP",        # HY·Baa 스프레드·기대인플레 근사(÷ IEF)
}
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval=5m"
UA = "Mozilla/5.0"           # 국채선물 수집기와 같은 짧은 UA(전체 브라우저 UA 는 429, 10-06 실측)
STEP, FULL_AFTER_S, GAP_S = 300, 4 * 86400, 1.5
DDL = ("CREATE TABLE IF NOT EXISTS yahoo_5m(ts_sec INTEGER NOT NULL, symbol TEXT NOT NULL, open REAL, high REAL, low REAL, "
       "close REAL, volume INTEGER, PRIMARY KEY(symbol, ts_sec))",
       "CREATE TABLE IF NOT EXISTS polls(ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, status TEXT NOT NULL, bars INTEGER)")


def rows_of(key: str, chart: dict) -> list[tuple]:
    """Yahoo chart JSON → yahoo_5m 행. 종가 없는 봉·5분 경계가 아닌 시각은 버리고, 남은 것 중 마지막 봉(진행 중일 수 있다)도 버린다."""
    r = chart["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    ts = r.get("timestamp") or []
    vol = q.get("volume") or [0] * len(ts)          # 지수(^VIX 등)는 거래량 열이 비어 올 수 있다 -- zip 이 봉을 통째로 잃지 않게
    rows = [(t, key, o, h, lo, c, v or 0)
            for t, o, h, lo, c, v in zip(ts, q["open"], q["high"], q["low"], q["close"], vol)
            if c is not None and t % STEP == 0]
    return rows[:-1]


def pick_range(last_ts: int | None, now: float) -> str:
    return "5d" if last_ts is not None and now - last_ts < FULL_AFTER_S else "60d"


def main() -> int:
    sqlite_init(DB)                       # auto_vacuum·WAL 은 표보다 먼저(새 파일에서만 먹는다)
    ok = 0
    with rw_connect(DB) as con:
        for ddl in DDL:
            con.execute(ddl)
        last = dict(con.execute("SELECT symbol, max(ts_sec) FROM yahoo_5m GROUP BY symbol").fetchall())
        for i, (key, ticker) in enumerate(SYMS.items()):
            if i:
                time.sleep(GAP_S)
            rng = pick_range(last.get(key), time.time())
            try:
                req = urllib.request.Request(URL.format(urllib.parse.quote(ticker), rng), headers={"User-Agent": UA})
                rows, status = rows_of(key, json.load(urllib.request.urlopen(req, timeout=30))), "ok"
            except Exception as e:  # 한 종목 실패가 나머지를 막지 않게 -- 다음 실행이 5일치로 다시 메운다
                rows, status = [], f"error:{e!r}"[:200]
                print(f"{time.strftime('%F %T')} {key} 실패: {e!r}", file=sys.stderr)
            con.execute("BEGIN")
            con.executemany("INSERT OR REPLACE INTO yahoo_5m VALUES(?,?,?,?,?,?,?)", rows)
            con.execute("INSERT INTO polls VALUES(?,?,?,?)", (int(time.time() * 1000), key, status, len(rows)))
            con.execute("COMMIT")
            ok += status == "ok"
            if rows and rng == "60d":
                print(f"{time.strftime('%F %T')} {key} 백필 {len(rows)}봉 "
                      f"{time.strftime('%F %H:%M', time.gmtime(rows[0][0]))} ~ {time.strftime('%F %H:%M', time.gmtime(rows[-1][0]))} UTC")
    return 0 if ok else 1


def selftest() -> None:
    import sqlite3
    import tempfile
    chart = {"chart": {"result": [{"timestamp": [300, 600, 900, 1000, 1200, 1500],
             "indicators": {"quote": [{"open": [1.0, None, 3.0, 3.1, 4.0, 5.0], "high": [1.5, None, 3.5, 3.2, 4.5, 5.0],
                                       "low": [0.5, None, 2.5, 3.0, 3.5, 5.0], "close": [1.2, None, 3.1, 3.1, 4.2, 5.0],
                                       "volume": [5, None, None, 1, 2, 0]}]}}]}}
    rows = rows_of("USDJPY", chart)
    assert [r[0] for r in rows] == [300, 900, 1200], rows      # 종가 없는 봉(600)·경계 아닌 시각(1000)·마지막 봉(1500) 버림
    assert rows[1][6] == 0 and rows[0][1] == "USDJPY", rows
    assert rows_of("X", {"chart": {"result": [{"indicators": {"quote": [{"open": [], "high": [], "low": [], "close": [],
                                                                         "volume": []}]}}]}}) == []   # 휴장(timestamp 없음)
    assert pick_range(None, 1e9) == "60d" and pick_range(int(1e9) - 86400, 1e9) == "5d" and pick_range(int(1e9) - 5 * 86400, 1e9) == "60d"
    assert all("=" not in k and "^" not in k and "/" not in k for k in SYMS), SYMS   # lake 경로 coin=
    with tempfile.TemporaryDirectory() as tmp:                   # 다시 받은 봉은 덮어쓴다
        con = sqlite3.connect(Path(tmp) / "t.sqlite")
        con.execute(DDL[0])
        con.executemany("INSERT OR REPLACE INTO yahoo_5m VALUES(?,?,?,?,?,?,?)", [rows[0], (*rows[0][:5], 9.9, 7)])
        assert con.execute("SELECT close, volume FROM yahoo_5m").fetchall() == [(9.9, 7)]
        con.close()
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())

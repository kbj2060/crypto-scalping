#!/usr/bin/env python3
"""**미 국채선물 1분봉 수집기** — ZT·ZF·ZN·ZB(2·5·10·30년) Yahoo 1분봉(2026-10-06, 사용자 «선물 4개 수집기 서버에 만들어줘»).

왜: 10-05 23:26~23:35 KST 금리↑(ZN −18bp·ZB −30bp 가격) 동안 ETH −82bp. 장중 금리↔ETH 를 여러 날로 검정하려면
  쌓아야 한다 -- Yahoo 1분봉은 7일치만 남는다. 금리 지수(^TNX 등)는 정규장만·15분 지연이라 빼고, 거의 24시간 거래되는
  표준 국채선물만 받는다(소형 금리선물 10Y=F·30Y=F 는 체결이 드물어 봉이 빈다).
무엇: bars(sym, ts 초, OHLCV, contract=«10-Year T-Note Futures,Dec-2026» 같은 근월물 이름 -- 연속 티커 롤 감지용).
  값은 가격(금리 아님): 금리↑ = 가격↓. CME 무료 시세라 ~10분 지연 -- 사후 연구용이지 라이브 신호용이 아니다.
저장: data/hot/treasury_futures.sqlite. cron 매시 한 번, range=5d 를 받아 덮어쓴다(마지막 봉이 미완성일 수 있어 REPLACE ·
  서버가 며칠 꺼져도 5일 안이면 메워진다). 주문·바이낸스 호출 없음.
  python scripts/live_treasury_futures_collector_20261006.py
  python scripts/live_treasury_futures_collector_20261006.py --selftest
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("TREASURY_FUTURES_DB", ROOT / "data/hot/treasury_futures.sqlite"))
SYMS = ["ZT=F", "ZF=F", "ZN=F", "ZB=F"]
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=5d&interval=1m"
UA = "Mozilla/5.0"  # 전체 브라우저 UA 는 429(10-06 실측) -- 짧은 UA 만 통과


def rows_of(sym: str, chart: dict) -> list[tuple]:
    """Yahoo chart JSON → bars 행. 종가 없는 분(체결 없음)과 진행 중 봉(ts 가 분 경계가 아님, 10-06 실측 18:11:43)은 버린다."""
    r = chart["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    contract = r["meta"].get("shortName") or ""
    return [(sym, t, o, h, lo, c, v or 0, contract)
            for t, o, h, lo, c, v in zip(r.get("timestamp") or [], q["open"], q["high"], q["low"], q["close"], q["volume"])
            if c is not None and t % 60 == 0]


def main() -> int:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS bars(sym TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL,
                   volume INTEGER, contract TEXT, fetched_ms INTEGER, PRIMARY KEY(sym, ts))""")
    failed = 0
    for sym in SYMS:
        try:
            req = urllib.request.Request(URL.format(urllib.parse.quote(sym)), headers={"User-Agent": UA})
            rows = rows_of(sym, json.load(urllib.request.urlopen(req, timeout=30)))
        except Exception as e:  # 한 종목 실패가 나머지를 막지 않게 -- 다음 시간에 5일치로 다시 메운다
            print(f"{time.strftime('%F %T')} {sym} 실패: {e!r}", file=sys.stderr)
            failed += 1
            continue
        now = int(time.time() * 1000)
        with con:
            con.executemany("INSERT OR REPLACE INTO bars VALUES(?,?,?,?,?,?,?,?,?)", [(*r, now) for r in rows])
        print(f"{time.strftime('%F %T')} {sym} {len(rows)}봉 {rows[-1][7] if rows else ''}")
    con.close()
    return 1 if failed == len(SYMS) else 0


def selftest() -> None:
    chart = {"chart": {"result": [{"meta": {"shortName": "10-Year T-Note Futures,Dec-2026"}, "timestamp": [60, 120, 180, 203],
             "indicators": {"quote": [{"open": [1.0, None, 3.0, 3.1], "high": [1.5, None, 3.5, 3.2], "low": [0.5, None, 2.5, 3.0],
                                       "close": [1.2, None, 3.1, 3.1], "volume": [5, None, None, 1]}]}}]}}
    rows = rows_of("ZN=F", chart)
    assert [r[1] for r in rows] == [60, 180], rows          # 체결 없는 분·진행 중 봉(203) 버림
    assert rows[1][6] == 0 and rows[0][7].endswith("Dec-2026"), rows
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())

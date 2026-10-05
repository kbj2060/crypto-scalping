#!/usr/bin/env python3
"""**미 국채선물 1분봉 수집기** — ZT·ZF·ZN·ZB(2·5·10·30년) Yahoo 1분봉(2026-10-06, 사용자 «선물 4개 수집기 서버에 만들어줘»).

왜: 10-05 23:26~23:35 KST 금리↑(ZN −18bp·ZB −30bp 가격) 동안 ETH −82bp. 장중 금리↔ETH 를 여러 날로 검정하려면
  쌓아야 한다 -- Yahoo 1분봉은 7일치만 남는다. 금리 지수(^TNX 등)는 정규장만·15분 지연이라 빼고, 거의 24시간 거래되는
  표준 국채선물만 받는다(소형 금리선물 10Y=F·30Y=F 는 체결이 드물어 봉이 빈다).
무엇: treasury_fut_1m(ts_sec, symbol=ZT|ZF|ZN|ZB, OHLCV, contract=«10-Year T-Note Futures,Dec-2026» 근월물 이름 -- 연속
  티커 롤 감지용). 값은 가격(금리 아님): 금리↑ = 가격↓. CME 무료 시세라 ~10분 지연 -- 사후 연구용이지 라이브 신호용이 아니다.
  polls(ts_ms, symbol, status, bars) = 호출 한 번. 주말·휴장엔 봉이 안 늘어나므로 감시기는 봉이 아니라 이 표의 ok 를 본다.
저장(10-01 저장 재설계 규약): hot = data/hot/treasury_futures.sqlite(WAL) 8일 → seal_to_lake 가 lake/cme/treasury_fut_1m/
  coin=ZN/date=/part.parquet 로 봉인. symbol 은 «=F» 를 뗀다(lake 경로 coin= 에 «=» 가 두 번 들어가지 않게).
cron 매시 한 번, range=5d 를 받아 덮어쓴다(마지막 봉이 미완성일 수 있어 REPLACE · 서버가 며칠 꺼져도 5일 안이면 메워진다 --
  봉인기 재봉인 8일 안이라 lake 도 따라 고쳐진다). 주문·바이낸스 호출 없음.
  python scripts/live_treasury_futures_collector_20261006.py
  python scripts/live_treasury_futures_collector_20261006.py --selftest
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_store import ROOT, rw_connect, sqlite_init  # noqa: E402

DB = Path(os.environ.get("TREASURY_FUTURES_DB", ROOT / "data/hot/treasury_futures.sqlite"))
SYMS = ["ZT", "ZF", "ZN", "ZB"]
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}%3DF?range=5d&interval=1m"
UA = "Mozilla/5.0"  # 전체 브라우저 UA 는 429(10-06 실측) -- 짧은 UA 만 통과
DDL = ("CREATE TABLE IF NOT EXISTS treasury_fut_1m(ts_sec INTEGER NOT NULL, symbol TEXT NOT NULL, open REAL, high REAL, "
       "low REAL, close REAL, volume INTEGER, contract TEXT, PRIMARY KEY(symbol, ts_sec))",
       "CREATE TABLE IF NOT EXISTS polls(ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, status TEXT NOT NULL, bars INTEGER)")


def rows_of(sym: str, chart: dict) -> list[tuple]:
    """Yahoo chart JSON → treasury_fut_1m 행. 종가 없는 분(체결 없음)과 진행 중 봉(ts 가 분 경계가 아님, 10-06 실측
    18:11:43)은 버린다."""
    r = chart["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    contract = r["meta"].get("shortName") or ""
    return [(t, sym, o, h, lo, c, v or 0, contract)
            for t, o, h, lo, c, v in zip(r.get("timestamp") or [], q["open"], q["high"], q["low"], q["close"], q["volume"])
            if c is not None and t % 60 == 0]


def main() -> int:
    sqlite_init(DB)                       # auto_vacuum·WAL 은 표보다 먼저(새 파일에서만 먹는다)
    ok = 0
    with rw_connect(DB) as con:
        for ddl in DDL:
            con.execute(ddl)
        for sym in SYMS:
            try:
                req = urllib.request.Request(URL.format(sym), headers={"User-Agent": UA})
                rows, status = rows_of(sym, json.load(urllib.request.urlopen(req, timeout=30))), "ok"
            except Exception as e:  # 한 종목 실패가 나머지를 막지 않게 -- 다음 시간에 5일치로 다시 메운다
                rows, status = [], f"error:{e!r}"[:200]
                print(f"{time.strftime('%F %T')} {sym} 실패: {e!r}", file=sys.stderr)
            con.execute("BEGIN")
            con.executemany("INSERT OR REPLACE INTO treasury_fut_1m VALUES(?,?,?,?,?,?,?,?)", rows)
            con.execute("INSERT INTO polls VALUES(?,?,?,?)", (int(time.time() * 1000), sym, status, len(rows)))
            con.execute("COMMIT")
            ok += status == "ok"
            if rows:
                print(f"{time.strftime('%F %T')} {sym} {len(rows)}봉 {rows[-1][7]}")
    return 0 if ok else 1


def selftest() -> None:
    chart = {"chart": {"result": [{"meta": {"shortName": "10-Year T-Note Futures,Dec-2026"}, "timestamp": [60, 120, 180, 203],
             "indicators": {"quote": [{"open": [1.0, None, 3.0, 3.1], "high": [1.5, None, 3.5, 3.2], "low": [0.5, None, 2.5, 3.0],
                                       "close": [1.2, None, 3.1, 3.1], "volume": [5, None, None, 1]}]}}]}}
    rows = rows_of("ZN", chart)
    assert [r[0] for r in rows] == [60, 180], rows          # 체결 없는 분·진행 중 봉(203) 버림
    assert rows[1][6] == 0 and rows[0][1] == "ZN" and rows[0][7].endswith("Dec-2026"), rows
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())

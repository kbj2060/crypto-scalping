#!/usr/bin/env python3
"""**Bybit 청산 수집기** — `allLiquidation` 스트림(2026-10-03, 사용자 «청산 데이터를 더 많이»).

왜: 바이낸스 `@forceOrder` 는 종목마다 **1초에 마지막 1건만** 보낸다(공식 문서) -- 몇 개를 어느 시각에 붙여도
  같은 스냅샷이다. Bybit `allLiquidation` 은 1초로 묶지 않고 청산을 모아 보낸다(공식 v5 문서, 500ms 묶음).
  10-03 실측(토요일 10분, 10종목): Bybit 4건 $27k · 바이낸스 29건 $13k -- 둘 다 1초 안 여러 건은 없었다(조용한 장).
  «급변 때 바이낸스가 놓치는 몫»은 평일 급변 구간이 쌓인 뒤 비교한다.
무엇: 종목별 청산 한 건 = (거래소 시각 ms, 종목, 쪽, 수량, 가격, 받은 시각 ms). `side` 는 Bybit 규약 그대로 --
  **청산된 포지션의 방향**(Buy = 롱 청산)이다. 바이낸스 `S`(강제 주문 방향, 롱 청산 = SELL)와 **반대**라 섞을 때 뒤집는다.
저장: data/hot/bybit_liq.sqlite(WAL) · 2초마다 한 트랜잭션 · 끊긴 구간은 gaps 표. 주문 없음·바이낸스 호출 없음.
  python scripts/live_bybit_liquidation_collector_20261003.py
  BYBIT_LIQ_SYMBOLS=ETHUSDT,BTCUSDT python scripts/live_bybit_liquidation_collector_20261003.py
  python scripts/live_bybit_liquidation_collector_20261003.py --selftest
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("BYBIT_LIQ_DB", ROOT / "data/hot/bybit_liq.sqlite"))
SYMBOLS = [s.strip().upper() for s in os.environ.get("BYBIT_LIQ_SYMBOLS", "ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT").split(",") if s.strip()]
WS = "wss://stream.bybit.com/v5/public/linear"
FLUSH_S, PING_S, IDLE_S = 2.0, 20.0, 60.0


def parse(msg: dict, recv_ms: int) -> list[tuple]:
    """allLiquidation 메시지 → 행. 다른 메시지(구독 응답·pong)는 []."""
    if not str(msg.get("topic", "")).startswith("allLiquidation."):
        return []
    return [(int(d["T"]), d["s"], d["S"], float(d["v"]), float(d["p"]), recv_ms) for d in msg.get("data") or []]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS bybit_liquidations(
        ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)""")
    con.execute("CREATE INDEX IF NOT EXISTS bybit_liq_sym_ts ON bybit_liquidations(symbol, ts_ms)")
    con.execute("CREATE TABLE IF NOT EXISTS gaps(from_ms INTEGER, to_ms INTEGER, reason TEXT)")
    return con


def write(con: sqlite3.Connection, rows: list[tuple], gaps: list[tuple] = ()) -> None:
    con.execute("BEGIN")
    con.executemany("INSERT INTO bybit_liquidations VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.executemany("INSERT INTO gaps VALUES (?, ?, ?)", list(gaps))
    con.execute("COMMIT")


async def run() -> None:
    import websockets
    con = connect(DB)
    buf: list[tuple] = []
    down: tuple[int, str] | None = None            # (끊긴 시각 ms, 이유)
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(WS, ping_interval=None, max_size=2**22) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{s}" for s in SYMBOLS]}))
                if down:
                    write(con, [], [(down[0], int(time.time() * 1000), down[1])]); down = None
                print(f"구독 {','.join(SYMBOLS)} → {DB}", flush=True)
                backoff, last_ping, last_flush, last_msg = 1.0, time.time(), time.time(), time.time()
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 5)
                        last_msg = time.time()
                        buf += parse(json.loads(raw), int(last_msg * 1000))
                    except asyncio.TimeoutError:
                        if time.time() - last_msg > IDLE_S:      # pong 도 안 오면 죽은 연결
                            raise ConnectionError("응답 없음 60초")
                    now = time.time()
                    if now - last_ping >= PING_S:
                        await ws.send(json.dumps({"op": "ping"})); last_ping = now
                    if buf and now - last_flush >= FLUSH_S:
                        write(con, buf); buf = []; last_flush = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 다시 붙는다(끊긴 구간은 gaps 에)
            if buf:
                write(con, buf); buf = []
            down = down or (int(time.time() * 1000), f"{type(exc).__name__}: {exc}"[:160])
            print(f"끊김 {exc!r} -- {backoff:.0f}초 뒤 재접속", flush=True)
            await asyncio.sleep(backoff); backoff = min(backoff * 2, 60.0)


def selftest() -> None:
    import tempfile
    m = {"topic": "allLiquidation.ETHUSDT", "type": "snapshot", "ts": 1,
         "data": [{"T": 1000, "s": "ETHUSDT", "S": "Buy", "v": "0.5", "p": "2700.1"},
                  {"T": 1001, "s": "ETHUSDT", "S": "Sell", "v": "1", "p": "2701"}]}
    rows = parse(m, 5)
    assert rows == [(1000, "ETHUSDT", "Buy", 0.5, 2700.1, 5), (1001, "ETHUSDT", "Sell", 1.0, 2701.0, 5)], rows
    assert parse({"op": "pong"}, 5) == [] and parse({"topic": "publicTrade.ETHUSDT", "data": [{}]}, 5) == []
    with tempfile.TemporaryDirectory() as d:
        con = connect(Path(d) / "t.sqlite"); write(con, rows, [(1, 2, "test")])
        assert con.execute("SELECT count(*), sum(qty) FROM bybit_liquidations").fetchone() == (2, 1.5)
        assert con.execute("SELECT count(*) FROM gaps").fetchone()[0] == 1
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        asyncio.run(run())

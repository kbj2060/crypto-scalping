#!/usr/bin/env python3
"""**나스닥100 무기한(Hyperliquid xyz:XYZ100) 실시간 수집기** — 1분봉 + 초당 가격 (2026-10-14).

왜: 사용자 «나스닥 선물과 금리 차트를 실시간으로» → «XYZ100 실시간 차트 붙여줘. 데이터도 수집하고».
  CME 무료 시세(Yahoo NQ=F)는 규정상 10분 늦다. HIP-3 거래소 trade.xyz 의 XYZ100 은 24시간 무기한이라 지연이 없고,
  10-13 실측 5분 수익이 NQ 선물과 상관 0.978(정규장 0.993 · 장외 0.954 · 1봉 지연 0.02 · 하루 거래대금 $1.7억).
  같은 대조에서 OKX QQQ 0.969 · OKX US100 0.47(얇음) · 금리 쪽 HL·OKX 상품은 0.03~0.33 이라 안 받는다.
  🔴주말엔 CME 가 쉬어 이 값도 거의 안 움직인다(5분 |수익| 평일의 1/4) -- 주말 움직임은 코인 쪽 투기, 나스닥이 아니다.
무엇: hot = data/hot/hl_xyz.sqlite(WAL)
  hl_xyz_1m(coin, ts_sec = 1분봉 시작, OHLCV, n) -- 웹소켓 candle 갱신을 그 분에 덮어쓴다(진행 중 봉도 계속 바뀜).
  hl_xyz_ctx_1s(coin, recv_sec, mid, mark, oracle) -- activeAssetCtx(~1/초). 🔴이 메시지엔 거래소 시각이 없다 -- 받은 시각이라
    열 이름이 recv_sec 다(거래소 시각인 척하면 지연 분석이 거짓이 된다, 09-23 HL 호가 수집기와 같은 규약).
  접속할 때마다 REST candleSnapshot 으로 마지막 저장 봉(없으면 72시간 전)부터 메운다 -- 재기동·끊김 구멍. 메움이 실패(429 등)해도
    웹소켓은 그대로 붙는다(실시간이 REST 에 묶이지 않게, 10-14 검토). 30초 조용하면 ping(HL 호가 수집기와 같은 규약).
  seal_to_lake 가 lake/hl/xyz_1m · xyz_ctx_1s / coin=xyz_XYZ100 으로 봉인, hot 은 8일. 주문·바이낸스 호출 없음(읽기 전용 시장 데이터).
  python scripts/live_xyz100_collector_20261014.py              # 상주(supervisor_xyz100.sh 가 띄운다)
  python scripts/live_xyz100_collector_20261014.py --selftest   # 네트워크 없이
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_store import ROOT, rw_connect, sqlite_init  # noqa: E402

DB = Path(os.environ.get("HL_XYZ_DB", ROOT / "data/hot/hl_xyz.sqlite"))
COIN = os.environ.get("HL_XYZ_COIN", "xyz:XYZ100")
WS_URL, INFO_URL = "wss://api.hyperliquid.xyz/ws", "https://api.hyperliquid.xyz/info"
FLUSH_S, BACKFILL_H = 2.0, 72                   # candleSnapshot 은 5000봉까지 -- 72시간(4,320봉)이면 긴 끊김도 메운다
DDL = ("CREATE TABLE IF NOT EXISTS hl_xyz_1m(coin TEXT NOT NULL, ts_sec INTEGER NOT NULL, open REAL, high REAL, low REAL, close REAL, "
       "volume REAL, n INTEGER, PRIMARY KEY(coin, ts_sec))",
       "CREATE TABLE IF NOT EXISTS hl_xyz_ctx_1s(coin TEXT NOT NULL, recv_sec INTEGER NOT NULL, mid REAL, mark REAL, oracle REAL, "
       "PRIMARY KEY(coin, recv_sec))")
UPSERT_1M = "INSERT OR REPLACE INTO hl_xyz_1m VALUES (?,?,?,?,?,?,?,?)"
UPSERT_CTX = "INSERT OR REPLACE INTO hl_xyz_ctx_1s VALUES (?,?,?,?,?)"


def candle_row(c: dict) -> tuple | None:
    """웹소켓 candle·REST candleSnapshot 공통 원소({t ms, o,h,l,c,v 문자열, n}) → hl_xyz_1m 행."""
    try:
        return (c.get("s") or COIN, int(c["t"]) // 1000, float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]),
                float(c.get("v") or 0), int(c.get("n") or 0))
    except (KeyError, TypeError, ValueError):
        return None


def ctx_row(data: dict, recv_ms: int) -> tuple | None:
    """activeAssetCtx → (coin, 받은 초, mid, mark, oracle). 셋 다 없으면 버린다."""
    ctx = data.get("ctx") or {}
    f = lambda k: float(ctx[k]) if ctx.get(k) not in (None, "") else None   # noqa: E731
    try:
        mid, mark, ora = f("midPx"), f("markPx"), f("oraclePx")
    except (TypeError, ValueError):
        return None
    return None if mid is None and mark is None and ora is None else (data.get("coin") or COIN, recv_ms // 1000, mid, mark, ora)


def backfill_rows(last_sec: int | None, now_ms: int) -> list[tuple]:
    """마지막 저장 봉(포함, 진행 중이었을 수 있다)부터 지금까지 REST 1분봉. 없으면 BACKFILL_H 시간 전부터. 스레드에서 부른다(DB 안 만짐)."""
    start = max(now_ms - BACKFILL_H * 3_600_000, (last_sec or 0) * 1000)
    body = json.dumps({"type": "candleSnapshot", "req": {"coin": COIN, "interval": "1m", "startTime": start, "endTime": now_ms}}).encode()
    req = urllib.request.Request(INFO_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return [x for x in (candle_row(c) for c in json.loads(r.read())) if x]


def write(bars: list[tuple], ctxs: list[tuple]) -> None:
    """한 트랜잭션으로 덮어쓴다 -- 스레드에서 부른다(fsync·잠금 대기가 이벤트 루프를 막지 않게)."""
    with rw_connect(DB) as con:
        con.execute("BEGIN")
        con.executemany(UPSERT_1M, bars)
        con.executemany(UPSERT_CTX, ctxs)
        con.execute("COMMIT")


async def backfill() -> None:
    """REST 메움 -- 실패해도 로그만 남기고 돌아간다(다음 재접속 때 다시)."""
    try:
        with rw_connect(DB) as con:
            last = con.execute("SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = ?", (COIN,)).fetchone()[0]
        rows = await asyncio.to_thread(backfill_rows, last, int(time.time() * 1000))
        await asyncio.to_thread(write, rows, [])
        print(f"[xyz100] {time.strftime('%F %T')} 접속 · REST 메움 {len(rows)}봉", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[xyz100] {time.strftime('%F %T')} REST 메움 실패 {type(e).__name__}: {e} -- 웹소켓은 그대로 붙는다", flush=True)


async def run() -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType
    sqlite_init(DB)
    with rw_connect(DB) as con:
        for ddl in DDL:
            con.execute(ddl)
    backoff = 1.0
    async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=15)) as session:
        while True:
            try:
                await backfill()
                ws = await asyncio.wait_for(session.ws_connect(WS_URL, heartbeat=25, max_msg_size=0), 30)   # 핸드셰이크가 멈추면 30초에 끊는다
                async with ws:
                    for sub in ({"type": "candle", "coin": COIN, "interval": "1m"}, {"type": "activeAssetCtx", "coin": COIN}):
                        await ws.send_json({"method": "subscribe", "subscription": sub})
                    bars: dict[int, tuple] = {}
                    ctxs: dict[int, tuple] = {}
                    last_flush, quiet = time.monotonic(), 0
                    while True:
                        try:
                            msg = await ws.receive(timeout=30)
                        except asyncio.TimeoutError:                       # ctx 는 ~1초마다 온다 -- 30초 조용하면 ping, 세 번이면 끊김
                            quiet += 1
                            if quiet >= 3:
                                raise ConnectionError("90초 무소식")
                            await ws.send_json({"method": "ping"})
                            continue
                        if msg.type != WSMsgType.TEXT:
                            raise ConnectionError(f"ws {msg.type}")
                        d = json.loads(msg.data)
                        ch, data = d.get("channel"), d.get("data")
                        if ch == "error":                                  # 구독 거부 등 -- 조용히 버리면 칸이 영구 «끊김»인데 원인이 안 남는다
                            print(f"[xyz100] {time.strftime('%F %T')} 거래소 오류 {str(data)[:200]}", flush=True)
                        elif ch == "candle" and isinstance(data, dict) and (r := candle_row(data)):
                            bars[r[1]] = r; quiet = 0
                        elif ch == "activeAssetCtx" and isinstance(data, dict) and (r := ctx_row(data, int(time.time() * 1000))):
                            ctxs[r[1]] = r; quiet = 0
                        if time.monotonic() - last_flush >= FLUSH_S and (bars or ctxs):
                            await asyncio.to_thread(write, list(bars.values()), list(ctxs.values()))
                            bars.clear(); ctxs.clear(); last_flush = time.monotonic()
                            backoff = 1.0
            except Exception as e:  # noqa: BLE001 -- 끊김·DNS·핸드셰이크 실패 모두 다시 붙는다
                print(f"[xyz100] {time.strftime('%F %T')} 끊김 {type(e).__name__}: {e} -- {backoff:.0f}초 뒤 재접속", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)


def selftest() -> None:
    import sqlite3
    import tempfile
    r = candle_row({"t": 1791909180000, "T": 1791909239999, "s": "xyz:XYZ100", "i": "1m", "o": "30590.0", "c": "30594.0",
                    "h": "30594.0", "l": "30590.0", "v": "0.0079", "n": 4})
    assert r == ("xyz:XYZ100", 1791909180, 30590.0, 30594.0, 30590.0, 30594.0, 0.0079, 4), r
    assert candle_row({"t": "x"}) is None
    c = ctx_row({"coin": "xyz:XYZ100", "ctx": {"midPx": "30593.5", "markPx": "30592.0", "oraclePx": "30583.0"}}, 1791909190999)
    assert c == ("xyz:XYZ100", 1791909190, 30593.5, 30592.0, 30583.0), c
    assert ctx_row({"coin": "xyz:XYZ100", "ctx": {"midPx": None, "markPx": "", "oraclePx": None}}, 0) is None
    with tempfile.TemporaryDirectory() as tmp:                                 # 같은 분 덮어쓰기(진행 중 봉 → 마감 봉)
        con = sqlite3.connect(Path(tmp) / "t.sqlite")
        for ddl in DDL:
            con.execute(ddl)
        con.executemany(UPSERT_1M, [r, (*r[:5], 30600.0, 1.0, 9)])
        assert con.execute("SELECT close, n FROM hl_xyz_1m").fetchall() == [(30600.0, 9)]
        con.close()
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else asyncio.run(run())

"""바이낸스 맥락 수집기: OI 1초(REST)·마크 1초(WS)·강제청산 원시(WS) -> data/hot/binance_ctx.sqlite (2026-10-01 저장 재설계 4d).

대시보드(dashboard/server.py)가 직접 하던 저장을 옮겼다 -- 배포로 대시보드가 재시작돼도 기록이 끊기지 않고,
대시보드는 이 파일(WAL, 락 없음)을 읽는다. 수집 규칙은 대시보드의 것 그대로다:
- OI: stamp(`time`)의 **집합**으로 중복을 거른다(도착 순서가 뒤바뀐다, 2026-09-19 실측 3.5%). PK(ts_ms, symbol).
  🔴REST 가중치 1 × 0.25초 × 종목 -- **이 프로세스만** 폴링한다(대시보드와 둘이면 IP 한도 2,400/분을 넘는다).
  밴·한도는 공용 가드(scripts/binance_ban_guard.py)를 따른다.
- 마크: `/market/ws/<sym>@markPrice@1s`(🔴`/ws/` 는 연결돼도 이벤트 0). PK(ts_ms, symbol).
- 청산: `/market/ws/!forceOrder@arr` 전 종목에서 COIN_CONFIG 종목만. 수량 = z(주문 누적 체결량, 없으면 q), 가격 = ap.
  롱 청산 = 시장에 SELL. 옛 data/live/liq_events[_<coin>].jsonl 과 같은 칸.

  python scripts/live_binance_ctx_collector_20261001.py [--selftest]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.binance_ban_guard as ban_guard  # noqa: E402
from scripts import data_store as ds  # noqa: E402
from scripts.coin_config import COIN_CONFIG  # noqa: E402

HOT_DB = ROOT / "data" / "hot" / "binance_ctx.sqlite"
OI_URL = "https://fapi.binance.com/fapi/v1/openInterest"
OI_POLL_S = 0.25
OI_SYMBOLS = [s.strip().upper() for s in os.getenv("BN_CTX_OI_SYMBOLS", "ETHUSDT,SOLUSDT,XRPUSDT").split(",") if s.strip()]
MARK_SYMBOLS = [s.strip().lower() for s in os.getenv("BN_CTX_MARK_SYMBOLS", "ethusdt").split(",") if s.strip()]
WS_BASE = "wss://fstream.binance.com/market/ws/"
LIQ_SYMBOLS = {c["binance_symbol"] for c in COIN_CONFIG.values()}
FLUSH_S = 10.0
PENDING_CAP = 200_000

DDL = (
    "CREATE TABLE IF NOT EXISTS oi_1s(ts_ms INTEGER, symbol TEXT, open_interest REAL, PRIMARY KEY (ts_ms, symbol)) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS mark_price_1s(ts_ms INTEGER, symbol TEXT, mark REAL, index_px REAL, funding_rate REAL, "
    "next_funding_ms INTEGER, PRIMARY KEY (ts_ms, symbol)) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS liquidations(ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL, symbol TEXT)",
)
INDEXES = ("CREATE INDEX IF NOT EXISTS oi_1s_sym ON oi_1s(symbol, ts_ms)",
           "CREATE INDEX IF NOT EXISTS mark_sym ON mark_price_1s(symbol, ts_ms)",
           "CREATE INDEX IF NOT EXISTS liq_sym ON liquidations(symbol, ts_ms)")
INSERTS = {"oi_1s": "INSERT OR IGNORE INTO oi_1s VALUES (?,?,?)",
           "mark_price_1s": "INSERT OR IGNORE INTO mark_price_1s VALUES (?,?,?,?,?,?)",
           "liquidations": "INSERT INTO liquidations VALUES (?,?,?,?,?,?)"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def init_db(path: Path = HOT_DB) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.sqlite_init(path)
    with ds.rw_connect(path) as con:
        for ddl in DDL + INDEXES:
            con.execute(ddl)


def write(path: Path, batch: dict[str, list[tuple]]) -> None:
    with ds.rw_connect(path) as con:
        con.execute("BEGIN")
        for table, rows in batch.items():
            if rows:
                con.executemany(INSERTS[table], rows)
        con.execute("COMMIT")


def parse_mark(o: dict) -> tuple | None:
    if o.get("e") != "markPriceUpdate":
        return None
    return (int(o["E"]), str(o["s"]).lower(), float(o["p"]), float(o["i"]), float(o["r"]), int(o.get("T") or 0))


def parse_force(msg: dict) -> tuple | None:
    o = msg.get("o") or {}
    if o.get("s") not in LIQ_SYMBOLS:
        return None
    qty, price = float(o.get("z") or o.get("q") or 0.0), float(o.get("ap") or o.get("p") or 0.0)
    return (int(o.get("T") or time.time() * 1000), "long" if o.get("S") == "SELL" else "short",
            qty, price, qty * price, o.get("s"))


class Pending:
    """테이블별 보류 행. flush 실패하면 앞에 되돌려 붙인다(유실 없음, 상한 넘으면 말하고 버림)."""

    def __init__(self, path: Path) -> None:
        self.path, self.rows = path, {t: [] for t in INSERTS}

    def add(self, table: str, row: tuple) -> None:
        self.rows[table].append(row)

    async def flush(self) -> None:
        batch, self.rows = self.rows, {t: [] for t in INSERTS}
        if not any(batch.values()):
            return
        try:
            await asyncio.to_thread(write, self.path, batch)
        except Exception as exc:  # noqa: BLE001
            for t, rows in batch.items():
                self.rows[t] = rows + self.rows[t]
            n = sum(len(v) for v in self.rows.values())
            if n > PENDING_CAP:
                self.rows = {t: [] for t in INSERTS}
                log(f"⚠️쓰기가 계속 막혀 {n}행 버림: {exc!r}")
            else:
                log(f"쓰기 보류 {n}행, 다음에 재시도: {type(exc).__name__}")


async def poll_oi(session, symbol: str, pend: Pending) -> None:
    seen: set[int] = set()
    while True:
        try:
            left = ban_guard.ban_remaining()
            if left > 0:                                    # 다른 프로세스가 받은 차단도 따른다
                await asyncio.sleep(min(left + 1, 300))
                continue
            async with session.get(OI_URL, params={"symbol": symbol}) as resp:
                data = await resp.json()
            if "time" not in data:
                ban_guard.note(resp.status, json.dumps(data), resp.headers.get("Retry-After"))
                m = re.search(r"banned until (\d+)", str(data.get("msg", "")))
                wait = max(5.0, int(m.group(1)) / 1000 - time.time() + 1) if m else 30.0
                log(f"oi {symbol}: 거래소 거절 {resp.status} {data.get('code')} -- {wait:.0f}초 쉰다")
                await asyncio.sleep(wait)
                continue
            ts_ms = int(data["time"])
            if ts_ms not in seen:
                seen.add(ts_ms)
                if len(seen) > 8192:
                    seen = {s for s in seen if s >= ts_ms - 600_000}
                pend.add("oi_1s", (ts_ms, symbol.lower(), float(data["openInterest"])))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 한 번의 실패로 멈추지 않는다
            log(f"oi {symbol} poll failed (will retry): {exc!r}")
            await asyncio.sleep(1.0)
        await asyncio.sleep(OI_POLL_S)


async def ws_loop(session, url: str, table: str, parse, pend: Pending) -> None:
    from aiohttp import WSMsgType
    while True:
        try:
            async with session.ws_connect(url, heartbeat=30) as ws:
                log(f"{table} ws connected")
                async for msg in ws:
                    if msg.type is not WSMsgType.TEXT:
                        break
                    row = parse(json.loads(msg.data) or {})
                    if row is not None:
                        pend.add(table, row)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 끊기면 3초 뒤 다시
            log(f"{table} ws: {exc!r}")
        await asyncio.sleep(3.0)


async def run(path: Path = HOT_DB) -> None:
    from aiohttp import ClientSession, ClientTimeout
    init_db(path)
    pend = Pending(path)
    log(f"시작 -- OI {OI_SYMBOLS} 0.25초 · 마크 {MARK_SYMBOLS} · 청산 {sorted(LIQ_SYMBOLS)} -> {path}")
    async with ClientSession(timeout=ClientTimeout(total=10)) as rest, \
            ClientSession(timeout=ClientTimeout(total=None)) as wss:
        tasks = [asyncio.create_task(poll_oi(rest, s, pend)) for s in OI_SYMBOLS]
        tasks += [asyncio.create_task(ws_loop(wss, f"{WS_BASE}{s}@markPrice@1s", "mark_price_1s", parse_mark, pend))
                  for s in MARK_SYMBOLS]
        tasks.append(asyncio.create_task(ws_loop(wss, f"{WS_BASE}!forceOrder@arr", "liquidations", parse_force, pend)))
        try:
            while True:
                await asyncio.sleep(FLUSH_S)
                await pend.flush()
        finally:
            for t in tasks:
                t.cancel()
            await pend.flush()                             # 종료(재기동) 때 보류분을 버리지 않는다


def selftest() -> None:
    import tempfile
    assert parse_mark({"e": "x"}) is None
    m = parse_mark({"e": "markPriceUpdate", "E": 1000, "s": "ETHUSDT", "p": "2500.1", "i": "2500.0", "r": "0.0001", "T": 9})
    assert m == (1000, "ethusdt", 2500.1, 2500.0, 0.0001, 9), m
    f = parse_force({"o": {"s": "ETHUSDT", "S": "SELL", "z": "2", "q": "5", "ap": "2500", "p": "2400", "T": 7}})
    assert f == (7, "long", 2.0, 2500.0, 5000.0, "ETHUSDT"), f
    assert parse_force({"o": {"s": "DOGEUSDT"}}) is None, "COIN_CONFIG 밖 종목은 버린다"
    d = Path(tempfile.mkdtemp()) / "c.sqlite"
    init_db(d)
    init_db(d)
    pend = Pending(d)
    pend.add("oi_1s", (1000, "ethusdt", 1.5))
    pend.add("oi_1s", (1000, "ethusdt", 1.5))               # 같은 stamp -> PK 가 거른다
    pend.add("mark_price_1s", m)
    pend.add("liquidations", f)
    asyncio.run(pend.flush())
    assert ds.read_rows(d, "SELECT count(*) FROM oi_1s") == [(1,)]
    assert ds.read_rows(d, "SELECT side, usd FROM liquidations") == [("long", 5000.0)]
    import sqlite3
    c = sqlite3.connect(d)
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal" and c.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        asyncio.run(run())

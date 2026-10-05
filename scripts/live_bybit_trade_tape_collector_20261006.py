#!/usr/bin/env python3
"""**Bybit 체결 테이프 + 맥락(OI·마크·펀딩) 수집기** (2026-10-06, 사용자 «OKX 와 바이비트도 합산» → «권장 순서대로»).

왜: 바이낸스는 ETH 테이커 플로우의 38~43%뿐이고 Bybit 가 19.1%다(2026-09-17~19 |순델타| 점유, OKX 수집기 머리말).
  합산 수급·OI 의 세 번째 거래소. 🔴OKX 와 같다 -- 고치는 것은 «설명»이지 «예측»이 아니다(CVD 는 동행지표).
무엇: 공개 WS 하나(`publicTrade` + `tickers`, 5코인 18KB/s · 2026-10-06 서버 60초 실측)로
  · 체결 → data/hot/bybit_tape.sqlite `trade_tape_1s`(바이낸스·OKX 테이프와 **같은 표·같은 칸 뜻**)
  · tickers → data/hot/bybit_ctx.sqlite `bybit_oi`(값이 바뀔 때) · `bybit_mark`(종목당 1초 1점) · `bybit_funding`(바뀔 때)
⭐묶는 단위 = 연속된 (체결 시각 ms, 방향, 가격). 테이커 주문 하나의 체결은 시각이 같다(실측 5코인 전부 같은 seq 안 시각 동일)
  -- 가격까지 묶으면 바이낸스 aggTrade · OKX `trades` 와 **같은 단위**(한 주문의 한 가격)가 된다. `seq` 대신 시각을 쓰는 이유는
  과거 일별 덤프(public.bybit.com/trading)에 seq 가 없어서다 -- 검정(맞대결 합산판, 2026-10-06)과 라이브가 같은 규칙이어야 한다.
  경계·가격빈은 바이낸스 수집기에서 import 한다.
🔴`S` 는 **테이커** 방향이다(Buy = 테이커 매수) -- 뒤집지 않는다. `v` 는 기초자산 수량(USDT 선형은 계약 = 코인 1개).
🔴청산은 따로 돈다(live_bybit_liquidation_collector_20261003.py) -- 여기서 받지 않는다.
ponytail: REST 복구 없음 -- Bybit 최근 체결 REST 는 1000건뿐이라 분 단위 재구성이 안 된다. 끊긴 구간은 gaps 에 적고
  1분봉 대조(verify_1m)로 유실을 드러낸다. 필요해지면 public.bybit.com/trading 일별 덤프로 덮는다.
봇과 완전 분리: 주문 없음 · 바이낸스 호출 없음(데이터는 USDT 선형 = 데이터 규칙).
  python scripts/live_bybit_trade_tape_collector_20261006.py
  BYBIT_TAPE_SYMBOLS=ETHUSDT python scripts/live_bybit_trade_tape_collector_20261006.py
  python scripts/live_bybit_trade_tape_collector_20261006.py --selftest
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_bn = importlib.import_module("scripts.live_trade_tape_collector_20260916")
TapeBuffer, TapeStore, log = _bn.TapeBuffer, _bn.TapeStore, _bn.log

TAPE_DB = Path(os.environ.get("BYBIT_TAPE_DB", ROOT / "data/hot/bybit_tape.sqlite"))
CTX_DB = Path(os.environ.get("BYBIT_CTX_DB", ROOT / "data/hot/bybit_ctx.sqlite"))
SYMBOLS = [s.strip().upper() for s in os.environ.get("BYBIT_TAPE_SYMBOLS", "ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT").split(",") if s.strip()]
WS = "wss://stream.bybit.com/v5/public/linear"
KLINE_URL = "https://api.bybit.com/v5/market/kline"
FLUSH_S, PING_S, IDLE_S, VERIFY_S = 5.0, 20.0, 60.0, 300.0
MARK_EVERY_MS = 1000           # ponytail: 마크는 종목당 1초 1점(tickers 는 100ms) -- 1초 해상도면 베이시스·가격차에 충분


class OrderGrouper:
    """체결 → (시각 ms, 방향, 가격) 묶음. 다른 키가 오면 앞 묶음을 닫는다(같은 주문의 체결은 연달아 온다).
    메시지 경계를 넘는 묶음도 이어 붙는다 -- 닫는 건 «다른 키» 또는 `flush()`(1초 넘게 조용할 때)."""

    def __init__(self) -> None:
        self.key, self.ts, self.qty, self.n, self.last = None, 0, 0.0, 0, 0.0

    def add(self, sell: bool, px: float, qty: float, ts_ms: int) -> tuple | None:
        k, done = (ts_ms, sell, px), None
        if k != self.key:
            done = self.flush()
            self.key, self.ts, self.qty, self.n = k, ts_ms, 0.0, 0
        self.qty += qty; self.n += 1; self.last = time.monotonic()
        return done

    def flush(self) -> tuple | None:
        """(ts_ms, px, qty, sell, 체결수) 또는 None."""
        if self.key is None:
            return None
        out = (self.ts, self.key[2], self.qty, self.key[1], self.n)
        self.key = None
        return out


def parse_trades(msg: dict) -> list[tuple]:
    """publicTrade → [(심볼, seq, sell, px, qty, ts_ms)]. 못 믿을 행은 버린다."""
    out = []
    for d in msg.get("data") or []:
        try:
            px, qty, ts, side = float(d["p"]), float(d["v"]), int(d["T"]), d["S"]
        except (KeyError, TypeError, ValueError):
            continue
        if px > 0 and qty > 0 and ts > 0 and side in ("Buy", "Sell"):
            out.append((d.get("s"), d.get("seq", d.get("i")), side == "Sell", px, qty, ts))
    return out


def ctx_conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: 쓰기는 to_thread(flush) 에서도 한다 -- 둘이 동시에 돌지는 않는다(flush 를 await 한다)
    con = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    con.execute("PRAGMA auto_vacuum = INCREMENTAL")   # 새 파일에서만, WAL 전환보다 먼저(봉인기 prune_hot 이 공간을 돌려받게)
    con.execute("PRAGMA journal_mode=WAL")
    for ddl in ("CREATE TABLE IF NOT EXISTS bybit_oi(inst TEXT, ts_ms INTEGER, oi_base REAL, oi_usd REAL)",
                "CREATE TABLE IF NOT EXISTS bybit_mark(inst TEXT, ts_ms INTEGER, mark_px REAL, index_px REAL)",
                "CREATE TABLE IF NOT EXISTS bybit_funding(inst TEXT, ts_ms INTEGER, funding_rate REAL, next_funding_ms INTEGER)",
                "CREATE TABLE IF NOT EXISTS gaps(from_ms INTEGER, to_ms INTEGER, reason TEXT)",
                "CREATE INDEX IF NOT EXISTS bybit_oi_inst_ts ON bybit_oi(inst, ts_ms)",
                "CREATE INDEX IF NOT EXISTS bybit_mark_inst_ts ON bybit_mark(inst, ts_ms)",
                "CREATE INDEX IF NOT EXISTS bybit_funding_inst_ts ON bybit_funding(inst, ts_ms)"):
        con.execute(ddl)
    return con


class TickerState:
    """tickers 는 첫 메시지만 스냅샷이고 뒤는 «바뀐 칸만» 온다 -- 종목별로 합쳐 들고 있다가 행으로 낸다."""

    def __init__(self) -> None:
        self.cur: dict[str, dict] = {}
        self.last_oi: dict[str, float] = {}
        self.last_fr: dict[str, tuple] = {}
        self.last_mark_ms: dict[str, int] = {}

    def rows(self, msg: dict) -> dict[str, list[tuple]]:
        d, ts = msg.get("data") or {}, int(msg.get("ts") or 0)
        s = d.get("symbol")
        if not s or ts <= 0:
            return {}
        c = self.cur.setdefault(s, {})
        c.update({k: v for k, v in d.items() if v not in (None, "")})
        f = lambda k: float(c[k]) if k in c else None   # noqa: E731
        out: dict[str, list[tuple]] = {}
        oi = f("openInterest")
        if oi is not None and oi != self.last_oi.get(s):
            self.last_oi[s] = oi
            out["bybit_oi"] = [(s, ts, oi, f("openInterestValue"))]
        fr = (f("fundingRate"), int(f("nextFundingTime") or 0))
        if fr[0] is not None and fr != self.last_fr.get(s):
            self.last_fr[s] = fr
            out["bybit_funding"] = [(s, ts, *fr)]
        if "markPrice" in d and ts - self.last_mark_ms.get(s, 0) >= MARK_EVERY_MS:
            self.last_mark_ms[s] = ts
            out["bybit_mark"] = [(s, ts, f("markPrice"), f("indexPrice"))]
        return out


async def verify_recent(session, stores: dict[str, TapeStore]) -> None:
    """닫힌 분의 테이프 합을 Bybit 1분봉 거래량과 대조(verify_1m). 실패해도 수집은 계속."""
    for s, store in stores.items():
        minutes = await asyncio.to_thread(store.unverified_minutes)
        if not minutes:
            continue
        try:
            async with session.get(KLINE_URL, params={"category": "linear", "symbol": s, "interval": "1",
                                                      "start": min(minutes) * 1000, "end": max(minutes) * 1000,
                                                      "limit": 1000}) as r:
                body = await r.json()
            kvol = {int(k[0]) // 1000: float(k[5]) for k in body["result"]["list"]}
        except Exception as exc:  # noqa: BLE001
            log(f"{s} 1분봉 조회 실패(수집은 계속): {type(exc).__name__}")
            continue
        bad = [r for r in await asyncio.to_thread(store.verify, kvol, minutes) if abs(r[1]) > _bn.VERIFY_TOLERANCE and not r[2]]
        for m, rel, _ in bad:
            log(f"⚠️{s} 완전성 {time.strftime('%H:%M', time.localtime(m))} rel_err {rel:+.4%} -- 체결 유실")


async def run() -> None:
    import aiohttp
    stores = {s: TapeStore(TAPE_DB, s, _bn.BUCKETS.get(s.lower(), 0.01)) for s in SYMBOLS}
    bufs = {s: TapeBuffer(_bn.BUCKETS.get(s.lower(), 0.01), s) for s in SYMBOLS}
    groups = {s: OrderGrouper() for s in SYMBOLS}
    for s, st in stores.items():
        with st._connect() as con:
            for k, v in ((f"max_unit:{s}", "taker_order_price_level"), (f"source:{s}", "bybit v5 publicTrade grouped by consecutive (T ms, side, price)")):
                con.execute("DELETE FROM meta WHERE key = ?", [k]); con.execute("INSERT INTO meta VALUES (?, ?)", [k, v])
    ctx, tick = ctx_conn(CTX_DB), TickerState()
    last_ms = {s: st.last_ts_ms() for s, st in stores.items()}
    ctx_buf: dict[str, list[tuple]] = {}
    down_ms, vtask = None, None

    def add_order(s: str, o: tuple | None) -> None:
        if o:
            bufs[s].add_agg(o[0], o[1], o[2], o[3], o[4])

    def flush(force: bool = False) -> None:
        now = time.monotonic()
        for s, g in groups.items():
            if g.key is not None and (force or now - g.last > 1.0):
                add_order(s, g.flush())
        for s, st in stores.items():
            st.write(bufs[s].take_closed(everything=force))
        if ctx_buf:
            ctx.execute("BEGIN")
            try:
                for t, rows in ctx_buf.items():
                    ctx.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
                ctx.execute("COMMIT")
            except Exception:
                ctx.execute("ROLLBACK")   # 열린 트랜잭션을 남기면 다음 BEGIN 이 전부 실패한다(10-06 리뷰) -- 행은 ctx_buf 에 남아 다음에 다시
                raise
            ctx_buf.clear()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        while True:
            try:
                async with session.ws_connect(WS, heartbeat=None, max_msg_size=2**22) as ws:
                    await ws.send_json({"op": "subscribe", "args": [f"publicTrade.{s}" for s in SYMBOLS] + [f"tickers.{s}" for s in SYMBOLS]})
                    log(f"구독 {','.join(SYMBOLS)} → {TAPE_DB.name} · {CTX_DB.name}")
                    first = set(SYMBOLS)
                    t_ping = t_flush = t_verify = t_msg = time.monotonic()
                    while True:
                        try:
                            m = await ws.receive(timeout=5)
                        except asyncio.TimeoutError:
                            m = None
                        now = time.monotonic()
                        if m is not None:
                            if m.type != aiohttp.WSMsgType.TEXT:
                                raise ConnectionError(f"ws {m.type}")
                            t_msg = now
                            msg = json.loads(m.data)
                            topic = str(msg.get("topic", ""))
                            if topic.startswith("publicTrade."):
                                for s, seq, sell, px, qty, ts in parse_trades(msg):
                                    if s not in groups:
                                        continue
                                    if s in first:          # 연결 뒤 첫 체결 -- 그 앞은 «모름»
                                        first.discard(s)
                                        stores[s].record_gap(last_ms[s] or ts // 60_000 * 60_000, ts, "ws_reconnect" if last_ms[s] else "startup")
                                    add_order(s, groups[s].add(sell, px, qty, ts))
                                    last_ms[s] = ts
                            elif topic.startswith("tickers."):
                                for t, rows in tick.rows(msg).items():
                                    ctx_buf.setdefault(t, []).extend(rows)
                        elif now - t_msg > IDLE_S:
                            raise ConnectionError("응답 없음 60초")
                        if now - t_ping >= PING_S:
                            await ws.send_json({"op": "ping"}); t_ping = now
                        if now - t_flush >= FLUSH_S:
                            await asyncio.to_thread(flush); t_flush = now
                        if down_ms:
                            ctx.execute("INSERT INTO gaps VALUES (?, ?, ?)", (down_ms[0], int(time.time() * 1000), down_ms[1])); down_ms = None
                        if now - t_verify >= VERIFY_S and (vtask is None or vtask.done()):
                            t_verify = now   # 별도 태스크 -- Bybit REST 가 막혀도 WS 수신·ping·flush 가 멈추지 않는다(10-06 리뷰)
                            vtask = asyncio.create_task(verify_recent(session, stores))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log(f"연결 끊김, 3초 뒤 재시도: {type(exc).__name__} {exc}")
                down_ms = down_ms or (int(time.time() * 1000), type(exc).__name__)
            try:
                await asyncio.to_thread(flush, True)   # 열린 묶음·초를 버리지 않는다(재연결 뒤 첫 체결이 gap 을 적는다)
            except Exception as exc:  # noqa: BLE001 -- 여기서 죽으면 run() 이 끝난다. 행은 버퍼에 남아 다음 flush 가 다시 쓴다
                log(f"재연결 전 쓰기 실패(다음 주기 재시도): {type(exc).__name__} {exc}")
            for b in bufs.values():
                b.closed_before = max(b.closed_before, b.max_sec + 1)
            await asyncio.sleep(3)


def selftest() -> None:
    # 묶기: 연속된 같은 (시각, 방향, 가격)만 한 주문 · 메시지 경계를 넘어도 이어진다
    g, out = OrderGrouper(), []
    for sell, px, q, ts in [(False, 100.0, 1, 10), (False, 100.0, 2, 10), (False, 100.5, 3, 10),
                            (True, 100.5, 4, 11), (True, 100.5, 5, 12)]:
        o = g.add(sell, px, q, ts)
        if o:
            out.append(o)
    out.append(g.flush())
    assert out == [(10, 100.0, 3.0, False, 2), (10, 100.5, 3.0, False, 1), (11, 100.5, 4.0, True, 1), (12, 100.5, 5.0, True, 1)], out
    assert g.flush() is None
    # 파싱: S 는 테이커 방향 그대로 · 못 믿을 행은 버린다
    msg = {"topic": "publicTrade.ETHUSDT", "data": [
        {"T": 1791218171553, "s": "ETHUSDT", "S": "Sell", "v": "0.01", "p": "2691.78", "seq": 7},
        {"T": 1791218171553, "s": "ETHUSDT", "S": "Buy", "v": "2", "p": "2691.79", "seq": 8},
        {"T": 0, "s": "ETHUSDT", "S": "Buy", "v": "2", "p": "1", "seq": 9},
        {"T": 1, "s": "ETHUSDT", "S": "X", "v": "2", "p": "1", "seq": 9}]}
    assert parse_trades(msg) == [("ETHUSDT", 7, True, 2691.78, 0.01, 1791218171553), ("ETHUSDT", 8, False, 2691.79, 2.0, 1791218171553)]
    # 크기 구간: ETH 경계($1만/$10만)가 바이낸스 표와 같은 뜻으로 들어간다
    b = TapeBuffer(0.1, "ETHUSDT")
    b.add_agg(5_000_000, 2500.0, 40.0, False, 3)    # $100k 고래
    b.add_agg(5_000_100, 2500.0, 3.9, True, 1)      # $9,750 리테일
    b.add_agg(5_001_000, 2500.0, 1.0, False, 1)     # 다음 초 -- 위 초를 닫는다
    r = b.take_closed()[0]
    assert (r[2], r[3], r[10], r[9]) == (40.0, 3.9, 40.0, 3.9), r
    # tickers: 스냅샷 + 델타 합치기 · OI·펀딩은 바뀔 때만 · 마크는 1초 1점
    t = TickerState()
    a = t.rows({"ts": 1000, "data": {"symbol": "ETHUSDT", "openInterest": "10", "openInterestValue": "25000",
                                     "fundingRate": "0.0001", "nextFundingTime": "9000", "markPrice": "2500", "indexPrice": "2501"}})
    assert a == {"bybit_oi": [("ETHUSDT", 1000, 10.0, 25000.0)], "bybit_funding": [("ETHUSDT", 1000, 0.0001, 9000)],
                 "bybit_mark": [("ETHUSDT", 1000, 2500.0, 2501.0)]}, a
    assert t.rows({"ts": 1500, "data": {"symbol": "ETHUSDT", "markPrice": "2502"}}) == {}, "1초 안 마크·안 바뀐 OI 는 안 쓴다"
    assert t.rows({"ts": 2100, "data": {"symbol": "ETHUSDT", "openInterest": "11", "markPrice": "2503"}}) == {
        "bybit_oi": [("ETHUSDT", 2100, 11.0, 25000.0)], "bybit_mark": [("ETHUSDT", 2100, 2503.0, 2501.0)]}
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            log("종료")

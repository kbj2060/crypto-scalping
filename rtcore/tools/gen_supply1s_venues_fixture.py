#!/usr/bin/env python3
"""rtcore 1초 수급 «나머지 원천» 일치 시험의 정답지 (2026-11-20).

원천 다섯 개의 합성 메시지 열(벽시계 순으로 섞음)을 파이썬 쪽 수신 코드에 먹여 supply_1s_payload 가 낼 칸을 만든다.
rtcore/tests/supply1s_venues_parity.rs 가 같은 메시지 원문을 Rust 수신 경로(feed::parse* → State::apply)에 넣어 대조한다.

  원천                      파이썬 쪽(정답)
  bn    바이낸스 선물 @trade   coin_flow.py collect_footprint **원본**(liq 창의 끝 = 이 원천의 가장 새 초)
  force 바이낸스 !forceOrder  server.py collect_force_orders 본문 -- **옮겨 적음**(server.py 는 import 하지 않는다)
  spot  바이낸스 현물 @aggTrade coin_flow.py collect_spot_flow **원본**
  okx   OKX trades·OI·청산    coin_flow.py collect_okx_flow **원본**(기동 ctVal REST 대조는 실패시켜 «표 값으로 계속» 길)
  bybit Bybit allLiquidation  수집기 parse·connect·write **원본**(임시 SQLite) → server.py _bybit_liq_refresh 의 SELECT·변환 옮겨 적음
payload 조립(supply_1s_payload 의 liq·spot·okx·okxOi·okxLiq·bybitLiq·partial·meta)도 server.py 에서 옮겨 적었다(아래 위치 주석).

네트워크 없음(gen_supply1s_fixture 의 소켓 차단을 그대로 쓴다) · `.env` 안 읽음 · 합성 값만.
    python3 rtcore/tools/gen_supply1s_venues_fixture.py     # 저장소 루트에서. 약 10초(재연결 대기 3초 x3)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import sqlite3
import sys
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.pop("BYBIT_LIQ_SYMBOLS", None)      # 수집기 기본 종목표를 정답지에 싣는다
import gen_supply1s_fixture as g1  # noqa: E402 -- 소켓 차단 · ROOT · payload_bn
cf = g1.cf
from aiohttp import WSMsgType  # noqa: E402
import scripts.live_bybit_liquidation_collector_20261003 as bybit_col  # noqa: E402
from scripts.coin_config import COIN_CONFIG  # noqa: E402
from scripts.live_okx_trade_tape_collector_20260923 import CT_VALS  # noqa: E402

ROOT = g1.ROOT
OUT = ROOT / "rtcore" / "tests" / "fixtures" / "supply1s_venues_eth.json"
SEED = 20261121
SYM = "ETHUSDT"
INST = "ETH-USDT-SWAP"
T0 = 1_763_600_400_000        # 합성 -- 실제 장과 무관
N_SEC = 760                   # 660초 보관 정리를 건드리려면 660 보다 길어야 한다


def fix_wall(w: int) -> int:
    """파이썬 time.time()*1000 이 정수 ms 로 되돌아오고(int(..) 내림이 초·ms 를 안 바꾸게) 초 경계가 아니게."""
    while w % 1000 == 0 or (w / 1000) * 1000 != w:
        w += 1
    return w


# ── 합성 메시지 열: 한 줄 = [원천, 연결번호, 수신 벽시계 ms, 원문] ─────────────────────────────
LATE: set[str] = set()      # 600초 넘게 늦은 현물·OKX 메시지 원문 -- 바로 뒤에서 중간 대조


def gen() -> list[list]:
    rng = random.Random(SEED)
    rows: list[list] = []
    conn = {"bn": 0, "force": 0, "spot": 0, "okx": 0, "bybit": 0}
    st = {"tid": 7_000_000, "agg": 900_000, "okx_tid": 400_000}

    def put(src: str, w: int, text: str) -> None:
        rows.append([src, conn[src], fix_wall(w), text])

    mid = 3000.0
    oi = 1_234_567.8
    for s in range(N_SEC):
        base = T0 + s * 1000
        mid = round(mid + rng.gauss(0, 0.5), 2)
        wall = base + 300     # 이 초의 메시지들이 도착하는 벽시계(대략)

        # bn @trade -- liq 창의 끝을 정한다(초마다 한두 건)
        for _ in range(rng.choice([0, 0, 1, 1])):
            st["tid"] += 1
            T = base + rng.randint(0, 900)
            put("bn", wall + rng.randint(1, 600), json.dumps({"e": "trade", "E": T + 5, "T": T, "s": SYM, "t": st["tid"],
                                                              "p": f"{mid:.2f}", "q": f"{rng.uniform(0.01, 3):.3f}", "X": "MARKET",
                                                              "m": rng.random() < 0.5}))
        if s in (N_SEC - 1 - 660, N_SEC - 1):   # 마지막 초(newest)와 정확히 newest − 660 -- 보관 경계 칸이 남는지(양성 대조로 확인)
            st["tid"] += 1
            put("bn", base + 950, json.dumps({"e": "trade", "E": base + 901, "T": base + 900, "s": SYM, "t": st["tid"],
                                              "p": f"{mid:.2f}", "q": "0.100", "X": "MARKET", "m": False}))

        # force -- 바이낸스 강제청산
        def fo(T, S="SELL", q=None, z=None, ap=None, p=None, sym=SYM, **extra):
            o = {"s": sym, "S": S, "o": "LIMIT", "f": "IOC", "X": "FILLED", "l": "0.1", "L": f"{mid:.2f}", **extra}
            for k, v in (("q", q), ("z", z), ("ap", ap), ("p", p), ("T", T)):
                if v is not None:
                    o[k] = v
            return json.dumps({"e": "forceOrder", "E": (T or base) + 3, "o": o})
        # 특수 경우는 마지막 660초 창(s >= 100) 안에 둔다 -- 창 밖이면 스냅숏 대조가 못 본다(양성 대조로 확인)
        if s == 140:     # 같은 ms 세 건 · 롱·숏 섞임
            for S in ("SELL", "SELL", "BUY"):
                put("force", wall, fo(base + 10, S, q="2.000", z="2.000", ap=f"{mid:.2f}", p=f"{mid - 5:.2f}"))
        elif s == 141:   # z 없음 -> q · ap 빈 문자열 -> p · T 없음 -> 수신 시각 · z "0"(참인 문자열 -> 수량 0 그대로)
            put("force", wall, fo(base + 20, q="1.250", ap=f"{mid:.2f}"))
            put("force", wall + 1, fo(base + 30, "BUY", q="3.000", z="", ap="", p=f"{mid + 1:.2f}"))
            put("force", wall + 2, fo(None, q="0.500", z="0.500", ap=f"{mid:.2f}"))
            put("force", wall + 3, fo(base + 40, q="9.000", z="0", ap=f"{mid:.2f}"))
        elif s == 142:   # 다른 종목 · o 없음
            put("force", wall, fo(base + 50, sym="SOLUSDT", q="10", z="10", ap="150.1"))
            put("force", wall + 1, json.dumps({"e": "forceOrder", "E": base}))
        elif s == 300:   # 연결 0 의 끝: 깨진 JSON(파이썬은 예외 -> 재연결, 그 연결 남은 것 없음)
            put("force", wall, fo(base + 5, q="4.000", z="4.000", ap=f"{mid:.2f}"))
            put("force", wall + 1, "{broken")
            conn["force"] += 1
        elif s == 500:   # 늦게 온 아주 오래된 청산(창 밖) · 몇 초 전(창 안 닫힌 초)
            put("force", wall, fo(base - 700_000, q="5.000", z="5.000", ap=f"{mid:.2f}"))
            put("force", wall + 1, fo(base - 4000, "BUY", q="6.000", z="6.000", ap=f"{mid:.2f}"))
        elif s == N_SEC - 1:   # 마지막 초(= newest) -- 파이썬 창 밖(진행 중인 초)
            put("force", wall, fo(base + 100, q="1.000", z="1.000", ap=f"{mid:.2f}"))
        elif rng.random() < 0.12:
            for _ in range(rng.randint(1, 2)):
                q = f"{rng.uniform(0.05, 60):.3f}"
                put("force", wall + rng.randint(0, 500), fo(base + rng.randint(0, 999), rng.choice(["SELL", "BUY"]), q=q, z=q,
                                                              ap=f"{mid + rng.uniform(-2, 2):.2f}", p=f"{mid:.2f}"))

        # spot -- 바이낸스 현물 @aggTrade
        def agg(T, p, q, m):
            st["agg"] += 1
            d = {"e": "aggTrade", "E": (T or base) + 2, "s": SYM, "a": st["agg"], "p": p, "q": q, "f": 1, "l": 2, "m": m, "M": True}
            if T is not None:
                d["T"] = T
            return json.dumps(d)
        if s == 160:     # 경계: $100,000 = 고래(>=) · $10,000 = 중형 · $9,999.6 = 리테일 · 가격 0 · 다른 사건 · T 없음
            put("spot", wall, agg(base + 1, "2500.00", "40.0000", False))
            put("spot", wall + 1, agg(base + 2, "2500.00", "4.0000", True))
            put("spot", wall + 2, agg(base + 3, "2500.00", "3.9998", False))
            put("spot", wall + 3, agg(base + 4, "0", "1.0", False))
            put("spot", wall + 4, json.dumps({"e": "kline", "E": base, "s": SYM}))
            put("spot", wall + 5, agg(None, f"{mid:.2f}", "0.3000", True))
        elif s == 170:   # 늦게 온 몇 초 전(닫힌 초가 고쳐진다)
            put("spot", wall, agg(base - 3000 + 7, f"{mid:.2f}", "2.5000", False))
        elif s == 200:   # 700초 전(보관 밖 -- 칸을 만들고 다음 새 초에 지워진다)
            put("spot", wall, agg(base - 700_000, f"{mid:.2f}", "8.0000", True))
            LATE.add(rows[-1][3])
        elif s == 350:   # 연결 0 끝: 깨진 JSON
            put("spot", wall, "{nope")
            conn["spot"] += 1
        elif s in (N_SEC - 2 - 660, N_SEC - 2):   # 현물 마지막 초와 그 − 660(보관 경계 칸)
            put("spot", wall, agg(base + 11, f"{mid:.2f}", "0.2000", False))
        elif s < N_SEC - 2 and rng.random() < 0.3:
            for _ in range(rng.randint(1, 2)):
                q = rng.choice([rng.uniform(0.001, 1.5), rng.uniform(1, 9), rng.uniform(30, 45)])
                put("spot", wall + rng.randint(0, 500), agg(base + rng.randint(0, 999), f"{mid + 0.4 + rng.choice([-0.01, 0, 0.01]):.2f}",
                                                             f"{q:.4f}", rng.random() < 0.5))

        # okx
        def okx_trades(entries):
            return json.dumps({"arg": {"channel": "trades", "instId": INST}, "data": entries})

        def ot(T, px, sz, side):
            st["okx_tid"] += 1
            d = {"instId": INST, "tradeId": str(st["okx_tid"]), "px": px, "sz": sz, "side": side, "count": "1"}
            if T is not None:
                d["ts"] = str(T)
            return d
        if s == 5:
            put("okx", wall - 200, json.dumps({"event": "subscribe", "arg": {"channel": "trades", "instId": INST}, "connId": "a1"}))
            put("okx", wall - 199, "pong")
        if s == 170:     # 한 메시지 여러 건 · 방향 이상 · 가격 0 · ts 없음 · 고래 경계(334계약 x 0.1 x 3000 >= 100,000)
            put("okx", wall, okx_trades([ot(base + 1, f"{mid:.1f}", "15", "buy"), ot(base + 2, f"{mid:.1f}", "15", "hold"),
                                         ot(base + 3, "0", "15", "sell"), ot(None, f"{mid:.1f}", "7", "sell"),
                                         ot(base + 5, "3000.0", "334", "sell"), ot(base + 6, "3000.0", "33", "buy")]))
        elif s == 220:   # 늦은 체결(닫힌 초)
            put("okx", wall, okx_trades([ot(base - 5000 + 9, f"{mid:.1f}", "40", "sell")]))
        elif s == 250:   # 700초 전 체결 · 700초 전 OI
            put("okx", wall, okx_trades([ot(base - 700_000, f"{mid:.1f}", "11", "buy")]))
            put("okx", wall + 1, json.dumps({"arg": {"channel": "open-interest", "instId": INST},
                                             "data": [{"instId": INST, "instType": "SWAP", "oi": "1", "oiCcy": "999.5", "oiUsd": "1",
                                                       "ts": str(base - 700_000)}]}))
            LATE.update((rows[-2][3], rows[-1][3]))
        elif s == 400:   # 연결 0 끝: 앞 건은 들어가고 셋째 건 px 가 숫자가 아님(파이썬 float() 예외 -> 재연결)
            put("okx", wall, okx_trades([ot(base + 1, f"{mid:.1f}", "3", "buy"), ot(base + 2, f"{mid:.1f}", "4", "sell"),
                                         ot(base + 3, "abc", "5", "buy")]))
            conn["okx"] += 1
        elif s in (N_SEC - 6 - 660, N_SEC - 6):   # OKX 마지막 초와 그 − 660(보관 경계 칸)
            put("okx", wall, okx_trades([ot(base + 12, f"{mid:.1f}", "2", "buy")]))
        elif s < N_SEC - 6 and rng.random() < 0.25:   # 끝 5초는 OKX 체결 없음 -> okxNow != now(okxLiq 창이 제 초로 잘리는지)
            put("okx", wall + rng.randint(0, 400), okx_trades([
                ot(base + rng.randint(0, 999), f"{mid + rng.choice([-0.1, 0, 0.1]):.1f}", str(rng.choice([1, 3, 10, 40, 150, 400])),
                   rng.choice(["buy", "sell"])) for _ in range(rng.randint(1, 3))]))
        if s % 6 == 1 or s in (190, 191):   # 660 = 6 x 110 -> 창 경계(top − 660) 초에 OI 가 있다     # OI: 같은 초 두 번(마지막 값) · 0 · ts 0
            oi = round(oi + rng.uniform(-50, 50), 1)
            d = [{"instId": INST, "instType": "SWAP", "oi": "1", "oiCcy": f"{oi:.1f}", "oiUsd": "1", "ts": str(base + 500)}]
            if s == 190:
                d.append({"instId": INST, "instType": "SWAP", "oi": "1", "oiCcy": f"{oi + 3:.1f}", "oiUsd": "1", "ts": str(base + 900)})
                d.append({"instId": INST, "instType": "SWAP", "oi": "1", "oiCcy": "0", "oiUsd": "1", "ts": str(base + 901)})
                d.append({"instId": INST, "instType": "SWAP", "oi": "1", "oiCcy": "5", "oiUsd": "1", "ts": "0"})
            put("okx", wall + 450, json.dumps({"arg": {"channel": "open-interest", "instId": INST}, "data": d}))
        if s in (180, 181, N_SEC - 2) or (s > 100 and rng.random() < 0.06):
            dets = [{"bkLoss": "0", "bkPx": f"{mid + rng.uniform(-3, 3):.1f}", "ccy": "", "posSide": rng.choice(["long", "short"]),
                     "side": "sell", "sz": str(rng.choice([1, 5, 20, 300])), "ts": str(base + rng.randint(0, 999))}
                    for _ in range(rng.randint(1, 3))]
            if s == 180:  # 수량 0 · bkPx 없음 · ts 없음
                dets += [{"posSide": "long", "sz": "0", "bkPx": "3000", "ts": str(base)},
                         {"posSide": "short", "sz": "7", "ts": str(base + 2)},
                         {"posSide": "long", "sz": "2", "bkPx": f"{mid:.1f}"}]
            data = [{"instId": INST, "instType": "SWAP", "details": dets}]
            if s == 181:  # 다른 종목(전 종목이 온다)
                data.append({"instId": "BTC-USDT-SWAP", "instType": "SWAP",
                             "details": [{"posSide": "long", "sz": "9", "bkPx": "90000", "ts": str(base)}]})
            put("okx", wall + 470, json.dumps({"arg": {"channel": "liquidation-orders", "instType": "SWAP"}, "data": data}))

        # bybit
        def by(entries):
            return json.dumps({"topic": f"allLiquidation.{SYM}", "type": "snapshot", "ts": base, "data": entries})

        def be(T, S, v, p, sym=SYM):
            return {"T": T, "s": sym, "S": S, "v": v, "p": p}
        if s == 3:
            put("bybit", wall, json.dumps({"success": True, "ret_msg": "", "conn_id": "x", "op": "subscribe"}))
        if s == 130:     # 같은 ms 두 건 · 다른 종목 · 늦게 온 앞 시각(정렬 순서로 더해진다)
            put("bybit", wall, by([be(base + 5, "Buy", "1.5", f"{mid:.2f}"), be(base + 5, "Sell", "2.25", f"{mid:.2f}"),
                                   be(base + 6, "Buy", "4", "150.5", sym="SOLUSDT")]))
            put("bybit", wall + 1, by([be(base - 2000, "Buy", "0.333", f"{mid:.2f}")]))
        elif s == 320:   # 연결 0 끝: 한 건에 v 가 없다 -> 메시지 통째로 빠진다(파이썬 KeyError -> 재연결)
            put("bybit", wall, by([be(base + 1, "Sell", "1.0", f"{mid:.2f}"), {"T": base + 2, "s": SYM, "S": "Buy", "p": "1"}]))
            conn["bybit"] += 1
        elif s >= N_SEC - 3:   # 벽시계 − 3초 보다 새 것(파이썬 창 밖)
            put("bybit", wall + 500, by([be(base + 1, "Sell", "2", f"{mid:.2f}")]))
        elif s == 400:
            put("bybit", wall, json.dumps({"success": True, "ret_msg": "pong", "op": "ping"}))
        elif rng.random() < 0.09:
            put("bybit", wall + rng.randint(0, 500), by([be(base + rng.randint(0, 999), rng.choice(["Buy", "Sell"]),
                                                            f"{rng.uniform(0.01, 80):.3f}", f"{mid + rng.uniform(-1, 1):.2f}")
                                                         for _ in range(rng.randint(1, 2))]))
    rows.sort(key=lambda r: r[2])     # 안정 정렬 -- 같은 원천 안 순서는 그대로
    return rows


class FakeWS:
    """aiohttp ws 흉내. 연결 끝은 예외(끊김 -- 파이썬 errors +1 · 3초 뒤 재접속), 마지막 연결은 끝에서 멈춘다."""

    def __init__(self, idx, rows, clock, cps_at, cps, snap, last, done):
        self.idx, self.rows, self.clock, self.k = idx, rows, clock, 0
        self.cps_at, self.cps, self.snap, self.last, self.done = cps_at, cps, snap, last, done

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.receive()

    async def send_json(self, _obj):
        pass

    async def receive(self, timeout=None):
        if self.k and self.idx[self.k - 1] + 1 in self.cps_at:
            self.cps.append({"after": self.idx[self.k - 1] + 1, **self.snap()})
        if self.k == len(self.idx):
            if self.last:
                self.done.set()
                await asyncio.Event().wait()
            raise ConnectionResetError("시험: 연결 끝")
        i = self.idx[self.k]
        self.k += 1
        self.clock["now"] = self.rows[i][2] / 1000
        return SimpleNamespace(type=WSMsgType.TEXT, data=self.rows[i][3])


async def run_collector(collect, url_tail: str, rows, src, clock, cps_at, cps, snap) -> asyncio.Task:
    """원천 하나의 원본 수집 함수를 그 원천 메시지로 끝까지 돌린다."""
    conns: list[list[int]] = []
    for i, r in enumerate(rows):
        if r[0] == src:
            while r[1] >= len(conns):
                conns.append([])
            conns[r[1]].append(i)
    done = asyncio.Event()

    class FakeSession:
        n = 0

        def __init__(self, *a, **k):
            pass

        def ws_connect(self, url, **k):
            assert url.endswith(url_tail), url
            c = FakeSession.n
            FakeSession.n += 1
            return FakeWS(conns[c], rows, clock, cps_at, cps, snap, c == len(conns) - 1, done)

        def get(self, *a, **k):
            raise RuntimeError("REST 금지(시험)")     # OKX ctVal 대조 -> «표 값으로 계속»

        async def close(self):
            pass

    cf.ClientSession = FakeSession
    task = asyncio.create_task(collect(None))
    await asyncio.wait_for(done.wait(), 60)
    assert FakeSession.n == len(conns), (src, FakeSession.n, len(conns))
    return task     # 마지막 연결에 붙은 채(connected=True) -- payload 를 뜬 뒤에 끊는다


# ── server.py collect_force_orders 1831~1841 -- 옮겨 적음(2026-11-20 판). 예외면 그 연결의 남은 메시지는 없다(재연결). ──
def force_collect(rows, clock) -> dict:
    LIQ_ASSET_BY_SYMBOL = {c["binance_symbol"]: a for a, c in COIN_CONFIG.items()}  # noqa: N806 -- server.py 450
    liq_events_by = {a: deque(maxlen=5000) for a in COIN_CONFIG}                      # server.py 1769
    dead = None
    for r in rows:
        if r[0] != "force" or r[1] == dead:
            continue
        clock["now"] = r[2] / 1000
        try:
            o = (json.loads(r[3]) or {}).get("o") or {}
            asset = LIQ_ASSET_BY_SYMBOL.get(o.get("s"))
            if not asset:
                continue
            qty, price = float(o.get("z") or o.get("q") or 0.0), float(o.get("ap") or o.get("p") or 0.0)
            ev = {"ts_ms": int(o.get("T") or cf.time.time() * 1000),
                  "side": "long" if o.get("S") == "SELL" else "short",
                  "qty": qty, "price": price, "usd": qty * price, "symbol": o.get("s")}
            liq_events_by[asset].append(ev)
        except Exception:  # noqa: BLE001 -- 파이썬은 끊고 다시 붙는다
            dead = r[1]
    return liq_events_by


# ── Bybit: 수집기 원본(parse · connect · write) → 임시 SQLite → server.py _bybit_liq_refresh 800~804 옮겨 적음 ──
def bybit_collect(rows, now_s: float) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        con = bybit_col.connect(Path(d) / "bybit_liq.sqlite")
        dead = None
        for r in rows:
            if r[0] != "bybit" or r[1] == dead:
                continue
            try:
                bybit_col.write(con, bybit_col.parse(json.loads(r[3]), r[2]))
            except Exception:  # noqa: BLE001 -- 수집기 run(): 예외면 끊고 다시 붙는다
                dead = r[1]
        since = int((now_s - cf.FOOTPRINT_KEEP_BARS * 300) * 1000)
        got = con.execute("SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? AND ts_ms >= ? "
                          "ORDER BY ts_ms", [SYM, since]).fetchall()
        con.close()
    return [{"ts_ms": int(t), "side": "long" if sd == "Buy" else "short", "qty": float(q), "price": float(px),
             "usd": float(q) * float(px)} for t, sd, q, px in got]


# ── server.py supply_1s_payload 4252~4337 의 since=0 몫 -- 옮겨 적음(2026-11-20 판). oi·symbol 은 코어가 안 옮긴다. ──
def payload(ns, liq_events, bybit_events, now_s: float) -> dict:
    SUPPLY_1S_SECONDS = cf.SUPPLY_1S_SECONDS  # noqa: N806
    by_sec, okx_sec, okx_oi_1s, okx_liq_events, okx_state = (ns.footprint_state["sec"], ns.okx_sec, ns.okx_oi_1s,
                                                              ns.okx_liq_events, ns.okx_state)
    spot_sec, spot_state = ns.spot_sec, ns.spot_state
    since = since_liq = since_spot = since_okx = since_okx_oi = since_okx_liq = 0
    _bnow = int(now_s) - 3
    _bfloor = max(0, _bnow - SUPPLY_1S_SECONDS)
    _bcells: dict[int, list[float]] = {}
    for e in bybit_events:
        x = e["ts_ms"] // 1000
        if _bfloor < x <= _bnow:
            c = _bcells.setdefault(x, [0.0, 0.0, 0.0, 0.0]); lo = e["side"] == "long"
            c[0 if lo else 1] += e["qty"]; c[2 if lo else 3] += e["usd"]
    bybit_liq_rows = [[x, round(v[0], 3), round(v[1], 3), round(v[2]), round(v[3])] for x, v in sorted(_bcells.items())]
    spot_newest = max(spot_sec) if spot_sec else 0
    _part = lambda d, x: [x] + [round(v, 3) for v in d[x]] if x in d else None  # noqa: E731
    partial = {"okx": _part(okx_sec, max(okx_sec)) if okx_sec else None,
               "spot": _part(spot_sec, spot_newest) if spot_sec else None,
               "bn": _part(by_sec, max(by_sec)) if by_sec else None}
    spot_floor = max(since_spot, spot_newest - SUPPLY_1S_SECONDS)
    spot_rows = [[x] + [round(v, 3) for v in spot_sec[x]] for x in sorted(spot_sec) if spot_floor < x < spot_newest]
    okx_newest = max(okx_sec) if okx_sec else 0
    okx_floor = max(since_okx, okx_newest - SUPPLY_1S_SECONDS)
    okx_rows = [[x] + [round(v, 3) for v in okx_sec[x]] for x in sorted(okx_sec) if okx_floor < x < okx_newest]
    okx_oi_floor = max(since_okx_oi, (max(okx_oi_1s) if okx_oi_1s else 0) - SUPPLY_1S_SECONDS)
    okx_oi_rows = [[x, okx_oi_1s[x]] for x in sorted(okx_oi_1s) if x > okx_oi_floor]
    okx_liq_floor = max(since_okx_liq, okx_newest - SUPPLY_1S_SECONDS)
    okx_liq_by_sec: dict[int, list[float]] = {}
    for e in list(okx_liq_events):
        x = int(e.get("ts_ms") or 0) // 1000
        if not (okx_liq_floor < x < okx_newest):
            continue
        c = okx_liq_by_sec.setdefault(x, [0.0, 0.0, 0.0, 0.0])
        lo = e.get("side") == "long"
        c[0 if lo else 1] += float(e.get("qty") or 0.0)
        c[2 if lo else 3] += float(e.get("usd") or 0.0)
    okx_liq_rows = [[x, round(v[0], 3), round(v[1], 3), round(v[2]), round(v[3])] for x, v in sorted(okx_liq_by_sec.items())]
    _now_ms = now_s * 1000
    _age = lambda ms: round(max(0.0, (_now_ms - ms) / 1000), 1) if ms else None  # noqa: E731
    okx_meta = {"connected": bool(okx_state["connected"]), "tradeAge": _age(okx_state["last_trade_ms"]),
                "oiAge": _age(okx_state["last_oi_ms"]), "inst": ns.spec.okx_inst, "errors": okx_state["errors"]}
    spot_meta = {"connected": bool(spot_state["connected"]), "tradeAge": _age(spot_state["last_trade_ms"]),
                 "errors": spot_state["errors"]}
    out = {"now": 0, "liq": [], "okx": okx_rows, "okxOi": okx_oi_rows, "okxLiq": okx_liq_rows, "okxNow": okx_newest,
           "bybitLiq": bybit_liq_rows, "okxMeta": okx_meta, "spot": spot_rows, "spotNow": spot_newest, "spotMeta": spot_meta,
           "partial": partial, "seconds": []}
    if not by_sec:
        return out
    newest = max(by_sec)
    liq_floor = max(since_liq, newest - SUPPLY_1S_SECONDS)
    liq_by_sec: dict[int, list[float]] = {}
    for e in list(liq_events):
        s = int(e.get("ts_ms") or 0) // 1000
        if not (liq_floor < s < newest):
            continue
        cell = liq_by_sec.setdefault(s, [0.0, 0.0, 0.0, 0.0])
        long_ = e.get("side") == "long"
        cell[0 if long_ else 1] += float(e.get("qty") or 0.0)
        cell[2 if long_ else 3] += float(e.get("usd") or 0.0)
    out["liq"] = [[s, round(v[0], 3), round(v[1], 3), round(v[2]), round(v[3])] for s, v in sorted(liq_by_sec.items())]
    out["now"] = newest
    out["seconds"] = g1.payload_bn(by_sec)["seconds"]
    return out


async def run(rows: list[list]) -> dict:
    clock = {"now": T0 / 1000}
    cf.time = SimpleNamespace(time=lambda: clock["now"])
    cf.TCPConnector = lambda **k: None

    async def no_rest(*_a, **_k):
        raise RuntimeError("REST 금지(시험)")
    ns = cf.make_coin_flow(dataclasses.replace(cf.FLOW_SPECS["eth"],
                                               snapshot_path=Path("/nonexistent/rtcore_fixture_footprint.json")), no_rest, {})
    assert ns.spec.okx_inst == INST
    free = dict(zip(ns.start.__code__.co_freevars, ns.start.__closure__))
    fn = {k: free[k].cell_contents for k in ("collect_footprint", "collect_spot_flow", "collect_okx_flow")}
    # 중간 대조: 아주 늦은 사건(600초 넘게 앞) 바로 뒤 -- 그 칸은 다음 새 초에 지워지므로 여기서 떠야 «만들고 지움»이 보인다
    late = {i + 1 for i, r in enumerate(rows) if r[3] in LATE}
    cps: list[dict] = []
    cells = lambda d: {str(k): list(v) for k, v in sorted(d.items())}  # noqa: E731
    tasks = [await run_collector(fn["collect_footprint"], "ethusdt@trade", rows, "bn", clock, set(), [], dict)]
    tasks.append(await run_collector(fn["collect_spot_flow"], "ethusdt@aggTrade", rows, "spot", clock, late, cps,
                        lambda: {"src": "spot", "sec": cells(ns.spot_sec)}))
    tasks.append(await run_collector(fn["collect_okx_flow"], "/ws/v5/public", rows, "okx", clock, late, cps,
                        lambda: {"src": "okx", "sec": cells(ns.okx_sec), "oi": {str(k): v for k, v in sorted(ns.okx_oi_1s.items())}}))
    liq = force_collect(rows, clock)["eth"]
    final_ms = fix_wall(rows[-1][2] + 1234)
    now_s = final_ms / 1000
    assert now_s * 1000 == final_ms
    clock["now"] = now_s
    out = payload(ns, liq, bybit_collect(rows, now_s), now_s)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return {"checkpoints": cps, "final_ms": final_ms, "final": out,
            "final_cells": {"bn": cells(ns.footprint_state["sec"]), "spot": cells(ns.spot_sec), "okx": cells(ns.okx_sec),
                            "okx_oi": {str(k): v for k, v in sorted(ns.okx_oi_1s.items())}},
            "liq_ring": [liq.maxlen, ns.okx_liq_events.maxlen]}


def main() -> None:
    rows = gen()
    got = asyncio.run(run(rows))
    fx = {
        "about": "합성 다원천 메시지와 파이썬 수신·payload 코드가 낸 1초 수급(rtcore/tools/gen_supply1s_venues_fixture.py, 시드 %d)" % SEED,
        "symbol": SYM.lower(),
        "okx_ct_vals": CT_VALS,
        "no_spot": sorted(s.symbol.lower() for s in cf.FLOW_SPECS.values() if s.spot_url is None),
        "bybit_symbols": [s.lower() for s in bybit_col.SYMBOLS],
        "rows": rows,
        **got,
    }
    OUT.write_text(json.dumps(fx, separators=(",", ":"), ensure_ascii=False))
    f = got["final"]
    per = {s: sum(1 for r in rows if r[0] == s) for s in ("bn", "force", "spot", "okx", "bybit")}
    print(f"{OUT.relative_to(ROOT)}: 메시지 {per} · 중간 대조 {len(got['checkpoints'])} · liq {len(f['liq'])} · spot {len(f['spot'])} · "
          f"okx {len(f['okx'])} · okxOi {len(f['okxOi'])} · okxLiq {len(f['okxLiq'])} · bybitLiq {len(f['bybitLiq'])} · "
          f"{OUT.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()

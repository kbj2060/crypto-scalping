#!/usr/bin/env python3
"""rtcollect 맥락 수집기 셋(OKX 맥락 · 바이낸스 맥락 · xyz100) 일치 시험의 입력·정답지 (2026-11-21).

gen_venue_fixture.py 와 같은 방식: 정해진 시드로 합성 사건 열을 만들고 **원본 수집 루프를 그대로** 돌려(파싱·보류·플러시·재연결·공백 전부 원본)
임시 SQLite 를 덤프한다. 바뀌는 것은 바깥뿐 -- 소켓 connect 를 막고, aiohttp 는 가짜 모듈(사건 열을 차례로 내놓고 보낸 글자를 적는다),
시계는 수집기 모듈의 `time.time`·`time.monotonic` 만 «사건 시각»으로, `asyncio.sleep` 은 0(웹소켓) 또는 가짜 시계를 민다(REST 폴링).
REST 는 기본 꺼짐인 Rust 와 맞추려고 웹소켓 정답지에서는 빈 코루틴(OKX assert_ct_val · xyz backfill), REST 정답지는 따로(가짜 응답 → 요청 열).
rtcollect/tests/ctx_parity.rs 가 같은 입력을 Rust 에 넣어 행 단위·스키마 글자·보낸 글자·요청 열까지 대조한다. 합성 자료만.

    python3 rtcollect/tools/gen_ctx_fixture.py [okx_ctx binance_ctx xyz]     # tests/fixtures/<종류>_{input,golden}.json
    python3 rtcollect/tools/gen_ctx_fixture.py --check-readers               # Rust 가 만든 SQLite 를 파이썬 읽는 쪽으로 교차 확인
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 막기 · Ev · _ns · _exc · dump · close_shared · _fake_modules · _quiet

ROOT, FIX = gv.ROOT, gv.FIX
SEED = 20261123
W0 = gv.W0


def _mod(name):
    return __import__(f"scripts.{name}", fromlist=["x"])


# ── 가짜 aiohttp(웹소켓) ──────────────────────────────────────────────────────
class _Feed:
    """사건 열 하나. 'o'·'f' = 접속 · 't'·'w'·'c' = 수신('w' 시한 · 'c' 글자 아닌 메시지/끝). 끝나면 CancelledError."""

    def __init__(self, events, clock):
        self.ev, self.i, self.clock, self.sent = events, 0, clock, []

    def nxt(self, ks):
        if self.i >= len(self.ev):
            raise asyncio.CancelledError()
        t, k, body = self.ev[self.i]
        if gv.LENIENT["on"] and k in "of" and "o" not in ks:
            return "c", ""          # 고친 수집기는 원본이 끊었던 자리에서 안 끊는다 -- 그 접속은 서버가 닫은 것으로(먹지 않는다)
        gv._want(k in ks, (self.i, k, ks))
        self.i += 1
        self.clock[0] = t
        return k, body


class _MT:
    TEXT, CLOSED = "TEXT", "CLOSED"


def fake_aiohttp(feed_of, rest=None):
    """feed_of(url) -> _Feed. rest = 가짜 REST(get) -- 바이낸스 OI 폴링."""
    class WS:
        def __init__(self, f):
            self.f = f

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def send_json(self, o):
            self.f.sent.append(json.dumps(o))

        async def receive(self, timeout=None):
            k, body = self.f.nxt("twc")
            if k == "w":
                raise asyncio.TimeoutError()
            if k == "c":
                return types.SimpleNamespace(type=_MT.CLOSED, data=None)
            return types.SimpleNamespace(type=_MT.TEXT, data=body)

        def __aiter__(self):
            return self

        async def __anext__(self):      # aiohttp: CLOSE·CLOSING·CLOSED 면 반복 끝, 그 밖은 내놓는다
            m = await self.receive()
            if m.type != _MT.TEXT:
                raise StopAsyncIteration
            return m

    class Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def ws_connect(self, url, *a, **k):
            f = feed_of(url)
            k_, body = f.nxt("of")
            if k_ == "f":
                raise gv._exc(body or "OSError: 가짜 접속 실패")
            f.sent.append(f"<connect {url}>")
            return WS(f)

        def get(self, url, params=None, **k):
            return rest(url, params)

    return types.SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **k: None, WSMsgType=_MT)


def _j(o):
    return o if isinstance(o, str) else json.dumps(o, separators=(",", ":"), ensure_ascii=False)


# ── ① OKX 맥락 ───────────────────────────────────────────────────────────────
class OkxCtx(gv.Ev):
    def __init__(self, rng, t0, inst, px, ct):
        super().__init__(rng, t0)
        self.inst, self.px, self.ct, self.oi = inst, px, ct, 5_941_713.19

    def oi_msg(self, dt, **over):
        self.oi += self.rng.gauss(0, 300)
        oi = round(self.oi, 2)
        d = {"instId": self.inst, "instType": "SWAP", "oi": f"{oi}", "oiCcy": repr(round(oi * self.ct, 6)),
             "oiUsd": f"{oi * self.ct * self.px:.1f}", "ts": str(self.ms(0.3))}
        d.update(over)
        self.msg(dt, {"arg": {"channel": "open-interest", "instId": self.inst}, "data": [{k: v for k, v in d.items() if v is not gv._DROP}]})

    def mark(self, dt, **over):
        self.px *= 1 + self.rng.gauss(0, 2e-5)
        d = {"instId": self.inst, "instType": "SWAP", "markPx": f"{self.px:.2f}", "ts": str(self.ms(0.05))}
        d.update(over)
        self.msg(dt, {"arg": {"channel": "mark-price", "instId": self.inst}, "data": [{k: v for k, v in d.items() if v is not gv._DROP}]})

    def funding(self, dt, **over):
        d = {"instId": self.inst, "instType": "SWAP", "fundingRate": f"{self.rng.gauss(5e-5, 2e-5):.16f}",
             "fundingTime": str((int(self.t) // 28800 + 1) * 28_800_000), "minFundingRate": "-0.0075", "maxFundingRate": "0.0075",
             "method": "current_period", "nextFundingTime": "0", "ts": str(self.ms(0.2))}
        d.update(over)
        self.msg(dt, {"arg": {"channel": "funding-rate", "instId": self.inst}, "data": [{k: v for k, v in d.items() if v is not gv._DROP}]})

    def det(self, **over):
        d = {"bkLoss": "0", "bkPx": f"{self.px * (1 + self.rng.gauss(0, 0.003)):.2f}", "ccy": "", "posSide": self.rng.choice(("long", "short")),
             "side": self.rng.choice(("buy", "sell")), "sz": f"{self.rng.lognormvariate(1, 1.2):.2f}", "ts": str(self.ms(0.4))}
        d.update(over)
        return {k: v for k, v in d.items() if v is not gv._DROP}

    def liq(self, dt, *rows):
        self.msg(dt, {"arg": {"channel": "liquidation-orders", "instType": "SWAP"}, "data": list(rows)})

    def lrow(self, inst, family, *dets):
        d = {"instId": inst, "instFamily": family, "instType": "SWAP", "details": list(dets)}
        return {k: v for k, v in d.items() if v is not gv._DROP}

    def second(self, liq=0.03):
        for _ in range(2):
            self.mark(self.rng.uniform(0.3, 0.6))
        if self.rng.random() < 0.18:
            self.oi_msg(self.rng.uniform(0.05, 0.2))
        if self.rng.random() < 0.02:
            self.funding(0.05)
        if self.rng.random() < liq:
            self.liq(0.05, self.lrow(self.inst, self.inst[:-5], self.det()))


def gen_okx_ctx(rng):
    D = gv._DROP
    runs = []
    e = OkxCtx(rng, W0, "ETH-USDT-SWAP", 2745.02, 0.1)
    e.add(2.0, "o")                                        # 시작 → 첫 접속 2초 = 공백 «reconnect» 한 줄
    e.msg(0.05, {"event": "subscribe", "arg": {"channel": "open-interest", "instId": e.inst}, "connId": "c1"})
    for _ in range(25):
        e.second(0.1)
    # 계속 가는 것: pong · 깨진 JSON · 이벤트 · 다른 채널 · arg 없음/빈 것/None · data 없음/None/빈 것
    for m in ("pong", '{"arg":{"channel":"mark-price"},"data":[{"markPx":"1"', {"event": "channel-conn-count", "connCount": "1"},
              {"arg": {"channel": "tickers", "instId": e.inst}, "data": [{"x": 1}]}, {"data": [{"instId": e.inst, "markPx": "1", "ts": "1"}]},
              {"arg": {}, "data": []}, {"arg": None}, {"arg": {"channel": "mark-price"}}, {"arg": {"channel": "mark-price"}, "data": None},
              {"arg": {"channel": "mark-price"}, "data": []}, {"arg": {"channel": "mark-price"}, "data": ""}, {"arg": {"channel": "mark-price"}, "data": {}},
              {"event": "", "arg": {"channel": "mark-price"}, "data": []}, {"arg": {"channel": 5}, "data": [{}]},
              {"event": "subscribe", "arg": {"channel": "mark-price"}, "data": [{"instId": e.inst, "markPx": "9", "ts": "1763683100000"}]},
              {"event": "", "arg": {"channel": "mark-price"}, "data": [{"instId": e.inst, "markPx": "8", "ts": "1763683100000"}]}):
        e.msg(0.2, m)
    # 행 모양 -- 그 행만 버림 / 그대로 묶음
    ts = str(e.ms(0.1))
    e.msg(0.3, {"arg": {"channel": "mark-price"}, "data": [
        {"instId": e.inst, "markPx": "2745.1", "ts": ts}, {"instId": e.inst, "markPx": "0", "ts": ts}, {"instId": e.inst, "markPx": "-1", "ts": ts},
        {"instId": e.inst, "markPx": "nan", "ts": ts}, {"instId": e.inst, "markPx": "abc", "ts": ts}, {"instId": e.inst, "ts": ts},
        {"instId": e.inst, "markPx": 2745.3, "ts": int(ts)}, {"instId": e.inst, "markPx": " 2745.4 ", "ts": f"{int(ts)}.9"},
        {"instId": e.inst, "markPx": "2745.5", "ts": "0"}, {"instId": e.inst, "markPx": "2745.5", "ts": "-5"}, {"instId": e.inst, "markPx": "2745.5"},
        {"instId": e.inst, "markPx": "2745.5", "ts": "abc"}, {"instId": e.inst, "markPx": "2745.5", "ts": float(ts) + 0.7},
        {"instId": "", "markPx": "2745.6", "ts": ts}, {"markPx": "2745.6", "ts": ts}, {"instId": None, "markPx": "2745.7", "ts": ts},
        {"instId": 7, "markPx": "2745.8", "ts": ts}, {"instId": True, "markPx": "1e1", "ts": ts}, {"instId": 1.5, "markPx": True, "ts": True},
        {"instId": e.inst, "markPx": "2745.9", "ts": "1.79e12"}]})
    e.msg(0.3, {"arg": {"channel": "open-interest"}, "data": [
        {"instId": e.inst, "oi": "5941713.19000001727", "oiCcy": "594171.319000001727", "oiUsd": "1631065629.5", "ts": ts},
        {"instId": e.inst, "oi": "100", "oiCcy": "100", "ts": ts},                     # 단위 갈라짐(로그만) · oiUsd 없음 = NULL
        {"instId": e.inst, "oi": "abc", "oiCcy": "1", "ts": ts}, {"instId": e.inst, "oiCcy": "1", "ts": ts}, {"instId": e.inst, "oi": "1", "ts": ts},
        {"instId": e.inst, "oi": "nan", "oiCcy": "inf", "oiUsd": "x", "ts": ts}, {"instId": e.inst, "oi": "0", "oiCcy": "0", "oiUsd": None, "ts": ts},
        {"instId": "XRP-USDT-SWAP", "oi": "12", "oiCcy": "1200", "oiUsd": 3, "ts": ts}, {"instId": "PENGU-USDT-SWAP", "oi": "1", "oiCcy": "5", "ts": ts},
        {"instId": e.inst, "oi": "1", "oiCcy": "0.1", "ts": "0"}, {"instId": e.inst, "oi": False, "oiCcy": True, "ts": ts}]})
    e.msg(0.3, {"arg": {"channel": "funding-rate"}, "data": [
        {"instId": e.inst, "fundingRate": "0.0000588298968880", "fundingTime": "1790092800000", "minFundingRate": "-0.0075",
         "maxFundingRate": "0.0075", "method": "current_period", "ts": ts},
        {"instId": e.inst, "fundingRate": "0.0001"},                                         # ts 없음 = 벽시계 · 나머지 0/NULL/""
        {"instId": e.inst, "fundingRate": "0.0001", "ts": "-3", "method": None, "fundingTime": "1.5"},
        {"instId": e.inst, "fundingRate": "0.0001", "ts": "0", "method": 5, "minFundingRate": "", "maxFundingRate": [1]},
        {"instId": e.inst, "fundingRate": "x", "ts": ts}, {"instId": e.inst, "ts": ts}, {"fundingRate": "0.1", "ts": ts},
        {"instId": e.inst, "fundingRate": 0, "fundingTime": 1790092800000.6, "method": True, "ts": ts}]})
    e.liq(0.3, e.lrow(e.inst, "ETH-USDT", e.det(), e.det(sz="0"), e.det(sz="-1"), e.det(ts="0"), e.det(sz=D), e.det(bkPx=D, bkLoss="abc"),
                      e.det(side=D, posSide=None), e.det(sz="nan"), e.det(sz=7)),
          e.lrow("PENGU-USDT-SWAP", "PENGU-USDT", e.det(bkPx="0.009151", sz="388")),       # ctVal 모름 = sz_base NULL
          e.lrow("XRP-USDT-SWAP", "XRP-USDT", e.det(sz="3")), e.lrow("BTC-USDT-SWAP", D, e.det()),
          {"instId": "SOL-USDT-SWAP", "details": None}, {"instId": "SOL-USDT-SWAP"}, {"instId": "SOL-USDT-SWAP", "details": []},
          {"instFamily": "X", "details": [e.det()]}, {"instId": None, "details": [e.det()]})
    # 플러시는 처리한 메시지 뒤에만 -- pong 만 12초 오면 그동안 안 쓴다
    for _ in range(12):
        e.msg(1.0, "pong")
    e.mark(0.1)
    for _ in range(10):
        e.second()
    # 재연결 길: 구독 거부 · arg 참인 비사전 · data 숫자·사전·글자 · 원소 비사전 · 세부 비사전 · 사전 아닌 JSON · ts 무한대·NaN
    bad = [{"event": "error", "msg": "bad", "code": "60012"}, {"arg": [1], "data": []}, {"arg": {"channel": "mark-price"}, "data": 7},
           {"arg": {"channel": "mark-price"}, "data": {"a": 1}}, {"arg": {"channel": "mark-price"}, "data": "ab"},
           {"arg": {"channel": "mark-price"}, "data": [{"instId": e.inst, "markPx": "1", "ts": ts}, 5]},
           {"arg": {"channel": "liquidation-orders"}, "data": [e.lrow(e.inst, "ETH-USDT", e.det(), "x")]},
           {"arg": {"channel": "liquidation-orders"}, "data": [{"instId": e.inst, "details": {"a": 1}}]},
           {"arg": {"channel": "open-interest"}, "data": [{"instId": e.inst, "oi": "1", "oiCcy": "1", "ts": "inf"}]},
           {"arg": {"channel": "mark-price"}, "data": [{"instId": e.inst, "markPx": "1", "ts": "nan"}]},
           [1, 2], '"abc"', "5", "null"]
    for b in bad:
        e.second()
        e.msg(0.1, b)
        e.add(rng.choice((0.5, 1.2, 2.0, 3.0)), "o")                 # 1.2초 = 1.5초 문턱 아래(안 적는다)
        for _ in range(2):
            e.second()
    # 40초 시한 · 글자 아닌 메시지 · 접속 실패 둘(공백은 «마지막 실패 ~ 붙은 시각»만 적힌다 -- 원본 그대로)
    e.add(40.0, "w")
    e.add(3.0, "o")
    for _ in range(5):
        e.second()
    e.add(0.3, "c")
    e.add(3.0, "f", "OSError: 가짜 접속 실패")
    e.add(3.0, "f", "TimeoutError: ")
    e.add(3.0, "o")
    for _ in range(30):
        e.second()
    e.add(0.3, "c")
    runs.append({"inst": e.inst, "t0": W0, "events": e.ev})
    # BTC(청산 구독 없음 -- 그래도 오면 적는다) · XRP(ctVal 100)
    for inst, px, ct in (("BTC-USDT-SWAP", 90000.0, 0.01), ("XRP-USDT-SWAP", 2.1234, 100.0)):
        x = OkxCtx(rng, W0 + 5.0, inst, px, ct)
        x.add(0.1, "o")
        for _ in range(20):
            x.second(0.05)
        x.add(0.3, "c")
        runs.append({"inst": inst, "t0": W0 + 5.0, "events": x.ev})
    # ETH 재시작(같은 파일) -- meta 갈아 끼움 · 시작 공백
    r = OkxCtx(rng, e.t + 30.0, "ETH-USDT-SWAP", e.px, 0.1)
    r.add(4.0, "o")
    for _ in range(15):
        r.second()
    r.add(0.3, "c")
    runs.append({"inst": r.inst, "t0": e.t + 30.0, "events": r.ev})
    return runs


def run_okx_ctx(runs, d: Path) -> list:
    m = _mod("live_okx_context_collector_20260923")
    clock = [0.0]
    m.time = gv._ns(time, time=lambda: clock[0], monotonic=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
    m.assert_ct_val = gv._noop
    sent = []
    feed = {"f": None}
    with gv._fake_modules(aiohttp=fake_aiohttp(lambda url: feed["f"])), gv._quiet():
        for run in runs:
            feed["f"] = _Feed(run["events"], clock)
            clock[0] = run["t0"]
            try:
                asyncio.run(m.collect(run["inst"], d / "okx_ctx.sqlite"))
            except asyncio.CancelledError:
                pass
            assert feed["f"].i == len(run["events"]), "사건을 다 먹지 못했다"
            sent.append(feed["f"].sent)
    return sent


OKX_CTX_TABLES = {"okx_oi": "rowid", "okx_mark": "rowid", "okx_funding": "rowid", "okx_liquidations": "rowid", "gaps": "rowid", "meta": "rowid"}



# ── ② 바이낸스 맥락 ──────────────────────────────────────────────────────────
M0 = 50_000.0                   # 가짜 단조 시계 시작(REST 폴링)


def gen_binance_ctx(rng):
    D = gv._DROP
    feeds = []
    for sym, px in (("ETHUSDT", 2500.0), ("BTCUSDT", 90000.0)):
        e = gv.Ev(rng, W0)

        def mk(dt, **over):
            ms = e.ms(0.01) // 1000 * 1000
            d = {"e": "markPriceUpdate", "E": ms, "s": sym, "p": f"{px * (1 + rng.gauss(0, 1e-4)):.8f}", "P": "0", "i": f"{px:.8f}",
                 "r": f"{rng.gauss(1e-4, 3e-5):.8f}", "T": (ms // 28_800_000 + 1) * 28_800_000}
            d.update(over)
            e.msg(dt, {k: v for k, v in d.items() if v is not D})

        e.add(0.3, "o")
        for _ in range(40):
            mk(1.0)
        if sym == "ETHUSDT":
            # 같은 시각 다시(PK 가 거른다 -- 먼저 온 값) · 값 모양
            E = e.ms() // 1000 * 1000
            for over in ({"E": E, "p": "1.0"}, {"E": E, "p": "2.0"}, {"E": str(E + 1)}, {"E": float(E + 2) + 0.9}, {"E": True},
                         {"T": D}, {"T": None}, {"T": 0}, {"T": "5"}, {"s": 7}, {"s": None}, {"s": "EthUsdt"}, {"p": " 2500.5 "},
                         {"r": True}, {"i": 1}, {"p": "1e3"}):
                mk(0.2, **over)
            for m in ({"e": "x"}, {}, [], 0, None, '""', {"e": "markPriceUpdate!"}, {"e": 5}):
                e.msg(0.2, m)
            # 재연결: 깨진 JSON · 참인 비사전 · 열쇠 없음 · 값 못 읽음
            for bad in ('{"e":"markPriceUpdate","E":1', [1], "abc", 5, {"e": "markPriceUpdate"}, {"e": "markPriceUpdate", "E": 1, "s": "a"},
                        {"e": "markPriceUpdate", "E": "x", "s": "a", "p": "1", "i": "1", "r": "1"},
                        {"e": "markPriceUpdate", "E": 1, "s": "a", "p": "x", "i": "1", "r": "1"},
                        {"e": "markPriceUpdate", "E": 1, "s": "a", "p": "1", "i": "1", "r": "1", "T": "x"},
                        {"e": "markPriceUpdate", "E": None, "s": "a", "p": "1", "i": "1", "r": "1"}):
                mk(0.5)
                e.msg(0.1, bad)
                e.add(3.0, "o")
            e.add(0.5, "c")
            e.add(3.0, "f", "OSError: 가짜")
            e.add(3.0, "o")
            for _ in range(10):
                mk(1.0)
        e.add(0.4, "c")
        feeds.append({"kind": "mark", "url": f"{sym.lower()}@markPrice@1s", "events": e.ev})
    # 강제청산
    e = gv.Ev(rng, W0)

    def fo(dt, **over):
        s = over.pop("s", rng.choice(("ETHUSDT", "BTCUSDT", "SOLUSDT", "DOGEUSDT", "ZECUSDT", "HYPEUSDT")))
        q = f"{rng.lognormvariate(0, 1):.3f}"
        o = {"s": s, "S": rng.choice(("SELL", "BUY")), "o": "LIMIT", "f": "IOC", "q": q, "p": f"{rng.uniform(1, 3000):.2f}",
             "ap": f"{rng.uniform(1, 3000):.2f}", "X": "FILLED", "l": q, "z": q, "T": e.ms(0.1)}
        o.update(over)
        e.msg(dt, {"e": "forceOrder", "E": e.ms(), "o": {k: v for k, v in o.items() if v is not D}})

    e.add(0.2, "o")
    for _ in range(40):
        fo(rng.uniform(0.2, 2.0))
    for over in ({"s": "ETHUSDT", "z": "0", "q": "5"}, {"s": "ETHUSDT", "z": D, "q": "5"}, {"s": "ETHUSDT", "z": D, "q": D},
                 {"s": "ETHUSDT", "z": "", "q": 2}, {"s": "ETHUSDT", "ap": D}, {"s": "ETHUSDT", "ap": "0", "p": D}, {"s": "ETHUSDT", "ap": 0, "p": 0},
                 {"s": "ETHUSDT", "T": D}, {"s": "ETHUSDT", "T": 0}, {"s": "ETHUSDT", "T": "7"}, {"s": "ETHUSDT", "T": 7.9},
                 {"s": "ETHUSDT", "S": D}, {"s": "ETHUSDT", "S": "sell"}, {"s": "ZECUSDT"}, {"s": "SOXLUSDT"}, {"s": "ethusdt"}, {"s": 5},
                 {"s": None}, {"s": "ETHUSDT", "z": True, "ap": " 2400.5 "}):
        fo(0.3, **over)
    for m in ({"e": "forceOrder"}, {"o": None}, {"o": {}}, {"o": 0}, {"o": ""}, {}, None, [], 0):
        e.msg(0.2, m)
    for bad in ({"o": [1]}, {"o": "x"}, {"o": {"s": ["ETHUSDT"]}}, {"o": {"s": "ETHUSDT", "z": "abc"}}, {"o": {"s": "ETHUSDT", "ap": [1]}},
                {"o": {"s": "ETHUSDT", "T": "x"}}, [1], '{"o":', "true"):
        fo(0.5)
        e.msg(0.1, bad)
        e.add(3.0, "o")
    for _ in range(10):
        fo(1.0)
    e.add(0.4, "c")
    feeds.append({"kind": "force", "url": "!forceOrder@arr", "events": e.ev})

    # OI REST 폴링 시나리오
    T = int(W0 * 1000)
    J = "application/json"

    def ok(ts, oi, **k):
        return {"status": 200, "ctype": J, "raw": json.dumps({"symbol": "ETHUSDT", "openInterest": oi, "time": ts}), **k}

    def raw(status, body, ctype=J, **k):
        return {"status": status, "ctype": ctype, "raw": body if isinstance(body, str) else json.dumps(body), **k}

    oi = [
        {"name": "정상·중복·순서 뒤바뀜", "symbol": "ETHUSDT", "period": 0.5, "responses": [
            ok(T + 1000, "123456.789"), ok(T + 1000, "999"), ok(T + 5000, "123460.1"), ok(T + 3000, "123458.0"), ok(T + 5000, "1"),
            ok(T + 9000, 123470.5), ok(str(T + 13000), "123471.25"), ok(float(T + 17000) + 0.5, " 123472 ")]},
        {"name": "오류·밴", "symbol": "BTCUSDT", "period": 0.5, "responses": [
            raw(500, "<html>500</html>", "text/html"), raw(200, "", J), raw(200, "null"), raw(200, []), raw(200, '"no-t1me"'),
            raw(200, '"has time"'), raw(200, {"time": "abc", "openInterest": "1"}), raw(200, {"time": T + 1000}),
            ok(T + 1000, "77.0"),                                                                   # seen 에 이미 있다 = 버려진다(원본 그대로)
            ok(T + 1500, "77.5", ctype="text/plain"),                                               # JSON 이지만 본문 형식이 아니다 = ContentTypeError
            raw(200, {"time": T + 2000, "openInterest": "x"}), raw(200, "{broken", J), ok(T + 3000, "78.5", ctype="application/vnd.x+json"),
            raw(418, {"code": -1003, "msg": f"Way too many requests; IP(1.2.3.4) banned until {T + 40_000}. Please use the websocket."}),
            ok(T + 50_000, "80.0"),
            raw(429, {"code": -1003, "msg": "Too many requests"}, headers={"Retry-After": "7"}),
            ok(T + 90_000, "81.0"), {"timeout": True}, ok(T + 95_000, "82.0"),
            raw(400, {"code": -1121, "msg": "Invalid symbol."}), ok(T + 99_000, "83.0"),
            raw(418, {"code": -1003, "msg": "banned until 12"}), ok(T + 120_000, "84.0")]},
        {"name": "느린 종목 5초", "symbol": "ZECUSDT", "period": 5.0, "responses": [ok(T + 1000 * k, f"{100 + k}.5") for k in range(6)]},
    ]
    return {"ws": feeds, "oi": oi, "slow": ["ZECUSDT", "SNDKUSDT", "SOXLUSDT", "SKHYNIXUSDT", "MUUSDT"]}


class _Resp:
    """aiohttp 응답 흉내: json() 은 본문 형식(^application/(…+)?json)을 보고, 빈 본문이면 None."""

    def __init__(self, r):
        self.r, self.status, self.headers = r, r["status"], {"Content-Type": r["ctype"], **r.get("headers", {})}

    async def __aenter__(self):
        if self.r.get("timeout"):
            raise asyncio.TimeoutError()
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        import re
        if not re.match(r"^application/(?:[\w.+-]+?\+)?json", self.r["ctype"].split(";")[0].strip().lower()):
            raise RuntimeError("ContentTypeError")
        return json.loads(self.r["raw"]) if self.r["raw"].strip() else None


def run_binance_ctx(inp, d: Path) -> dict:
    from urllib.parse import urlencode, urlparse
    m = _mod("live_binance_ctx_collector_20261001")
    bg = m.ban_guard
    m.LIQ_SYMBOLS = {c["binance_symbol"] for c in m.COIN_CONFIG.values()} | set(inp["slow"])
    clock = [0.0, 0.0]          # 단조(쓰지 않음) · 벽
    m.init_db(d / "binance_ctx.sqlite")
    pend = m.Pending(d / "binance_ctx.sqlite")
    out = {"sent": [], "requests": []}
    feed = {"f": None}
    with gv._fake_modules(aiohttp=fake_aiohttp(lambda url: feed["f"])), gv._quiet():
        m.time = gv._ns(time, time=lambda: clock[1])
        m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
        for f in inp["ws"]:
            clk = [0.0]
            feed["f"] = _Feed(f["events"], clk)
            m.time = gv._ns(time, time=lambda: clk[0])
            table, parse = ("mark_price_1s", m.parse_mark) if f["kind"] == "mark" else ("liquidations", m.parse_force)
            try:
                asyncio.run(m.ws_loop(sys.modules["aiohttp"].ClientSession(), m.WS_BASE + f["url"], table, parse, pend))
            except asyncio.CancelledError:
                pass
            assert feed["f"].i == len(f["events"]), "사건을 다 먹지 못했다"
            out["sent"].append(feed["f"].sent)
        # OI 폴링: 가짜 시계(잠 = 두 시계를 민다) · 밴 파일은 임시
        for sc in inp["oi"]:
            clock[0], clock[1] = M0, W0
            bg.BAN_FILE = d / f"ban_{sc['symbol']}"
            bg.time = gv._ns(time, time=lambda: clock[1])
            m.time = gv._ns(time, time=lambda: clock[1])

            async def sleep(s, *a, **k):
                clock[0] += s
                clock[1] += s
                await asyncio.sleep(0)

            m.asyncio = gv._ns(asyncio, sleep=sleep)
            left, reqs = list(sc["responses"]), []

            def rest(url, params):
                if not left:
                    raise asyncio.CancelledError()
                reqs.append([clock[0], clock[1], f"{urlparse(url).path}?{urlencode(params)}"])
                return _Resp(left.pop(0))

            sess = fake_aiohttp(None, rest).ClientSession()
            try:
                asyncio.run(m.poll_oi(sess, sc["symbol"], pend, sc["period"]))
            except asyncio.CancelledError:
                pass
            out["requests"].append(reqs)
            out.setdefault("ban", []).append(bg.BAN_FILE.read_text() if bg.BAN_FILE.exists() else None)
        asyncio.run(pend.flush())
    return out


BINANCE_CTX_TABLES = {"oi_1s": "ts_ms, symbol", "mark_price_1s": "ts_ms, symbol", "liquidations": "rowid"}


# ── ③ xyz100(Hyperliquid xyz:*) ─────────────────────────────────────────────
XYZ_COINS = ["xyz:XYZ100", "xyz:SP500", "xyz:JPY", "xyz:EUR"]


def gen_xyz(rng):
    D = gv._DROP
    e = gv.Ev(rng, W0)
    px = {"xyz:XYZ100": 30590.0, "xyz:SP500": 6800.0, "xyz:JPY": 151.234, "xyz:EUR": 1.08123}

    def candle(dt, coin, /, **over):
        m0 = int(e.t) // 60 * 60_000
        p = px[coin] * (1 + rng.gauss(0, 1e-4))
        d = {"t": m0, "T": m0 + 59_999, "s": coin, "i": "1m", "o": f"{p:.1f}", "c": f"{p * 1.0001:.1f}", "h": f"{p * 1.0002:.1f}",
             "l": f"{p * 0.9998:.1f}", "v": f"{rng.lognormvariate(-3, 1):.4f}", "n": rng.randint(1, 30)}
        d.update(over)
        e.msg(dt, {"channel": "candle", "data": {k: v for k, v in d.items() if v is not D}})

    def ctx(dt, coin_, /, **over):
        coin = coin_
        p = px[coin]
        c = {"funding": "0.0000125", "openInterest": "1234.5", "prevDayPx": f"{p:.1f}", "dayNtlVlm": "170000000.0", "premium": "0.0003",
             "oraclePx": f"{p * 0.9997:.2f}", "markPx": f"{p * 1.0001:.2f}", "midPx": f"{p * 1.00015:.2f}", "impactPxs": ["1", "2"], "dayBaseVlm": "5555.1"}
        c.update(over.pop("ctx", {}))
        d = {"coin": coin, "ctx": {k: v for k, v in c.items() if v is not D}}
        d.update(over)
        e.msg(dt, {"channel": "activeAssetCtx", "data": {k: v for k, v in d.items() if v is not D}})

    def second():
        for c in XYZ_COINS:
            if rng.random() < 0.8:
                ctx(rng.uniform(0.02, 0.2), c)
            if rng.random() < 0.3:
                candle(rng.uniform(0.02, 0.1), c)

    e.add(0.0, "o")
    for c in XYZ_COINS:
        for t in ("candle", "activeAssetCtx"):
            e.msg(0.02, {"channel": "subscriptionResponse", "data": {"method": "subscribe", "subscription": {"type": t, "coin": c}}})
    for _ in range(20):
        second()
    # 값 모양
    X = "xyz:XYZ100"
    for over in ({"s": D}, {"t": str(int(e.t) // 60 * 60_000)}, {"t": float(int(e.t) // 60 * 60_000) + 0.5}, {"t": "x"}, {"t": D}, {"v": D}, {"v": ""},
                 {"v": 0}, {"n": D}, {"n": "4"}, {"n": "4.0"}, {"n": 4.9}, {"o": "abc"}, {"h": None}, {"s": 5}, {"s": ""}, {"s": True}, {"c": " 30600.5 "}):
        candle(0.1, X, **over)
    for over in ({"coin": D}, {"coin": ""}, {"ctx": {"midPx": D, "markPx": D, "oraclePx": D}}, {"ctx": {"midPx": "", "markPx": None, "oraclePx": D}},
                 {"ctx": {"midPx": "x"}}, {"ctx": {"markPx": 0}}, {"ctx": {"midPx": False, "markPx": [1]}}, {"ctx": {"oraclePx": " 1.5 "}}):
        ctx(0.1, X, **over)
    # 양성 대조 보강: 다른 봉이 덮지 않는 분(2시간 전 · EUR)에 n 을 못 읽는 candle -- 원본은 그 봉을 버린다
    e.msg(0.1, {"channel": "candle", "data": {"t": (int(e.t) // 60 - 120) * 60_000, "s": "xyz:EUR", "o": "1.08", "h": "1.09", "l": "1.07", "c": "1.08", "v": "5", "n": "4.0"}})
    e.msg(0.1, {"channel": "activeAssetCtx", "data": {"coin": X}})                  # ctx 없음 = 셋 다 None
    e.msg(0.1, {"channel": "activeAssetCtx", "data": {"coin": X, "ctx": {}}})
    e.msg(0.1, {"channel": "activeAssetCtx", "data": {"coin": X, "ctx": None}})
    e.msg(0.3, {"channel": "activeAssetCtx", "data": {"coin": 1, "ctx": {"midPx": "1"}}})       # 1 · True · 1.0 = 같은 열쇠
    e.msg(0.0, {"channel": "activeAssetCtx", "data": {"coin": True, "ctx": {"midPx": "2"}}})
    e.msg(0.0, {"channel": "activeAssetCtx", "data": {"coin": 1.0, "ctx": {"midPx": "3"}}})
    for m in ({"channel": "pong"}, {"channel": "error", "data": "Invalid subscription {\"type\":\"candle\"}"}, {"channel": "candle", "data": [1]},
              {"channel": "candle", "data": None}, {"channel": "activeAssetCtx", "data": "x"}, {"channel": "trades", "data": {}}, {}, {"data": {}}):
        e.msg(0.3, m)
    second()
    # 시한: 둘 → 자료 → 셋(끊김)
    e.add(30.0, "w")
    e.add(30.0, "w")
    second()
    for _ in range(3):
        e.add(30.0, "w")
    e.add(1.0, "o")
    for _ in range(5):
        second()
    # 재연결: 깨진 JSON · 사전 아닌 JSON · ctx 참인 비사전 · 종목 목록(hash 불가) -- 마지막 플러시 뒤 행은 버려진다
    for bad in ('{"channel":', "[1]", "null", '"x"', {"channel": "activeAssetCtx", "data": {"coin": X, "ctx": [1]}},
                {"channel": "activeAssetCtx", "data": {"coin": X, "ctx": "abc"}}, {"channel": "candle", "data": {"s": ["a"], "t": 1, "o": 1, "h": 1, "l": 1, "c": 1}},
                {"channel": "activeAssetCtx", "data": {"coin": {"a": 1}, "ctx": {"midPx": "1"}}}):
        for _ in range(2):
            second()
        e.msg(0.1, bad)
        e.add(1.0, rng.choice(("o", "o", "f")))
        if e.ev[-1][1] == "f":
            e.add(2.0, "o")
    e.add(0.5, "c")
    e.add(1.0, "f", "OSError: 가짜")
    e.add(2.0, "f", "TimeoutError: ")
    e.add(4.0, "f", "ClientConnectorError: x")
    e.add(8.0, "o")
    for _ in range(12):
        second()
    e.add(0.2, "c")
    e.add(1.0, "o")
    for _ in range(4):
        second()
    e.add(0.2, "c")
    return {"coins": XYZ_COINS, "events": e.ev, "rest": gen_xyz_rest(rng)}


def gen_xyz_rest(rng):
    now = W0 + 7200.25
    T = int(now * 1000)

    def bars(coin, m0, n, /, **over):
        out = []
        for k in range(n):
            t = (m0 + k * 60) * 1000
            p = 30590.0 + k
            d = {"t": t, "T": t + 59_999, "s": coin, "i": "1m", "o": f"{p}", "c": f"{p + 1}", "h": f"{p + 2}", "l": f"{p - 1}", "v": f"{0.01 * k:.2f}", "n": k}
            d.update(over)
            out.append(d)
        return out

    m = int(now) // 60 * 60
    J = lambda body, status=200: {"status": status, "raw": json.dumps(body)}  # noqa: E731
    return [
        {"name": "처음(72시간 전부터)·마지막 봉부터", "wall": now, "setup": [["xyz:SP500", m - 600, 6800.0]],
         "responses": [J(bars("xyz:XYZ100", m - 300, 6)), J(bars("xyz:SP500", m - 600, 4) + bars("xyz:SP500", m - 600, 1, o="1.5")),
                       J([]), J({})]},
        {"name": "실패: 500·시간초과·JSON·원소·종목 목록", "wall": now + 30, "setup": [],
         "responses": [{"status": 500, "raw": "x"}, {"timeout": True}, {"status": 200, "raw": "{broken"}, J([1, 2])]},
        {"name": "실패 계속: 사전·글자·숫자·빈 글자·목록 종목", "wall": now + 60, "setup": [],
         "responses": [J({"a": 1}), J("abc"), J(5), J("")]},
        {"name": "s 없음 = 요청 종목 · 못 읽는 원소는 버림", "wall": now + 90, "setup": [],
         "responses": [J([{k: v for k, v in b.items() if k != "s"} for b in bars("xyz:XYZ100", m, 3)] + [{"t": "x"}, {"t": 1}]),
                       J(bars("xyz:SP500", m, 2, s=["l"])), J(bars("xyz:JPY", m, 2, s=7)), J(bars("xyz:EUR", m, 1, v="", n=None))]},
    ]


def run_xyz(inp, d: Path) -> dict:
    import urllib.error
    import urllib.request
    m = _mod("live_xyz100_collector_20261014")
    clock = [0.0]
    m.time = gv._ns(time, time=lambda: clock[0], monotonic=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
    m.DB, m.COINS = d / "hl_xyz.sqlite", list(inp["coins"])
    out = {}
    real_backfill, m.backfill = m.backfill, gv._noop
    feed = _Feed(inp["events"], clock)
    clock[0] = inp["events"][0][0]
    with gv._fake_modules(aiohttp=fake_aiohttp(lambda url: feed)), gv._quiet():
        try:
            asyncio.run(m.run())
        except asyncio.CancelledError:
            pass
    assert feed.i == len(inp["events"]), "사건을 다 먹지 못했다"
    out["sent"] = feed.sent
    m.backfill = real_backfill
    # REST 메움: 시나리오마다 새 파일(rest_<k>.sqlite) -- 가짜 urlopen(보낸 본문을 적는다)
    real = urllib.request.urlopen
    out["rest"] = []
    try:
        for k, sc in enumerate(inp["rest"]):
            m.DB = d / f"rest_{k}.sqlite"
            m.sqlite_init(m.DB)
            with m.rw_connect(m.DB) as con:
                for ddl in m.DDL:
                    con.execute(ddl)
            m.write([(c, t, p, p, p, p, 1.0, 1) for c, t, p in sc["setup"]], [])
            clock[0] = sc["wall"]
            left, bodies = list(sc["responses"]), []

            def urlopen(req, timeout=None):
                gv._want(timeout == 20 and req.full_url == m.INFO_URL and req.headers.get("Content-type") == "application/json", req.headers)
                bodies.append(req.data.decode())
                r = left.pop(0)
                if r.get("timeout"):
                    raise TimeoutError("timed out")
                if r["status"] >= 400:
                    raise urllib.error.HTTPError(req.full_url, r["status"], "X", {}, None)
                import io

                class R(io.BytesIO):
                    pass
                return R(r["raw"].encode())

            urllib.request.urlopen = urlopen
            with gv._quiet():
                asyncio.run(m.backfill())
            assert not left, sc["name"]
            out["rest"].append(bodies)
    finally:
        urllib.request.urlopen = real
    return out


XYZ_TABLES = {"hl_xyz_1m": "rowid", "hl_xyz_ctx_1s": "rowid"}


# ── ④ 하이퍼리퀴드 자산 맥락(HL_BT_CTX_ONLY) ─────────────────────────────────
def gen_hl_ctx(rng):
    D = gv._DROP
    runs = []
    for coin, px, n in (("ETH", 2747.4, 1), ("BTC", 90000.0, 0)):
        e = gv.Ev(rng, W0 + (0 if coin == "ETH" else 3))

        def ctx(dt, coin_=coin, **over):
            c = {"funding": f"{rng.gauss(1.25e-5, 3e-6):.10f}", "openInterest": f"{1144067.9 + rng.gauss(0, 50):.4f}", "prevDayPx": "2758.6",
                 "dayNtlVlm": "1344109438.2353596687", "premium": f"{rng.gauss(4e-4, 1e-4):.10f}", "oraclePx": f"{px:.1f}",
                 "markPx": f"{px * 1.0001:.1f}", "midPx": f"{px * 1.00012:.2f}", "impactPxs": ["1", "2"], "dayBaseVlm": "487476.9786999995"}
            c.update(over.pop("ctx", {}))
            d = {"coin": coin_, "ctx": {k: v for k, v in c.items() if v is not D}}
            d.update(over)
            e.msg(dt, {"channel": "activeAssetCtx", "data": {k: v for k, v in d.items() if v is not D}})

        e.add(0.2, "o")
        e.msg(0.05, {"channel": "subscriptionResponse", "data": {"method": "subscribe", "subscription": {"type": "activeAssetCtx", "coin": coin}}})
        for _ in range(40):
            ctx(rng.uniform(0.8, 1.2))
        if n:
            # 값 모양: OI 없음/못 읽음 = 버림 · 다른 칸 없음 = NULL · coin 없음 = "" · None · 숫자
            for over in ({"ctx": {"openInterest": D}}, {"ctx": {"openInterest": "x"}}, {"ctx": {"openInterest": None}}, {"ctx": {"funding": D, "premium": "x"}},
                         {"ctx": {"markPx": None, "midPx": [1]}}, {"coin": D}, {"coin": None}, {"coin": 5}, {"ctx": {"openInterest": 7}}, {"ctx": {"openInterest": True}},
                         {"ctx": {"openInterest": " 12.5 "}}):
                ctx(0.3, **over)
            for m in ({"channel": "activeAssetCtx", "data": {"coin": "ETH"}}, {"channel": "activeAssetCtx", "data": {"coin": "ETH", "ctx": [1]}},
                      {"channel": "activeAssetCtx", "data": None}, {"channel": "activeAssetCtx"}, {"channel": "pong"}, {"channel": "error", "data": "x"},
                      '{"channel":', {}, {"channel": "activeAssetCtx", "data": {}}):
                e.msg(0.3, m)
            # 30초 시한 = ping(끊지 않음) · 시한 뒤에는 플러시 판정 없음
            e.add(30.0, "w")
            e.add(30.0, "w")
            for _ in range(5):
                ctx(1.0)
            # 재연결: 참인 비사전 data · 사전 아닌 JSON · 끊김 · 접속 실패 -- 남은 행은 끊길 때 쓴다
            for bad in ({"channel": "activeAssetCtx", "data": [1]}, {"channel": "activeAssetCtx", "data": "abc"}, "[1]", "null"):
                for _ in range(3):
                    ctx(1.0)
                e.msg(0.1, bad)
                e.add(3.0, "o")
            for _ in range(20):
                ctx(1.0)
            e.add(0.3, "c")
            e.add(3.0, "f", "OSError: 가짜")
            e.add(3.0, "o")
            for _ in range(35):
                ctx(1.0)
        e.add(0.3, "c")
        runs.append({"coin": coin, "t0": e.ev[0][0] - 0.2, "events": e.ev})
    return runs


def run_hl_ctx(runs, d: Path) -> list:
    import logging
    logging.disable(logging.CRITICAL)
    m = _mod("live_hyperliquid_book_ticker_collector_20260923")
    clock = [0.0]
    m.time = gv._ns(time, time=lambda: clock[0], monotonic=lambda: clock[0])
    m.asyncio = gv._ns(asyncio, sleep=gv._fast_sleep)
    m.CTX_ONLY = True
    sent, feed = [], {"f": None}
    with gv._fake_modules(aiohttp=fake_aiohttp(lambda url: feed["f"])), gv._quiet():
        for run in runs:
            feed["f"] = _Feed(run["events"], clock)
            clock[0] = run["t0"]
            try:
                asyncio.run(m.run(run["coin"], d / "bt", d / "hl_ctx.sqlite"))
            except asyncio.CancelledError:
                pass
            assert feed["f"].i == len(run["events"]), "사건을 다 먹지 못했다"
            sent += feed["f"].sent
    return sent


# ── 의도한 차이(결함은 Rust 에서만 고친다 -- gv.FixedSrc · rtcollect/SHADOW.md §11) ──────────
def _gap_sym(url):
    tail = url.rsplit("/", 1)[-1]
    return None if tail.startswith("!") else tail.split("@")[0]


def fixes(kind):
    if kind == "okx_ctx":
        m = _mod("live_okx_context_collector_20260923")
        return gv.FixedSrc(m, [(m, "_ws_loop", [
            ("    wrote = 0\n", "    wrote = 0\n    up = False\n"),
            ('                log("스트림 연결됨")\n', '                log("스트림 연결됨")\n                up = True\n'),
            ("        down_from = int(time.time() * 1000)\n        await store.aflush()",
             "        if up:\n            down_from = int(time.time() * 1000)\n        up = False\n        await store.aflush()")])])
    if kind == "binance_ctx":
        m = _mod("live_binance_ctx_collector_20261001")
        inserts = {**m.INSERTS, "gaps": "INSERT INTO gaps VALUES (?,?,?,?)"}
        ddl = m.DDL + ("CREATE TABLE IF NOT EXISTS gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)",)
        return gv.FixedSrc(m, [
            (m, "ws_loop", [
                ("    from aiohttp import WSMsgType\n", "    from aiohttp import WSMsgType\n    down = None\n    gsym = _rt_gap_sym(url)\n"),
                ('                log(f"{table} ws connected")\n',
                 '                log(f"{table} ws connected")\n                if down is not None and int(time.time() * 1000) - down >= 1500:\n'
                 '                    pend.add("gaps", (gsym, down, int(time.time() * 1000), "reconnect"))\n                down = None\n'),
                ("                    row = parse(json.loads(msg.data) or {})\n",
                 "                    try:\n                        obj = json.loads(msg.data)\n                    except ValueError:\n                        continue\n"
                 "                    row = parse(obj or {})\n"),
                ("        await asyncio.sleep(3.0)", "        if down is None:\n            down = int(time.time() * 1000)\n        await asyncio.sleep(3.0)")]),
            (m, "poll_oi", [
                ("                seen.add(ts_ms)\n", '                row = (ts_ms, symbol.lower(), float(data["openInterest"]))\n                seen.add(ts_ms)\n'),
                ('                pend.add("oi_1s", (ts_ms, symbol.lower(), float(data["openInterest"])))', '                pend.add("oi_1s", row)')])],
            {"_rt_gap_sym": _gap_sym, "INSERTS": inserts, "DDL": ddl})
    if kind == "xyz":
        m = _mod("live_xyz100_collector_20261014")
        return gv.FixedSrc(m, [(m, "run", [
            ("            try:\n                await backfill()", "            bars, ctxs = {}, {}\n            try:\n                await backfill()"),
            ("            except Exception as e:  # noqa: BLE001 -- 끊김·DNS·핸드셰이크 실패 모두 다시 붙는다\n",
             "            except Exception as e:  # noqa: BLE001 -- 끊김·DNS·핸드셰이크 실패 모두 다시 붙는다\n"
             "                if bars or ctxs:\n                    try:\n                        await asyncio.to_thread(write, list(bars.values()), list(ctxs.values()))\n"
             "                    except Exception:  # noqa: BLE001\n                        pass\n")])])
    return None

# ── 공통 ─────────────────────────────────────────────────────────────────────
KINDS = {
    "okx_ctx": (gen_okx_ctx, run_okx_ctx, {"okx_ctx.sqlite": OKX_CTX_TABLES}),
    "binance_ctx": (gen_binance_ctx, run_binance_ctx, {"binance_ctx.sqlite": BINANCE_CTX_TABLES}),
    "xyz": (gen_xyz, run_xyz, {"hl_xyz.sqlite": XYZ_TABLES, **{f"rest_{k}.sqlite": XYZ_TABLES for k in range(4)}}),
    "hl_ctx": (gen_hl_ctx, run_hl_ctx, {"hl_ctx.sqlite": {"hl_asset_ctx": "rowid", "meta": "rowid"}}),
}


def python_reference(kind, inp, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    extra = KINDS[kind][1](inp, d)
    gv.close_shared()
    return extra


def dump_dir(kind, d: Path) -> dict:
    import sqlite3
    out = {}
    for name, t in KINDS[kind][2].items():
        con = sqlite3.connect(f"file:{d / name}?mode=ro", uri=True)
        have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        con.close()
        out[name] = gv.dump(d / name, {k: v for k, v in {**t, "gaps": "rowid"}.items() if k in have})   # 🔧바이낸스 맥락 gaps = 더한 표
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--check-readers", action="store_true")
    a = ap.parse_args()
    if a.check_readers:
        return check_readers(a.kinds)
    for kind in a.kinds:
        inp = KINDS[kind][0](random.Random(SEED + list(KINDS).index(kind)))
        with tempfile.TemporaryDirectory() as td:
            extra = python_reference(kind, inp, Path(td) / "orig")
            golden = {"db": dump_dir(kind, Path(td) / "orig"), "extra": extra}
            fx = fixes(kind)
            if fx:
                gv.LENIENT["on"] = True
                with fx:
                    extra = python_reference(kind, inp, Path(td) / "fixed")
                gv.LENIENT["on"] = False
                fixed = {"db": dump_dir(kind, Path(td) / "fixed"), "extra": extra}
                orig_db = golden["db"]
                golden = fixed
                golden["intended"] = gv.intended_diff(orig_db, fixed["db"])
        (FIX / f"{kind}_input.json").write_text(json.dumps({"seed": SEED, "input": inp}, separators=(",", ":"), ensure_ascii=False))
        (FIX / f"{kind}_golden.json").write_text(json.dumps(golden, separators=(",", ":"), ensure_ascii=False))
        sizes = [(FIX / f"{kind}_{x}.json").stat().st_size for x in ("input", "golden")]
        print(f"{kind}: " + " · ".join(f"{db} {t} {len(v)}행" for db, g in golden["db"].items() for t, v in g["tables"].items() if v)
              + f" · 파일 {sizes[0] // 1024}+{sizes[1] // 1024}KB")


def cargo_replay(kind, inp: Path, out: Path) -> None:
    import subprocess
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 120 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release",
                        "--example", "replay_ctx", "--", kind, str(inp), str(out)], cwd=ROOT / "rtcollect", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo replay_ctx 실패:\n{r.stderr}")


def ws_only(kind, inp):
    """읽는 쪽 교차 확인용 입력 -- REST 몫을 뺀다(Rust 예제는 웹소켓 몫만 먹인다)."""
    if kind == "binance_ctx":
        return {**inp, "oi": []}
    if kind == "xyz":
        return {**inp, "rest": []}
    return inp


WIN = [int((W0 - 3600) * 1000), int((W0 + 20 * 3600) * 1000)]   # 합성 자료의 시각 창(이상한 ts 행은 빼고 범위를 잰다)


def readers(kind, d: Path, inp) -> dict:
    """원본 파이썬이 그 표를 읽는 함수들(네트워크 없이 부를 수 있는 것)로 읽는다. 범위는 자료 범위(하루 넘으면 멈춘다 -- gv._span)."""
    from scripts import data_store
    for name in ("ops_watchdog", "research_eth_liqmap_venue_leverage_validation_20261008"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import scripts.ops.data_quality_monitor_20261022 as dqm
    dqm.HOT = d
    out = {}
    if kind == "okx_ctx":
        import dashboard.coin_flow as cf
        import scripts.live_liqmap_scorer_20261008 as lms
        db = d / "okx_ctx.sqlite"
        lo, hi = gv._span(db, "SELECT min(ts_ms) / 1000, max(ts_ms) / 1000 FROM okx_mark WHERE ts_ms BETWEEN ? AND ?", WIN)
        for t in OKX_CTX_TABLES:
            out[f"read_rows:{t}"] = data_store.read_rows(db, f"SELECT * FROM {t} ORDER BY rowid")
        # coin_flow.okx_tape_footprint: 봉별 OI 첫·끝 · 청산(빈 테이프 파일과 함께 -- 원본 TapeStore 로 만든 표)
        tape = d / "empty_tape.sqlite"
        _mod("live_trade_tape_collector_20260916").TapeStore(tape, "ETH-USDT-SWAP", 0.1)
        gv.close_shared()
        for inst in ("ETH-USDT-SWAP", "BTC-USDT-SWAP", "XRP-USDT-SWAP", "PENGU-USDT-SWAP"):
            fp = cf.okx_tape_footprint(tape, db, inst, lo // 300 * 300, (hi + 1) * 1000, 300, 0.5, 0.1)
            out[f"okx_tape_footprint:{inst}"] = [sorted(fp["oi"].items()), fp["liq"], fp["ctx_ok"]]
            out[f"dqm.ctx_stats:mark:okx:{inst}"] = dqm.ctx_stats(*dqm.CTX["mark:okx"][:4], inst, lo, hi + 1)
            out[f"dqm.minute_marks:{inst}"] = dqm.minute_marks(*dqm.CTX["mark:okx"][:4], inst, lo, hi + 1, "mark_px")
            out[f"dqm.LIQS:okx:{inst}"] = dqm.q(dqm.LIQS["okx"][0], dqm.LIQS["okx"][2], (inst, lo * 1000, (hi + 1) * 1000))
        lms.HOT = d
        q = lms._q("okx_ctx.sqlite", "SELECT ts_ms, pos_side AS side, sz_base AS qty FROM okx_liquidations WHERE inst_id='ETH-USDT-SWAP' AND ts_ms>=? AND ts_ms<?",
                   (lo * 1000, (hi + 1) * 1000))
        out["liqmap_scorer._q"] = [list(map(repr, r)) for r in q.itertuples(index=False)]
        # 대시보드 시장 맥락(server.py:2529-2531)·청산 탭(live_liq_tab:172)과 같은 SQL(서버 모듈은 import 하지 않는다)
        for name, sql in (("okx_fund", "SELECT funding_rate, funding_time FROM okx_funding WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1"),
                          ("okx_mark", "SELECT mark_px, ts_ms FROM okx_mark WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1"),
                          ("okx_oi", "SELECT oi_base FROM okx_oi WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1")):
            out[f"server:{name}"] = data_store.read_rows(db, sql, ["ETH-USDT-SWAP"])
    elif kind == "binance_ctx":
        import scripts.live_liqmap_scorer_20261008 as lms
        db = d / "binance_ctx.sqlite"
        lo, hi = gv._span(db, "SELECT min(ts_ms) / 1000, max(ts_ms) / 1000 FROM mark_price_1s WHERE ts_ms BETWEEN ? AND ?", WIN)
        for t, o in BINANCE_CTX_TABLES.items():
            out[f"read_rows:{t}"] = data_store.read_rows(db, f"SELECT * FROM {t} ORDER BY {o}")
        for c in ("ETH", "BTC"):
            for name in ("mark:binance", "oi:binance"):
                out[f"dqm.ctx_stats:{name}:{c}"] = dqm.ctx_stats(*dqm.CTX[name][:4], dqm.CTX[name][4](c), lo, hi + 1)
            out[f"dqm.minute_marks:{c}"] = dqm.minute_marks(*dqm.CTX["mark:binance"][:4], f"{c.lower()}usdt", lo, hi + 1, "mark")
        for c in ("ETH", "BTC", "SOL", "ZEC", "HYPE"):
            out[f"dqm.LIQS:binance:{c}"] = dqm.q(dqm.LIQS["binance"][0], dqm.LIQS["binance"][2], (f"{c}USDT", lo * 1000, (hi + 1) * 1000))
        lms.HOT = d
        for name, sql in (("liq", "SELECT ts_ms, side, qty FROM liquidations WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?"),
                          ("mark", "SELECT ts_ms, mark FROM mark_price_1s WHERE symbol='ethusdt' AND ts_ms>=? AND ts_ms<?")):
            q = lms._q("binance_ctx.sqlite", sql, (lo * 1000 - 5000, (hi + 1) * 1000))
            out[f"liqmap_scorer._q:{name}"] = [list(map(repr, r)) for r in q.itertuples(index=False)]
    elif kind == "hl_ctx":
        db = d / "hl_ctx.sqlite"
        lo, hi = gv._span(db, "SELECT min(recv_ms) / 1000, max(recv_ms) / 1000 FROM hl_asset_ctx WHERE recv_ms BETWEEN ? AND ?", WIN)
        out["read_rows"] = data_store.read_rows(db, "SELECT * FROM hl_asset_ctx ORDER BY rowid")
        for c in ("ETH", "BTC"):
            out[f"dqm.ctx_stats:ctx:hl:{c}"] = dqm.ctx_stats(*dqm.CTX["ctx:hl"][:4], c, lo, hi + 1)
            # 대시보드 시장 맥락(server.py:2533)과 같은 SQL
            out[f"server:hl_ctx:{c}"] = data_store.read_rows(db, "SELECT funding, open_interest, premium, mark_px, mid_px, recv_ms FROM hl_asset_ctx WHERE coin = ? ORDER BY recv_ms DESC LIMIT 1", [c])
    else:
        db = d / "hl_xyz.sqlite"
        lo, hi = gv._span(db, "SELECT min(ts_sec), max(ts_sec) FROM hl_xyz_1m WHERE ts_sec BETWEEN ? AND ?", [w // 1000 for w in WIN])
        for t in XYZ_TABLES:
            out[f"read_rows:{t}"] = data_store.read_rows(db, f"SELECT * FROM {t} ORDER BY rowid")
        # 대시보드 /api/xyz100(server.py:628-631)과 같은 SQL
        for coin in ("xyz:XYZ100", "xyz:SP500", "xyz:JPY", "xyz:EUR"):
            out[f"server:1m:{coin}"] = data_store.read_rows(db, """SELECT ts_sec, close FROM hl_xyz_1m WHERE coin = ? AND close IS NOT NULL
                                      AND ts_sec >= max(?, (SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = ?) - ?) ORDER BY ts_sec""", [coin, lo, coin, 86400])
            out[f"server:ctx:{coin}"] = data_store.read_rows(db, "SELECT recv_sec, mid, mark, oracle FROM hl_xyz_ctx_1s WHERE coin = ? ORDER BY recv_sec DESC LIMIT 1", [coin])
    out["integrity"] = [data_store.read_rows(p, "PRAGMA integrity_check") for p in sorted(d.glob("*.sqlite"))]
    return out


def check_readers(kinds) -> None:
    for kind in kinds:
        inp = ws_only(kind, json.loads((FIX / f"{kind}_input.json").read_text())["input"])
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ip = td / "input.json"
            ip.write_text(json.dumps({"input": inp}))
            fx = fixes(kind)                         # 🔧정답지와 같게: 결함 고친 파이썬(의도한 차이)
            gv.LENIENT["on"] = bool(fx)
            if fx:
                with fx:
                    python_reference(kind, inp, td / "py")
            else:
                python_reference(kind, inp, td / "py")
            gv.LENIENT["on"] = False
            cargo_replay(kind, ip, td / "rs")
            a, b = readers(kind, td / "py", inp), readers(kind, td / "rs", inp)
            assert a.keys() == b.keys()
            bad = [k for k in a if a[k] != b[k]]
            for k in a:
                n = len(a[k]) if isinstance(a[k], (list, dict)) else 1
                print(f"  {kind}  {'같음' if k not in bad else '다름'}  {k}: {n}")
            assert not bad, bad
            assert all(x == [("ok",)] for x in a["integrity"]), a["integrity"]
            # 섞어 쓰기: 파이썬이 앞 판을 쓰고 Rust 가 같은 파일에 이어 쓴다(OKX 맥락 -- 종목 실행이 여럿)
            if kind == "okx_ctx":
                k = len(inp) - 1
                fx = fixes(kind)
                gv.LENIENT["on"] = bool(fx)
                if fx:
                    with fx:
                        KINDS[kind][1](inp[:k], td / "mix")
                else:
                    KINDS[kind][1](inp[:k], td / "mix")
                gv.LENIENT["on"] = False
                gv.close_shared()
                mp = td / "mix_input.json"
                mp.write_text(json.dumps({"input": inp[k:]}))
                cargo_replay(kind, mp, td / "mix")
                same = dump_dir(kind, td / "mix") == dump_dir(kind, td / "py")
                print(f"  {kind}  {'같음' if same else '다름'}  섞어 쓰기(파이썬 {k}판 + Rust {len(inp) - k}판) vs 전부 파이썬")
                assert same
    print("check-readers OK")


if __name__ == "__main__":
    main()

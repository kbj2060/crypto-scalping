#!/usr/bin/env python3
"""rtcollect 수집기 셋(Bybit 청산 · Bybit 체결 테이프 · OKX 체결 테이프) 일치 시험의 입력·정답지 (2026-11-20).

gen_tape_fixture.py(바이낸스)와 같은 방식: 정해진 시드로 합성 사건 열을 만들고, **원본 수집 루프를 그대로** 돌려
(`run()` · `collect()` -- 파싱·묶기·플러시·재연결·공백 기록 전부 원본) 임시 SQLite 를 만든 뒤 표 전체를 덤프한다.
rtcollect/tests/venue_parity.rs 가 같은 사건 열을 Rust 에 넣어 행 단위·스키마 글자까지 대조한다.

네트워크 없음: 소켓 connect 를 막고, 웹소켓 라이브러리(websockets · aiohttp)는 가짜 모듈로 바꾼다(사건 열을 차례로 내놓는다).
REST(1분봉 대조 verify_recent · ctVal 대조 assert_ct_val · 복구 backfill_loop)는 빈 코루틴으로. 시계는 수집기 모듈의
`time.time`·`time.monotonic` 만 «사건 시각»으로(두 시계가 같은 값), `asyncio.sleep`(재연결 대기)은 모듈 안에서만 0 으로.
Bybit 테이프의 멈춤 감시 스레드(실제 30초 sleep 후 os._exit)는 끈다.

사건 = [초, 종류, 본문]: 'o' 접속 성공 · 'f' 접속 실패 · 't' 글자 메시지 · 'w' 수신 시한(Bybit 5초 · OKX 25초) ·
'c' 연결 끝(Bybit 청산 = 예외, 본문 «이름: 글» · Bybit 테이프·OKX = 글자 아닌 메시지).
연결 안에서 예외가 나는 메시지 뒤에는 생성기가 반드시 'o'·'f' 를 둔다(가짜 접속이 그렇지 않으면 멈춘다).

    python3 rtcollect/tools/gen_venue_fixture.py                    # 셋 다 다시 만들기(tests/fixtures/<종류>_{input,golden}.json)
    python3 rtcollect/tools/gen_venue_fixture.py --check-readers    # Rust 가 만든 SQLite 를 파이썬 읽는 쪽으로 교차 확인
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtcollect" / "tests" / "fixtures"
SEED = 20261121
W0 = 1_763_683_200.0 - 40.0      # 2025-11-21 00:00:00 UTC 40초 앞(벽시계 초). 합성 -- 실제 장과 무관. 날짜 경계를 건넌다


def _no_net(*_a, **_k):
    raise RuntimeError("gen_venue_fixture: 네트워크 금지")


socket.socket.connect = _no_net
socket.create_connection = _no_net
sys.path.insert(0, str(ROOT))


# ── 사건 열 ──────────────────────────────────────────────────────────────────
class Ev:
    def __init__(self, rng: random.Random, t0: float):
        self.rng, self.t, self.ev = rng, t0, []

    def add(self, dt: float, kind: str, body: str = "") -> float:
        self.t = round(self.t + dt, 6)
        self.ev.append([self.t, kind, body])
        return self.t

    def msg(self, dt: float, obj) -> float:
        return self.add(dt, "t", obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"), ensure_ascii=False))

    def ms(self, lag=0.0) -> int:
        """지금 받은 메시지의 거래소 시각 ms(수신보다 lag 초 앞)."""
        return int((self.t - lag) * 1000)


# ── ① Bybit 청산 ─────────────────────────────────────────────────────────────
def gen_liq(rng: random.Random) -> list:
    e = Ev(rng, W0)
    syms = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT", "DOGEUSDT"]   # DOGE = 구독 밖(원본은 거르지 않는다)
    px = {"ETHUSDT": 2500.0, "BTCUSDT": 90000.0, "SOLUSDT": 140.0, "XRPUSDT": 2.1, "HYPEUSDT": 35.0, "DOGEUSDT": 0.16}

    def one(**over):
        s = rng.choice(syms)
        d = {"T": e.ms(rng.uniform(0.05, 0.6)), "s": s, "S": rng.choice(("Buy", "Sell")),
             "v": f"{rng.lognormvariate(0, 1.5) * 1000 / px[s]:.3f}", "p": f"{px[s] * (1 + rng.gauss(0, 0.002)):.4f}"}
        d.update(over)
        return d

    def liq(dt, *rows):
        e.msg(dt, {"topic": "allLiquidation.ETHUSDT", "type": "snapshot", "ts": e.ms() + 1, "data": list(rows)})

    def burst(n):
        for _ in range(n):
            k = rng.random()
            if k < 0.15:
                e.add(5.0, "w")
            else:
                liq(rng.uniform(0.05, 1.6), *[one() for _ in range(rng.choice((1, 1, 1, 2, 3)))])

    e.add(0.0, "o")
    e.msg(0.05, {"success": True, "ret_msg": "", "conn_id": "c1", "op": "subscribe"})
    burst(40)
    e.msg(0.3, {"success": True, "ret_msg": "pong", "conn_id": "c1", "op": "ping"})
    # 값 모양: v 숫자 · p 공백 · T 문자열 · T 실수 · 쪽 null · 종목 숫자 · 쪽 참거짓 → 원본은 그대로 묶는다(INTEGER·NULL)
    liq(0.4, one(v=1.25), one(p=" 2500.5 "), one(T=str(e.ms(0.1))), one(T=float(e.ms(0.2))))
    liq(0.4, one(S=None), one(s=123), one(S=True), one(v="1e-3"))
    for m in ({"topic": 5, "data": [one()]}, {"data": [one()]}, {"topic": "allLiquidation.ETHUSDT", "data": None},
              {"topic": "allLiquidation.ETHUSDT", "data": []}, {"topic": "allLiquidation.ETHUSDT"},
              {"topic": "publicTrade.ETHUSDT", "data": [{}]}, {"topic": "allLiquidation.ETHUSDT", "data": 0},
              {"topic": "allLiquidationX.ETHUSDT", "data": [one()]},
              {"topic": "allLiquidation.ETHUSDT", "data": {}}):
        e.msg(0.2, m)
    burst(10)
    # 플러시 2초: 2초 안에 몰린 메시지는 한 번에 · 시한(w) 에서도 플러시
    for _ in range(6):
        liq(0.3, one())
    e.add(5.0, "w")
    # 연결 안 예외 → 남은 행은 except 에서 쓰고, 처음 끊긴 시각·이유를 들고 접속 실패 둘 → 붙으면 공백 한 줄
    liq(0.5, one())
    e.msg(0.3, '{"topic":"allLiquidation.ETHUSDT","data":[{"T":1')          # 깨진 JSON
    e.add(1.0, "f", "OSError: 가짜 접속 실패")
    e.add(2.0, "f", "TimeoutError: ")
    e.add(4.0, "o")
    burst(8)
    bad = [{"topic": "allLiquidation.ETHUSDT", "data": {"a": 1}},          # 사전 → 열쇠 글자 → TypeError
           {"topic": "allLiquidation.ETHUSDT", "data": "ab"},              # 글자 → TypeError
           {"topic": "allLiquidation.ETHUSDT", "data": 7},                 # 숫자 → TypeError(반복 불가)
           {"topic": "allLiquidation.ETHUSDT", "data": [one(), 5]},        # 원소가 숫자 → TypeError(앞 원소도 같이 빠진다)
           {"topic": "allLiquidation.ETHUSDT", "data": [{k: v for k, v in one().items() if k != "T"}]},   # KeyError
           {"topic": "allLiquidation.ETHUSDT", "data": [one(v="abc")]},    # ValueError
           {"topic": "allLiquidation.ETHUSDT", "data": [one(T="1.5")]},    # int("1.5") ValueError
           {"topic": "allLiquidation.ETHUSDT", "data": [one(p=None)]},     # float(None) TypeError
           [1, 2], "abc"]                                                  # JSON 이지만 사전이 아님 → AttributeError
    for b in bad:
        liq(0.3, one())
        e.msg(0.2, b)
        e.add(1.0, "o")
        burst(3)
    # 긴 이유(160 글자 자르기, 한글) · 연결 끝 예외 · 버퍼 비었을 때
    burst(5)
    e.add(0.4, "c", "ConnectionClosedError: " + "끊김 " * 90)
    e.add(1.0, "f", "OSError: x")
    e.add(2.0, "o")
    burst(6)
    # 60초 넘게 아무것도 안 오면(시한 5초 × 13) ConnectionError
    liq(0.3, one())                      # 마지막 수신 시각을 정해 둔다 → 5초 × 12 = 60 은 아직, 13번째(65초)에 끊는다
    for j in range(13):
        e.add(4.9 if j == 0 else 5.0, "w")   # 12번째 = 59.9초(아직) · 13번째 = 64.9초 > 60 (부동소수 경계를 피한다)
    e.add(1.0, "o")
    burst(20)
    for _ in range(3):                    # 플러시(2초) 사이에 끝난다 → 남은 행은 except 의 쓰기로만 남는다
        liq(0.2, one())
    e.add(0.5, "c", "ConnectionClosedOK: received 1000 (OK)")
    e.add(1.0, "o")
    liq_tail(e, one)
    return e.ev


def liq_tail(e: "Ev", one) -> None:
    """🔧맨 끝: 바인딩 못 하는 종목(목록)이 든 메시지 → 끊김 → 다시 붙음 → 정상 셋 → 시한. 원본은 그 행이 버퍼에 남아 그 뒤 쓰기가 전부 실패
    (정상 행까지 못 씀 = 기록 멈춤) · 고친 쪽은 그 행만 버리고 다 쓴다. 시한 뒤 입력이 끝나 두 쪽 다 같은 자리에서 멈춘다."""
    e.msg(0.3, {"topic": "allLiquidation.ETHUSDT", "data": [one(), one(s=["ETHUSDT"]), one()]})
    e.add(0.2, "c", "ConnectionClosedError: 가짜")
    e.add(1.0, "o")
    for _ in range(3):
        e.msg(0.3, {"topic": "allLiquidation.ETHUSDT", "data": [one()]})
    e.add(5.0, "w")


def run_liq(events: list, db: Path) -> None:
    m = __import__("scripts.live_bybit_liquidation_collector_20261003", fromlist=["x"])
    clock = [0.0]
    m.time = _ns(time, time=lambda: clock[0])
    m.asyncio = _ns(asyncio, sleep=_fast_sleep)
    m.DB = db
    st = {"i": 0}

    def nxt(kinds):
        i = st["i"]
        if i >= len(events):
            raise asyncio.CancelledError()
        t, k, body = events[i]
        _want(k in kinds, (i, k, kinds))
        st["i"] = i + 1
        clock[0] = t
        return k, body

    class WS:
        async def send(self, _s):
            pass

        async def recv(self):
            k, body = nxt("twc")
            if k == "t":
                return body
            if k == "w":
                raise asyncio.TimeoutError()
            raise _exc(body)

    class Conn:
        async def __aenter__(self):
            k, body = nxt("of")
            if k == "f":
                raise _exc(body)
            return WS()

        async def __aexit__(self, *a):
            return False

    with _fake_modules(websockets=types.SimpleNamespace(connect=lambda *a, **k: Conn())), _quiet():
        try:
            asyncio.run(m.run())
        except asyncio.CancelledError:
            pass
    assert st["i"] == len(events), "사건을 다 먹지 못했다"


# ── ② Bybit 체결 테이프 + 맥락 ────────────────────────────────────────────────
BYBIT_SYMS = ["ETHUSDT", "BTCUSDT", "XRPUSDT"]


class BybitStream(Ev):
    PX = {"ETHUSDT": (2500.0, 0.01, 2, 2), "BTCUSDT": (90000.0, 0.1, 1, 3), "XRPUSDT": (2.1234, 0.0001, 4, 1),
          "SOLUSDT": (140.0, 0.01, 2, 1)}

    def __init__(self, rng, t0):
        super().__init__(rng, t0)
        self.mid = {s: v[0] for s, v in self.PX.items()}
        self.seq = 1000

    def row(self, sym_, ts_, side_, p_, v_, /, **over):
        self.seq += 1
        d = {"T": ts_, "s": sym_, "S": side_, "v": v_, "p": p_, "L": "PlusTick", "i": f"id{self.seq}", "BT": False, "seq": self.seq}
        d.update(over)
        return {k: x for k, x in d.items() if x is not _DROP}

    def trades(self, dt, rows, s=None):
        self.msg(dt, {"topic": f"publicTrade.{s or rows[0].get('s', 'ETHUSDT')}", "type": "snapshot", "ts": self.ms() + 2,
                      "data": rows})

    def order_rows(self, s, T, big=False):
        """한 테이커 주문 = 같은 (T, 방향, 가격) 체결 여럿 · 가끔 가격 여럿(호가 쓸기)."""
        mid, tick, dec, qdec = self.PX[s]
        self.mid[s] += self.rng.gauss(0, 2) * tick
        side = self.rng.choice(("Buy", "Sell"))
        p0 = self.mid[s] + self.rng.randint(-2, 2) * tick
        out = []
        for lv in range(self.rng.choice((1, 1, 1, 2, 3))):
            p = f"{p0 + (lv if side == 'Buy' else -lv) * tick:.{dec}f}"
            for _ in range(self.rng.choice((1, 1, 2, 3))):
                q = self.rng.lognormvariate(-1.0 if big else -3.5, 1.5) * (3000.0 / mid)
                out.append(self.row(s, T, side, p, f"{max(q, 10 ** -qdec):.{qdec}f}"))
        return out

    def second(self, syms=BYBIT_SYMS, k=3):
        for _ in range(self.rng.randint(0, k)):
            s = self.rng.choice(syms)
            rows = self.order_rows(s, self.ms(self.rng.uniform(0.01, 0.3)), big=self.rng.random() < 0.08)
            # 한 주문을 메시지 둘로 쪼개 보내기도 한다(메시지 경계를 넘어 이어 붙는다)
            if len(rows) > 1 and self.rng.random() < 0.3:
                cut = self.rng.randint(1, len(rows) - 1)
                self.trades(self.rng.uniform(0.01, 0.2), rows[:cut])
                self.trades(self.rng.uniform(0.0, 0.05), rows[cut:])
            else:
                self.trades(self.rng.uniform(0.01, 0.3), rows)
        if self.rng.random() < 0.5:
            self.ticker(self.rng.uniform(0.01, 0.2), self.rng.choice(syms))

    def ticker(self, dt, s, typ="delta", **data):
        mid = self.mid[s]
        if not data:
            data = {"markPrice": f"{mid * (1 + self.rng.gauss(0, 1e-4)):.6g}", "indexPrice": f"{mid * (1 + self.rng.gauss(0, 1e-4)):.6g}"}
            if self.rng.random() < 0.3:
                data["openInterest"] = f"{self.rng.choice((100000.5, 100001.0, 100002.25)):.2f}"
                data["openInterestValue"] = f"{mid * 100000:.2f}"
            if self.rng.random() < 0.1:
                data["fundingRate"] = self.rng.choice(("0.0001", "0.00012", "-0.00005"))
        ts = int(round(self.t + dt, 6) * 1000) - 200      # 이 메시지를 받는 시각보다 0.2초 앞(🔴처음엔 직전 메시지 시각이라 마크 1초 규칙이 엇나갔다)
        self.msg(dt, {"topic": f"tickers.{s}", "type": typ, "data": {"symbol": s, **data}, "cs": 1, "ts": ts})

    def snapshot(self, s):
        mid = self.mid[s]
        self.ticker(0.01, s, "snapshot", lastPrice=f"{mid}", markPrice=f"{mid}", indexPrice=f"{mid}",
                    openInterest="100000.5", openInterestValue=f"{mid * 100000.5:.2f}", fundingRate="0.0001",
                    nextFundingTime="1763712000000", volume24h="1")


_DROP = object()


def gen_bybit_tape(rng: random.Random) -> list:
    runs = []
    e = BybitStream(rng, W0)
    e.add(0.0, "o")
    e.msg(0.05, {"success": True, "ret_msg": "", "conn_id": "c1", "req_id": "", "op": "subscribe"})
    for s in BYBIT_SYMS:
        e.snapshot(s)
    for s in range(30):
        e.second()
    # 묶기: 같은 (T, 방향, 가격)이 메시지 사이로 이어지고, 다른 키가 오면 닫힌다 · 1초 넘게 조용한 묶음은 플러시가 닫는다
    T = e.ms(0.1)
    e.trades(0.2, [e.row("ETHUSDT", T, "Buy", "2500.00", "1.00"), e.row("ETHUSDT", T, "Buy", "2500.00", "2.00")])
    e.trades(0.1, [e.row("ETHUSDT", T, "Buy", "2500.00", "3.00"), e.row("ETHUSDT", T, "Buy", "2500.01", "1.00")])
    e.add(5.0, "w")                       # 시한 -- 플러시(1초 넘게 조용한 묶음 닫기)
    e.add(5.0, "w")
    # 크기 경계 양쪽(ETH $10k·$100k 정확히, 한 주문 = 한 체결)
    for q, side in (("3.99", "Buy"), ("4.00", "Buy"), ("39.99", "Sell"), ("40.00", "Sell")):
        e.trades(0.05, [e.row("ETHUSDT", e.ms(0.01), side, "2500.00", q)])
    # 섞인 메시지: 종목 여럿 · 구독 밖 종목(SOL) · s 없음 · s 숫자 · 못 믿을 행(그 행만 버림)
    T = e.ms(0.05)
    e.trades(0.3, [e.row("BTCUSDT", T, "Sell", "90000.5", "0.050"), e.row("SOLUSDT", T, "Buy", "140.00", "1.0"),
                   e.row("ETHUSDT", T, "Sell", "2500.10", "0.50", s=_DROP), e.row("ETHUSDT", T, "Sell", "2500.10", "0.50", s=7),
                   e.row("ETHUSDT", T, "Buy", "0", "1"), e.row("ETHUSDT", T, "Buy", "2500.0", "0"),
                   e.row("ETHUSDT", 0, "Buy", "2500.0", "1"), e.row("ETHUSDT", T, "X", "2500.0", "1"),
                   e.row("ETHUSDT", T, "Buy", "abc", "1"), e.row("ETHUSDT", T, "Buy", None, "1"),
                   e.row("ETHUSDT", T, "Buy", "2500.0", "1", T=_DROP), e.row("ETHUSDT", "1.5", "Buy", "2500.0", "1"),
                   "x", 5, None, [1], e.row("ETHUSDT", str(T), "Buy", " 2500.20 ", 0.5),
                   e.row("XRPUSDT", float(T), "Sell", 2.1234, "100"), e.row("ETHUSDT", T, "Sell", "2500.30", "1e-2", S="Sell")])
    for d in ({"a": 1}, "abc", None, [], 0):          # 사전·글자 = 원소 하나하나 TypeError → 그 원소만 / 거짓 = 빈 목록
        e.msg(0.1, {"topic": "publicTrade.ETHUSDT", "data": d})
    for m in ({"topic": 5, "data": [e.row("ETHUSDT", T, "Buy", "2500.0", "9")]}, {"op": "pong", "ret_msg": "pong"},
              {"topic": "orderbook.50.ETHUSDT", "data": {}}):
        e.msg(0.1, m)
    # 맥락: 바뀐 칸만 · OI 그대로면 행 없음 · 펀딩(율, 다음 시각) 쌍이 바뀔 때 · 마크 1초 1점 · 빈 값은 합치지 않는다
    e.ticker(0.2, "ETHUSDT", openInterest="100000.5", markPrice="2500.5")
    e.ticker(0.3, "ETHUSDT", markPrice="2500.6")
    e.ticker(1.1, "ETHUSDT", markPrice="", indexPrice="2500.7")          # markPrice 열쇠는 있다 → 마크 행(값은 옛 값)
    e.ticker(0.4, "ETHUSDT", nextFundingTime="1763740800000")
    e.ticker(0.4, "ETHUSDT", fundingRate="0.0001", nextFundingTime=None)
    e.ticker(0.4, "BTCUSDT", openInterest="nan", markPrice="90001")      # NaN 은 늘 «바뀜» → 행마다 NULL
    e.ticker(1.2, "BTCUSDT", openInterest="nan", markPrice="90002")
    e.ticker(0.3, "BTCUSDT", openInterest="200.0", openInterestValue=None)
    e.ticker(0.3, "XRPUSDT", fundingRate="1e-4", nextFundingTime="0")
    e.ticker(0.3, "XRPUSDT", fundingRate="Infinity")
    e.ticker(1.3, "XRPUSDT", markPrice=2.2, indexPrice=True)
    for m in ({"topic": "tickers.ETHUSDT", "data": {"markPrice": "1"}, "ts": e.ms()},        # symbol 없음
              {"topic": "tickers.ETHUSDT", "data": {"symbol": "ETHUSDT", "markPrice": "1"}},  # ts 없음
              {"topic": "tickers.ETHUSDT", "data": {"symbol": "ETHUSDT", "markPrice": "1"}, "ts": 0},
              {"topic": "tickers.ETHUSDT", "data": {"symbol": "", "markPrice": "1"}, "ts": e.ms()},
              {"topic": "tickers.ETHUSDT", "data": None, "ts": e.ms()},
              {"topic": "tickers.ETHUSDT", "data": {"symbol": "ETHUSDT", "markPrice": "2501"}, "ts": str(e.ms())},
              {"topic": "tickers.ETHUSDT", "data": {"symbol": "ETHUSDT", "markPrice": "2502"}, "ts": float(e.ms() + 5000)}):
        e.msg(0.3, m)
    for s in range(20):
        e.second()
    # 연결 안 예외 → 재연결(끊긴 시각·이름 → 맥락 gaps · 열린 묶음·진행 중 초까지 쓰고 그 초를 닫는다 → 재연결 뒤 같은 초 체결은 버린다)
    poison = [
        {"topic": "publicTrade.ETHUSDT", "data": 7},                                           # TypeError(반복 불가)
        {"topic": "publicTrade.ETHUSDT", "data": [e.row(["ETHUSDT"], T, "Buy", "2500.0", "1")]},  # s 가 목록 → unhashable
        '{"topic":"publicTrade.ETHUSDT","data":[{"T":1',                                       # JSONDecodeError
        [1, 2], "abc",                                                                          # AttributeError
        {"topic": "tickers.ETHUSDT", "data": [1], "ts": 5},                                    # AttributeError
        {"topic": "tickers.ETHUSDT", "data": {"symbol": "ETHUSDT"}, "ts": "abc"},             # ValueError
        {"topic": "tickers.ETHUSDT", "data": {"symbol": ["ETHUSDT"], "markPrice": "1"}, "ts": 5},  # unhashable
    ]
    for b in poison:
        rows = e.order_rows("ETHUSDT", e.ms(0.05))
        e.trades(0.3, rows)
        e.msg(0.1, b)
        e.add(0.5, "o")
        e.trades(0.05, [e.row("ETHUSDT", rows[0]["T"], "Buy", "2500.00", "9.99")])   # 닫힌 초 → 버림
        e.trades(0.4, [e.row("ETHUSDT", e.ms(0.01), "Buy", "2500.00", "0.50")])
        for _ in range(3):
            e.second()
    # 값이 남아 계속 터지는 맥락(합친 상태에 못 읽는 값이 남는다) -- 새 값이 올 때까지 매 메시지 재연결
    e.ticker(0.3, "ETHUSDT", openInterest="abc")
    e.add(0.5, "o")
    e.ticker(0.3, "ETHUSDT", markPrice="2503")
    e.add(0.5, "o")
    e.ticker(0.3, "ETHUSDT", openInterest="100003")
    e.ticker(0.3, "XRPUSDT", nextFundingTime="inf")                  # int(inf) OverflowError
    e.add(0.5, "o")
    e.ticker(0.3, "XRPUSDT", nextFundingTime="nan")                  # int(nan) ValueError
    e.add(0.5, "o")
    e.ticker(0.3, "XRPUSDT", nextFundingTime="1763740800000")
    e.ticker(0.3, "ETHUSDT", openInterest="100004", openInterestValue="x")   # OI 상태는 바뀐 뒤 예외 → 행 없음
    e.add(0.5, "o")
    e.ticker(0.3, "ETHUSDT", openInterest="100004", openInterestValue="250000")   # 같은 OI → 행 없음
    for _ in range(10):
        e.second()
    # 끊김(글자 아닌 메시지) → 접속 실패 둘 → 붙음(맥락 gaps 의 끝 = 붙은 뒤 첫 수신·시한)
    e.add(0.4, "c", "ConnectionError")
    e.add(3.0, "f", "OSError")
    e.add(3.0, "f", "TimeoutError")
    e.add(3.0, "o")
    # 플러시 시각 경계(붙은 시각 + 5초): 3.9초에 다음다음 초 체결(묶음) · 4.4초에 그 묶음을 닫는 체결 · 4.6초에 처음 초로 늦은 체결 → 5초 플러시면 살고 4초면 버려진다
    T = e.ms(0.05)
    e.trades(0.1, [e.row("ETHUSDT", T, "Buy", "2500.00", "0.30")])
    e.trades(3.8, [e.row("ETHUSDT", T + 2000, "Buy", "2500.10", "0.30")])   # 3.9초 -- 4초 플러시보다 앞
    e.trades(0.5, [e.row("ETHUSDT", T + 2001, "Sell", "2500.15", "0.30")])   # 앞 묶음을 닫아 버퍼의 최대 초를 올린다(묶음은 닫혀야 버퍼에 든다)
    e.trades(0.2, [e.row("ETHUSDT", T + 10, "Sell", "2500.20", "0.30")])
    e.trades(0.5, [e.row("ETHUSDT", T + 2100, "Buy", "2500.10", "0.30")])
    # 묶음 닫기(1초 넘게 조용하면 플러시가 닫는다): 붙고 3.5초에 묶음, 5초 플러시(1.5초 조용) → 닫힘 → 같은 열쇠가 다시 오면 새 주문
    e.add(0.3, "c", "ConnectionError")
    e.add(3.0, "o")
    T = e.ms(0.05) + 3000
    e.trades(3.5, [e.row("BTCUSDT", T, "Buy", "90000.0", "0.100")])
    e.add(1.5, "w")
    e.trades(0.1, [e.row("BTCUSDT", T, "Buy", "90000.0", "0.200")])
    e.trades(0.1, [e.row("BTCUSDT", T + 1, "Sell", "90000.0", "0.100")])
    e.add(5.0, "w")
    for _ in range(10):
        e.second()
    # 60초 침묵(시한 13번) → ConnectionError
    e.trades(0.3, [e.row("ETHUSDT", e.ms(0.01), "Buy", "2500.00", "0.10")])   # 마지막 수신 시각을 정해 둔다
    for j in range(13):
        e.add(4.9 if j == 0 else 5.0, "w")   # 12번째 = 59.9초(아직) · 13번째 = 64.9초 > 60 (부동소수 경계를 피한다)
    e.add(3.0, "o")
    for _ in range(25):
        e.second()
    e.add(0.4, "c", "ConnectionError")
    runs.append({"symbols": BYBIT_SYMS, "t0": W0 - 0.5, "events": e.ev})

    # 재시작(같은 파일) -- last_ts_ms 로 ws_reconnect 공백 · meta 다시 쓰기 · 종목 하나 빠짐(BTC)
    r = BybitStream(rng, e.t + 20.0)
    r.add(0.0, "o")
    for s in ("ETHUSDT", "XRPUSDT"):
        r.snapshot(s)
    r.trades(0.2, [r.row("ETHUSDT", 0, "Buy", "2500.0", "1"), r.row("ETHUSDT", r.ms(0.05), "Buy", "2500.00", "0.20")])   # 새 판 첫 체결 앞 시각 0
    for _ in range(20):
        r.second(["ETHUSDT", "XRPUSDT"])
    r.add(0.4, "c", "ConnectionError")
    runs.append({"symbols": ["ETHUSDT", "XRPUSDT"], "t0": e.t + 19.5, "events": r.ev})
    return runs


def run_bybit_tape(runs: list, d: Path) -> None:
    m = __import__("scripts.live_bybit_trade_tape_collector_20261006", fromlist=["x"])
    clock = [0.0]
    m.time = _ns(time, time=lambda: clock[0], monotonic=lambda: clock[0])
    m.asyncio = _ns(asyncio, sleep=_fast_sleep)
    m.start_stall_guard = lambda _beat: None
    m.verify_recent = _noop
    st = {"ev": [], "i": 0}
    fake = _fake_aiohttp(st, clock, "twc")
    with _fake_modules(aiohttp=fake), _quiet():
        for run in runs:
            m.TAPE_DB, m.CTX_DB, m.SYMBOLS = d / "bybit_tape.sqlite", d / "bybit_ctx.sqlite", list(run["symbols"])
            st["ev"], st["i"] = run["events"], 0
            clock[0] = run["t0"]
            try:
                asyncio.run(m.run())
            except asyncio.CancelledError:
                pass
            assert st["i"] == len(st["ev"]), "사건을 다 먹지 못했다"


# ── ③ OKX 체결 테이프 ────────────────────────────────────────────────────────
class OkxStream(Ev):
    def __init__(self, rng, t0, inst, mid, tick, dec, szdec):
        super().__init__(rng, t0)
        self.inst, self.mid, self.tick, self.dec, self.szdec, self.tid = inst, mid, tick, dec, szdec, 5_000_000

    def row(self, ts_, side_, px_, sz_, /, count=None, **over):
        self.tid += 1
        d = {"instId": self.inst, "tradeId": str(self.tid), "px": px_, "sz": sz_, "side": side_, "ts": ts_,
             "count": count if count is not None else str(self.rng.choice((1, 1, 1, 2, 3, 7))), "source": "0", "seqId": self.tid}
        d.update(over)
        return {k: x for k, x in d.items() if x is not _DROP}

    def trades(self, dt, rows, **arg):
        self.msg(dt, {"arg": {"channel": "trades", "instId": self.inst, **arg}, "data": rows})

    def second(self, k=4):
        self.mid += self.rng.gauss(0, 2) * self.tick
        for _ in range(self.rng.randint(0, k)):
            rows = []
            ts = self.ms(self.rng.uniform(0.01, 0.2))
            for _ in range(self.rng.choice((1, 1, 2, 3))):
                px = f"{self.mid + self.rng.randint(-3, 3) * self.tick:.{self.dec}f}"
                sz = self.rng.lognormvariate(1.0 if self.rng.random() < 0.08 else -1.0, 1.5)
                rows.append(self.row(str(ts), self.rng.choice(("buy", "sell")), px, f"{max(sz, 10 ** -self.szdec):.{self.szdec}f}"))
            self.trades(self.rng.uniform(0.02, 0.35), rows)


def gen_okx(rng: random.Random) -> list:
    runs = []
    e = OkxStream(rng, W0, "ETH-USDT-SWAP", 2500.0, 0.01, 2, 2)
    e.add(0.0, "o")
    e.msg(0.05, {"event": "subscribe", "arg": {"channel": "trades", "instId": e.inst}, "connId": "a1"})
    for _ in range(30):
        e.second()
    # 계속 가는 것(continue): pong · 깨진 JSON · 이벤트 · 다른 채널 · arg 없음 · data 사전·글자·빈 것
    for m in ("pong", '{"arg":{"channel":"trades"},"data":[{"px":"1"', '{"event":"subscribe"}',
              {"event": "channel-conn-count", "channel": "trades", "connCount": "1"},
              {"arg": {"channel": "tickers", "instId": e.inst}, "data": [e.row(str(e.ms()), "buy", "2500", "1")]},
              {"data": [e.row(str(e.ms()), "buy", "2500", "1")]}, {"arg": {}, "data": []}, {"arg": None},
              {"arg": {"channel": "trades"}, "data": {"a": 1}}, {"arg": {"channel": "trades"}, "data": "ab"},
              {"arg": {"channel": "trades"}, "data": None}, {"event": "", "arg": {"channel": "trades"}, "data": []},
              {"event": "subscribe", "arg": {"channel": "trades", "instId": e.inst}, "data": [e.row(str(e.ms()), "buy", "2500", "7")]}):
        e.msg(0.2, m)
    # 행 모양: 대문자 방향 · count 변형 · 숫자 값 · 못 믿을 행(그 행만)
    ts = str(e.ms(0.05))
    e.trades(0.3, [e.row(ts, "BUY", "2500.10", "3", count="5"), e.row(ts, "Sell", "2500.10", "3", count=""),
                   e.row(ts, "buy", "2500.10", "3", count="0"), e.row(ts, "buy", "2500.10", "3", count="abc"),
                   e.row(ts, "buy", "2500.10", "3", count="3.0"), e.row(ts, "sell", "2500.10", "3", count=-2),
                   e.row(ts, "sell", "2500.10", "3", count=4.7), e.row(ts, "sell", "2500.10", "3", count=_DROP),
                   e.row(ts, "sell", "2500.10", "3", count=None), e.row(ts, "sell", "2500.10", "3", count=[1]),
                   e.row(int(ts), "buy", 2500.2, 4), e.row(ts, "buy", " 2500.30 ", "1e1"),
                   e.row(ts, "buy", "0", "1"), e.row(ts, "buy", "2500", "0"), e.row("0", "buy", "2500", "1"),
                   e.row(ts, "NA", "2500", "1"), e.row(ts, None, "2500", "1"), e.row(ts, "buy", "abc", "1"),
                   e.row(ts, "buy", None, "1"), e.row(ts, "buy", "2500", "1", px=_DROP), e.row("1.5", "buy", "2500", "1"),
                   e.row(ts, "buy", "2500", "1", side=_DROP), "x", 5, None, [1]])
    # 크기 경계 양쪽(ETH ctVal 0.1 → 40 계약 = 4 ETH = $10k · 400 계약 = $100k)
    for sz, side in (("39.9", "buy"), ("40", "buy"), ("399.9", "sell"), ("400", "sell")):
        e.trades(0.05, [e.row(str(e.ms(0.01)), side, "2500.00", sz, count="1")])
    # 플러시는 체결 채널 메시지 뒤에만 -- 체결이 아닌 메시지만 6초 오면 그동안 안 쓴다
    for _ in range(12):
        e.msg(0.5, "pong")
    e.trades(0.1, [e.row(str(e.ms(0.01)), "buy", "2500.00", "1")])     # 6초 만의 체결 메시지 → 여기서 플러시(시각이 정해진다)
    # 플러시 시각 경계(+5초): 4.5초에 다음다음 초, 4.7초에 처음 초로 늦은 체결 → 5초 플러시면 살고 4초면 버려진다
    ts = e.ms(0.01) + 100
    e.trades(0.1, [e.row(str(ts), "buy", "2500.00", "1")])
    e.trades(4.4, [e.row(str(ts + 2000), "buy", "2500.00", "1")])
    e.trades(0.2, [e.row(str(ts + 10), "sell", "2500.05", "1")])
    e.trades(0.5, [e.row(str(ts + 2100), "buy", "2500.00", "1")])
    # 닫힌 초로 늦게 온 행 → 버림
    for _ in range(8):
        e.second()
    e.trades(0.2, [e.row(str(e.ms(7.0)), "sell", "2500.00", "9")])
    for _ in range(5):
        e.second()
    # 재연결 길: 구독 거부(error) · arg 가 참인 비사전 · data 가 숫자 · 사전 아닌 JSON → 다음 연결의 첫 체결이 공백을 적는다
    for b in ({"event": "error", "msg": "bad", "code": "60012"}, {"arg": [1], "data": []}, {"arg": {"channel": "trades"}, "data": 7},
              [1, 2], '"abc"'):
        e.second()
        e.msg(0.1, b)
        e.add(2.0, "o")
        for _ in range(3):
            e.second()
    # 25초 시한 → 재연결 · 글자 아닌 메시지(끊김) · 접속 실패
    e.add(25.0, "w")
    e.add(3.0, "o")
    for _ in range(5):
        e.second()
    e.add(0.3, "c")
    e.add(3.0, "f")
    e.add(3.0, "f")
    e.add(3.0, "o")
    e.trades(0.4, [e.row(str(e.ms(0.01)), "buy", "2500.00", "1")])
    # 1.0~1.5초 공백은 안 적는다
    e.add(0.2, "c")
    e.add(0.1, "o")
    e.trades(1.2, [e.row(str(e.ms(0.01)), "buy", "2500.00", "1")])
    for _ in range(25):
        e.second()
    e.add(0.3, "c")
    runs.append({"inst": e.inst, "t0": W0 - 0.5, "events": e.ev})

    # XRP(ctVal 100 · 칸 0.0001) · BTC(ctVal 0.01 · 칸 1.0 -- .5 반올림 동률)
    for inst, mid, tick, dec, szdec, n in (("XRP-USDT-SWAP", 2.1234, 0.0001, 4, 0, 40), ("BTC-USDT-SWAP", 90000.0, 0.1, 1, 1, 40)):
        x = OkxStream(rng, W0 + 10.0, inst, mid, tick, dec, szdec)
        x.add(0.0, "o")
        for s in range(n):
            if inst.startswith("BTC") and s % 8 == 0:
                ts = str(x.ms(0.01))
                x.trades(0.1, [x.row(ts, "buy", p, "5") for p in ("90000.5", "90001.5", "89999.5")])
            x.second()
        x.add(0.3, "c")
        runs.append({"inst": inst, "t0": W0 + 9.5, "events": x.ev})

    # ETH 재시작(같은 파일) -- last_ts_ms 로 ws_reconnect 공백 · 겹치는 초는 INSERT OR REPLACE
    r = OkxStream(rng, e.t + 10.0, "ETH-USDT-SWAP", e.mid, 0.01, 2, 2)
    r.add(0.0, "o")
    for _ in range(20):
        r.second()
    r.add(0.3, "c")
    runs.append({"inst": r.inst, "t0": e.t + 9.5, "events": r.ev})
    return runs


def run_okx(runs: list, d: Path) -> None:
    m = __import__("scripts.live_okx_trade_tape_collector_20260923", fromlist=["x"])
    clock = [0.0]
    m.time = _ns(time, time=lambda: clock[0], monotonic=lambda: clock[0])
    m.asyncio = _ns(asyncio, sleep=_fast_sleep)
    m.assert_ct_val = _noop
    m.verify_recent = _noop
    bn_backfill, m._bn.backfill_loop = m._bn.backfill_loop, _noop
    st = {"ev": [], "i": 0}
    try:
        with _fake_modules(aiohttp=_fake_aiohttp(st, clock, "twc")), _quiet():
            for run in runs:
                st["ev"], st["i"] = run["events"], 0
                clock[0] = run["t0"]
                try:
                    asyncio.run(m.collect(run["inst"], d / "okx_tape.sqlite"))
                except asyncio.CancelledError:
                    pass
                assert st["i"] == len(st["ev"]), "사건을 다 먹지 못했다"
    finally:
        m._bn.backfill_loop = bn_backfill


# ── 가짜 모듈 ────────────────────────────────────────────────────────────────
class HarnessError(BaseException):
    """가짜 모듈의 단언 실패. 🔴Exception 이면 수집기의 `except Exception` 이 받아 재연결을 끝없이 돈다(2026-11-20 처음 그랬다)."""


def _want(ok: bool, why) -> None:
    if not ok:
        raise HarnessError(why)


def _ns(mod, **over):
    ns = types.SimpleNamespace(**{k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")})
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


async def _fast_sleep(_s, *a, **k):
    await asyncio.sleep(0)


async def _noop(*_a, **_k):
    return None


def _exc(body: str) -> Exception:
    """«이름: 글» → 그 이름의 예외(type(exc).__name__ · str(exc) 가 그대로 나온다)."""
    name, _, text = body.partition(": ")
    # 이름만 같은 새 예외 -- 내장 TimeoutError 가 아니라서 asyncio.TimeoutError 처리에 안 걸린다(실제 접속 실패처럼 일반 예외 길)
    return type(name, (Exception,), {})(text) if text else type(name, (Exception,), {})()


LENIENT = {"on": False}   # 고친 정답지를 만들 때만: 다음 접속 사건이 수신 중에 오면 «서버가 닫음»으로 본다(Rust replay 도 같게)


class _MT:
    TEXT, CLOSED = "TEXT", "CLOSED"


def _fake_aiohttp(st: dict, clock: list, kinds: str):
    """aiohttp 흉내: ws_connect 가 'o'(성공)·'f'(실패)를 먹고, receive 가 't'·'w'(시한)·'c'(글자 아닌 메시지)를 먹는다."""
    def nxt(ks):
        ev, i = st["ev"], st["i"]
        if i >= len(ev):
            raise asyncio.CancelledError()          # 입력 끝 = 수집 끝
        t, k, body = ev[i]
        if LENIENT["on"] and k in "of" and "o" not in ks:
            return "c", ""                          # 고친 수집기는 원본이 끊었던 자리에서 안 끊는다 -- 그 접속은 서버가 닫은 것으로(먹지 않는다)
        _want(k in ks, (i, k, ks))
        st["i"] = i + 1
        clock[0] = t
        return k, body

    class WS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def send_json(self, _o):
            pass

        async def receive(self, timeout=None):
            k, body = nxt(kinds)
            if k == "w":
                raise asyncio.TimeoutError()
            if k == "c":
                return types.SimpleNamespace(type=_MT.CLOSED, data=None)
            return types.SimpleNamespace(type=_MT.TEXT, data=body)

    class Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def ws_connect(self, *a, **k):
            k_, body = nxt("of")
            if k_ == "f":
                raise _exc(body or "OSError: 가짜 접속 실패")
            return WS()

    return types.SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **k: None, WSMsgType=_MT)


class _fake_modules:
    def __init__(self, **mods):
        self.mods, self.old = mods, {}

    def __enter__(self):
        for k, v in self.mods.items():
            self.old[k] = sys.modules.get(k)
            sys.modules[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _quiet:
    def __enter__(self):
        self.old, self.dev = sys.stdout, open(os.devnull, "w")
        sys.stdout = self.dev

    def __exit__(self, *a):
        sys.stdout = self.old
        self.dev.close()


def close_shared() -> None:
    """바이낸스 모듈이 붙들고 있는 hot 연결을 닫는다(TapeStore 공유 연결 -- 덤프 전에 WAL 을 정리)."""
    tc = __import__("scripts.live_trade_tape_collector_20260916", fromlist=["x"])
    for ent in tc._SQLITE.values():
        if ent[1] is not None:
            ent[1].close()
            ent[1] = None
    tc._SQLITE.clear()



# ── 의도한 차이(사용자 결정 2026-11-21: 결함은 Rust 에서만 고친다) ─────────────────────────────
# 정답지는 «원본 + 결함 고침 한 곳» 파이썬으로 만든다: 원본 함수의 글자에서 정해 둔 몇 줄만 바꿔 다시 정의한다(바꾼 글자 = rtcollect/SHADOW.md §11).
# 고치지 않은 원본으로도 한 번 더 돌려, 두 결과가 어디서 몇 행 다른지를 정답지 "intended" 에 적는다(차이가 0 이면 그 결함을 시험이 안 건드린다는 뜻).
class FixedSrc:
    """with FixedSrc(모듈, [(주인, 이름, [(옛 글자, 새 글자)…]), …], 더할 이름들): -- 들어갈 때 바꾸고 나올 때 되돌린다."""

    def __init__(self, mod, fixes, inject=None):
        self.mod, self.fixes, self.inject, self.old = mod, fixes, inject or {}, []

    def __enter__(self):
        import inspect
        import textwrap
        self.saved = {k: getattr(self.mod, k, None) for k in self.inject}
        for k, v in self.inject.items():
            setattr(self.mod, k, v)
        for owner, name, reps in self.fixes:
            f = getattr(owner, name)
            src = textwrap.dedent(inspect.getsource(f))
            for a, b in reps:
                assert src.count(a) == 1, (name, a, src.count(a))
                src = src.replace(a, b)
            ns = self.mod.__dict__
            keep = ns.get(name)
            exec(compile(src, f"<고침 {name}>", "exec"), ns)
            new = ns[name]
            if owner is not self.mod:
                ns[name] = keep
                if keep is None:
                    ns.pop(name)
            self.old.append((owner, name, f))
            setattr(owner, name, new)
        return self

    def __exit__(self, *a):
        for owner, name, f in reversed(self.old):
            setattr(owner, name, f)
        self.old.clear()
        for k, v in self.saved.items():
            if v is None:
                delattr(self.mod, k)
            else:
                setattr(self.mod, k, v)


def intended_diff(orig: dict, fixed: dict) -> dict:
    """{파일.표: [원본 행 수, 고친 행 수, 다른 행 수(앞에서부터 짝지어)]} -- 0 이 아닌 표만."""
    out = {}
    for db, g in fixed.items():
        for t, rows in g["tables"].items():
            o = orig.get(db, {}).get("tables", {}).get(t, [])
            n = sum(a != b for a, b in zip(o, rows)) + abs(len(o) - len(rows))
            if n:
                out[f"{db}.{t}"] = [len(o), len(rows), n]
    return out


def _bindable(r) -> bool:
    return all(isinstance(x, (str, int, float, type(None))) for x in r[1:3])


_NUM = ("openInterest", "openInterestValue", "fundingRate", "nextFundingTime", "markPrice", "indexPrice")


def _num_ok(k, v) -> bool:
    import math
    if k not in _NUM:
        return True
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return k != "nextFundingTime" or math.isfinite(x)


def fixes(kind: str):
    if kind == "bybit_liq":
        m = __import__("scripts.live_bybit_liquidation_collector_20261003", fromlist=["x"])
        return FixedSrc(m, [(m, "parse", [(
            '    return [(int(d["T"]), d["s"], d["S"], float(d["v"]), float(d["p"]), recv_ms) for d in msg.get("data") or []]',
            '    return [r for r in [(int(d["T"]), d["s"], d["S"], float(d["v"]), float(d["p"]), recv_ms) for d in msg.get("data") or []] if _rt_bindable(r)]')])],
            {"_rt_bindable": _bindable})
    if kind == "bybit_tape":
        m = __import__("scripts.live_bybit_trade_tape_collector_20261006", fromlist=["x"])
        return FixedSrc(m, [(m.TickerState, "rows", [(
            'c.update({k: v for k, v in d.items() if v not in (None, "")})',
            'c.update({k: v for k, v in d.items() if v not in (None, "") and _rt_num_ok(k, v)})')])],
            {"_rt_num_ok": _num_ok})
    return None

# ── 종류 ─────────────────────────────────────────────────────────────────────
TAPE_TABLES = {"trade_tape_1s": "symbol, ts_sec, price_bin", "gaps": "rowid", "verify_1m": "rowid",
               "backfill_1m": "rowid", "meta": "rowid"}
KINDS = {
    "bybit_liq": (gen_liq, run_liq, {"bybit_liq.sqlite": {"bybit_liquidations": "rowid", "gaps": "rowid"}}),
    "bybit_tape": (gen_bybit_tape, run_bybit_tape, {"bybit_tape.sqlite": TAPE_TABLES,
                                                    "bybit_ctx.sqlite": {"bybit_oi": "rowid", "bybit_mark": "rowid",
                                                                         "bybit_funding": "rowid", "gaps": "rowid"}}),
    "okx": (gen_okx, run_okx, {"okx_tape.sqlite": TAPE_TABLES}),
}


def python_reference(kind: str, inp, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    _, run, dbs = KINDS[kind]
    run(inp, (d / next(iter(dbs))) if kind == "bybit_liq" else d)
    close_shared()


def _cell(x):
    """JSON 에 없는 무한대 REAL 은 {"f": "inf"|"-inf"} 로(tests/common 이 읽는다). NaN 은 SQLite 가 NULL 로 쌓아 안 나온다."""
    return {"f": repr(x)} if isinstance(x, float) and x in (float("inf"), float("-inf")) else x


def dump(db: Path, tables: dict) -> dict:
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {"pragma": {p: con.execute(f"PRAGMA {p}").fetchone()[0]
                           for p in ("journal_mode", "auto_vacuum", "page_size", "encoding", "user_version")},
                "schema": [list(r) for r in con.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")],
                "order": tables,
                "tables": {t: [[_cell(x) for x in r] for r in con.execute(f"SELECT * FROM {t} ORDER BY {o}")] for t, o in tables.items()}}
    finally:
        con.close()


def dump_dir(kind: str, d: Path) -> dict:
    return {name: dump(d / name, t) for name, t in KINDS[kind][2].items()}


def cargo_replay(kind: str, inp: Path, out: Path) -> None:
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 120 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release",
                        "--example", "replay_venue", "--", kind, str(inp), str(out)],
                       cwd=ROOT / "rtcollect", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo replay_venue 실패:\n{r.stderr}")


# ── 파이썬 읽는 쪽 교차 확인 ───────────────────────────────────────────────────
def _span(db: Path, sql: str, args=()) -> tuple[int, int]:
    """자료가 있는 구간 [lo, hi) -- 🔴읽기 함수에 넘길 범위는 늘 이것으로(2026-11-20 0~2^40초로 불러 13.7GB OOM).
    하루를 넘으면 멈춘다."""
    from scripts import data_store
    lo, hi = data_store.read_rows(db, sql, list(args))[0]
    assert lo is not None and 0 <= hi - lo < 86_400, (db.name, lo, hi)
    return int(lo), int(hi) + 1


def readers(kind: str, d: Path, inp) -> dict:
    from scripts import data_store
    # 읽는 쪽 모듈이 머리에서 부르는 것 중 비밀 설정 파일을 읽는 것(ops_watchdog = load_dotenv)·연구 코드는 빈 모듈로 막는다
    for name in ("ops_watchdog", "research_eth_liqmap_venue_leverage_validation_20261008"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import scripts.ops.data_quality_monitor_20261022 as dqm
    dqm.HOT = d
    out: dict = {}
    if kind == "bybit_liq":
        import scripts.live_liqmap_scorer_20261008 as lms
        db = d / "bybit_liq.sqlite"
        lo, hi = _span(db, "SELECT min(ts_ms) / 1000, max(ts_ms) / 1000 FROM bybit_liquidations WHERE typeof(ts_ms) = 'integer'")
        out["read_rows"] = data_store.read_rows(db, "SELECT * FROM bybit_liquidations ORDER BY rowid")
        # 연결 안 메시지 예외(깨진 JSON 등)의 글은 구현마다 다르다 -- tests/venue_parity.rs liq_reason 과 같이 그 넷만 이름으로
        parse_err = ("JSONDecodeError", "TypeError", "ValueError", "AttributeError")
        out["gaps"] = [(a_, b_, r.split(":")[0] if r.split(":")[0] in parse_err else r)
                       for a_, b_, r in data_store.read_rows(db, "SELECT * FROM gaps ORDER BY rowid")]
        lms.HOT = d
        q = lms._q("bybit_liq.sqlite", "SELECT ts_ms, side, qty FROM bybit_liquidations WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",
                   (lo * 1000, hi * 1000))
        out["liqmap_scorer._q"] = [list(map(repr, r)) for r in q.itertuples(index=False)]
        for c in ("ETH", "BTC", "SOL", "XRP"):
            out[f"dqm.LIQS:{c}"] = dqm.q(dqm.LIQS["bybit"][0], dqm.LIQS["bybit"][2], (f"{c}USDT", lo * 1000, hi * 1000))
    else:
        import dashboard.coin_flow as cf
        tape = d / ("bybit_tape.sqlite" if kind == "bybit_tape" else "okx_tape.sqlite")
        syms = sorted({s for r in inp for s in r["symbols"]}) if kind == "bybit_tape" else sorted({r["inst"] for r in inp})
        for sym in syms:
            lo, hi = _span(tape, "SELECT min(ts_sec), max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [sym])
            lo = lo // 300 * 300
            bucket = 0.01
            if kind == "okx":
                bucket = __import__("scripts.live_okx_trade_tape_collector_20260923", fromlist=["x"]).BUCKETS[sym]
            else:
                bucket = __import__("scripts.live_trade_tape_collector_20260916", fromlist=["x"]).BUCKETS.get(sym.lower(), 0.01)
            out[f"read_rows:{sym}"] = data_store.read_rows(tape, "SELECT * FROM trade_tape_1s WHERE symbol = ? ORDER BY 2, 3", [sym])
            out[f"gaps:{sym}"] = data_store.read_rows(tape, "SELECT * FROM gaps WHERE symbol = ? ORDER BY rowid", [sym])
            out[f"footprint_5m:{sym}"] = data_store.read_rows(tape, "SELECT * FROM footprint_5m WHERE symbol = ? ORDER BY 2, 3", [sym])
            secs, top = cf.tape_seconds(tape, sym, lo, hi, bucket)
            out[f"tape_seconds:{sym}"] = [sorted(secs.items()), top]
            fp = cf.okx_tape_footprint(tape, d / "_no_ctx.duckdb", sym, lo, hi * 1000, 300, 0.5, bucket)
            out[f"okx_tape_footprint:{sym}"] = [sorted((b, sorted(c.items())) for b, c in fp["bars"].items()), fp["tape_max"], fp["ctx_ok"]]
            out[f"dqm.tape_stats:{sym}"] = dqm.tape_stats(tape.name, sym, lo, hi)
        if kind == "bybit_tape":
            ctx = d / "bybit_ctx.sqlite"
            lo, hi = _span(ctx, "SELECT min(ts_ms) / 1000, max(ts_ms) / 1000 FROM bybit_mark")
            for t in ("bybit_oi", "bybit_mark", "bybit_funding", "gaps"):
                out[f"ctx:{t}"] = data_store.read_rows(ctx, f"SELECT * FROM {t} ORDER BY rowid")
            for name in ("mark:bybit", "oi:bybit"):
                db_, table, tcol, kcol, key = dqm.CTX[name]
                for c in ("ETH", "BTC", "XRP"):
                    out[f"dqm.ctx_stats:{name}:{c}"] = dqm.ctx_stats(db_, table, tcol, kcol, key(c), lo, hi)
                    out[f"dqm.minute_marks:{name}:{c}"] = dqm.minute_marks(db_, table, tcol, kcol, key(c), lo, hi,
                                                                          "mark_px" if table == "bybit_mark" else "oi_base")
    out["integrity"] = [data_store.read_rows(d / n, "PRAGMA integrity_check") for n in KINDS[kind][2]]
    return out


def check_readers(kinds) -> None:
    for kind in kinds:
        inp_path = FIX / f"{kind}_input.json"
        inp = json.loads(inp_path.read_text())["input"]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fx = fixes(kind)                         # 🔧정답지와 같게: 결함 고친 파이썬(의도한 차이)
            LENIENT["on"] = bool(fx)
            if fx:
                with fx:
                    python_reference(kind, inp, td / "py")
            else:
                python_reference(kind, inp, td / "py")
            LENIENT["on"] = False
            cargo_replay(kind, inp_path, td / "rs")
            a, b = readers(kind, td / "py", inp), readers(kind, td / "rs", inp)
            assert a.keys() == b.keys()
            bad = [k for k in a if a[k] != b[k]]
            for k in a:
                n = len(a[k]) if isinstance(a[k], (list, dict)) else 1
                print(f"  {kind}  {'같음' if k not in bad else '다름'}  {k}: {n}")
            assert not bad, bad
            assert all(x == [("ok",)] for x in a["integrity"]), a["integrity"]
            # 섞어 쓰기: 파이썬이 앞 판을 쓰고 연결을 열어 둔 채(WAL 미정리) Rust 가 뒤 판을 이어 쓴다 → 전부 파이썬과 같아야 한다
            if kind != "bybit_liq":
                k = len(inp) - 1
                fx = fixes(kind)
                LENIENT["on"] = bool(fx)
                if fx:
                    with fx:
                        KINDS[kind][1](inp[:k], td / "mix")
                else:
                    KINDS[kind][1](inp[:k], td / "mix")
                LENIENT["on"] = False
                assert (td / "mix" / next(iter(KINDS[kind][2]))).with_suffix(".sqlite-wal").exists(), "파이썬 연결이 열려 있어야 한다"
                mix_inp = td / "mix_input.json"
                mix_inp.write_text(json.dumps({"input": inp[k:]}))
                cargo_replay(kind, mix_inp, td / "mix")
                close_shared()
                same = dump_dir(kind, td / "mix") == dump_dir(kind, td / "py")
                print(f"  {kind}  {'같음' if same else '다름'}  섞어 쓰기(파이썬 {k}판 + 연결 열린 채 Rust {len(inp) - k}판) vs 전부 파이썬")
                assert same
    print("check-readers OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--check-readers", action="store_true")
    a = ap.parse_args()
    if a.check_readers:
        return check_readers(a.kinds)
    for i, kind in enumerate(a.kinds):
        gen = KINDS[kind][0]
        inp = gen(random.Random(SEED + list(KINDS).index(kind)))
        with tempfile.TemporaryDirectory() as td:
            python_reference(kind, inp, Path(td) / "orig")
            orig = dump_dir(kind, Path(td) / "orig")
            fx = fixes(kind)
            if fx:
                LENIENT["on"] = True
                with fx:
                    python_reference(kind, inp, Path(td) / "fixed")
                LENIENT["on"] = False
            golden = dump_dir(kind, Path(td) / ("fixed" if fx else "orig"))
            if fx:
                golden["intended"] = intended_diff(orig, golden)
        FIX.mkdir(parents=True, exist_ok=True)
        (FIX / f"{kind}_input.json").write_text(json.dumps({"seed": SEED, "input": inp}, separators=(",", ":"), ensure_ascii=False))
        (FIX / f"{kind}_golden.json").write_text(json.dumps(golden, separators=(",", ":"), ensure_ascii=False))
        n_ev = len(inp) if kind == "bybit_liq" else sum(len(r["events"]) for r in inp)
        print(f"{kind}: 사건 {n_ev:,} · " + " · ".join(f"{db} {t} {len(v)}행" for db, g in golden.items() if db != "intended" for t, v in g["tables"].items() if v)
              + (f" · 의도한 차이 {golden['intended']}" if "intended" in golden else ""))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rtcollect 체결 테이프 일치 시험의 입력·정답지를 만든다 (2026-11-20).

정해진 시드로 합성 웹소켓 메시지 열을 만들고, **원본 수집 루프** `scripts/live_trade_tape_collector_20260916.py::collect()`
를 그대로 돌려(파싱·가격 0 건너뛰기·주문 되묶기·재연결 리셋·공백 기록·5초 플러시·INSERT OR REPLACE 전부 원본)
임시 SQLite 를 만든 뒤 표 전체를 덤프한다. rtcollect/tests/tape_parity.rs 가 같은 입력을 Rust 에 넣어 행 단위로 대조한다.

네트워크 없음: 소켓 connect 를 막고, aiohttp 는 가짜 모듈로 바꾸고(받은 메시지 열을 차례로 내놓는 웹소켓),
REST 를 부르는 `verify_recent`·`backfill_loop` 는 빈 코루틴으로 바꾼다(이번 범위 밖 -- REST 는 코어가 따로 정한다).
시계: 수집기 모듈의 `time.monotonic` 만 «메시지 수신 시각»(입력의 recv)으로 바꾼다 -- 5초 플러시가 입력으로 정해진다.
`asyncio.sleep`(재연결 3초 대기)도 모듈 안에서만 0 으로. 심볼(실행)은 차례로 돌린다 -- 원본도 심볼마다 상태가 따로이고
같은 파일·연결만 나눈다(그래서 gaps·meta 의 rowid 순서가 정해진다).

    python3 rtcollect/tools/gen_tape_fixture.py                    # 입력·정답지(tests/fixtures/tape_*.json) 다시 만들기
    python3 rtcollect/tools/gen_tape_fixture.py --check-readers    # Rust 가 만든 SQLite 를 파이썬 읽는 쪽으로 교차 확인
    python3 rtcollect/tools/gen_tape_fixture.py --bench 300000     # 부하 입력(스크래치) + 파이썬 원본 처리 시간
"""
from __future__ import annotations

import argparse
import asyncio
import enum
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtcollect" / "tests" / "fixtures"
SEED = 20261120
DAY = 1_763_683_200_000          # 2025-11-21 00:00:00 UTC(ms). 합성 -- 실제 장과 무관. 날짜 경계를 건너게 그 150초 앞에서 시작
DBS = {"futures": "binance_tape.sqlite", "spot": "binance_spot_tape.sqlite"}


def _no_net(*_a, **_k):
    raise RuntimeError("gen_tape_fixture: 네트워크 금지")


socket.socket.connect = _no_net
socket.create_connection = _no_net


# ── 합성 메시지 열 ───────────────────────────────────────────────────────────
class Stream:
    """한 실행(=수집기 한 판)의 사건 열. 사건 = [recv(초, 단조), 종류, 본문] -- 't' 글자 메시지 · 'c' 글자 아닌 메시지(끊김) ·
    'f' 접속 실패. recv 는 체결 시각 + 수신 지연(단조 증가)."""

    def __init__(self, rng, T, mid, tick, dec, qdec, spot=False):
        self.rng, self.T, self.mid, self.tick, self.dec, self.qdec, self.spot = rng, T, mid, tick, dec, qdec, spot
        self.tid, self.aid, self.recv, self.ev = 70_000_000, 9_000_000, 5000.0 + (T - DAY) / 1000, []

    def _at(self, T):
        self.recv = max(self.recv, 5000.0 + (T - DAY) / 1000 + self.rng.uniform(0.001, 0.05))
        return round(self.recv, 6)

    def raw(self, T, text):
        self.ev.append([self._at(T), "t", text])

    def close(self):
        self.ev.append([self.recv, "c", ""])

    def fail(self):
        self.ev.append([self.recv, "f", ""])

    def px(self, p):
        return f"{p:.{self.dec}f}"

    def trade(self, ts, p, q, side, **over):
        if self.spot:
            n = self.rng.randint(1, 4)
            d = {"e": "aggTrade", "E": ts + 3, "a": self.aid, "p": p, "q": q, "f": self.tid + 1, "l": self.tid + n,
                 "T": ts, "m": side}
            self.aid += 1
            self.tid += n
        else:
            self.tid += 1
            d = {"e": "trade", "E": ts + 2, "T": ts, "t": self.tid, "p": p, "q": q, "m": side}
        d.update(over)
        d = {k: v for k, v in d.items() if v is not _DROP}
        self.raw(ts, json.dumps(d, separators=(",", ":")))

    def order(self, T, side, nfill, big=False):
        """같은 가격·방향 연속 체결(=테이커 주문 하나). 체결 사이 0~30ms(같은 ms 포함)."""
        p = self.px(self.mid + self.rng.randint(-3, 3) * self.tick)
        for _ in range(nfill):
            q = self.rng.lognormvariate(-1.0 if big else -4.0, 1.6) * (3000.0 / self.mid)
            self.trade(T, p, f"{max(q, 10 ** -self.qdec):.{self.qdec}f}", side)
            T += self.rng.choice((0, 0, 1, 3, 8, 30))
        return T

    def second(self, s0, k=6):
        """한 초 분량의 무작위 주문."""
        self.mid += self.rng.gauss(0, 2.5) * self.tick
        T = s0 + self.rng.randint(0, 200)
        for _ in range(self.rng.randint(0, k)):
            T = self.order(T, self.rng.random() < 0.5, self.rng.choice((1, 1, 2, 3, 5)), big=self.rng.random() < 0.08)
            T += self.rng.randint(0, 150)
            if T >= s0 + 1000:
                break


_DROP = object()


def gen_runs(seed=SEED):
    rng = random.Random(seed)
    runs = []
    # ① ETH 선물 -- 특수 상황 대부분
    T0 = DAY - 150_000 + 437
    e = Stream(rng, T0, 2500.0, 0.01, 2, 3)
    t = T0
    for s in range(200):
        s0 = T0 // 1000 * 1000 + s * 1000
        if s == 10:      # 같은 ms 여러 건(가격·방향 섞임)
            for p, m in (("2501.00", False), ("2501.00", False), ("2501.10", True), ("2501.00", False)):
                e.trade(s0 + 5, p, "0.250", m)
        elif s == 12:    # 가격 0 (바이낸스가 섞어 보낸다 -- 2026-09-16 실측 0.3%)
            e.trade(s0 + 1, "0", "0", False, X="NA", st=1)
            e.trade(s0 + 2, "2500.50", "0.000", True)       # 수량 0
        elif s == 15:    # 같은 메시지 두 번(체결 ID 중복)
            e.trade(s0 + 10, "2500.20", "1.000", False)
            e.ev.append([e._at(s0 + 11), "t", e.ev[-1][2]])
        elif s == 18:    # 체결 ID 건너뜀
            e.trade(s0 + 10, "2500.30", "1.000", True)
            e.tid += 5
            e.trade(s0 + 12, "2500.30", "1.000", True)
        elif s == 20:    # 체결 ID 역전
            e.trade(s0 + 10, "2500.30", "1.000", True)
            e.tid -= 3
            e.trade(s0 + 12, "2500.30", "1.000", True)
        elif s == 22:    # 같은 초 안에서 시각이 뒤로
            e.trade(s0 + 500, "2500.40", "0.500", False)
            e.trade(s0 + 450, "2500.40", "0.500", False)
        elif s == 25:    # 초 경계를 걸친 같은 가격·방향·연속 ID 주문 -> 초에서 잘린다
            e.trade(s0 + 995, "2500.60", "2.000", False)
            e.trade(s0 + 1002, "2500.60", "3.000", False)
        elif s == 28:    # 크기 경계 양쪽($10k·$100k 정확히)
            for j, (q, m) in enumerate((("3.999", False), ("4.000", False), ("39.999", True), ("40.000", True))):
                e.trade(s0 + 150 * (j + 1), "2500.00", q, m)
        elif s == 30:    # 체결 아닌 메시지 · e 없음 · e 가 숫자
            e.raw(s0 + 5, '{"e":"24hrTicker","E":%d,"c":"2500.0"}' % (s0 + 5))
            e.raw(s0 + 6, '{"T":%d,"p":"2500.0","q":"1.0","m":true}' % (s0 + 6))
            e.raw(s0 + 7, '{"e":5}')
        elif s == 33:    # 값 모양: p·q 가 JSON 숫자, T 가 문자열, m 이 0/1·문자열
            e.trade(s0 + 10, 2500.7, 0.125, 1, T=str(s0 + 10))
            e.trade(s0 + 20, 2500.7, 1, 0)
            e.trade(s0 + 30, "2500.70", "0.333", "")
            e.trade(s0 + 40, " 2500.70 ", "1e-3", "x")
        elif s == 35:    # 체결 ID 없음 · null
            e.trade(s0 + 10, "2500.80", "1.000", False, t=_DROP)
            e.trade(s0 + 15, "2500.80", "1.000", False, t=None)
        elif s == 40:    # 닫힌 초로 늦게 온 체결·주문 -> 버린다(dropped_late)
            e.trade(s0 + 100, "2500.90", "1.000", False)
            e.trade(s0 - 7_000, "2500.95", "9.000", True)          # 7초 전(이미 디스크로 나간 초)
            e.trade(s0 - 6_990, "2500.95", "9.000", True)
            e.trade(s0 + 200, "2500.90", "1.000", False)           # 위 늦은 주문을 닫는다 -> 그 초는 닫힘 -> 버림
        elif s == 50:    # 끊김(글자 아닌 메시지) -> 3초 뒤 재연결: 공백 기록 · 열린 묶음 버림
            e.order(s0 + 100, False, 3)
            e.close()
            s0 += 0
        elif s in (51, 52, 53):
            continue
        elif s == 60:    # 깨진 JSON -> 원본은 예외 = 재연결. 1초 미만 공백은 기록 안 함
            e.trade(s0 + 100, "2501.00", "1.000", False)
            e.raw(s0 + 101, '{"e":"trade","T":%d,"t":1,"p":"25' % (s0 + 101))
            e.trade(s0 + 900, "2501.00", "1.000", False)
        elif s == 70:    # 끊김 -> 접속 실패 2번 -> 붙은 첫 메시지에 m 이 없다(공백은 기록, 그다음 KeyError) -> 다시
            e.close(); e.fail(); e.fail()
            e.trade(s0 + 2_500, "2501.10", "1.000", False, m=_DROP)
        elif s in (71, 72):
            continue
        elif s == 80:    # JSON 이지만 객체가 아니다 -> AttributeError = 재연결
            e.raw(s0 + 10, "[1,2,3]")
        elif s == 90:    # 숫자로 못 읽는 가격 -> ValueError = 재연결
            e.trade(s0 + 10, "abc", "1.000", False)
        elif s == 95:    # 시각이 정수 문자열이 아니다(T="x") -> 재연결 · 체결 ID 가 문자열
            e.trade(s0 + 10, "2501.20", "1.000", False, T="x")
            e.trade(s0 + 20, "2501.20", "1.000", False, t=str(e.tid + 1))
        elif 120 <= s < 132:   # 플러시 시점이 보이게: 매 초 «정상 1건 + 직전 초로 늦은 1건» -- 늦은 건은 이 초에 플러시가 났을 때만 버려진다
            e.trade(s0 + 100, "2500.10", "0.100", False)
            e.trade(s0 - 500, "2500.20", "0.100", True)
        elif s == 140:   # 1.0~1.5초 공백은 기록 안 함(1.5초 문턱)
            e.trade(s0 + 100, "2500.10", "0.100", False)
            e.close()
            e.trade(s0 + 1_350, "2500.10", "0.100", False)
        elif s == 141:
            continue
        elif s in (197, 198, 199):   # ⑤ 재시작 판이 같은 초·칸을 다시 쓴다 -> INSERT OR REPLACE
            e.trade(s0 + 300, "2500.00", "0.700", False)
            e.second(s0)
        elif s == 98:    # 체결은 칸에 들어간 뒤 체결 ID 를 못 읽는다(t="z") -> 재연결(칸엔 남고 묶음엔 없다)
            e.trade(s0 + 10, "2501.30", "1.000", False, t="z")
        else:
            e.second(s0)
    e.close()
    runs.append({"symbol": "ethusdt", "market": "futures", "t0": round(e.ev[0][0] - 0.5, 6), "events": e.ev})
    end_a = T0 + 200_000

    # ② BTC 선물 -- 칸 1.0 이라 .5 가격이 반올림 동률(파이썬 round = 짝수 쪽) · 경계 $13k/$160k
    b = Stream(rng, DAY + 10_000, 90000.0, 0.1, 1, 3)
    for s in range(60):
        s0 = (DAY + 10_000) // 1000 * 1000 + s * 1000
        if s % 6 == 0:
            for p in ("90000.5", "90001.5", "89999.5", "90002.5"):
                b.trade(s0 + 50, p, "0.050", s % 12 == 0)
        if s == 20:
            for q in ("0.144", "0.145", "1.777", "1.778"):        # $13k·$160k 양쪽(90000 근처)
                b.trade(s0 + 300, "90000.0", q, False)
        b.second(s0, 4)
    b.close()
    runs.append({"symbol": "btcusdt", "market": "futures", "t0": round(b.ev[0][0] - 0.5, 6), "events": b.ev})
    btc_end = (DAY + 10_000) // 1000 * 1000 + 60_000

    # ③ XRP 선물(칸 0.0001) · ④ 표에 없는 심볼(기본 칸 0.01 · ETH 경계)
    for sym, mid, tick, dec, qdec, n in (("xrpusdt", 2.1234, 0.0001, 4, 1, 40), ("dogeusdt", 0.15432, 0.00001, 5, 0, 30)):
        x = Stream(rng, DAY + 20_000, mid, tick, dec, qdec)
        for s in range(n):
            x.second((DAY + 20_000) // 1000 * 1000 + s * 1000, 5)
        x.close()
        runs.append({"symbol": sym, "market": "futures", "t0": round(x.ev[0][0] - 0.5, 6), "events": x.ev})

    # ⑤ ETH 재시작(같은 파일) -- last_ts_ms 로 공백 · 이미 쓴 초와 겹치는 체결은 INSERT OR REPLACE 로 덮는다
    r = Stream(rng, end_a - 2_500, 2500.0, 0.01, 2, 3)
    for s in range(60):
        s0 = (end_a - 2_500) // 1000 * 1000 + s * 1000
        if s in (0, 1):
            r.trade(s0 + 400, "2500.00", "0.300", False)
        if s == 30:
            r.close()
            r.trade(s0 + 4_000, "2502.00", "1.000", True)          # 재연결 공백 4초
        r.second(s0)
    r.close()
    runs.append({"symbol": "ethusdt", "market": "futures", "t0": round(r.ev[0][0] - 0.5, 6), "events": r.ev})

    # ⑦ BTC 재시작 3초 뒤(앞 판 끝 + 1초 = last_ts_ms 부터 공백)
    b2 = Stream(rng, btc_end + 3_000, b.mid, 0.1, 1, 3)
    for s in range(10):
        b2.second((btc_end + 3_000) // 1000 * 1000 + s * 1000, 4)
    b2.close()
    runs.append({"symbol": "btcusdt", "market": "futures", "t0": round(b2.ev[0][0] - 0.5, 6), "events": b2.ev})

    # ⑥ SOL 현물(@aggTrade -- add_agg · 체결 수 l-f+1 · l/f 없음 · 선물 메시지 섞임은 건너뜀)
    p = Stream(rng, DAY + 30_000, 140.0, 0.01, 2, 2, spot=True)
    for s in range(60):
        s0 = (DAY + 30_000) // 1000 * 1000 + s * 1000
        if s == 10:
            p.trade(s0 + 10, "140.00", "60.00", False, l=_DROP)
            p.trade(s0 + 20, "140.00", "60.00", True, f=None, l=None)
            p.trade(s0 + 30, "140.00", "53.57", True, f=5, l=3)
            p.raw(s0 + 40, '{"e":"trade","T":%d,"t":1,"p":"140.0","q":"1.0","m":true}' % (s0 + 40))
        p.second(s0, 4)
    p.close()
    runs.append({"symbol": "solusdt", "market": "spot", "t0": round(p.ev[0][0] - 0.5, 6), "events": p.ev})
    return runs


def gen_bench(n_msgs, rate=5000.0, seed=1):
    """부하 입력: ETH 선물 한 판, 초당 `rate` 건."""
    rng = random.Random(seed)
    T0 = DAY + 1_000
    e = Stream(rng, T0, 2500.0, 0.01, 2, 3)
    s = 0
    while len(e.ev) < n_msgs:
        s0 = T0 // 1000 * 1000 + s * 1000
        while len(e.ev) < n_msgs and (not e.ev or e.ev[-1][0] < 5000.0 + (s0 + 1000 - DAY) / 1000):
            T = s0 + int(rng.random() * 1000)
            e.order(T, rng.random() < 0.5, rng.choice((1, 2, 3, 5)))
        e.ev.sort(key=lambda x: x[0])
        s += 1
        if s > n_msgs:
            break
    # 수신 시각을 초당 rate 건 간격으로 다시 매긴다(같은 순서)
    e.ev = [[round(5000.0 + i / rate, 6), k, t] for i, (_r, k, t) in enumerate(e.ev)]
    e.close()
    return [{"symbol": "ethusdt", "market": "futures", "t0": 4999.5, "events": e.ev}]


# ── 원본 수집 루프를 가짜 웹소켓으로 ───────────────────────────────────────────
def _load_collector():
    sys.path.insert(0, str(ROOT))
    import scripts.live_trade_tape_collector_20260916 as tc
    return tc


class _MT(enum.Enum):
    TEXT = 1
    CLOSED = 8


def python_reference(runs, out_dir: Path, timing: dict | None = None, quiet=True) -> None:
    """runs 를 원본 collect() 에 차례로 먹여 out_dir 에 SQLite 를 만든다."""
    tc = _load_collector()
    clock = [0.0]
    real_time = time
    tc.time = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("__")})
    tc.time.monotonic = lambda: clock[0]

    async def _fast_sleep(_s, *a, **k):
        await asyncio.sleep(0)
    tc.asyncio = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
    tc.asyncio.sleep = _fast_sleep

    async def _noop(*_a, **_k):
        return None
    tc.verify_recent = _noop          # REST(1분봉 대조) -- 이번 범위 밖
    tc.backfill_loop = _noop          # REST(aggTrades 복구) -- 이번 범위 밖
    if timing is not None:
        orig_write = tc.TapeStore.write

        def timed_write(self, rows):
            t = real_time.perf_counter()
            orig_write(self, rows)
            timing.setdefault("writes", []).append((len(rows), real_time.perf_counter() - t))
        tc.TapeStore.write = timed_write

    state = {"ev": [], "i": 0}

    class FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            ev, i = state["ev"], state["i"]
            if i >= len(ev) or ev[i][1] == "f":
                raise StopAsyncIteration
            state["i"] = i + 1
            clock[0] = ev[i][0]
            if ev[i][1] == "c":
                return types.SimpleNamespace(type=_MT.CLOSED, data=None)
            return types.SimpleNamespace(type=_MT.TEXT, data=ev[i][2])

    class FakeSession:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def ws_connect(self, *a, **k):
            ev, i = state["ev"], state["i"]
            if i >= len(ev):
                raise asyncio.CancelledError()      # 입력 끝 = 수집 끝
            if ev[i][1] == "f":
                state["i"] = i + 1
                raise OSError("가짜 접속 실패")
            return FakeWS()

    real_aiohttp = sys.modules.get("aiohttp")
    sys.modules["aiohttp"] = types.SimpleNamespace(ClientSession=FakeSession, ClientTimeout=lambda **k: None,
                                                   WSMsgType=_MT)
    out_dir.mkdir(parents=True, exist_ok=True)
    devnull = open(os.devnull, "w")
    for run in runs:
        state["ev"], state["i"] = run["events"], 0
        clock[0] = run["t0"]
        old = sys.stdout
        if quiet:
            sys.stdout = devnull
        try:
            asyncio.run(tc.collect(run["symbol"], out_dir / DBS[run["market"]], run["market"]))
        except asyncio.CancelledError:
            pass
        finally:
            sys.stdout = old
    devnull.close()
    if real_aiohttp is None:                 # 읽는 쪽(coin_flow)이 진짜 aiohttp 를 import 할 수 있게 되돌린다
        sys.modules.pop("aiohttp", None)
    else:
        sys.modules["aiohttp"] = real_aiohttp


def close_shared() -> None:
    tc = _load_collector()
    for ent in tc._SQLITE.values():
        if ent[1] is not None:
            ent[1].close()
            ent[1] = None


# ── 덤프 ─────────────────────────────────────────────────────────────────────
TABLES = {"trade_tape_1s": "symbol, ts_sec, price_bin", "gaps": "rowid", "verify_1m": "rowid",
          "backfill_1m": "rowid", "meta": "rowid"}


def dump(db: Path) -> dict:
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out = {"pragma": {p: con.execute(f"PRAGMA {p}").fetchone()[0]
                          for p in ("journal_mode", "auto_vacuum", "page_size", "encoding", "user_version")},
               "schema": [list(r) for r in con.execute(
                   "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")],
               "tables": {t: [list(r) for r in con.execute(f"SELECT * FROM {t} ORDER BY {o}")]
                          for t, o in TABLES.items()}}
    finally:
        con.close()
    return out


def dump_dir(d: Path) -> dict:
    return {name: dump(d / name) for name in sorted(set(DBS.values())) if (d / name).exists()}


# ── 파이썬 읽는 쪽 교차 확인 ───────────────────────────────────────────────────
def readers(d: Path, runs) -> dict:
    """원본 읽기 함수로 읽은 결과. data_store.read_rows(원시 질의) · coin_flow.tape_seconds(1초 수급 복원) ·
    coin_flow.okx_tape_footprint(바이낸스 풋프린트 테이프 복원 -- footprint_tape_restore 가 부르는 것)."""
    sys.path.insert(0, str(ROOT))
    from scripts import data_store
    import dashboard.coin_flow as cf
    tc = _load_collector()
    out = {}
    db = d / DBS["futures"]
    for sym in sorted({r["symbol"] for r in runs if r["market"] == "futures"}):
        # 🔴범위는 자료가 있는 구간으로 좁힌다. okx_tape_footprint 는 [lo_sec, t0_ms//1000] 의 «모든 분»을 집합에 넣는다
        #   (coin_flow.py:175-178) -- 2026-11-20 처음엔 lo=0·hi=2**40 초를 넘겨 1.8e10 분을 돌다 13.7GB 로 개발기 WSL 을 OOM 으로 내렸다.
        #   단위: lo_sec·hi 는 초, t0_ms 는 밀리초(footprint_tape_restore 의 실제 호출 coin_flow.py:546-549 과 같다).
        mn, mx = data_store.read_rows(db, "SELECT min(ts_sec), max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [sym])[0]
        lo, hi = mn // 300 * 300, mx + 1
        assert hi - lo < 86_400, (sym, lo, hi)
        bucket = tc.BUCKETS.get(sym, 0.01)
        out[f"read_rows:{sym}"] = data_store.read_rows(
            db, "SELECT * FROM trade_tape_1s WHERE symbol = ? ORDER BY ts_sec, price_bin", [sym])
        out[f"gaps:{sym}"] = data_store.read_rows(db, "SELECT from_ms, to_ms, reason FROM gaps WHERE symbol = ? "
                                                      "ORDER BY rowid", [sym])
        out[f"footprint_5m:{sym}"] = data_store.read_rows(db, "SELECT * FROM footprint_5m WHERE symbol = ? "
                                                              "ORDER BY bar_start, price_bin_5", [sym])
        secs, top = cf.tape_seconds(db, sym, lo, hi, bucket)
        out[f"tape_seconds:{sym}"] = [sorted(secs.items()), top]
        fp = cf.okx_tape_footprint(db, d / "_no_ctx.duckdb", sym, lo, hi * 1000, 300, 0.5, bucket)
        out[f"okx_tape_footprint:{sym}"] = [sorted((b, sorted(c.items())) for b, c in fp["bars"].items()),
                                            fp["tape_max"], fp["ctx_ok"]]
    spot = d / DBS["spot"]
    out["read_rows:spot"] = data_store.read_rows(spot, "SELECT * FROM trade_tape_1s ORDER BY 1, 2, 3")
    out["integrity"] = [data_store.read_rows(d / n, "PRAGMA integrity_check") for n in sorted(set(DBS.values()))]
    return out


def cargo_replay(inp: Path, out: Path, *extra: str) -> str:
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["cargo", "run", "-q", "-j", "2", "--release", "--example", "replay", "--", str(inp), str(out), *extra],
                       cwd=ROOT / "rtcollect", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo replay 실패:\n{r.stderr}")
    return r.stdout


def check_readers() -> None:
    runs = json.loads((FIX / "tape_input.json").read_text())["runs"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        python_reference(runs, td / "py")
        close_shared()
        cargo_replay(FIX / "tape_input.json", td / "rs")
        a, b = readers(td / "py", runs), readers(td / "rs", runs)
        assert a.keys() == b.keys()
        bad = [k for k in a if a[k] != b[k]]
        for k in a:
            n = len(a[k]) if isinstance(a[k], list) else 1
            print(f"  {'같음' if k not in bad else '다름'}  {k}: {n}")
        assert not bad, bad
        assert a["integrity"] == [[("ok",)], [("ok",)]], a["integrity"]

        # 섞어 쓰기: 파이썬이 앞 실행들을 쓰고 연결을 **열어 둔 채**(WAL 미정리) Rust 가 나머지 실행을 이어 쓴다
        #   (⑤ ETH 재시작이 파이썬이 쓴 행에서 last_ts_ms 를 읽는다) -> 전부 파이썬으로 만든 것과 덤프가 같아야 한다.
        k = 4
        python_reference(runs[:k], td / "mix")
        wal = (td / "mix" / DBS["futures"]).with_name(DBS["futures"] + "-wal")
        assert wal.exists(), "파이썬 연결이 열린 채라 WAL 파일이 있어야 한다"
        cargo_replay(FIX / "tape_input.json", td / "mix", "--runs", f"{k}:")
        same = dump_dir(td / "mix") == dump_dir(td / "py")
        print(f"  {'같음' if same else '다름'}  섞어 쓰기(파이썬 {k}판 + 연결 열린 채 Rust {len(runs) - k}판) vs 전부 파이썬")
        assert same
        close_shared()
    print("check-readers OK")


# ── 속도 ─────────────────────────────────────────────────────────────────────
def bench(n: int) -> None:
    import statistics
    scratch = Path(os.environ.get("RTCOLLECT_SCRATCH") or tempfile.gettempdir()) / "rtcollect_bench"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    runs = gen_bench(n)
    inp = scratch / "bench_input.json"
    inp.write_text(json.dumps({"runs": runs}, separators=(",", ":")))
    msgs = sum(1 for e in runs[0]["events"] if e[1] == "t")
    timing: dict = {}
    c0, w0 = time.process_time(), time.perf_counter()
    python_reference(runs, scratch / "py", timing)
    c1, w1 = time.process_time(), time.perf_counter()
    close_shared()
    ws = timing["writes"]
    wt = [t for _n, t in ws if _n]
    print(f"python: 메시지 {msgs:,} · 벽 {w1 - w0:.3f}s · CPU {c1 - c0:.3f}s · 메시지당 CPU {(c1 - c0) / msgs * 1e6:.2f}µs · "
          f"쓰기 {len(wt)}회 중앙 {statistics.median(wt) * 1e3:.2f}ms 최대 {max(wt) * 1e3:.2f}ms · "
          f"쓰기 제외 메시지당 벽 {(w1 - w0 - sum(t for _n, t in ws)) / msgs * 1e6:.2f}µs")
    print("rust:  " + cargo_replay(inp, scratch / "rs", "--bench").strip())
    same = dump_dir(scratch / "py") == dump_dir(scratch / "rs")
    print(f"부하 입력 결과 동일: {same}")
    assert same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-readers", action="store_true")
    ap.add_argument("--bench", type=int)
    a = ap.parse_args()
    if a.check_readers:
        return check_readers()
    if a.bench:
        return bench(a.bench)
    runs = gen_runs()
    with tempfile.TemporaryDirectory() as td:
        python_reference(runs, Path(td))
        close_shared()
        golden = dump_dir(Path(td))
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "tape_input.json").write_text(json.dumps({"seed": SEED, "runs": runs}, separators=(",", ":"),
                                                    ensure_ascii=False))
    (FIX / "tape_golden.json").write_text(json.dumps(golden, separators=(",", ":"), ensure_ascii=False))
    n_ev = sum(len(r["events"]) for r in runs)
    print(f"실행 {len(runs)} · 사건 {n_ev:,} · " + " · ".join(
        f"{db} {t} {len(v)}행" for db, g in golden.items() for t, v in g["tables"].items() if v))


if __name__ == "__main__":
    main()

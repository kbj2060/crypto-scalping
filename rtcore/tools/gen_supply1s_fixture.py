#!/usr/bin/env python3
"""rtcore 1초 수급 일치 시험의 정답지를 만든다 (2026-11-20).

같은 합성 `@trade` 열을 **파이썬 원본 수집 함수**(dashboard/coin_flow.py 의 collect_footprint 를 make_coin_flow
클로저에서 꺼내 그대로 돌림 -- 파싱·가격 0 건너뛰기·주문 되묶기·재연결 리셋·footprint_add 전부 원본)에 먹여
`footprint_state["sec"]` 를 중간중간 떠 두고, 끝에 supply_1s_payload 의 바이낸스 선물 몫(seconds·partial.bn·now)을 만든다.
rtcore/tests/supply1s_parity.rs 가 같은 입력을 Rust 누적기에 넣어 대조한다.

네트워크 없음: 소켓 connect 를 막고, aiohttp ClientSession 은 가짜로 바꾸고, REST 백필(fetch)은 바로 예외를 낸다.
dashboard/server.py 는 import 하지 않는다 -- 그래서 payload 조립 몇 줄은 아래 payload_bn() 에 **옮겨 적었다**(원본 위치 주석).
`.env` 를 읽지 않는다(coin_flow import 경로에 dotenv 없음 -- 2026-11-20 확인).

    python3 rtcore/tools/gen_supply1s_fixture.py      # 저장소 루트에서. rtcore/tests/fixtures/supply1s_eth.json 을 쓴다(약 6초: 재연결 대기 3초 x2)
    python3 rtcore/tools/gen_supply1s_fixture.py xrp  # 칸 폭이 작은 종목(btc = 가격이 큰 종목) -- tests/fixtures/gen/ (커밋 안 함, 시험 전에 만든다)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
# 심볼별 합성 장(2026-11-20 XRP·BTC 추가 -- ETH 칸 폭 0.5·가격 소수 2자리에서는 «응답 가격 반올림» 변형이 값을 안 바꿔 못 잡혔다).
# (중간가, 가격 소수 자리, 수량 배율, 수량 소수 자리, 경계 체결 3건 [(가격, 수량, 매도)] = 명목이 고래 하한 · 리테일 상한과 정확히 같음 · 바로 밑)
# 가격 움직임은 ETH 를 칸 폭 비율(PX = 칸 폭 / 0.5)로 줄이고 늘린다. ETH 는 PX = 1.0 · 수량 배율 1.0 이라 예전 정답지와 바이트까지 같다.
MARKETS = {
    "eth": (3000.0, 2, 1.0, 3, [("2500.00", "40.000", False), ("2500.00", "4.000", True), ("2500.00", "3.999", False)]),
    "xrp": (2.1, 4, 330.0, 1, [("2.0000", "11500.0", False), ("2.0000", "1700.0", True), ("2.0000", "1699.9", False)]),
    "btc": (90000.0, 1, 0.05, 3, [("80000.0", "2.000", False), ("104000.0", "0.125", True), ("104000.0", "0.124", False)]),
}
# ponytail: 코인은 import 할 때 argv 로 정한다 -- gen_footprint_fixture 가 같은 인자로 이 모듈을 가져다 쓴다.
COIN = sys.argv[1] if len(sys.argv) > 1 else "eth"
MID, PD, QS, QD, EDGES = MARKETS[COIN]
OUT = ROOT / "rtcore" / "tests" / "fixtures" / ("supply1s_eth.json" if COIN == "eth" else f"gen/supply1s_{COIN}.json")


def pfmt(x: float) -> str:
    return f"{x:.{PD}f}"


def qfmt(q: float) -> str:
    return f"{max(q * QS, 10 ** -QD):.{QD}f}"


def _no_net(*_a, **_k):
    raise RuntimeError("gen_supply1s_fixture: 네트워크 금지")


socket.socket.connect = _no_net          # 무엇이 접속을 시도하면 곧바로 터진다
socket.create_connection = _no_net
sys.path.insert(0, str(ROOT))
import dashboard.coin_flow as cf  # noqa: E402
from aiohttp import WSMsgType  # noqa: E402
from scripts.live_trade_tape_collector_20260916 import BUCKETS, size_bands  # noqa: E402

SEED = 20261120
SYM = f"{COIN.upper()}USDT"
PX = cf.FLOW_SPECS[COIN].bucket / 0.5


def _no_tape(*_a, **_k):
    raise RuntimeError("gen_supply1s_fixture: 체결 테이프 SQLite 읽기 금지")


cf.okx_tape_footprint = cf.tape_seconds = _no_tape   # ETH 외 코인은 재연결 때 REST 대신 테이프로 메운다 -- 개발기 파일을 읽지 않게
T0 = 1_763_600_000_123        # 2025-11-20 무렵(ms). 합성 -- 실제 장과 무관
N_SEC = 900                   # 660 초 보관 정리를 건드리려면 660 보다 길어야 한다
CONN_AT = (300, 600)          # 이 초에서 재연결(열린 주문 묶음은 버려진다)


# ── 합성 체결 열 ─────────────────────────────────────────────────────────────
# 한 줄 = [연결번호, 종류(0 체결 · 1 체결 아닌 메시지), T(ms), t(체결ID), p, q, m, 수신 벽시계(ms)]
def gen() -> list[list]:
    rng = random.Random(SEED)
    out: list[list] = []
    st = {"tid": 50_000_000, "wall": T0, "mid": MID, "conn": 0}

    def put(T: int, p: str, q: str, m: bool, *, late: bool = False, skip_tid: int = 0) -> None:
        st["tid"] += 1 + skip_tid
        w = st["wall"] if late else max(st["wall"], T)
        w += rng.randint(1, 40)
        if w % 1000 == 0:      # 파이썬 time.time()*1000 의 float 내림이 초 경계를 바꾸지 않게
            w += 1
        st["wall"] = w
        out.append([st["conn"], 0, T, st["tid"], p, q, m, w])

    def fills(T: int, px: float, side: bool, qtys: list[float], gaps: list[int]) -> int:
        for i, q in enumerate(qtys):
            if i:
                T += gaps[i - 1]
            put(T, pfmt(px), qfmt(q), side)
        return T

    for s in range(N_SEC):
        if s in CONN_AT:
            st["conn"] += 1
        base = T0 // 1000 * 1000 + s * 1000
        st["mid"] += rng.gauss(0, 0.4) * PX
        mid = round(st["mid"], PD)
        if s == 50:      # 초 경계를 걸친 같은 가격·방향·연속ID 주문 → 초에서 잘린다
            fills(base + 995, mid, False, [2.0, 3.0], [4]); fills(base + 1002, mid, False, [40.0, 1.0], [8])
            continue
        if s == 51:
            continue     # (50 에서 다음 초까지 썼다)
        if s == 60:      # 가격 0 체결이 묶음 중간에 -- 체결ID 가 하나 빠진 셈이라 묶음이 끊긴다
            put(base + 10, pfmt(mid), qfmt(20.0), True)
            put(base + 11, "0", "0", False)
            put(base + 12, pfmt(mid), qfmt(20.0), True)
            continue
        if s == 70:      # 체결ID 건너뜀(가격 0 없이)
            put(base + 5, pfmt(mid), qfmt(18.0), False); put(base + 6, pfmt(mid), qfmt(18.0), False, skip_tid=1)
            continue
        if s == 80:      # 경계값(ETH): $100,000 = 고래(>=) · $10,000 = 중형(< 아님) · $9,999 = 리테일 -- 다른 코인은 MARKETS 경계 체결
            for ms, e in zip((100, 400, 700), EDGES):
                put(base + ms, *e)
            put(base + 900, pfmt(mid), qfmt(0.01), True)
            continue
        if s == 90:      # 같은 가격, 방향만 번갈아 -- 묶이지 않는다
            for i in range(6):
                put(base + 20 + i, pfmt(mid), qfmt(7.5), i % 2 == 0)
            continue
        if s == 100:     # 뒤늦게 온 과거 시각: 같은 초 앞 ms · 3초 전(이미 닫힌 초) · 그다음 정상
            put(base + 500, pfmt(mid), qfmt(1.0), False)
            put(base + 200, pfmt(mid), qfmt(2.0), True, late=True)
            put(base - 3000 + 400, pfmt(mid + 0.5 * PX), qfmt(50.0), False, late=True)
            put(base + 600, pfmt(mid), qfmt(0.5), True)
            continue
        if s == 200:     # 700초 전 체결(보관 660초 밖 -- 파이썬은 칸을 만들고 다음 새 초에 지운다)
            put(base - 700_000 + 300, pfmt(mid), qfmt(5.0), False, late=True)
        if s == 400:     # 25시간 전 체결 -- 24시간 봉 창 밖이라 통째로 버려진다
            put(base - 25 * 3600_000 + 300, pfmt(mid), qfmt(60.0), True, late=True)
        if s == 450:     # 급증 초: 큰 주문 하나가 같은 가격 체결 수백 건으로 쪼개짐
            fills(base + 1, mid, True, [0.4] * 300, [rng.randint(0, 2) for _ in range(299)])
            continue
        if s == 299 or s == 599:   # 연결 끝에 열린 채 남는 고래 주문 묶음(재연결 리셋이 버린다)
            fills(base + 980, mid, False, [30.0, 30.0], [1])
            continue
        if rng.random() < 0.08:
            continue     # 빈 초
        if rng.random() < 0.05:
            out.append([st["conn"], 1, base + 1, 0, "", "", False, st["wall"] + 1])   # 체결 아닌 메시지
        ms = rng.randint(0, 30)
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.01:   # 바이낸스가 섞어 보내는 {"p":"0","q":"0","X":"NA"}(2026-09-16 실측 0.3%)
                put(base + ms, "0", "0", False)
            kind = rng.random()
            n = rng.randint(1, 3)
            if kind < 0.70:
                qtys = [rng.uniform(0.001, 1.2) for _ in range(n)]
            elif kind < 0.92:
                qtys = [rng.uniform(1.0, 9.0) for _ in range(n)]
            else:
                n = rng.randint(3, 10)
                qtys = [rng.uniform(3.0, 12.0) for _ in range(n)]
            gaps = [rng.choice([0, 0, 1, 3, 12, 40, 100, 101, 150]) for _ in range(n - 1)]
            if ms + sum(gaps) > 990:
                break
            px = round(mid + rng.choice([-0.02, -0.01, 0.0, 0.01, 0.02]) * PX * rng.randint(0, 3), PD)
            ms = fills(base + ms, px, rng.random() < 0.5, qtys, gaps) - base + rng.choice([0, 1, 5, 30, 120])
            if ms > 990:
                break
    return out


def text(row: list) -> str:
    _c, kind, T, tid, p, q, m, w = row
    if kind == 1:
        return json.dumps({"e": "markPriceUpdate", "E": T, "s": SYM})
    d = {"e": "trade", "E": w, "T": T, "s": SYM, "t": tid, "p": p, "q": q, "X": "MARKET", "m": m}
    if p == "0":
        d["X"] = "NA"
    return json.dumps(d)


# ── dashboard/server.py supply_1s_payload 의 바이낸스 선물 몫 -- 옮겨 적음(server.py 4248~4251 · 4290~4301 · 4335~4357, 2026-11-20 판) ──
def payload_bn(by_sec: dict, since: int = 0) -> dict:
    SUPPLY_1S_SECONDS = cf.SUPPLY_1S_SECONDS  # noqa: N806
    _part = lambda d, x: [x] + [round(v, 3) for v in d[x]] if x in d else None  # noqa: E731
    partial = {"bn": _part(by_sec, max(by_sec)) if by_sec else None}
    if not by_sec:
        return {"now": 0, "seconds": [], "partial": partial}
    newest = max(by_sec)
    floor = max(since, newest - SUPPLY_1S_SECONDS)
    return {"now": newest, "partial": partial,
            "seconds": [[s] + [round(x, 3) for x in by_sec[s]] for s in sorted(by_sec) if floor < s < newest]}


async def run(rows: list[list]) -> dict:
    clock = {"now": T0 / 1000}
    cf.time = SimpleNamespace(time=lambda: clock["now"])          # footprint_add 의 24시간 봉 창 = 수신 시각

    async def no_rest(*_a, **_k):
        raise RuntimeError("REST 금지(시험)")   # 재연결 백필(ETH aggTrades)은 실패로 끝나고 ready=False -> 스냅샷 저장도 안 한다
    # 스냅샷 경로는 없는 곳 -- footprint_load 가 아무것도 못 읽게(저장은 ready=False 라 안 일어난다)
    ns = cf.make_coin_flow(dataclasses.replace(cf.FLOW_SPECS[COIN],
                                               snapshot_path=Path("/nonexistent/rtcore_fixture_footprint.json")),
                           no_rest, {})
    free = dict(zip(ns.start.__code__.co_freevars, ns.start.__closure__))
    collect_footprint = free["collect_footprint"].cell_contents
    by_sec = ns.footprint_state["sec"]

    conns: list[list[int]] = []
    for i, r in enumerate(rows):
        if r[0] >= len(conns):
            conns.append([])
        conns[r[0]].append(i)
    cps_at = {len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4}
    # 아주 늦은 체결(700초·25시간 전) 바로 뒤 -- 그 칸은 다음 새 초에 지워지므로 여기서 떠야 «버림/만듦» 규칙이 보인다
    cps_at |= {i + 1 for i, r in enumerate(rows) if r[1] == 0 and r[7] - r[2] > 600_000}
    cps: list[dict] = []
    done = asyncio.Event()
    snap = lambda: {str(k): list(v) for k, v in sorted(by_sec.items())}   # noqa: E731

    class FakeWS:
        def __init__(self, idx: list[int], last: bool):
            self.idx, self.k, self.last = idx, 0, last

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.k and self.idx[self.k - 1] + 1 in cps_at:
                cps.append({"after": self.idx[self.k - 1] + 1, "sec": snap()})
            if self.k == len(self.idx):
                if self.last:
                    done.set()
                    await asyncio.Event().wait()
                raise StopAsyncIteration
            i = self.idx[self.k]
            self.k += 1
            clock["now"] = rows[i][7] / 1000
            return SimpleNamespace(type=WSMsgType.TEXT, data=text(rows[i]))

    class FakeSession:
        n = 0

        def __init__(self, *a, **k):
            pass

        def ws_connect(self, url, **k):
            assert url.endswith(f"{SYM.lower()}@trade"), url
            c = FakeSession.n
            FakeSession.n += 1
            return FakeWS(conns[c], c == len(conns) - 1)

        async def close(self):
            pass

    cf.ClientSession = FakeSession
    cf.TCPConnector = lambda **k: None
    task = asyncio.create_task(collect_footprint(None))
    await asyncio.wait_for(done.wait(), 60)
    final_sec = snap()
    final = payload_bn(by_sec)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert FakeSession.n == len(conns)
    return {"checkpoints": cps, "final_sec": final_sec, "final": final}


def main() -> None:
    rows = gen()
    got = asyncio.run(run(rows))
    zero = sum(1 for r in rows if r[4] == "0")
    fx = {
        "about": "합성 @trade 열과 dashboard/coin_flow.py 원본이 낸 1초 칸(rtcore/tools/gen_supply1s_fixture.py, 시드 %d)" % SEED,
        "symbol": SYM.lower(),
        "bands": {s: list(size_bands(s)) for s in sorted(BUCKETS)} | {"unknownusdt": list(size_bands("unknownusdt"))},
        "supply_1s_seconds": cf.SUPPLY_1S_SECONDS,
        "rows": rows,
        **got,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx, separators=(",", ":"), ensure_ascii=False))
    print(f"{OUT.relative_to(ROOT)}: 메시지 {len(rows)} (가격 0 {zero}) · 연결 {len(CONN_AT) + 1} · "
          f"중간 대조 {len(got['checkpoints'])} · 끝 칸 {len(got['final_sec'])} · seconds {len(got['final']['seconds'])} · "
          f"{OUT.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()

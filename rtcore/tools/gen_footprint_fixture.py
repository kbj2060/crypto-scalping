#!/usr/bin/env python3
"""rtcore 풋프린트(5분봉 × 가격 칸) 일치 시험의 정답지를 만든다 (2026-11-20).

gen_supply1s_fixture.py 와 같은 방식: 같은 합성 `@trade` 열을 **파이썬 원본 수집 함수**(coin_flow.py collect_footprint 를
make_coin_flow 클로저에서 꺼내 그대로 -- 파싱·가격 0·되묶기·재연결 리셋·footprint_add·봉 보관 전부 원본)에 먹이고,
중간중간 footprint_state["bars"] 와 1초 링을 떠 둔다. rtcore/tests/footprint_parity.rs 가 같은 입력을 Rust 에 넣어 대조한다.

server.py 는 import 하지 않는다 -- /api/footprint 조립(api_footprint)과 dhl(fp_dhl · bar_delta_hl)을 아래에 **옮겨 적었다**(원본 위치 주석).
fp_dhl 의 테이프 가지(hot SQLite 를 읽는 쪽)는 «수집기 테이프 = 같은 체결의 1초 합»으로 흉내 낸다(tape_update) -- 원본 SQL 의
sum() 은 SQLite 가 보정 합산(Kahan-Babuska-Neumaier)이라 끝자리가 다를 수 있다(round(x, 3) 뒤엔 거의 안 갈린다).
네트워크 없음(gen_supply1s_fixture 가 소켓 connect 를 막는다) · .env 안 읽음 · REST 백필은 바로 예외(ready=False).

    python3 rtcore/tools/gen_footprint_fixture.py     # 저장소 루트에서. rtcore/tests/fixtures/footprint_eth.json (약 10초: 재연결 대기 3초 x3)
    python3 rtcore/tools/gen_footprint_fixture.py xrp # 칸 폭이 작은 종목(btc = 가격이 큰 종목) -- tests/fixtures/gen/ (커밋 안 함, 시험 전에 만든다)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_supply1s_fixture import COIN, EDGES, MID, PD, PX, ROOT, SYM, T0, cf, pfmt, qfmt, text  # noqa: E402 -- 이 import 가 네트워크·테이프 읽기를 막는다
from aiohttp import WSMsgType  # noqa: E402

OUT = ROOT / "rtcore" / "tests" / "fixtures" / ("footprint_eth.json" if COIN == "eth" else f"gen/footprint_{COIN}.json")
SEED = 20261121
BS = cf.FOOTPRINT_BAR_SECONDS
WANT = cf.FOOTPRINT_MAX_WINDOW_BARS      # 화면이 묻는 bars=144(app.js CHART_PAN_BARS)
B0 = T0 // 1000 // BS * BS                # 첫 봉(초)


# ── 합성 체결 열 ─────────────────────────────────────────────────────────────
# 한 줄 = [연결번호, 종류(0 체결 · 1 체결 아닌 메시지), T(ms), t(체결ID), p, q, m, 수신 벽시계(ms)] -- gen_supply1s_fixture 와 같다.
def gen() -> tuple[list[list], set[int]]:
    rng = random.Random(SEED)
    out: list[list] = []
    marks: set[int] = set()                 # 중간 대조를 뜰 자리(특수 사례 바로 뒤)
    st = {"tid": 70_000_000, "wall": T0, "mid": MID, "conn": 0}

    def put(T: int, p: str, q: str, m: bool, *, late: bool = False, skip_tid: int = 0) -> None:
        st["tid"] += 1 + skip_tid
        w = st["wall"] if late else max(st["wall"], T)
        w += rng.randint(1, 40)
        if w % 1000 == 0:
            w += 1
        st["wall"] = w
        out.append([st["conn"], 0, T, st["tid"], p, q, m, w])

    def fills(T: int, px: float, side: bool, qtys: list[float], gaps: list[int]) -> int:
        for i, q in enumerate(qtys):
            if i:
                T += gaps[i - 1]
            put(T, pfmt(px), qfmt(q), side)
        return T

    def walk() -> float:
        st["mid"] += rng.gauss(0, 0.6) * PX
        return round(st["mid"], PD)

    def sparse_bar(k: int) -> None:
        """봉 하나에 주문 2~4개(드문 체결 -- 봉 수·보관 규칙용)."""
        secs = sorted(rng.sample(range(BS), rng.randint(1, 3)))
        for s in secs:
            mid = walk()
            px = round(mid + rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) * PX * rng.randint(0, 2), PD)
            qtys = [rng.choice([0.05, 0.8, 3.0, 12.0, 40.0]) * rng.uniform(0.5, 1.5) for _ in range(rng.randint(1, 3))]
            fills((B0 + k * BS + s) * 1000 + rng.randint(0, 900), px, rng.random() < 0.5, qtys, [0, 1, 30][:len(qtys) - 1])

    def mark() -> None:
        marks.add(len(out))

    # ① 봉 0~19 드문 체결 → 긴 공백 → 봉 270~ 다시. 봉 288 이 생기는 순간부터 24시간 창이 봉 0~ 을 지운다 --
    #    그때 봉이 적어(38봉) 지워지는 봉이 화면 창(144봉) 안이다(밀기 drop).
    for k in range(20):
        sparse_bar(k)
    for k in range(270, 560):
        if k == 400:
            st["conn"] += 1                  # 재연결
        if k == 450:
            continue                         # 빈 봉 -- 470 에서 늦은 체결이 창 한가운데에 이 봉을 만든다
        sparse_bar(k)
        if k == 288:
            mark()
        if k == 470:   # 늦게 온 과거 체결: 창 안의 빈 봉(450)을 새로 만든다 → 창이 한 봉 밀린다
            put((B0 + 450 * BS + 100) * 1000, pfmt(walk()), qfmt(2.0), False, late=True); mark()
        if k == 500:   # 창 밖(144봉 밖)·보관 안(288봉 안) 닫힌 봉(300) -- 칸은 바뀌지만 밀기·스냅숏에는 안 나온다
            put((B0 + 300 * BS + 7) * 1000, pfmt(walk()), qfmt(5.0), True, late=True); mark()
        if k == 520:   # 25시간 전 -- 24시간 봉 창 밖이라 통째로 버린다
            put((B0 + 520 * BS) * 1000 - 25 * 3600_000, pfmt(walk()), qfmt(60.0), True, late=True); mark()

    # ② 봉 560~563 촘촘한 체결(초마다) -- 1초 링(660초)이 봉을 덮었다 놓치는 dhl 규칙용. 특수 사례는 d(초)에.
    d0 = B0 + 560 * BS
    st["conn"] += 1
    for d in range(4 * BS):
        base = (d0 + d) * 1000
        mid = walk()
        if d == 299:   # 봉 경계를 걸친 같은 가격·방향·연속ID 체결: 초가 바뀌어 끊긴다 · 앞 묶음은 앞 봉, 뒤 묶음은 다음 봉
            fills(base + 990, mid, False, [20.0, 20.0], [5]); fills(base + 1001, mid, False, [20.0, 20.0], [3]); mark()
            continue
        if d == 300:
            continue
        if d == 310:   # 가격 칸 경계(반올림 동률): /0.5 가 x.5 -- 파이썬 round 는 짝수 쪽(6000.5→6000 · 6001.5→6002) · BTC 도 동률(/15) · XRP 는 4자리 가격으로 동률이 안 난다
            m0 = round(mid / PX) * PX
            for o in (0.25, 0.75, 0.24, 0.26, -0.25):
                put(base + 100, pfmt(m0 + o * PX), qfmt(1.0), True)
            mark()
            continue
        if d == 320:   # 같은 ms 여러 건(같은 가격·방향 = 한 주문 · 방향 바뀜 = 끊김) + 가격 0 이 가운데
            for i in range(6):
                put(base + 50, pfmt(mid), qfmt(6.0), i == 3)
            put(base + 50, "0", "0", False)
            put(base + 50, pfmt(mid), qfmt(6.0), False)
            mark()
            continue
        if d == 330:   # 고래·리테일 경계값(ETH): $100,000 = 고래(>=) · $10,000 = 중형 · $9,997.5 = 리테일 -- 다른 코인은 MARKETS 경계 체결
            for ms, e in zip((100, 400, 700), EDGES):
                put(base + ms, *e)
            put(base + 900, pfmt(mid), qfmt(0.01), True)
            mark()
            continue
        if d == 340:   # 뒤늦게: 같은 봉 닫힌 초(3초 전) · 앞 봉(560, 아직 링이 덮음)
            put(base + 500, pfmt(mid), qfmt(1.0), False)
            put(base - 3000 + 200, pfmt(mid + 0.5 * PX), qfmt(9.0), False, late=True)
            put((d0 + 250) * 1000 + 300, pfmt(mid), qfmt(30.0), True, late=True)
            mark()
            continue
        if d == 450:   # 큰 주문 하나가 같은 가격 체결 300건으로
            fills(base + 1, mid, True, [0.4] * 300, [rng.randint(0, 2) for _ in range(299)]); mark()
            continue
        if d == 599:   # 연결 끝에 열린 채 남는 고래 묶음 → 재연결 리셋이 버린다
            fills(base + 980, mid, False, [30.0, 30.0], [1])
            st["conn"] += 1
            mark()
            continue
        if d == 900:   # 700초 전 체결(1초 링 660초 밖) -- 파이썬은 그 초 칸을 만들고 다음 새 초에 지운다 · 봉(557)은 링 밖
            put(base - 700_000 + 300, pfmt(mid), qfmt(5.0), False, late=True); mark()
        if d == 1000:  # 링이 놓친 봉(560, dhl 을 얼려 둠)에 늦은 체결 -- 코어는 얼린 값을 지운다(몸통만)
            put((d0 + 100) * 1000 + 5, pfmt(mid), qfmt(3.0), False, late=True); mark()
        if rng.random() < 0.06:
            continue
        if rng.random() < 0.04:
            out.append([st["conn"], 1, base + 1, 0, "", "", False, st["wall"] + 1])
        ms = rng.randint(0, 30)
        for _ in range(rng.randint(1, 2)):
            if rng.random() < 0.01:
                put(base + ms, "0", "0", False)
            kind, n = rng.random(), rng.randint(1, 2)
            if kind < 0.70:
                qtys = [rng.uniform(0.001, 1.2) for _ in range(n)]
            elif kind < 0.92:
                qtys = [rng.uniform(1.0, 9.0) for _ in range(n)]
            else:
                n = rng.randint(3, 8)
                qtys = [rng.uniform(3.0, 12.0) for _ in range(n)]
            gaps = [rng.choice([0, 0, 1, 3, 12, 40, 100, 101, 150]) for _ in range(n - 1)]
            if ms + sum(gaps) > 990:
                break
            px = round(mid + rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]) * PX * rng.randint(0, 4), PD)
            ms = fills(base + ms, px, rng.random() < 0.5, qtys, gaps) - base + rng.choice([0, 1, 5, 30, 120])
            if ms > 990:
                break
    return out, marks


# ── dashboard/server.py 옮겨 적음(2026-11-20 판) ─────────────────────────────
def bar_delta_hl(deltas) -> list[float]:
    """server.py bar_delta_hl(237~243) 그대로."""
    c = lo = hi = 0.0
    for d in deltas:
        c += d
        lo, hi = min(lo, c), max(hi, c)
    return [round(lo, 3), round(hi, 3), round(c, 3)]


def fp_dhl(sec: dict, tape: dict, sent: list, now: float) -> dict[int, list]:
    """server.py fp_dhl(3952~3991). 링 가지는 그대로, 테이프 가지는 tape 흉내(머리 주석) · 테이프 기억(fp_dhl_cache)은 뺐다.
    돌려주는 값: 봉 → [가지, 값 또는 None]. 가지 = ring(링으로 잼) · tape(테이프로 잼) · wait(아직 6분 안 지나 파이썬은 안 냄)."""
    bs, out = BS, {}
    items = sorted(sec.items())
    s0 = items[0][0] if items else None
    cd = {b: sum(v[0] - v[1] for v in cells.values()) for b, cells in sent}
    ok = lambda b, v: v is not None and abs(v[2] - cd[b]) <= max(0.5, 0.01 * abs(cd[b]))   # noqa: E731
    for b, _ in sent:
        if s0 is not None and s0 <= b:
            v = bar_delta_hl([c[4] - c[5] for t, c in items if b <= t < b + bs])
            out[b] = ["ring", v if ok(b, v) else None]
        elif b + bs <= now - 360:
            ts = sorted(t for t in tape if b <= t < b + bs)
            v = bar_delta_hl([tape[t][0] - tape[t][1] for t in ts]) if ts else None
            out[b] = ["tape", v if ok(b, v) else None]
        else:
            out[b] = ["wait", None]
    return out


def payload(fs: dict, tape: dict, now: float) -> dict:
    """server.py api_footprint(3993~4059)의 since=0 · bars=144 몫. dhl 은 [가지, 값] 으로 따로(시험이 가지별로 다르게 본다)."""
    want, bucket, dp = WANT, cf.FLOW_SPECS[COIN].bucket, cf.FLOW_SPECS[COIN].price_dp
    recent = sorted(fs["bars"].items())[-want:]
    dhl = fp_dhl(fs["sec"], tape, recent, now)
    return {
        "bucket": bucket, "barSeconds": BS, "barsExpected": want,
        "bars": [{"time": bar, "levels": [[round(k * bucket, dp)] + [round(x, 3) for x in v] for k, v in sorted(cells.items())],
                  "dhl": dhl[bar]} for bar, cells in recent],
    }


def payload_1m(fs: dict) -> dict:
    """server.py footprint_1m(3997~4016)의 since=0 몫(2026-11-24 옮겨 적음 · ready 는 코어 판이라 뺐다)."""
    bucket, dp = cf.FLOW_SPECS[COIN].bucket, cf.FLOW_SPECS[COIN].price_dp
    floor = fs.get("m1_from") or 0
    bars = sorted((b, c) for b, c in fs["bars1m"].items() if b >= floor and b in fs["ohlc1m"])
    return {"bucket": bucket, "barSeconds": 60, "barsExpected": 120, "m1_from": fs["m1_from"],
            "bars": [{"time": b, "ohlc": [round(x, dp) for x in fs["ohlc1m"][b][:4]],
                      "levels": [[round(k * bucket, dp)] + [round(x, 3) for x in v] for k, v in sorted(c.items())]} for b, c in bars]}


def tape_update(tape: dict, base: dict, sec: dict, prev: set) -> set:
    """수집기 테이프 흉내: 초 → [총매수, 총매도] 를 지우지 않고 든다. 링이 지웠다가 다시 만든 초(아주 늦은 체결)는 이어 더한다."""
    for s, c in sec.items():
        if s not in prev and s in tape:
            base[s] = list(tape[s])
        b = base.get(s)
        tape[s] = [b[0] + c[4], b[1] + c[5]] if b else [c[4], c[5]]
    return set(sec)


async def run(rows: list[list], marks: set[int]) -> dict:
    clock = {"now": T0 / 1000}
    cf.time = SimpleNamespace(time=lambda: clock["now"])

    async def no_rest(*_a, **_k):
        raise RuntimeError("REST 금지(시험)")
    ns = cf.make_coin_flow(dataclasses.replace(cf.FLOW_SPECS[COIN],
                                               snapshot_path=Path("/nonexistent/rtcore_fixture_footprint.json")),
                           no_rest, {})
    free = dict(zip(ns.start.__code__.co_freevars, ns.start.__closure__))
    collect_footprint = free["collect_footprint"].cell_contents
    fs = ns.footprint_state
    tape: dict[int, list] = {}
    tbase: dict[int, list] = {}
    prev: set[int] = set()

    conns: list[list[int]] = []
    for i, r in enumerate(rows):
        while r[0] >= len(conns):
            conns.append([])
        conns[r[0]].append(i)
    cps_at = marks | {len(rows) * j // 10 for j in range(1, 10)}
    cps: list[dict] = []
    seen: dict[int, dict] = {}          # 지난 대조 때 봉 칸 -- 그 뒤 바뀐 봉만 싣는다(정답지 크기)
    seen1: dict[int, dict] = {}         # 1분봉 몫(2026-11-24)
    done = asyncio.Event()

    def checkpoint(after: int) -> None:
        bars = {b: {k: list(v) for k, v in cells.items()} for b, cells in fs["bars"].items()}
        changed = {str(b): {str(k): v for k, v in sorted(c.items())} for b, c in sorted(bars.items()) if seen.get(b) != c}
        seen.clear()
        seen.update(bars)
        b1 = {b: {k: list(v) for k, v in cells.items()} for b, cells in fs["bars1m"].items()}
        changed1 = {str(b): {str(k): v for k, v in sorted(c.items())} for b, c in sorted(b1.items()) if seen1.get(b) != c}
        seen1.clear()
        seen1.update(b1)
        sec = fs["sec"]
        top = max(sec) if sec else 0
        win = min((s for s in sec if s >= top - cf.SUPPLY_1S_SECONDS), default=None)
        recent = sorted(fs["bars"].items())[-WANT:]
        cps.append({"after": after, "now": clock["now"], "bars": sorted(bars), "cells": changed,
                    # 파이썬 덮임 판정(링의 가장 오래된 초)이 창 기준과 다른 순간 -- 시험은 이때 dhl 을 안 본다(footprint.rs «덮임 판정»)
                    "stale": bool(sec) and min(sec) != win,
                    # dhl 은 맨 끝 6봉만(링 덮임이 바뀌는 곳) -- 창 전체는 끝 스냅숏이 본다(정답지 크기)
                    # 1분봉(2026-11-24): 봉 목록 · 바뀐 봉 칸 · OHLC 전부([시, 고, 저, 종, 첫 ms, 끝 ms] 반올림 전)
                    "bars1m": sorted(b1), "cells1m": changed1, "ohlc1m": {str(b): list(v) for b, v in sorted(fs["ohlc1m"].items())},
                    "dhl": [[b, *v] for b, v in sorted(fp_dhl(sec, tape, recent, clock["now"]).items())][-6:]})

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
            nonlocal prev
            if self.k:
                prev = tape_update(tape, tbase, fs["sec"], prev)
                if self.idx[self.k - 1] + 1 in cps_at:
                    checkpoint(self.idx[self.k - 1] + 1)
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
    final_cells = {str(b): {str(k): list(v) for k, v in sorted(c.items())} for b, c in sorted(fs["bars"].items())}
    final = payload(fs, tape, clock["now"])
    final_1m = payload_1m(fs)
    final_ohlc1m = {str(b): v for b, v in sorted(fs["ohlc1m"].items())}
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert FakeSession.n == len(conns)
    # ETH 외 코인은 테이프 복원이 실패해도 ready 가 된다(footprint_tape_restore 끝) -- 값에는 안 닿는다(스냅샷 저장만 · 없는 경로라 실패)
    assert COIN != "eth" or not fs["ready"], "REST 가 막혔는데 ready -- 정답지 전제가 깨졌다"
    return {"checkpoints": cps, "final_cells": final_cells, "final": final, "final_ms": rows[-1][7], "final_1m": final_1m, "final_ohlc1m": final_ohlc1m}


def main() -> None:
    rows, marks = gen()
    got = asyncio.run(run(rows, marks))
    fx = {
        "about": "합성 @trade 열과 dashboard/coin_flow.py 원본이 낸 풋프린트 봉·칸(rtcore/tools/gen_footprint_fixture.py, 시드 %d)" % SEED,
        "symbol": SYM.lower(),
        "specs": {s.symbol.lower(): [s.bucket, s.price_dp] for s in cf.FLOW_SPECS.values()},
        "keep_bars": cf.FOOTPRINT_KEEP_BARS, "window_bars": WANT, "bar_seconds": BS,
        # 시각은 T0 를 뺀 ms(정답지 크기) -- 시험이 더한다
        "t0": T0, "rows": [[c, k, T - T0, tid, p, q, m, w - T0] for c, k, T, tid, p, q, m, w in rows],
        **got,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx, separators=(",", ":"), ensure_ascii=False))
    kinds: dict[str, int] = {}
    for b in got["final"]["bars"]:
        kinds[b["dhl"][0] + ("" if b["dhl"][1] else "·없음")] = kinds.get(b["dhl"][0] + ("" if b["dhl"][1] else "·없음"), 0) + 1
    print(f"{OUT.relative_to(ROOT)}: 메시지 {len(rows)} · 연결 {rows[-1][0] + 1} · 중간 대조 {len(got['checkpoints'])} · "
          f"끝 봉 {len(got['final_cells'])} · 창 {len(got['final']['bars'])}봉 · dhl {kinds} · {OUT.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()

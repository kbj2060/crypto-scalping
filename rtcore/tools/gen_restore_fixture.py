#!/usr/bin/env python3
"""rtcore 재시작 복원 일치 시험의 입력·정답지를 만든다 (2026-11-21).

같은 합성 `@trade` 열(SOL -- 테이프 칸 0.01 → 풋프린트 칸 0.02 라 홀수 센트가 반올림 동률: 2026-10-20 칸 키 사고 자리)로
① 체결 테이프 SQLite 를 **원본 수집기** `scripts/live_trade_tape_collector_20260916.py::collect()` 로 만든다
   (rtcollect/tools/gen_tape_fixture.py python_reference 를 옮겨 적음 -- 가짜 aiohttp · 수신 시각 = 단조 시계 · REST 는 빈 코루틴).
   테이프 세 벌: full · gaps(수집기가 200초 끊김 → gaps 표 · 그 사이 행 없음) · behind(full 사본에서 재시작 20초 앞부터 지움 = 수집기가 덜 따라옴).
② 사례마다 **원본 대시보드 함수**로 정답을 만든다(dashboard/coin_flow.py make_coin_flow 의 클로저를 그대로):
   - 재시작 전: collect_footprint 로 rows[:S] 를 먹이고 stop() 이 쓰는 스냅숏(footprint_save) -- REST 백필이 도는 «eth» 몫(asset) 으로
     만들어 재연결 메우기가 테이프를 안 읽게 한다(REST 는 바로 예외 → 아무것도 안 더함 = 코어의 «메우기 없는 라이브»와 같다).
   - 재시작 뒤: 같은 SOL 사양(테이프 메우기 몫)으로 footprint_load → collect_footprint(첫 체결 → footprint_tape_restore) + supply_1s_restore.
     M 번째 메시지 뒤에 두 복원이 끝나기를 기다려(그 전엔 메시지 루프가 양보하지 않으므로 합치는 자리가 정해진다) 봉·1초 칸·dhl 을 뜬다.
   - dhl = dashboard/server.py fp_dhl 을 옮겨 적은 것(테이프 가지는 원문 SQL 을 data_store.read_rows 로 · 기억(fp_dhl_cache)만 뺌).
rtcore/tests/restore_parity.rs 가 같은 입력으로 코어를 돌려(재시작 전 코어 → 코어 스냅숏 파일 → 새 코어 → 같은 테이프) 대조한다.

네트워크 없음(소켓 connect 를 막는다) · .env 안 읽음 · 실제 data/ 폴더 안 건드림(테이프·스냅숏·DB 경로를 전부 임시 폴더로 바꾼다) ·
원본 읽기 함수(okx_tape_footprint · tape_seconds)는 범위가 하루를 넘으면 멈추는 단언으로 감싼다(2026-11-20 13.7GB 사고).

    python3 rtcore/tools/gen_restore_fixture.py    # 저장소 루트에서. rtcore/tests/fixtures/gen/restore/ (커밋 안 함 -- 시험 전에 만든다 · 약 1분)
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import os
import random
import shutil
import socket
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtcore" / "tests" / "fixtures" / "gen" / "restore"


def _no_net(*_a, **_k):
    raise RuntimeError("gen_restore_fixture: 네트워크 금지")


socket.socket.connect = _no_net
socket.create_connection = _no_net
sys.path.insert(0, str(ROOT))
import dashboard.coin_flow as cf  # noqa: E402
import scripts.live_trade_tape_collector_20260916 as tc  # noqa: E402
from aiohttp import WSMsgType  # noqa: E402
from scripts.data_store import read_rows  # noqa: E402

SPEC = cf.FLOW_SPECS["sol"]
SYM = SPEC.symbol
SYML = SYM.lower()
BS = cf.FOOTPRINT_BAR_SECONDS
WANT = cf.FOOTPRINT_MAX_WINDOW_BARS
T0 = 1_763_596_800_000          # 2025-11-20 00:00 UTC(ms) -- 합성, 실제 장과 무관. 봉 경계에 맞춤
SEED = 20261121
DUR = 27 * 3600
M = 40                          # 재시작 뒤 이 메시지 수만큼 라이브가 쌓인 다음 복원이 합쳐진다
END_AFTER = 420                 # 끝 대조 = 재시작 + 420초(1초 링에 되살린 초가 아직 240초 남은 때)
DAY = 86_400

# 파이썬 정답 한 벌 = (스냅숏 시각 S, 재시작 첫 체결 시각 R, 테이프, 스냅숏 씀). 시각 = T0 부터 초(소수 = ms).
# S 의 ms 600 바로 앞에 고래 묶음 두 체결(560·580ms) → 스냅숏 때 열려 있는 주문 · 그 뒤 700·850ms 체결 = 스냅숏 뒤에 센 «경계 초» 몫.
# R 의 ms 400: 390ms 첫 체결과 405ms 둘째 체결이 한 주문(재시작이 가른다) · 100·250ms = 첫 체결 앞이라 잃는 «경계 초» 몫.
RUNS = {
    "short": (50400.6, 50410.4, "full", True),        # 스냅숏 직후 재시작(공백 10초)
    "bar": (52180.6, 52650.4, "full", True),           # 공백이 봉 경계 둘을 넘음
    "long": (54000.6, 56400.4, "full", True),          # 공백 40분(> 11분 -- 1초 칸 전부 테이프)
    "day": (4300.6, 94530.4, "full", True),            # 공백 25시간(스냅숏 봉 전부 창 밖 → 테이프로 창 전체)
    "gaps": (58000.6, 59800.4, "gaps", True),          # 테이프에 수집기 재연결 공백
    "nosnap": (None, 61200.4, "full", False),          # 스냅숏 없음(테이프만 -- 창 144봉 전부)
    "notape": (63000.6, 63120.4, "none", True),        # 테이프 없음(스냅숏만)
    "none": (None, 64800.4, "none", False),            # 둘 다 없음
    "nosnap_long": (None, 56400.4, "full", False),     # 판 번호 다름·깨짐의 정답(= 스냅숏 없음)
    "behind": (66000.6, 66200.4, "behind", True),      # 테이프가 재시작 20초 앞에서 멈춤
}
# 코어 사례 = (이름, 재시작 전 코어를 돌릴 S 의 정답 이름, 파이썬 정답, 코어 스냅숏 처리)
CASES = [
    ("short", "short", "short", "ok"), ("bar", "bar", "bar", "ok"), ("long", "long", "long", "ok"), ("day", "day", "day", "ok"),
    ("gaps", "gaps", "gaps", "ok"), ("nosnap", None, "nosnap", "none"), ("notape", "notape", "notape", "ok"),
    ("none", None, "none", "none"), ("badver", "long", "nosnap_long", "badver"), ("corrupt", "long", "nosnap_long", "corrupt"),
    ("behind", "behind", "behind", "ok"),
]
GAP_T = (58700.0, 58900.0)      # gaps 테이프: 이 사이 체결을 수집기가 못 받음
BEHIND_S = 20


# ── 합성 체결 열 ─────────────────────────────────────────────────────────────
# 한 줄 = [T(ms), 체결ID, p, q, m, 수신 벽시계(ms)] -- 연결 하나(재연결은 기존 정답지들이 본다).
def gen() -> tuple[list[list], dict[str, int]]:
    rng = random.Random(SEED)
    rows: list[list] = []
    marks: dict[str, int] = {}
    st = {"tid": 90_000_000, "wall": T0, "mid": 140.0}
    dense: list[tuple[float, float]] = []
    snaps = {int(v[0]): v[0] for v in RUNS.values() if v[0] is not None}
    rests = {int(v[1]): v[1] for v in RUNS.values()}
    for S, R, _t, _u in RUNS.values():
        if S is not None:
            dense.append((S - 700, S + 60))
        dense.append((R - 700, R + 800))

    def put(T: int, p: str, q: str, m: bool, *, late: bool = False) -> None:
        st["tid"] += 1
        w = st["wall"] if late else max(st["wall"], T)
        w += rng.randint(1, 40)
        if w % 1000 == 0:
            w += 1
        st["wall"] = w
        rows.append([T, st["tid"], p, q, m, w])

    pf = lambda x: f"{x:.2f}"                     # noqa: E731
    qf = lambda q: f"{max(q, 0.01):.2f}"          # noqa: E731

    def qty() -> float:
        r = rng.random()
        return rng.uniform(0.05, 3.0) if r < 0.6 else rng.uniform(3.0, 50.0) if r < 0.9 else rng.uniform(50.0, 200.0)

    def order(base: int, ms: int, mid: float) -> int:
        """주문 하나(같은 가격·방향 연속 체결). 5% 는 고래(3~6체결 × 80~150 SOL ≥ $55k)."""
        px = round(mid + rng.choice([-2, -1, 0, 1, 2]) * 0.01 * rng.randint(0, 3), 2)   # 홀수 센트 = 칸 키 동률
        side = rng.random() < 0.5
        whale = rng.random() < 0.05
        n = rng.randint(3, 6) if whale else rng.choice([1, 1, 2, 3])
        for i in range(n):
            if i:
                ms += rng.choice([0, 1, 3, 12, 40])
            put(base + ms, pf(px), qf(rng.uniform(80, 150) if whale else qty()), side)
        return ms

    for s in range(DUR):
        st["mid"] += rng.gauss(0, 0.6) * 0.04
        mid = round(st["mid"], 2)
        base = T0 + s * 1000
        if s in snaps:
            put(base + 100, pf(mid), qf(1.5), True)
            put(base + 300, pf(mid + 0.01), qf(4.0), False)
            p = pf(mid - 0.01)
            put(base + 560, p, "250.00", False)
            put(base + 580, p, "250.00", False)
            marks[f"S{snaps[s]}"] = len(rows)
            put(base + 700, pf(mid + 0.01), qf(1.0), True)
            put(base + 850, pf(mid - 0.02), qf(2.0), False)
            continue
        if s in rests:
            put(base + 100, pf(mid + 0.03), qf(2.5), True)
            put(base + 250, pf(mid - 0.01), qf(0.7), False)
            p = pf(mid)
            put(base + 390, p, "30.00", True)
            marks[f"R{rests[s]}"] = len(rows)
            put(base + 405, p, "30.00", True)
            put(base + 600, pf(mid + 0.01), qf(5.0), False)
            put(base + 800, pf(mid - 0.01), qf(1.0), True)
            continue
        if s in (20000, 30000, 40000):     # 늦게 온 체결(3초 · 700초 전) -- 수집기는 이미 쓴 초면 버린다(dropped_late)
            put(base + 500, pf(mid), qf(2.0), False)
            put(base - 3000 + 200, pf(mid + 0.03), qf(7.0), True, late=True)
            put(base - 700_000 + 300, pf(mid), qf(3.0), False, late=True)
            continue
        if any(a <= s < b for a, b in dense):
            ms = rng.randint(0, 100)
            for _ in range(rng.randint(1, 2)):
                ms = order(base, ms, mid) + rng.randint(5, 300)
                if ms > 950:
                    break
        elif rng.random() < 1 / 25:
            order(base, rng.randint(0, 900), mid)
    return rows, marks


def text(r: list) -> str:
    T, tid, p, q, m, w = r
    return json.dumps({"e": "trade", "E": w, "T": T, "s": SYM, "t": tid, "p": p, "q": q, "X": "MARKET", "m": m})


# ── 체결 테이프: 원본 collect() 를 가짜 웹소켓으로(rtcollect/tools/gen_tape_fixture.py python_reference 옮겨 적음) ──
class _MT(enum.Enum):
    TEXT = 1
    CLOSED = 8


def build_tape(rows: list[list], path: Path, drop: tuple[int, int] | None = None) -> None:
    ev: list[list] = []
    closed = False
    for r in rows:
        if drop and drop[0] <= r[0] < drop[1]:
            if not closed:
                ev.append([ev[-1][0], "c", ""])    # 끊김 → 수집기는 쓰고 3초 쉬고(여기선 0) 다시 붙는다 → 첫 체결에 gaps 한 줄
                closed = True
            continue
        ev.append([r[5] / 1000, "t", text(r)])
    clock = [ev[0][0] - 0.5]
    real_time = tc.time
    tc.time = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("__")})
    tc.time.monotonic = lambda: clock[0]

    async def _fast_sleep(_s, *a, **k):
        await asyncio.sleep(0)
    real_asyncio = tc.asyncio
    tc.asyncio = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
    tc.asyncio.sleep = _fast_sleep

    async def _noop(*_a, **_k):
        return None
    tc.verify_recent = _noop          # REST(1분봉 대조) -- 범위 밖
    tc.backfill_loop = _noop          # REST(aggTrades 복구) -- 범위 밖
    state = {"i": 0}

    class FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            i = state["i"]
            if i >= len(ev):
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
            if state["i"] >= len(ev):
                raise asyncio.CancelledError()      # 입력 끝 = 수집 끝
            return FakeWS()

    real_aiohttp = sys.modules.get("aiohttp")
    sys.modules["aiohttp"] = types.SimpleNamespace(ClientSession=FakeSession, ClientTimeout=lambda **k: None, WSMsgType=_MT)
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        asyncio.run(tc.collect(SYML, path, "futures"))
    except asyncio.CancelledError:
        pass
    finally:
        sys.stdout.close()
        sys.stdout = old
        sys.modules["aiohttp"] = real_aiohttp
        tc.time, tc.asyncio = real_time, real_asyncio
    for ent in tc._SQLITE.values():       # 붙든 쓰기 연결을 닫아 WAL 을 본 파일로 넣는다
        if ent[1] is not None:
            ent[1].close()
            ent[1] = None


# ── dashboard/server.py 옮겨 적음(2026-11-20 판) ─────────────────────────────
def bar_delta_hl(deltas) -> list[float]:
    """server.py bar_delta_hl(237~243) 그대로."""
    c = lo = hi = 0.0
    for d in deltas:
        c += d
        lo, hi = min(lo, c), max(hi, c)
    return [round(lo, 3), round(hi, 3), round(c, 3)]


def fp_dhl(fs: dict, sent: list, tape: Path, now: float) -> dict[int, list]:
    """server.py fp_dhl(3952~3991) 동기판 -- 테이프 기억(fp_dhl_cache)만 뺐다(한 번 부르면 같다). 봉 → [가지, 값 또는 None]."""
    sec, bs, out, need = fs["sec"], BS, {}, []
    items = sorted(sec.items())
    s0 = items[0][0] if items else None
    cd = {b: sum(v[0] - v[1] for v in cells.values()) for b, cells in sent}
    ok = lambda b, v: v is not None and abs(v[2] - cd[b]) <= max(0.5, 0.01 * abs(cd[b]))   # noqa: E731
    for b, _ in sent:
        if s0 is not None and s0 <= b:
            v = bar_delta_hl([c[4] - c[5] for t, c in items if b <= t < b + bs])
            out[b] = ["ring", v if ok(b, v) else None]
        elif b + bs <= now - 360:
            need.append(b)
        else:
            out[b] = ["wait", None]
    if need:
        assert max(need) + bs - min(need) <= DAY, need
        try:
            rows = read_rows(tape,
                "WITH s AS (SELECT ts_sec - ts_sec % ? AS bar, ts_sec, sum(buy_qty) - sum(sell_qty) AS d FROM trade_tape_1s"
                " WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY ts_sec),"
                " c AS (SELECT bar, d, sum(d) OVER (PARTITION BY bar ORDER BY ts_sec) AS cum FROM s)"
                " SELECT bar, min(cum), max(cum), sum(d) FROM c GROUP BY bar",
                [bs, SYML, min(need), max(need) + bs])
        except Exception:  # noqa: BLE001 -- 원문과 같다(못 읽으면 몸통만)
            rows = None
        got = {int(b): [round(min(0.0, float(lo or 0.0)), 3), round(max(0.0, float(hi or 0.0)), 3), round(float(c or 0.0), 3)]
               for b, lo, hi, c in rows or []}
        for b in need:
            out[b] = ["tape", got[b] if ok(b, got.get(b)) else None]
    return out


# ── 원본 읽기 함수: 범위 단언(2026-11-20 okx_tape_footprint 0~2^40초 → 13.7GB 사고) ──
_orig_otf, _orig_ts = cf.okx_tape_footprint, cf.tape_seconds


def _otf(tape_db, ctx_db, inst, lo_sec, t0_ms, *a, **k):
    assert 0 <= t0_ms // 1000 - lo_sec <= DAY, (lo_sec, t0_ms)
    return _orig_otf(tape_db, ctx_db, inst, lo_sec, t0_ms, *a, **k)


def _ts(tape_db, symbol, lo_sec, hi_sec, tape_bucket):
    assert 0 <= hi_sec - lo_sec <= DAY, (lo_sec, hi_sec)
    return _orig_ts(tape_db, symbol, lo_sec, hi_sec, tape_bucket)


cf.okx_tape_footprint, cf.tape_seconds = _otf, _ts
_real_sleep = asyncio.sleep


async def _fast(_s, *a, **k):
    await _real_sleep(0)


cf.asyncio = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
cf.asyncio.sleep = _fast            # 복원의 5초·1초 대기·재연결 3초를 0 으로(횟수는 원본 그대로)


async def no_rest(*_a, **_k):
    raise RuntimeError("REST 금지(시험)")


def free(fn) -> dict:
    return {k: c.cell_contents for k, c in zip(fn.__code__.co_freevars, fn.__closure__)}


def checkpoint(fs: dict, tape: Path, now: float) -> dict:
    recent = sorted(fs["bars"].items())[-WANT:]
    return {"now": now,
            # list(v): 칸은 라이브가 계속 고치는 리스트다 -- 떠 둔 값이 끝 값으로 바뀌지 않게 복사
            "bars": {str(b): {str(k): list(v) for k, v in sorted(c.items())} for b, c in sorted(fs["bars"].items())},
            "sec": {str(s): list(v) for s, v in sorted(fs["sec"].items())},
            "dhl": [[b, kind, v] for b, (kind, v) in sorted(fp_dhl(fs, recent, tape, now).items())]}


async def oracle(rows: list[list], marks: dict, name: str, tapes: dict, work: Path) -> dict:
    S, R, tv, use_snap = RUNS[name]
    s_idx = marks[f"S{S}"] if S is not None else None
    r_idx = marks[f"R{R}"]
    e_idx = next(i for i in range(r_idx, len(rows)) if rows[i][0] >= T0 + int((R + END_AFTER) * 1000))
    tape = tapes.get(tv, work / "no_tape.sqlite")
    clock = {"now": T0 / 1000}
    cf.time = SimpleNamespace(time=lambda: clock["now"])
    cf.HOT_TAPE_DB = tape                         # make_coin_flow 가 TAPE_DB 로 잡는다
    cf.LIVE_DIR = work                            # footprint_tape_restore 의 «_no_ctx.duckdb»
    cf.OI_1S_DB_PATH = work / "no_oi.sqlite"
    snap = work / "footprint_sol.json"
    snap.unlink(missing_ok=True)
    nowhere = dict(okx_tape_db=work / "no_okx.sqlite", okx_ctx_db=work / "no_okxctx.sqlite")
    out: dict = {"S_idx": s_idx, "R_idx": r_idx, "E_idx": e_idx, "M": M, "tape": tv}

    async def feed(ns, part: list[list], hook) -> asyncio.Task:
        done = asyncio.Event()

        class FakeWS:
            k = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.k:
                    await hook(self.k)
                if self.k == len(part):
                    done.set()
                    await asyncio.Event().wait()
                r = part[self.k]
                self.k += 1
                clock["now"] = r[5] / 1000
                return SimpleNamespace(type=WSMsgType.TEXT, data=text(r))

        class FakeSession:
            def __init__(self, *a, **k):
                pass

            def ws_connect(self, url, **k):
                assert url.endswith(f"{SYML}@trade"), url
                out.setdefault("loaded_last_ms", ns.footprint_state["last_ms"])   # footprint_load 바로 뒤
                return FakeWS()

            async def close(self):
                pass

        cf.ClientSession = FakeSession
        cf.TCPConnector = lambda **k: None
        task = asyncio.create_task(free(ns.start)["collect_footprint"](None))
        await asyncio.wait_for(done.wait(), 300)
        return task

    async def nohook(_k):
        return None

    if s_idx is not None:
        # 재시작 전 판: REST 몫(asset=eth)이라 재연결 메우기가 테이프를 안 읽는다(바로 예외) = 코어의 메우기 없는 라이브
        ns1 = cf.make_coin_flow(dataclasses.replace(SPEC, asset="eth", snapshot_path=snap, **nowhere), no_rest, {})
        t1 = await feed(ns1, rows[:s_idx], nohook)
        ns1.footprint_state["ready"] = True
        await ns1.stop(None)                       # footprint_save
        t1.cancel()
        saved = json.loads(snap.read_text())
        out["py_snapshot"] = {"last_ms": saved["last_ms"], "bars": saved["bars"]}
        if not use_snap:
            snap.unlink()
        out.pop("loaded_last_ms", None)
    ns2 = cf.make_coin_flow(dataclasses.replace(SPEC, snapshot_path=snap if use_snap else work / "none.json", **nowhere), no_rest, {})
    fs = ns2.footprint_state
    out["restart_ms"] = rows[r_idx][5] - 500
    clock["now"] = out["restart_ms"] / 1000       # footprint_load 의 143봉 자르기 기준
    sup = asyncio.create_task(free(ns2.start)["supply_1s_restore"](None))
    part = rows[r_idx:e_idx]

    async def hook(k: int) -> None:
        if k == M:
            while not (fs["ready"] and sup.done()):
                await _real_sleep(0.002)
            out["merge"] = checkpoint(fs, tape, clock["now"])
        if k == len(part):
            out["end"] = checkpoint(fs, tape, clock["now"])
    t2 = await feed(ns2, part, hook)
    t2.cancel()
    for t in (t1 if s_idx is not None else None, t2):
        if t is not None:
            try:
                await t
            except asyncio.CancelledError:
                pass
    # 기대 구멍(코어만 내는 표시 -- 파이썬엔 없다): 원본 footprint_tape_restore 의 lo·hi 와 같은 식
    first_ms = rows[r_idx][0]
    last_ms = out["loaded_last_ms"]
    now_bar = first_ms // 1000 // BS * BS
    floor = now_bar - (WANT - 1) * BS
    lo, hi = (max(floor, last_ms // 1000 + 1) if last_ms else floor), first_ms // 1000
    out["lo"], out["hi"] = lo, hi
    holes = []
    if tv == "none":
        holes = [[lo, hi]] if lo < hi else []
    else:
        con = sqlite3.connect(f"file:{tape}?mode=ro", uri=True)
        for a, z in con.execute("SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND to_ms > ? AND from_ms < ?", [SYML, lo * 1000, hi * 1000]):
            x, y = max(a // 1000, lo), min(z // 1000 + 1, hi)
            if x < y:
                holes.append([x, y])
        top = con.execute("SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [SYML]).fetchone()[0]
        con.close()
        if top < hi - 1 and lo < hi:
            holes.append([max(lo, top + 1), hi])
        out["tape_top"] = top
    out["holes"] = holes
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, marks = gen()
    tapes = {}
    for v, drop in (("full", None), ("gaps", (T0 + int(GAP_T[0] * 1000), T0 + int(GAP_T[1] * 1000)))):
        p = OUT / f"tape_{v}.sqlite"
        for x in OUT.glob(f"tape_{v}.sqlite*"):
            x.unlink()
        build_tape(rows, p, drop)
        tapes[v] = p
    p = OUT / "tape_behind.sqlite"
    for x in OUT.glob("tape_behind.sqlite*"):
        x.unlink()
    shutil.copyfile(tapes["full"], p)
    cut = (T0 // 1000 + int(RUNS["behind"][1])) - BEHIND_S
    con = sqlite3.connect(p)
    con.execute("DELETE FROM trade_tape_1s WHERE ts_sec >= ?", [cut])
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    tapes["behind"] = p
    work = OUT / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    old = sys.stdout
    runs = {}
    for name in RUNS:
        sys.stdout = open(os.devnull, "w")
        (work / name).mkdir()              # 판마다 따로 -- 재시작 뒤 판도 ready 가 되면 30초마다 제 스냅숏 경로에 쓴다(다음 판이 읽으면 안 된다)
        try:
            runs[name] = asyncio.run(oracle(rows, marks, name, tapes, work / name))
        finally:
            sys.stdout.close()
            sys.stdout = old
        r = runs[name]
        print(f"{name}: 봉 {len(r['merge']['bars'])}·{len(r['end']['bars'])} · 1초 {len(r['merge']['sec'])}·{len(r['end']['sec'])} · "
              f"[{r['lo']}, {r['hi']}) {r['hi'] - r['lo']}초 · 구멍 {r['holes']}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    cases = []
    for name, part1, py, snapmode in CASES:
        c = dict(runs[py])
        c.update(name=name, snap=snapmode, S_idx=runs[part1]["S_idx"] if part1 else None,
                 py_snapshot=runs[part1].get("py_snapshot") if part1 else None)
        cases.append(c)
    fx = {"about": "합성 SOL @trade 열 · 원본 수집기가 만든 테이프 · 원본 coin_flow 복원 정답(rtcore/tools/gen_restore_fixture.py, 시드 %d)" % SEED,
          "symbol": SYML, "t0": T0, "tape_buckets": dict(tc.BUCKETS), "supply_1s_seconds": cf.SUPPLY_1S_SECONDS,
          "window_bars": WANT, "rows": [[T - T0, tid, p, q, m, w - T0] for T, tid, p, q, m, w in rows],
          "tapes": {k: v.name for k, v in tapes.items()}, "cases": cases}
    (OUT / "restore_sol.json").write_text(json.dumps(fx, separators=(",", ":"), ensure_ascii=False))
    print(f"{(OUT / 'restore_sol.json').relative_to(ROOT)}: 체결 {len(rows)} · 사례 {len(cases)} · "
          f"{(OUT / 'restore_sol.json').stat().st_size // 1024}KB · 테이프 {[(k, v.stat().st_size // 1024) for k, v in tapes.items()]}KB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rtcore 나머지 원천 복원(OKX 1초 칸·OKX 미결제약정·OKX 청산·바이낸스 강제청산·Bybit 청산) 일치 시험의 입력·정답지 (2026-11-21).

① DB 네 개를 **원본 수집기**로 만든다 -- rtcollect/tools 의 원본 실행 틀(gen_venue_fixture · gen_ctx_fixture: 가짜 aiohttp/websockets ·
   가짜 시계 · REST 는 빈 코루틴)을 **가져다 쓴다**(읽기만 -- rtcollect 파일은 안 고친다):
   OKX 체결 테이프(live_okx_trade_tape_collector collect) · OKX 맥락(live_okx_context_collector collect -- 청산은 ETH 프로세스가 전 종목을 받는다,
   OI 는 SOL 프로세스) · 바이낸스 맥락(live_binance_ctx_collector ws_loop · 강제청산) · Bybit 청산(live_bybit_liquidation_collector run).
   OKX 테이프에는 1분봉 대조 행(verify_1m · rel_err 0)을 더한다 -- 원본 수집기가 REST 로 쓰는 행이라 가짜 판에선 비고,
   그게 없으면 원본 okx_footprint_restore 가 «온전한 봉»을 못 찾아 청산 복원을 안 한다(서버에선 5분마다 쌓인다).
② 재시작 시각 R 에서 **원본 복원 함수**로 정답을 만든다: dashboard/coin_flow.py supply_1s_restore(OKX 1초 칸·OKX OI -- 라이브 첫 초를 심어 둔다) ·
   okx_footprint_restore(OKX 청산 -- 첫 체결 시각 t0_ms 를 심어 둔다). server.py 는 import 하지 않는다 -- liq_events_load(강제청산)·
   supply_1s_payload 의 Bybit 몫(bybit_liq_events + 창)은 아래에 **옮겨 적었다**(원본 위치 주석).
rtcore/tests/restore_venues_parity.rs 가 같은 DB·같은 실시간 메시지로 코어를 돌려 대조한다.

네트워크 없음 · .env 안 읽음 · 실제 data/ 안 건드림(경로 전부 이 폴더) · 원본 읽기 함수는 범위가 하루를 넘으면 멈추는 단언으로 감싼다.
    python3 rtcore/tools/gen_restore_venues_fixture.py    # 저장소 루트에서. rtcore/tests/fixtures/gen/restore_venues/ (커밋 안 함 -- 시험 전에 만든다)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import shutil
import sqlite3
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtcore" / "tests" / "fixtures" / "gen" / "restore_venues"
sys.path.insert(0, str(ROOT / "rtcollect" / "tools"))
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 connect 를 막는다(import 할 때)
import gen_ctx_fixture as gc  # noqa: E402

import dashboard.coin_flow as cf  # noqa: E402
from scripts.data_store import read_rows  # noqa: E402
from scripts.live_okx_trade_tape_collector_20260923 import BUCKETS as OKX_BUCKETS  # noqa: E402

SEED = 20261124
SPEC = cf.FLOW_SPECS["sol"]
INST = SPEC.okx_inst                          # SOL-USDT-SWAP (ctVal 1)
SYM = SPEC.symbol                             # SOLUSDT
W = 1_763_700_000.0                           # 합성 시작(벽시계 초 · 봉 경계) -- 실제 장과 무관
SPAN = 1800.0
R = W + 1500.4                                # 코어·파이썬 재시작(이 시각 뒤 메시지만 실시간으로 받는다)
LIVE = 30.0                                   # 재시작 뒤 실시간 메시지 30초
DAY = 86_400


def okx_trades(rng: random.Random) -> list:
    e = gv.Ev(rng, W)
    e.add(0.0, "o")
    tid, mid = 7_000_000, 140.0
    while e.t < W + SPAN:
        mid += rng.gauss(0, 0.02)
        rows = []
        dt = rng.uniform(0.05, 0.6)
        ts = str(int((e.t + dt) * 1000) - 30)          # 수신보다 30ms 앞 -- 메시지 순서 = 시각 순서
        for _ in range(rng.choice((1, 1, 2, 3))):
            tid += 1
            sz = rng.lognormvariate(1.0 if rng.random() < 0.08 else -0.5, 1.4) * 10
            rows.append({"instId": INST, "tradeId": str(tid), "px": f"{mid + rng.randint(-3, 3) * 0.01:.2f}", "sz": f"{max(sz, 0.01):.2f}",
                         "side": rng.choice(("buy", "sell")), "ts": ts, "count": str(rng.choice((1, 1, 2, 5)))})
        e.msg(dt, {"arg": {"channel": "trades", "instId": INST}, "data": rows})
    return [{"inst": INST, "t0": W - 0.5, "events": e.ev}]


def okx_ctx(rng: random.Random) -> list:
    """청산: ETH 프로세스(전 종목 instType=SWAP) · OI: SOL 프로세스."""
    liq = gv.Ev(rng, W)
    liq.add(0.0, "o")
    while liq.t < W + SPAN:
        dets = [{"bkLoss": "0", "bkPx": f"{140 * (1 + rng.gauss(0, 0.003)):.2f}", "ccy": "", "posSide": rng.choice(("long", "short")),
                 "side": rng.choice(("buy", "sell")), "sz": f"{rng.lognormvariate(1, 1.2):.2f}", "ts": str(int((liq.t + 1.0) * 1000) - 400)}
                for _ in range(rng.choice((1, 1, 2)))]
        rows = [{"instId": INST, "instFamily": "SOL-USDT", "instType": "SWAP", "details": dets}]
        if rng.random() < 0.3:     # 다른 종목(코어는 SOL 만 쓴다)
            rows.append({"instId": "ETH-USDT-SWAP", "instFamily": "ETH-USDT", "instType": "SWAP",
                         "details": [{"bkPx": "2500.1", "posSide": "long", "side": "sell", "sz": "3", "ts": str(int(liq.t * 1000))}]})
        liq.msg(rng.uniform(3, 25), {"arg": {"channel": "liquidation-orders", "instType": "SWAP"}, "data": rows})
    oi = gv.Ev(rng, W)
    oi.add(0.0, "o")
    v = 5_000_000.0
    while oi.t < W + SPAN:
        v += rng.gauss(0, 300)
        x = round(v, 2)
        oi.msg(rng.uniform(1.5, 7), {"arg": {"channel": "open-interest", "instId": INST},
                                     "data": [{"instId": INST, "instType": "SWAP", "oi": f"{x}", "oiCcy": repr(x), "oiUsd": f"{x * 140:.1f}",
                                               "ts": str(int(oi.t * 1000) - 300)}]})
    return [{"inst": "ETH-USDT-SWAP", "t0": W - 0.5, "events": liq.ev}, {"inst": INST, "t0": W - 0.5, "events": oi.ev}]


def bn_force(rng: random.Random) -> dict:
    e = gv.Ev(rng, W)
    e.add(0.0, "o")
    while e.t < W + SPAN:
        s = rng.choice(("SOLUSDT", "SOLUSDT", "ETHUSDT"))
        q = f"{rng.lognormvariate(1, 1):.2f}"
        dt = rng.uniform(1, 15)
        ms = int((e.t + dt) * 1000) - 100
        e.msg(dt, {"e": "forceOrder", "E": ms + 5, "o": {"s": s, "S": rng.choice(("SELL", "BUY")), "o": "LIMIT", "f": "IOC", "q": q,
                                                          "p": f"{140 * (1 + rng.gauss(0, 0.01)):.2f}", "ap": f"{140 * (1 + rng.gauss(0, 0.004)):.3f}",
                                                          "X": "FILLED", "l": q, "z": q, "T": ms}})
    return {"ws": [{"kind": "force", "url": "!forceOrder@arr", "events": e.ev}], "oi": [], "slow": []}


def bybit(rng: random.Random) -> list:
    e = gv.Ev(rng, W)
    e.add(0.0, "o")
    e.msg(0.05, {"success": True, "ret_msg": "", "conn_id": "c1", "op": "subscribe"})
    while e.t < W + SPAN:
        dt = rng.uniform(0.5, 12)
        rows = [{"T": int((e.t + dt) * 1000) - rng.randint(50, 600), "s": rng.choice(("SOLUSDT", "SOLUSDT", "ETHUSDT")), "S": rng.choice(("Buy", "Sell")),
                 "v": f"{rng.lognormvariate(0, 1.5) * 10:.3f}", "p": f"{140 * (1 + rng.gauss(0, 0.002)):.4f}"} for _ in range(rng.choice((1, 1, 2)))]
        e.msg(dt, {"topic": "allLiquidation.SOLUSDT", "type": "snapshot", "ts": e.ms() + 1, "data": rows})
    e.add(0.5, "c", "ConnectionClosedOK: received 1000 (OK)")
    return e.ev


# ── 원본 읽기 함수: 범위 단언(2026-11-20 13.7GB 사고) ──
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
cf.asyncio.sleep = _fast


async def no_rest(*_a, **_k):
    raise RuntimeError("REST 금지(시험)")


def free(fn) -> dict:
    return {k: c.cell_contents for k, c in zip(fn.__code__.co_freevars, fn.__closure__)}


# ── dashboard/server.py 옮겨 적음(2026-11-20 판) ─────────────────────────────
def liq_events_load(bn_ctx: Path, now_ms: float) -> list:
    """server.py liq_events_load(1785~1815)의 1초 수급 링 몫(5분봉 누적·기록 시작점은 코어 범위 밖)."""
    cut = now_ms - cf.SUPPLY_1S_SECONDS * 1000
    rows = read_rows(bn_ctx, "SELECT ts_ms, side, qty, price, usd, symbol FROM liquidations WHERE symbol = ? "
                             "ORDER BY ts_ms DESC LIMIT ?", [SYM, 5000])
    return [[int(ts), side == "long", float(qty or 0.0), float(usd or 0.0)] for ts, side, qty, price, usd, sym in reversed(rows) if ts >= cut]


def bybit_rows(db: Path, now: float) -> tuple[list, list]:
    """server.py _bybit_liq_refresh(전체 다시 읽기 갈래 800~804) + supply_1s_payload 의 Bybit 창(4252~4260)."""
    since = int((now - cf.FOOTPRINT_KEEP_BARS * cf.FOOTPRINT_BAR_SECONDS) * 1000)
    rows = read_rows(db, "SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms", [SYM, since])
    ev = [{"ts_ms": int(t), "side": "long" if sd == "Buy" else "short", "qty": float(q), "price": float(px), "usd": float(q) * float(px)} for t, sd, q, px in rows]
    bnow = int(now) - 3
    bfloor = bnow - cf.SUPPLY_1S_SECONDS
    cells: dict[int, list[float]] = {}
    shown = []
    for e in ev:
        x = e["ts_ms"] // 1000
        if bfloor < x <= bnow:
            c = cells.setdefault(x, [0.0, 0.0, 0.0, 0.0]); lo = e["side"] == "long"
            c[0 if lo else 1] += e["qty"]; c[2 if lo else 3] += e["usd"]
            shown.append([e["ts_ms"], lo, e["qty"], e["usd"]])
    return [[x, round(v[0], 3), round(v[1], 3), round(v[2]), round(v[3])] for x, v in sorted(cells.items())], shown


async def oracle(d: Path, t0_ms: int, t0_oi: int) -> dict:
    clock = {"now": R + 20.0}
    cf.time = SimpleNamespace(time=lambda: clock["now"])
    cf.HOT_TAPE_DB = d / "없는_바이낸스_테이프.sqlite"     # 바이낸스 몫은 이 정답지 범위 밖(라이브 칸이 비어 기다리다 끝난다)
    cf.LIVE_DIR = d
    cf.OI_1S_DB_PATH = d / "없는_oi.sqlite"
    cf.OKX_LIQ_DB_PATH = d / "okx_ctx.sqlite"             # okx_footprint_restore 의 liq_db(모듈 상수)
    ns = cf.make_coin_flow(dataclasses.replace(SPEC, snapshot_path=d / "없음.json", okx_tape_db=d / "okx_tape.sqlite",
                                               okx_ctx_db=d / "okx_ctx.sqlite"), no_rest, {})
    planted = [0.0] * 7
    ns.okx_sec[t0_ms // 1000] = planted                    # 라이브 첫 초(supply_1s_restore 의 t0 = min)
    ns.okx_oi_1s[t0_oi] = -1.0                             # 라이브 첫 OI 초
    ns.okx_fp["t0_ms"] = t0_ms                             # 라이브 첫 OKX 체결(okx_footprint_restore 가 기다리는 값)
    fns = free(ns.start)
    await asyncio.wait_for(asyncio.gather(fns["supply_1s_restore"](None), fns["okx_footprint_restore"](None)), 120)
    return {"okx_sec": {str(s): v for s, v in sorted(ns.okx_sec.items()) if v is not planted},
            "okx_oi": {str(s): v for s, v in sorted(ns.okx_oi_1s.items()) if s != t0_oi},
            "okx_liq": [[e["ts_ms"], e["side"] == "long", e["qty"], e["usd"]] for e in ns.okx_liq_events]}


def main() -> None:
    rng = random.Random(SEED)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    trades, ctx, force, liq = okx_trades(rng), okx_ctx(rng), bn_force(rng), bybit(rng)
    gv.run_okx(trades, OUT)
    gc.run_okx_ctx(ctx, OUT)
    gc.run_binance_ctx(force, OUT)
    gv.run_liq(liq, OUT / "bybit_liq.sqlite")
    gv.close_shared()
    # 1분봉 대조 행(머리 주석 ①)
    con = sqlite3.connect(OUT / "okx_tape.sqlite")
    lo = int(W) // 60 * 60 - DAY
    con.executemany("INSERT INTO verify_1m VALUES (?,?,?,?,?,?)", [(INST, m, 1.0, 1.0, 0.0, "합성") for m in range(lo, int(W + SPAN) + 60, 60)])
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    # 재시작 뒤 실시간 메시지(코어가 받는 것 = OKX 체결 · OI · 청산) -- 수신 시각 순
    live = sorted([[t, b] for t, k, b in trades[0]["events"] + ctx[0]["events"] + ctx[1]["events"] if k == "t" and R <= t < R + LIVE])
    first = lambda ch: next(json.loads(b) for _t, b in live if json.loads(b)["arg"]["channel"] == ch)   # noqa: E731
    t0_ms = int(first("trades")["data"][0]["ts"])
    t0_oi = int(first("open-interest")["data"][0]["ts"]) // 1000
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        got = asyncio.run(oracle(OUT, t0_ms, t0_oi))
    finally:
        sys.stdout.close()
        sys.stdout = old
    rows, shown = bybit_rows(OUT / "bybit_liq.sqlite", R)
    fx = {"about": "합성 SOL OKX·청산 원천 · 원본 수집기가 만든 DB · 원본 coin_flow 복원 정답(rtcore/tools/gen_restore_venues_fixture.py, 시드 %d)" % SEED,
          "symbol": SYM.lower(), "inst": INST, "restart_s": R, "t0_ms": t0_ms, "t0_oi": t0_oi, "okx_tape_buckets": dict(OKX_BUCKETS),
          "live": live, **got, "bn_liq": liq_events_load(OUT / "binance_ctx.sqlite", R * 1000), "bybit_rows": rows, "bybit_events": shown}
    (OUT / "restore_venues_sol.json").write_text(json.dumps(fx, separators=(",", ":"), ensure_ascii=False))
    n_ok = sum(1 for e in got["okx_liq"] if e[0] >= t0_ms - 660_000)
    print(f"{(OUT / 'restore_venues_sol.json').relative_to(ROOT)}: 실시간 메시지 {len(live)} · OKX 1초 {len(got['okx_sec'])} · OKX OI {len(got['okx_oi'])} · "
          f"OKX 청산 {len(got['okx_liq'])}(660초 안 {n_ok}) · 강제청산 {len(fx['bn_liq'])} · Bybit 행 {len(rows)} 사건 {len(shown)}")
    assert got["okx_sec"] and got["okx_oi"] and n_ok and fx["bn_liq"] and rows, "정답지가 비었다"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rtcore OKX 풋프린트 봉 복원(restore.rs read_okx_fp · Supply::okx_fp_attempt) 일치 시험의 입력·정답지 (2026-11-21).

DB 는 gen_restore_venues_fixture 의 원본 수집기 틀로 만든다(그 모듈을 가져다 쓴다 -- 읽기만).
정답 = **원본 coin_flow.okx_footprint_restore** 를 가짜 시계 위에서 돌린 값. asyncio.sleep 을 바꿔
«잠들기 전 = 앞 회차가 끝난 상태»를 걸음마다 떠 두고, 시계를 잠든 만큼 앞으로 민다.
깨어날 때마다 그 시각까지 온 실시간 메시지를 원본 처리(coin_flow.py 760~832 옮겨 적음)로 넣는다.
1분봉 대조 표도 정해 둔 회차에 고친다:
- 진행 봉의 분: 처음엔 rel_err 가 나쁨 → 2회차에 고침.
- 한 분: 공백 + 대조 없음 → 3회차에 좋은 대조를 넣음.
- t0 의 분: 대조 없음 → «최근 15분·공백 없음» 길.
- 20봉 앞 묶음: 나쁨 → 5회차에 고침.
- 100봉 앞 한 분: 끝까지 나쁨 → 창 끝(lo)에 못 닿아 8회차까지 돈다.
코어 시험(tests/okx_fp_restore.rs)이 같은 DB·같은 고침·같은 메시지·같은 시각으로 회차를 부르고 걸음마다 대조한다.
네트워크 없음 · 실제 data/ 안 건드림.
    python3 rtcore/tools/gen_okx_fp_fixture.py     # 저장소 루트에서. rtcore/tests/fixtures/gen/okx_fp/ (커밋 안 함)
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
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_restore_venues_fixture as gr  # noqa: E402 -- cf 를 고친다(빠른 sleep·창 단언) · 아래에서 sleep 을 다시 바꾼다

cf, gv, gc = gr.cf, gr.gv, gr.gc
ROOT = gr.ROOT
OUT = ROOT / "rtcore" / "tests" / "fixtures" / "gen" / "okx_fp"
SEED = 20261125
W, SPAN, INST, SPEC = gr.W, gr.SPAN, gr.INST, gr.SPEC
R = W + 1750.4                       # 재시작 -- 진행 봉(W+1500) 안 다섯째 분
LIVE = 1500.0                        # 재시작 뒤 실시간 메시지 25분(회차 사이에도 온다)
RETAIL_MAX_USD, WHALE_MIN_USD = cf.size_bands(SPEC.symbol)
CT = cf.OKX_CT_VALS.get(INST)


def schedule(t0_ms: int, variant: str) -> tuple[list, dict]:
    """DB 고침을 SQL 로: 처음(정답지·시험이 같은 사본에서 시작) · 회차별(회차 → [[sql, 인자]]). 회차마다 «한 조건만» 갈리게 짰다.
    A1 테이프가 t0 뒤를 아직 안 씀 → A2·A3 진행 봉 조건 하나씩(변형 a·b) → 진행 봉 통과 뒤 3봉 앞 묶음에서 멈춤
    → A5 묶음 고침 + 이미 되살린 봉에 테이프 행 추가(다시 안 바꿔야 함) → A8 100봉 앞 영구 나쁜 분 고침(창 끝 도달).
    «최근 15분» 길은 지금 − 900초 안 분만 -- A4(t0 + 915초)부터는 진행 봉 분이 이 길로 못 통과한다(그래서 변형 둘).
    넷째 분은 대조 없음(최근 15분 길) · 50봉 앞 한 분은 rel_err 5e-5(허용 1e-4 안)."""
    start = t0_ms // 1000 // 300 * 300
    t0_sec, t0_min = t0_ms // 1000, t0_ms // 1000 // 60 * 60
    lo = int(W) // 60 * 60 - gr.DAY
    V = "INSERT INTO verify_1m VALUES (?, ?, 1.0, 1.0, ?, '합성')"
    D = "DELETE FROM verify_1m WHERE symbol = ? AND ts_min = ?"
    rel = {m: 0.0 for m in range(lo, int(W + SPAN) + 60 + int(LIVE), 60)}
    rel[start + 180] = None
    old = start - 3 * 300
    for m in range(old, old + 300, 60):
        rel[m] = 0.5
    rel[start - 100 * 300 + 120] = 2.0
    rel[start - 50 * 300 + 60] = 5e-5
    init = [[V, [INST, m, r]] for m, r in rel.items() if r is not None]
    init += [["CREATE TABLE behind AS SELECT * FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?", [INST, t0_sec]],
             ["DELETE FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?", [INST, t0_sec]]]
    assert t0_min >= start + 240, "t0 가 진행 봉 다섯째 분이어야 한다"
    gap = lambda m: ["INSERT INTO gaps VALUES (?, ?, ?, '합성')", [INST, m * 1000 + 5000, m * 1000 + 20000]]   # noqa: E731
    tail = {
        5: [*[x for m in range(old, old + 300, 60) for x in ([D, [INST, m]], [V, [INST, m, 0.0]])],
            ["INSERT OR REPLACE INTO trade_tape_1s (symbol, ts_sec, price_bin, buy_qty, sell_qty) VALUES (?, ?, ?, ?, ?)",
             [INST, start - 300 + 7, 13999, 5.0, 2.0]]],
        8: [[D, [INST, start - 100 * 300 + 120]], [V, [INST, start - 100 * 300 + 120, 0.0]]],
    }
    if variant == "a":
        # A2 t0 의 분 나쁨(진행 봉 조건이 t0 의 분까지) → A3 그 분 대조 없음 + 공백(최근 15분 길이 공백을 본다) → A4 좋은 대조
        fix = {2: [["INSERT INTO trade_tape_1s SELECT * FROM behind", []], [D, [INST, t0_min]], [V, [INST, t0_min, 1.0]]],
               3: [[D, [INST, t0_min]], gap(t0_min)],
               4: [[V, [INST, t0_min, 0.0]], [V, [INST, start + 180, 0.0]]], **tail}
    else:
        # A2 셋째 분 대조 없음 + 공백 → A3 공백 위에 좋은 대조 · 넷째 분은 «최근 15분» 길로만 온전(진행 봉이 이 길로 통과)
        fix = {2: [["INSERT INTO trade_tape_1s SELECT * FROM behind", []], [D, [INST, start + 120]], gap(start + 120)],
               3: [[V, [INST, start + 120, 0.0]]], **tail}
    return init, fix


def apply_sql(db: Path, rows: list) -> None:
    con = sqlite3.connect(db)
    for sql, args in rows:
        con.execute(sql, args)
    con.commit()
    con.close()


def feed(ns, msg: dict, now: float) -> None:
    """coin_flow.py collect_okx_flow 의 OKX 몫(760~832 옮겨 적음: 봉 칸·first_bar·t0 · OI 5분봉 · 청산 링)."""
    ch = msg["arg"]["channel"]
    if ch == "trades":
        for t in msg.get("data") or []:
            price = float(t.get("px") or 0.0)
            qty = float(t.get("sz") or 0.0) * CT
            side = str(t.get("side") or "")
            if not (price > 0 and qty > 0 and side in ("buy", "sell")):
                continue
            ts_ms = int(t.get("ts"))
            i = 1 if side == "sell" else 0
            notional = price * qty
            bar = ts_ms // 1000 // 300 * 300
            if not ns.okx_fp["first_bar"]:
                ns.okx_fp["first_bar"] = bar
                ns.okx_fp["t0_ms"] = ts_ms
            cells = ns.okx_bars.get(bar)
            if cells is None:
                cells = ns.okx_bars[bar] = {}
                cut = int(now * 1000) // 1000 // 300 * 300 - (cf.FOOTPRINT_KEEP_BARS - 1) * 300
                for ob in [b for b in ns.okx_bars if b < cut]:
                    del ns.okx_bars[ob]
            fc = cells.setdefault(int(round(price / SPEC.bucket)), [0.0] * 6)
            fc[i] += qty
            if notional >= WHALE_MIN_USD:
                fc[2 + i] += qty
            elif notional < RETAIL_MAX_USD:
                fc[4 + i] += qty
    elif ch == "open-interest":
        for r in msg.get("data") or []:
            oi = float(r.get("oiCcy") or 0.0)
            ts_ms = int(r.get("ts") or 0)
            if oi > 0 and ts_ms > 0:
                ob = ts_ms // 1000 // 300 * 300
                if ob in ns.okx_oi_5m:
                    ns.okx_oi_5m[ob][1] = oi
                else:
                    ns.okx_oi_5m[ob] = [oi, oi]
                    for old in [x for x in ns.okx_oi_5m if x < ob - cf.FOOTPRINT_KEEP_BARS * 300]:
                        del ns.okx_oi_5m[old]
    elif ch == "liquidation-orders":
        for r in msg.get("data") or []:
            if str(r.get("instId") or "") != INST:
                continue
            for det in r.get("details") or []:
                qty = float(det.get("sz") or 0.0) * CT
                px = float(det.get("bkPx") or 0.0)
                if qty <= 0:
                    continue
                ns.okx_liq_events.append({"ts_ms": int(det.get("ts")), "side": "long" if det.get("posSide") == "long" else "short",
                                          "qty": qty, "price": px, "usd": qty * px, "symbol": INST})


def okx_add(ns) -> dict:
    """server.py api_liquidation_5m_history 3963~3976 의 OKX 더하기(옮겨 적음) -- 봉마다 [롱, 숏, 건수], first_bar «뒤» 봉만."""
    okx_from = ns.okx_fp["first_bar"]
    add: dict[int, list] = {}
    if okx_from:
        for e in list(ns.okx_liq_events):
            bar = int(e["ts_ms"]) // 1000 // 300 * 300
            if bar > okx_from:
                a = add.setdefault(bar, [0.0, 0.0, 0])
                a[0 if e["side"] == "long" else 1] += float(e.get("usd") or 0.0)
                a[2] += 1
    return {str(b): list(v) for b, v in sorted(add.items())}


def bybit_add(db: Path, now: float) -> dict:
    """server.py bybit_liq_events(741~760) + merge_bybit_liq 의 add·first(763~775) 옮겨 적음 -- 봉 > first 만."""
    since = int((now - cf.FOOTPRINT_KEEP_BARS * 300) * 1000)
    start = gr.read_rows(db, "SELECT min(ts_ms) FROM bybit_liquidations", [])[0][0]
    rows = gr.read_rows(db, "SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms", [SPEC.symbol, since])
    ev = [{"ts_ms": int(t), "side": "long" if sd == "Buy" else "short", "usd": float(q) * float(px)} for t, sd, q, px in rows if int(t) <= now * 1000]
    if start is None:
        return {"from": None, "bars": {}}
    first = int(start) // 1000 // 300 * 300
    add: dict[int, list] = {}
    for e in ev:
        b = e["ts_ms"] // 1000 // 300 * 300
        a = add.setdefault(b, [0.0, 0.0, 0])
        a[0 if e["side"] == "long" else 1] += e["usd"]
        a[2] += 1
    return {"from": first, "bars": {str(b): v for b, v in sorted(add.items()) if b > first}}


def state(ns, now: float, attempt: int) -> dict:
    return {"now": now, "attempt": attempt, "first_bar": ns.okx_fp["first_bar"], "okx5m": okx_add(ns),
            # 🔴list(v): 칸 리스트를 그대로 넣으면 뒤 회차의 += 가 앞 걸음 값까지 바꾼다
            "bars": {str(b): {str(k): list(v) for k, v in sorted(c.items())} for b, c in sorted(ns.okx_bars.items())},
            "oi5m": {str(b): list(v) for b, v in sorted(ns.okx_oi_5m.items())},
            "liq": [[e["ts_ms"], e["side"] == "long", e["qty"], e["usd"]] for e in ns.okx_liq_events]}


async def oracle(d: Path, live: list, fix: dict) -> list:
    t_first = next(t for t, b in live if json.loads(b)["arg"]["channel"] == "trades")
    clock = {"now": t_first}
    cf.time = SimpleNamespace(time=lambda: clock["now"])
    cf.OKX_LIQ_DB_PATH = d / "okx_ctx.sqlite"
    ns = cf.make_coin_flow(dataclasses.replace(SPEC, snapshot_path=d / "없음.json", okx_tape_db=d / "okx_tape.sqlite",
                                               okx_ctx_db=d / "okx_ctx.sqlite"), gr.no_rest, {})
    li = [0]

    def feed_upto(t: float) -> None:
        while li[0] < len(live) and live[li[0]][0] <= t:
            feed(ns, json.loads(live[li[0]][1]), live[li[0]][0])
            li[0] += 1

    feed_upto(t_first)
    steps: list = []
    n = [0]

    async def sleep(s, *a, **k):
        if n[0]:
            steps.append(state(ns, clock["now"], n[0]))      # 앞 회차가 끝난 상태
        n[0] += 1
        clock["now"] += s
        feed_upto(clock["now"])
        if n[0] in fix:
            apply_sql(d / "okx_tape.sqlite", fix[n[0]])
        await gr._real_sleep(0)

    cf.asyncio.sleep = sleep
    await asyncio.wait_for(gr.free(ns.start)["okx_footprint_restore"](None), 120)
    steps.append(state(ns, clock["now"], n[0]))
    return steps


def build(variant: str) -> None:
    out = OUT / variant
    rng = random.Random(SEED)
    out.mkdir(parents=True)
    trades, ctx, liq = gr.okx_trades(rng), gr.okx_ctx(rng), gr.bybit(rng)
    gv.run_okx(trades, out)
    gc.run_okx_ctx(ctx, out)
    gv.run_liq(liq, out / "bybit_liq.sqlite")
    gv.close_shared()
    live = sorted([[t, b] for t, k, b in trades[0]["events"] + ctx[0]["events"] + ctx[1]["events"] if k == "t" and R <= t < R + LIVE])
    t0_ms = int(next(json.loads(b) for _t, b in live if json.loads(b)["arg"]["channel"] == "trades")["data"][0]["ts"])
    init, fix = schedule(t0_ms, variant)
    apply_sql(out / "okx_tape.sqlite", init)
    con = sqlite3.connect(out / "okx_tape.sqlite")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    shutil.copy(out / "okx_tape.sqlite", out / "okx_tape_start.sqlite")     # 시험은 이 사본에서 시작한다(고침을 다시 한다)
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        steps = asyncio.run(oracle(out, live, fix))
    finally:
        sys.stdout.close()
        sys.stdout = old
    fx = {"about": "합성 SOL OKX · 원본 okx_footprint_restore 걸음별(rtcore/tools/gen_okx_fp_fixture.py, 시드 %d · 변형 %s)" % (SEED, variant),
          "symbol": SPEC.symbol.lower(), "inst": INST, "restart_s": R, "t0_ms": t0_ms, "live": live,
          "fix": {str(k): v for k, v in fix.items()}, "steps": steps, "bybit5m": bybit_add(out / "bybit_liq.sqlite", R)}
    p = out / "okx_fp_sol.json"
    p.write_text(json.dumps(fx, separators=(",", ":"), ensure_ascii=False))
    fb = [(s["first_bar"] - steps[0]["first_bar"]) // 300 for s in steps]
    print(f"{p.relative_to(ROOT)}: 걸음 {len(steps)} · first_bar(진행 봉 기준 봉) {fb} · 봉 {[len(s['bars']) for s in steps]} · "
          f"OKX 청산봉 {[len(s['okx5m']) for s in steps]} · Bybit 봉 {len(fx['bybit5m']['bars'])} · OI 봉 {[len(s['oi5m']) for s in steps]} · "
          f"청산 {[len(s['liq']) for s in steps]} · 실시간 {len(live)} · {p.stat().st_size // 1024}KB")
    assert len(steps) == 8 and len(set(fb)) >= 4, "회차·first_bar 가 움직이지 않았다(정답지 무효)"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    for v in ("a", "b"):
        build(v)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rtserve 일봉·4시간 풋프린트(src/fpd.rs) 일치 시험의 입력·정답지 (2026-11-22).

정답 = **원본 dashboard/footprint_daily.py 를 그대로 부른 값**: DailyFootprint(read_lake = server.py _fpd_lake 와 같은 data_store.read ·
read_hot/read_ctx = server.py _fpd_rows 와 같은 data_store.read_rows) → days_payload · cells_payload · h4_payload · h4_cells_payload.
server.py 의 손잡이(_cols_since · _iso_day · _iso_h4 · parse_row 쓰임)는 시험이 따로 본다(옮겨 적음 -- 이 파일 아래 cols_since).
입력은 합성: 소급본 parquet(data/footprint_daily 꼴) · lake(binance/{tape,oi_1s,liquidations}/coin=ETH/date=…/part.parquet) ·
hot SQLite(trade_tape_1s · oi_1s · liquidations). 고정 시계로 걸음마다 부르고, 걸음 사이에 hot 에 행을 더해 캐시 동작(오늘 30초 ·
닫힌 날 한 번 · 4시간 칸 닫힘/형성/빈칸 10분)을 본다. 네트워크 없음.
🔴서버 판 파이썬으로 돌린다(numpy 2.2.5 · pandas 2.2.3 · duckdb 1.5.1 -- 합 차례가 서버와 같아야 한다):
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_fpd_fixture.py   # 저장소 루트에서
출력: rtserve/tests/fixtures/gen/fpd/ (커밋 안 함 · 시험 전에 만든다)
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "fpd"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
os.environ["DATA_LAKE"] = str(OUT / "lake")          # data_store.LAKE 는 import 때 정해진다
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scripts import data_store  # noqa: E402
import dashboard.footprint_daily as fpd_eth  # noqa: E402

SEED = 20261122
DAY = 86400
TODAY = date(2026, 11, 21)
HIST_END = date(2026, 11, 5)       # 소급본 마지막 날
LAKE_DAYS = [date(2026, 11, 6) + timedelta(days=i) for i in range(13)]   # 11-06 ~ 11-18 봉인
HOT_FROM = date(2026, 11, 17)      # hot = 11-17 ~ 오늘(11-17·18 은 lake 와 겹침 -- lake 가 이긴다)


def ts(d: date, s: int = 0) -> int:
    return int(datetime.combine(d, datetime.min.time(), timezone.utc).timestamp()) + s


def tape_rows(rng: random.Random, d: date, n: int, base: int, sec_lo: int = 0, sec_hi: int = DAY) -> list[tuple]:
    rows = []
    for _ in range(n):
        s = ts(d, rng.randint(sec_lo, sec_hi - 1))
        pb = base + rng.randint(-400, 400)
        nd = 4 if rng.random() < 0.5 else 3     # 4자리 수량 = np.round(3) 동률·마지막 비트가 갈리는 칸이 생긴다
        b = round(rng.expovariate(1 / 2), nd) if rng.random() < 0.7 else 0.0
        q = round(rng.expovariate(1 / 2), nd) if rng.random() < 0.7 else 0.0
        rows.append((s, pb, b, q))
    # 같은 초 여러 칸(실제 테이프 모양) · 순서는 섞어 둔다(원본이 stable argsort 로 정렬한다)
    for k in range(0, n, 9):
        s, pb, b, q = rows[k]
        rows.append((s, pb + 3, round(b / 2, 3), q))
    rng.shuffle(rows)
    return rows


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(asset: str) -> None:
    rng = random.Random(SEED + len(asset))
    m = fpd_eth.for_asset(asset)                        # 원본 코인별 모듈 사본(캐시도 따로)
    sym, coin, hs = m.SYMBOL, m.COIN, m.HOT_SYM
    B0 = {"eth": 30000, "xrp": 14000}[asset]           # 테이프 칸 번호(ETH 0.1 → $3,000 · XRP 0.0001 → $1.4)
    out = OUT / asset
    out.mkdir(parents=True)
    # ── 소급본(ETH 만 -- XRP 는 «소급본 없음» 갈래: 시작 = 오늘 − 30일) ──
    hd = [HIST_END - timedelta(days=i) for i in range(40)][::-1]
    days = pd.DataFrame({"day": hd, "open": [rng.uniform(2800, 3200) for _ in hd], "high": [rng.uniform(3200, 3300) for _ in hd],
                         "low": [rng.uniform(2700, 2800) for _ in hd], "close": [rng.uniform(2800, 3200) for _ in hd],
                         "buy_qty": [rng.uniform(1e5, 3e5) for _ in hd], "sell_qty": [rng.uniform(1e5, 3e5) for _ in hd],
                         "buy_quote": [rng.uniform(3e8, 9e8) for _ in hd], "sell_quote": [rng.uniform(3e8, 9e8) for _ in hd],
                         "oi": [np.nan if i < 5 else rng.uniform(1e6, 2e6) for i in range(len(hd))]})
    if asset == "eth":
        write_parquet(OUT / "hist" / f"{sym}_days.parquet", days)
    cl = []
    for d in hd:
        for b in sorted(rng.sample(range(2700, 3300), 40)):
            cl.append((d, b, round(rng.uniform(0, 500), 3), round(rng.uniform(0, 500), 3)))
    if asset == "eth":
        write_parquet(OUT / "hist" / f"{sym}_cells.parquet", pd.DataFrame(cl, columns=["day", "bin", "buy", "sell"]))
    # ── lake ──
    for d in LAKE_DAYS:
        tr = tape_rows(rng, d, 300, B0)
        write_parquet(data_store.lake_path("binance", "tape", coin, d.isoformat()),
                      pd.DataFrame(tr, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty"]))
        oi = [(ts(d, s) * 1000 + rng.randint(0, 999), 1.5e6 + rng.uniform(-1e5, 1e5)) for s in range(0, DAY, 3600)]
        write_parquet(data_store.lake_path("binance", "oi_1s", coin, d.isoformat()), pd.DataFrame(oi, columns=["ts_ms", "open_interest"]))
        lq = [(ts(d, rng.randint(0, DAY - 1)) * 1000, rng.choice(["long", "short"]), round(rng.uniform(10, 50000), 2)) for _ in range(12)]
        write_parquet(data_store.lake_path("binance", "liquidations", coin, d.isoformat()), pd.DataFrame(lq, columns=["ts_ms", "side", "usd"]))
    # ── hot ──
    tape = sqlite3.connect(out / "hot_tape.sqlite")
    tape.execute("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL)")
    ctx = sqlite3.connect(out / "hot_ctx.sqlite")
    ctx.execute("CREATE TABLE oi_1s(symbol TEXT, ts_ms INTEGER, open_interest REAL)")
    ctx.execute("CREATE TABLE liquidations(symbol TEXT, ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL)")
    d = HOT_FROM
    while d <= TODAY:
        hi = DAY if d < TODAY else 10 * 3600          # 오늘은 10시까지(첫 걸음 시각)
        # 오늘 06~08시 칸은 비운다(수집기 정지 = 4시간 칸 «없음» -- 나중에 메운다)
        rows = [r for r in tape_rows(rng, d, 300, B0 + 50, 0, hi) if not (d == TODAY and ts(d, 4 * 3600) <= r[0] < ts(d, 8 * 3600))]
        tape.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)", [(hs, *r) for r in rows])
        ctx.executemany("INSERT INTO oi_1s VALUES (?,?,?)", [(hs, ts(d, s) * 1000 + 7, 1.6e6 + rng.uniform(-1e5, 1e5)) for s in range(0, hi, 1800)])
        ctx.executemany("INSERT INTO liquidations VALUES (?,?,?,?,?,?)",
                        [(sym, ts(d, rng.randint(0, hi - 1)) * 1000, rng.choice(["long", "short"]), 1.0, 3000.0, round(rng.uniform(10, 50000), 2)) for _ in range(8)])
        d += timedelta(days=1)
    tape.commit(); ctx.commit()
    shutil.copy(out / "hot_tape.sqlite", out / "hot_tape_start.sqlite")   # 시험은 이 사본에서 시작해 걸음마다 같은 행을 더한다
    shutil.copy(out / "hot_ctx.sqlite", out / "hot_ctx_start.sqlite")

    def more(t0: int, t1: int, n: int, base: int | None = None) -> list:
        """걸음 사이에 hot 에 더할 행(오늘 [t0, t1))."""
        rows = [r for r in tape_rows(rng, TODAY, n, base or B0 + 60, t0 - ts(TODAY), t1 - ts(TODAY))]
        oi = [(hs, s * 1000 + 11, 1.7e6 + rng.uniform(-1e5, 1e5)) for s in range(t0, t1, 900)]
        lq = [(sym, rng.randint(t0, t1 - 1) * 1000, rng.choice(["long", "short"]), 1.0, 3000.0, round(rng.uniform(10, 50000), 2)) for _ in range(3)]
        return [(hs, *r) for r in rows], oi, lq

    # ── 걸음: (시각, 걸음 전에 hot 에 더할 것, 부를 것) ──
    T = ts(TODAY, 10 * 3600)
    plan = [
        (T, None),
        (T + 10, more(T, T + 10, 40)),                   # 오늘 30초 캐시 안 -- 새 행이 아직 안 보인다
        (T + 31, more(T + 10, T + 31, 40)),              # 30초 지나 다시 셈
        (T + 2 * 3600 + 100, more(T + 31, T + 2 * 3600 + 100, 200)),   # 12시 칸 경계 넘음(08시 칸 막 닫힘 · 정착 2분 전)
        # 정착(2분) 지남 → 08시 칸 굳음. 굳기 전에 08시 칸에 늦은 체결(수집기 늦게 씀)이 들어온다 -- 정착 뒤 다시 세면 들어가야 한다
        (T + 2 * 3600 + 260, ("late", more(T + 2 * 3600 + 100, T + 2 * 3600 + 260, 30), more(ts(TODAY, 8 * 3600), ts(TODAY, 12 * 3600), 15))),
        # 빈 칸(04시)을 수집기가 메움 → 10분 다시 보기에 나타남 · 굳은 08시 칸에 또 늦은 체결 -- 굳은 칸은 안 바뀌어야 한다(다시 세기가 04시부터라 08시도 읽는다)
        (T + 2 * 3600 + 900, ("fill", ts(TODAY, 4 * 3600), ts(TODAY, 8 * 3600), more(ts(TODAY, 8 * 3600), ts(TODAY, 12 * 3600), 15))),
    ]
    clock = {"now": 0.0}
    m.time = SimpleNamespace(time=lambda: clock["now"])   # _ctx_from_lake 의 1시간 캐시 시계(원본은 time.time)
    lake = lambda stream, start, end, cols: data_store.read("binance", stream, coin, start and start.isoformat(), end and end.isoformat(), columns=cols)  # noqa: E731
    def read_lake(stream, start, end, cols):              # server.py _fpd_lake 그대로(예외 = 없음)
        try:
            return lake(stream, start, end, cols)
        except Exception:  # noqa: BLE001
            return None
    rows_of = lambda db: (lambda sql, params: data_store.read_rows(db, sql, params) if db.exists() else [])  # noqa: E731 -- server.py _fpd_rows
    F = m.DailyFootprint(read_lake, rows_of(out / "hot_tape.sqlite"), rows_of(out / "hot_ctx.sqlite"), hist_dir=OUT / "hist")
    # 반올림 전 값(열은 4자리 · 칸은 3자리로 반올림돼 합 차례·정렬 안정성의 끝자리가 안 보인다): 원본 day_from_tape 를 같은 묶음에 그대로
    raw = []
    for d, g in m.tape_rows_by_day(read_lake, rows_of(out / "hot_tape.sqlite"), LAKE_DAYS[0], TODAY).items():
        a = (g.ts_sec.to_numpy(), g.price_bin.to_numpy(), g.buy_qty.to_numpy(float), g.sell_qty.to_numpy(float))
        sm, (cb, cbu, cse) = m.day_from_tape(*a)
        raw.append({"day": d, "in": [a[0].tolist(), a[1].tolist(), a[2].tolist(), a[3].tolist()], "summ": sm,
                    "cells": [cb.tolist(), cbu.tolist(), cse.tolist()]})
    steps = []
    adds = []
    for now, add in plan:
        if add is not None:
            if add[0] == "fill":
                fill = [(hs, *r) for r in tape_rows(rng, TODAY, 60, B0 + 55, add[1] - ts(TODAY), add[2] - ts(TODAY))]
                add = (fill + add[3][0], add[3][1], add[3][2])
            elif add[0] == "late":
                add = (add[1][0] + add[2][0], add[1][1] + add[2][1], add[1][2] + add[2][2])
            tape.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)", add[0])
            ctx.executemany("INSERT INTO oi_1s VALUES (?,?,?)", add[1])
            ctx.executemany("INSERT INTO liquidations VALUES (?,?,?,?,?,?)", add[2])
            tape.commit(); ctx.commit()
        adds.append(add)
        clock["now"] = float(now)
        st = {"now": float(now), "calls": []}
        def call(kind, *a):
            try:
                v = getattr(F, kind)(float(now), *a)
                st["calls"].append({"kind": kind, "args": list(a), "body": json.dumps(v)})
            except ValueError as e:
                st["calls"].append({"kind": kind, "args": list(a), "error": str(e)})
        call("days_payload")
        call("cells_payload", "2026-10-20", "2026-11-21", 5)
        call("cells_payload", "2026-11-15", "2026-11-21", 1)
        call("cells_payload", "2026-11-10", "2026-11-21", 7)          # 7칸 묶기: XRP 하한 k×0.0005 가 십진으로 안 떨어지는 칸(round 6 이 갈린다)
        call("cells_payload", "2026-11-21", "2026-11-20", 5)          # 거꾸로 = range
        call("cells_payload", "2015-01-01", "2026-11-21", 5)          # 4000일 넘음 = range
        call("h4_payload")
        h4 = json.loads(st["calls"][-1]["body"])["cols"]["d"]   # 바로 앞 = h4_payload
        call("h4_cells_payload", h4[0], h4[-1], 3)
        call("h4_cells_payload", h4[-3], h4[-1], 1)
        call("h4_cells_payload", h4[0], h4[-1], 7)
        steps.append(st)
    # 시험이 같은 hot 상태를 걸음마다 다시 만들 수 있게: 처음 DB 사본 + 걸음별 더한 행
    tape.close(); ctx.close()
    fx = {"about": "원본 footprint_daily.DailyFootprint 를 합성 lake·소급본·hot 위에서 고정 시계로 부른 값(rtserve/tools/gen_fpd_fixture.py, 시드 %d)" % SEED,
          "asset": asset, "steps": steps, "raw": raw, "adds": [None if a is None else {"tape": a[0], "oi": a[1], "liq": a[2]} for a in adds]}
    (out / f"fpd_{asset}.json").write_text(json.dumps(fx))
    n_calls = sum(len(s["calls"]) for s in steps)
    print(f"{out / f'fpd_{asset}.json'}: 원시 묶음 {len(raw)} · 걸음 {len(steps)} · 호출 {n_calls} · days 날 {len(json.loads(steps[0]['calls'][0]['body'])['cols']['d'])} · "
          f"h4 칸 {[len(json.loads(next(c for c in s['calls'] if c['kind'] == 'h4_payload')['body'])['cols']['d']) for s in steps]}")


def main() -> None:
    for a in ("eth", "xrp"):
        run(a)


if __name__ == "__main__":
    main()

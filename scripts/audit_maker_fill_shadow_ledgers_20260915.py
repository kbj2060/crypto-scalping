#!/usr/bin/env python3
"""peg-maker 집행 섀도우 원장 감사 (2026-09-15).

`docs/experiments/server_load_audit_and_shadow_retirements_20260915.md` 가 인용하는
숫자를 재현한다. 읽기 전용 -- 원장에 아무것도 쓰지 않는다.

## ⭐왜 all-in 인가
`filled == True` 만 평균내면 **체결률이 낮은 정책이 유리해 보인다**. static 은 8~10%가
타임아웃 후 taker 로 폴백하는데 그 8.7~17.0bp 가 통째로 빠지기 때문이다. 원장은 미체결
leg 에도 `cost_bp` 를 taker 폴백가로 남기므로, 정책 비교는 **필터 없이 전량 평균**이 맞다.
체결분만 보면 static 이 1.4~2.3bp 싸 보이지만 전량으로는 -0.03~+0.38bp 로 동률이다.

## 워커가 쓰기락을 쥐고 있다
DuckDB 는 read_only 연결도 락을 요구하므로 가동 중 워커의 DB 는 직접 못 연다. 사본을 뜬다.
⚠️**본체 mtime 이 오래됐다고 "안 쓰고 있다"로 읽지 말 것** -- 체크포인트가 안 됐을 뿐
WAL 이 자라고 있을 수 있다(2026-09-15 실제로 본체 12KB / WAL 7.9MB 였다). 사본은 .wal 도
함께 떠야 하고, 사본은 WAL 재생이 필요하므로 read_only 가 아니라 쓰기 모드로 연다.

사용:
    python scripts/audit_maker_fill_shadow_ledgers_20260915.py            # 전 심볼
    python scripts/audit_maker_fill_shadow_ledgers_20260915.py --symbols ETH BTC
"""
from __future__ import annotations

import argparse
import datetime
import math
import pathlib
import shutil
import statistics as st
import tempfile

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGERS = {"ETH": "", "BTC": "_btc", "SOL": "_sol", "XRP": "_xrp", "HYPE": "_hype"}


def load(suffix: str, workdir: pathlib.Path):
    """원장 사본을 떠서 legs 를 읽는다. .wal 이 있으면 함께 복사해 재생시킨다."""
    src = ROOT / f"data/live/maker_fill_shadow{suffix}.duckdb"
    if not src.exists():
        return None
    dst = workdir / src.name
    shutil.copy2(src, dst)
    wal = src.with_suffix(".duckdb.wal")
    if wal.exists():
        shutil.copy2(wal, dst.with_suffix(".duckdb.wal"))
    con = duckdb.connect(str(dst))
    rows = con.execute(
        "select recorded_at_utc, policy, filled, cost_bp, trigger "
        "from maker_fill_shadow_legs order by recorded_at_utc").fetchall()
    con.close()
    return rows


def ci95(xs: list[float]) -> tuple[float, float]:
    """평균과 95% 반폭. n<2 면 반폭 nan."""
    m = st.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def report(name: str, rows) -> None:
    if not rows:
        print(f"{name}: 원장 없음 / 행 0"); return
    span = (rows[-1][0] - rows[0][0]).total_seconds() / 86400
    trig = {}
    for r in rows:
        trig[r[4]] = trig.get(r[4], 0) + 1
    print(f"\n=== {name}  legs {len(rows):,}  {rows[0][0]} ~ {rows[-1][0]}  ({span:.1f}일)  trigger {trig}")

    allin = {}
    for pol in sorted({r[1] for r in rows}):
        g = [r for r in rows if r[1] == pol]
        a = [r[3] for r in g if r[3] is not None]                      # 전량(미체결 taker 폴백 포함)
        f = [r[3] for r in g if r[2] and r[3] is not None]             # 체결분만
        u = [r[3] for r in g if not r[2] and r[3] is not None]
        if not a:
            print(f"   {pol:8s} n={len(g):>6,}  비용 기록 0"); continue
        m, h = ci95(a)
        allin[pol] = 2 * m
        print(f"   {pol:8s} n={len(g):>6,}  체결률 {100*len(f)/len(g):5.1f}%  "
              f"미체결 {st.mean(u) if u else float('nan'):6.2f}bp  "
              f"전량왕복 {2*m:5.2f}bp (±{2*h:.2f})  체결만왕복 {2*st.mean(f):5.2f}bp  "
              f"착시 {2*(m-st.mean(f)):+.2f}")
    if "peg" in allin and "static" in allin:
        print(f"   └ 전량 왕복 peg−static = {allin['peg']-allin['static']:+.2f}bp")

    # 추가 데이터가 아직 추정치를 움직이는가 (peg 기준)
    peg = [(r[0], r[3]) for r in rows if r[1] == "peg" and r[3] is not None]
    if len(peg) >= 40:
        h_ = len(peg) // 2
        a = [v for _, v in peg[:h_]]
        b = [v for _, v in peg[h_:]]
        d = 2 * (st.mean(b) - st.mean(a))
        dh = 2 * 1.96 * math.sqrt((st.stdev(a)**2)/len(a) + (st.stdev(b)**2)/len(b))
        cut = peg[-1][0] - datetime.timedelta(days=5)
        early = [v for t, v in peg if t <= cut]
        full = [v for _, v in peg]
        moved = 2 * (st.mean(full) - st.mean(early)) if early else float("nan")
        print(f"   수렴: 전반 {2*st.mean(a):.2f} / 후반 {2*st.mean(b):.2f}  Δ {d:+.2f}bp (±{dh:.2f}) "
              f"{'유의' if abs(d) > dh else '무의'}  |  마지막5일이 옮긴 폭 {moved:+.2f}bp "
              f"(그 5일 표본 {len(full)-len(early):,})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=list(LEDGERS))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        wd = pathlib.Path(td)
        for name in a.symbols:
            report(name.upper(), load(LEDGERS[name.upper()], wd))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rtcore 미시 값(situation.rs micro · okx_oi_rows) 일치 시험의 입력·정답지 (2026-11-21). 커밋한다(tests/fixtures/micro_eth.json).

정답 = 원본 식:
- qi = micro_ref.qi(import).
- mid·spread·vol60·liq60 = server.py collect_micro_ref 1937~1980 줄(닫힌 함수 안 -- 옮겨 적음).
- OKX OI 5분봉 = coin_flow.py 805~812(옮겨 적음) · okx_d = server.py api_oi_5m(옮겨 적음).
입력은 합성 사건(@bookTicker · @trade(늦은 체결 섞음) · @forceOrder · OKX open-interest(봉 건너뜀 섞음)) + 1초 틱. 네트워크 없음.
    python3 rtcore/tools/gen_micro_fixture.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
import micro_ref as mref  # noqa: E402

SEED, BS, KEEP = 20261122, 300, 288


def okx_d_of(okx_oi_5m: dict) -> dict:
    """server.py api_oi_5m 의 okx_d(옮겨 적음)."""
    okx_d: dict[int, tuple[float, float]] = {}
    prev = None
    for ob in sorted(okx_oi_5m):
        o, c = okx_oi_5m[ob]
        okx_d[ob] = ((c - o) if prev is None or ob - prev[0] > BS else (c - prev[1]), c)
        prev = (ob, c)
    return okx_d


def main() -> None:
    rng = random.Random(SEED)
    t0 = 1_763_700_000 // BS * BS + 37
    events, ticks = [], []
    t = float(t0)
    px, oi = 3000.0, 2_000_000.0
    tid = 0
    for _ in range(3500):
        t += rng.expovariate(1 / 0.6)
        r = rng.random()
        ms = int(t * 1000)
        rx = ms + rng.randint(0, 300)
        if r < 0.55:
            px += rng.gauss(0, 0.4)
            tid += 1
            late = rng.random() < 0.03
            ex = ms - rng.randint(1000, 4000) if late else ms   # 늦은 체결(앞 초 칸)
            events.append(["trade", ex, rx, tid, round(px, 2), round(rng.expovariate(1 / 2), 3), rng.random() < 0.5])
        elif r < 0.85:
            b = round(px - rng.choice([0.01, 0.02, 0.05]), 2)
            events.append(["book", ms, rx, b, round(rng.uniform(0, 80), 3), round(b + rng.choice([0.01, 0.02, 0.1]), 2),
                           rng.choice([0.0, round(rng.uniform(0, 80), 3)])])
        elif r < 0.9:
            lm = int(t) * 1000 + 250 if rng.random() < 0.3 else ms   # 틱(x.25초)과 정확히 60초 떨어진 청산 = 경계 동률
            events.append(["liq", lm, max(rx, lm), rng.random() < 0.5, round(rng.expovariate(1 / 5), 3), round(px, 2)])
        else:
            if rng.random() < 0.02:
                t += rng.choice([400, 700])                      # OI 가 봉을 건너뜀(Δ = 봉 안)
            oi += rng.gauss(0, 300)
            events.append(["oi", int(t * 1000), int(t * 1000) + 50, round(oi, 4)])
    # 사건 받은 시각 순 · 틱은 1초마다(일부 건너뜀)
    events.sort(key=lambda e: e[2])
    end = events[-1][2] / 1000
    k = t0 + 5.25
    while k < end:
        ticks.append(round(k, 3))
        k += 1.0 if rng.random() < 0.9 else 3.0

    sec: dict[int, list[float]] = {}
    liq_events: deque = deque(maxlen=5000)
    okx_oi_5m: dict[int, list[float]] = {}
    book = None
    out = []
    ei = 0
    for now in ticks:
        while ei < len(events) and events[ei][2] / 1000 <= now:
            e = events[ei]
            if e[0] == "trade":
                cell = sec.setdefault(e[1] // 1000, [0.0] * 7)
                cell[4 + (1 if e[6] else 0)] += e[5]
                cell[6] = e[4]
            elif e[0] == "book":
                book = {"bidPrice": e[3], "bidQty": e[4], "askPrice": e[5], "askQty": e[6]}
            elif e[0] == "liq":
                liq_events.append({"ts_ms": e[1], "side": "long" if e[3] else "short", "usd": e[4] * e[5]})
            else:   # coin_flow.py 801~812 옮겨 적음
                v, ts_ms = e[3], e[1]
                if v > 0 and ts_ms > 0:
                    s = ts_ms // 1000
                    ob = s // BS * BS
                    if ob in okx_oi_5m:
                        okx_oi_5m[ob][1] = v
                    else:
                        okx_oi_5m[ob] = [v, v]
                        for old in [x for x in okx_oi_5m if x < ob - KEEP * BS]:
                            del okx_oi_5m[old]
            ei += 1
        now_sec = int(now)
        row = {"now": now}   # 틱마다 전부 싣는다(쌓은 값과 비교)
        if book is not None:   # server.py 1937·1938·1988 옮겨 적음
            row["qi"] = mref.qi(float(book["bidQty"]), float(book["askQty"]))
            row["mid"] = (float(book["bidPrice"]) + float(book["askPrice"])) / 2.0
            row["spread"] = float(book["askPrice"]) - float(book["bidPrice"])
        by_sec = sec   # 1966~1968
        vol1s = [(s, cell[4] + cell[5]) for s, cell in by_sec.items() if s >= now_sec - 90]
        row["vol60"] = sum(v for s, v in vol1s if now_sec - 61 < s < now_sec)
        cut = now * 1000 - 60_000   # 1977~1980
        ev60 = [e for e in liq_events if e["ts_ms"] >= cut]
        row["liq60"] = {"long": sum(e["usd"] for e in ev60 if e["side"] == "long"),
                        "short": sum(e["usd"] for e in ev60 if e["side"] == "short"), "n": len(ev60)}
        row["okx_d"] = [[ob, d, c] for ob, (d, c) in sorted(okx_d_of(okx_oi_5m).items())]
        out.append(row)
    p = ROOT / "rtcore" / "tests" / "fixtures" / "micro_eth.json"
    p.write_text(json.dumps({"events": events, "ticks": out}))
    print(f"{p}: 사건 {len(events)} · 틱 {len(out)} · OKX 봉 {len(okx_oi_5m)} · 청산 있는 틱 "
          f"{sum(1 for r in out if r['liq60']['n'])} · {p.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rtcore 30분 상황 근거(situation.rs) 일치 시험의 입력·정답지 (2026-11-21). 커밋한다(tests/fixtures/situation_eth.json).

정답 = 원본 함수를 부른 값:
- server.py `_situation_inputs` -- make_app 안 닫힌 함수라 import 가 안 된다 → 아래에 글자 그대로 옮겨 적음(주석만 줄임).
  server.py 를 고치면 이 사본도 맞춰야 한다.
- situation.classify · micro_ref.deriv_from_ring -- import.
- prev_dir 잇기 = compute_situation 의 `(situation_state.get("now") or {}).get("dir") or 0`(옮겨 적음 · 한 줄).
- 마크 링 = collect_mark_price 의 링 넣기·10초 정리(옮겨 적음 · 웹소켓 루프 안이라 떠서 돌릴 수 없다).
입력은 합성(가짜 체결 경로 · 캔들 빠진 봉 · 1초 링 · 청산 · BTC · 마크 사건). 네트워크 없음 · 실제 data/ 안 건드림.
    python3 rtcore/tools/gen_situation_fixture.py
"""
from __future__ import annotations

import json
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtcore" / "tests" / "fixtures"


def _no_net(*_a, **_k):
    raise RuntimeError("gen_situation_fixture: 네트워크 금지")


socket.socket.connect = _no_net
socket.create_connection = _no_net
sys.path.insert(0, str(ROOT / "dashboard"))
import micro_ref as mref  # noqa: E402
import situation as sit  # noqa: E402

SEED = 20261121
BS = 300
BUCKET = 0.5          # ETH 풋프린트 칸 폭(coin_flow FLOW_SPECS)
MARK_PRICE_RING_S, OI_1S_FLUSH_SECONDS = 7200, 10.0   # server.py 상수(옮겨 적음)


def _situation_inputs(now: float, asset: str = "eth") -> dict[str, Any]:
    """server.py make_app 안 `_situation_inputs`(2026-11-21 판, 2154~2210) 글자 그대로 옮겨 적음 -- 닫힌 함수라 import 가 안 된다.
    쓰는 바깥 이름(cctx·flows·sit·mref·FOOTPRINT_BAR_SECONDS)은 이 모듈 전역으로 넣는다."""
    C, F = cctx[asset], flows[asset]
    situation_state, micro_state, mark_ring = C.situation_state, C.micro_state, C.mark_ring
    footprint_state, FOOTPRINT_BUCKET = F.footprint_state, F.spec.bucket   # noqa: N806
    bar_start = int(now) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS
    fbars = footprint_state["bars"]
    oi_map = {int(b): (None if gap else float(dl)) for b, dl, _c, _n, gap in (situation_state.get("oi_5m") or [])}
    liq_map: dict[int, tuple[float, float]] = {}
    try:
        for r in ((situation_state.get("liq_5m") or {}).get("bars") or []):
            ts = int(datetime.fromisoformat(r["ts"]).timestamp())
            liq_map[ts // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS] = (float(r.get("long_usd") or 0), float(r.get("short_usd") or 0))
    except Exception:  # noqa: BLE001
        pass
    candles = {int(c["time"]): c for c in (situation_state.get("candles") or []) if not c.get("partial")}
    bars = []
    veto_prev = None
    want = [b for b in sorted(fbars) if b < bar_start][-(2 * sit.WINDOW + 1):]
    for b in want:
        cells = fbars[b]
        buy = sum(v[0] for v in cells.values()); sell = sum(v[1] for v in cells.values())
        c = candles.get(b)
        veto_prev = (c or {}).get("veto", veto_prev)
        prices = [k * FOOTPRINT_BUCKET for k in cells]
        bars.append({"time": b,
                     "high": float(c["high"]) if c else (max(prices) if prices else None),
                     "low": float(c["low"]) if c else (min(prices) if prices else None),
                     "close": float(c["close"]) if c else None,
                     "delta": buy - sell,
                     "oi_delta": oi_map.get(b), "liq_long": liq_map.get(b, (0, 0))[0], "liq_short": liq_map.get(b, (0, 0))[1],
                     "veto": veto_prev})
    by_sec = footprint_state["sec"]
    for bar, b in zip(bars, want):
        if bar["close"] is None:
            secs = [sc for sc in by_sec if b <= sc < b + FOOTPRINT_BAR_SECONDS and by_sec[sc][6]]
            if secs:
                bar["close"] = by_sec[max(secs)][6]
            else:
                cells = fbars[b]; tot = sum(v[0] + v[1] for v in cells.values())
                bar["close"] = (sum(k * FOOTPRINT_BUCKET * (v[0] + v[1]) for k, v in cells.items()) / tot) if tot > 0 else None
    btc: dict[str, float] = {}
    bc = {int(c["time"]): c for c in (situation_state.get("btc_candles") or [])}
    if len(want) >= sit.WINDOW + 1 and want[-sit.WINDOW - 1] in bc and want[-1] in bc:
        c0, c1 = bc[want[-sit.WINDOW - 1]], bc[want[-1]]
        span = [bc[t] for t in want[-sit.WINDOW:] if t in bc]
        btc = {"move_bp": (c1["close"] - c0["close"]) / c0["close"] * 1e4,
               "range_bp": (max(c["high"] for c in span) - min(c["low"] for c in span)) / c1["close"] * 1e4}
    mp = micro_state["payload"] if micro_state["payload"].get("available") else {}
    return {"bars": bars, "mid": mp.get("mid"), "book": situation_state.get("book") or {},
            "deriv": mref.deriv_from_ring(mark_ring, sit.WINDOW * FOOTPRINT_BAR_SECONDS, sit.BASIS_PCT), "btc": btc}


FOOTPRINT_BAR_SECONDS = BS
cctx: dict = {}
flows: dict = {}


def mark_push(ring: dict, st: dict, ts_ms: int, v: tuple, now: float) -> None:
    """collect_mark_price 의 링 몫(server.py 옮겨 적음)."""
    ring[ts_ms // 1000] = v
    if now - st["trimmed_at"] >= OI_1S_FLUSH_SECONDS:
        for old in [sc for sc in ring if sc < now - MARK_PRICE_RING_S]:
            del ring[old]
        st["trimmed_at"] = now


def make_case(rng: random.Random, kind: str) -> dict:
    nbars = rng.randint(10, 22)
    t0 = 1_763_700_000 // BS * BS + rng.randint(0, 50) * BS
    px = rng.uniform(2000, 3500)
    drift = rng.choice([-1, 1]) * rng.uniform(0, 6)
    bars: dict[int, dict[int, list[float]]] = {}
    candles: dict[int, list[float]] = {}
    sec: dict[int, list[float]] = {}
    tid = 0
    for i in range(nbars):
        b = t0 + i * BS
        if kind == "flip" and i % 4 == 0:
            drift = (-1 if (i // 4) % 2 else 1) * rng.uniform(10, 14)   # 강한 추세를 4봉마다 뒤집음(중립 없이 직접 반전)
        elif kind != "flip" and rng.random() < 0.15:
            drift = rng.choice([-1, 1]) * rng.uniform(0, 8)   # 국면 바꿈(슈미트 진입·이탈)
        cells: dict[int, list[float]] = {}
        h = lo = None
        last = None
        n = rng.randint(0 if kind == "sparse" else 5, 60)
        times = sorted(rng.uniform(b, b + BS) for _ in range(n))
        for t in times:
            px += drift * rng.uniform(0, 0.4) + rng.gauss(0, 1.2)
            p = round(px * 100) / 100
            q = rng.choice([0.5, 1.5, 2.5, 1.0]) if kind == "ties" else round(rng.expovariate(1 / 3), 3)   # ties: 델타 합이 .5 로 끝남(반올림 동률)
            sell = rng.random() < 0.5
            k = round(p / BUCKET)
            c = cells.setdefault(k, [0.0] * 6)
            c[1 if sell else 0] += q
            h = p if h is None else max(h, p)
            lo = p if lo is None else min(lo, p)
            last = p
            tid += 1
            s = int(t)
            row = sec.setdefault(s, [0.0] * 7)
            row[4 if not sell else 5] += q
            row[6] = p
        # 칸 키 순서로(코어 BTreeMap) -- situation.rs ponytail 주석
        bars[b] = {k: cells[k] for k in sorted(cells)}
        if cells and kind != "zero_sec" and rng.random() < 0.8:   # zero_sec: 캔들 없이 1초 링 대체만
            candles[b] = [h, lo, last]
    if kind == "zero_sec":
        for s in rng.sample(sorted(sec), k=len(sec) // 3):
            sec[s][6] = 0.0
    btc = None
    if kind != "no_btc":
        bp = rng.uniform(60000, 100000)
        bdrift = rng.uniform(-0.0015, 0.0015)   # BTC 추세(동행·단독 문턱 0.35 근처를 고루)
        btc = {}
        for i in range(nbars):
            o = bp
            bp *= 1 + bdrift + rng.gauss(0, 0.002)
            if rng.random() < 0.9:
                btc[t0 + i * BS] = [max(o, bp) * (1 + rng.uniform(0, 0.001)), min(o, bp) * (1 - rng.uniform(0, 0.001)), bp]
    liq = {t0 + i * BS: [rng.choice([0.0, rng.uniform(0, 5e5)]), rng.choice([0.0, rng.uniform(0, 5e5)])]
           for i in range(nbars) if rng.random() < 0.5}
    # 걸음: 봉 경계 근처를 여러 번(형성 봉 안 · 경계 직후) -- 슈미트가 이어지는지
    nows = []
    t = t0 + 4 * BS + rng.uniform(0, BS)   # 처음 몇 걸음은 완결 봉 < 7(ok=false)
    while t < t0 + nbars * BS + BS:
        nows.append(round(t, 3))
        t += 7.3 if kind == "grow" else rng.choice([1.0, 7.3, 61.0, 150.0, 299.0, 300.0])   # grow: 임계 표본이 59→60 을 넘는 순간을 촘촘히
    # 마크 사건(1초 · 가끔 빈 초) -- 긴 링(지평 1800초 Δ 표본 60개 이상)은 일부 경우만(파일 크기)
    span = {"long": 4500, "short": 1500, "grow": 2600}.get(kind, rng.choice([0, 400, 1200]))
    marks = []
    m0 = int(nows[0]) - span
    mk = px
    for s in range(m0, int(nows[-1]) + 1):
        if rng.random() < (0.06 if kind in ("long", "short", "grow") else 0.85):   # 긴 링 아닌 경우는 듬성듬성(파일 크기)
            continue
        mk += rng.gauss(0, 0.8)
        idx = mk * (1 - rng.gauss(0.0002, 0.0004))
        fr = rng.choice([0.0001, 0.00005, -0.0002, 0.00031, 0.0002])
        ms = s * 1000 + rng.randint(0, 999)
        lag = rng.randint(1, 400) if rng.random() < 0.8 else 1000 - ms % 1000   # 받은 시각이 정수 초(링 경계 7,200초 동률)
        marks.append([ms, round(mk, 2), round(idx, 4), fr, lag])
    mids = [None if rng.random() < 0.2 else round(px + rng.uniform(-1, 1), 2) for _ in nows]
    return {"kind": kind, "t0": t0, "bars": bars, "candles": candles, "sec": sec, "btc": btc, "liq": liq,
            "nows": nows, "mids": mids, "marks": marks, "mark_t0": m0 - 3.0}


def run_case(case: dict) -> list:
    ring: dict = {}
    st = {"trimmed_at": case["mark_t0"]}
    mi = 0
    out = []
    prev_res: dict | None = None
    for now, mid in zip(case["nows"], case["mids"]):
        # 받은 시각(ms 지연 포함)이 now 이하인 마크 사건까지 넣는다
        while mi < len(case["marks"]) and (case["marks"][mi][0] + case["marks"][mi][4]) / 1000 <= now:
            ts, mk, idx, fr, lag = case["marks"][mi]
            mark_push(ring, st, ts, (mk, idx, fr), (ts + lag) / 1000)
            mi += 1
        bar_start = int(now) // BS * BS
        fbars = {b: v for b, v in case["bars"].items() if b <= bar_start}
        sec = {s: v for s, v in case["sec"].items() if now - 660 < s <= now}
        ss = {"oi_5m": [], "book": {},
              "liq_5m": {"bars": [{"ts": datetime.fromtimestamp(b, timezone.utc).isoformat(), "long_usd": v[0], "short_usd": v[1]}
                                  for b, v in case["liq"].items() if b < bar_start]},
              "candles": [{"time": b, "high": v[0], "low": v[1], "close": v[2]} for b, v in case["candles"].items() if b < bar_start],
              "btc_candles": None if case["btc"] is None else
              [{"time": b, "high": v[0], "low": v[1], "close": v[2]} for b, v in case["btc"].items() if b < bar_start]}
        C = SimpleNamespace(situation_state=ss, mark_ring=ring,
                            micro_state={"payload": {"available": mid is not None, "mid": mid}})
        F = SimpleNamespace(footprint_state={"bars": fbars, "sec": sec}, spec=SimpleNamespace(bucket=BUCKET))
        cctx["eth"], flows["eth"] = C, F
        inp = _situation_inputs(now, "eth")
        inp["prev_dir"] = (prev_res or {}).get("dir") or 0
        res = sit.classify(inp)
        prev_res = res
        out.append({"now": now, "ring_keys": [min(ring), max(ring), len(ring)] if ring else None, "res": res})
    return out


def main() -> None:
    rng = random.Random(SEED)
    kinds = ["long", "long", "short", "sparse", "zero_sec", "zero_sec", "no_btc", "flip", "flip", "flip", "ties", "ties", "grow"] + ["mix"] * 24
    cases = []
    for kind in kinds:
        c = make_case(rng, kind)
        c["steps"] = run_case(c)
        cases.append(c)
    # 키를 글자로(JSON)
    for c in cases:
        c["bars"] = {str(b): {str(k): v for k, v in cells.items()} for b, cells in c["bars"].items()}
        for key in ("candles", "sec", "liq"):
            c[key] = {str(b): v for b, v in c[key].items()}
        if c["btc"] is not None:
            c["btc"] = {str(b): v for b, v in c["btc"].items()}
    FIX.mkdir(parents=True, exist_ok=True)
    p = FIX / "situation_eth.json"
    p.write_text(json.dumps({"bucket": BUCKET, "cases": cases}, ensure_ascii=False))
    steps = [s for c in cases for s in c["steps"]]
    ok = [s for s in steps if s["res"]["ok"]]
    print(f"{p}: 경우 {len(cases)} · 걸음 {len(steps)} · ok {len(ok)} · 방향 "
          f"{ {d: sum(1 for s in ok if s['res']['dir'] == d) for d in (-1, 0, 1)} } · "
          f"거부 {sum(1 for s in ok if s['res']['evidence']['reject'])} · 임계 있음 {sum(1 for s in ok if s['res']['evidence']['basis_thr_bp'] is not None)} · "
          f"BTC {sum(1 for s in ok if s['res']['evidence']['btc_rel'])} · lead≠0 {sum(1 for s in ok if s['res']['evidence']['lead'])} · "
          f"{p.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()

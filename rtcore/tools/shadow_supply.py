#!/usr/bin/env python3
"""실서버 그림자 대조: rtcore(ws://127.0.0.1:8788) 1초 수급 vs 파이썬 /api/supply-1s (2026-11-20). 읽기 전용 · aiohttp 만.

rtcore 는 구독 → 스냅숏 1회 → 밀기(바뀐 초만, 초 단위로 덮어씀)로 칸을 짓고, 파이썬은 10초마다 since=0 전량을 받아 덮어쓴다.
끝나면 칸(lane)마다 «양쪽 다 닫힌 초»만 필드별로 비교해 한 줄씩 찍는다.
  닫힌 초: bn·liq < 둘 중 작은 now · spot < 작은 spotNow · okx·okxLiq < 작은 okxNow · okxOi < 둘 중 작은 마지막 OI 초
           bybitLiq ≤ 파이썬 마지막 응답 시각 − 10초(파이썬은 수집기 2초 flush·1초 캐시·3초 지연 뒤에 본다)
  비교 범위 시작: rtcore 가 처음 본 초(bn 은 bn 칸 첫 초 · spot·okx 는 제 칸 첫 초 · 청산은 그 원천 체결 칸 첫 초)
     -- 파이썬은 재시작 때 테이프에서 11분을 되살리지만 rtcore 는 안 한다(그 앞은 비교하지 않는다).
  안 보는 것: partial(진행 중) · oi(코어가 안 옮김) · meta 는 끝에 나란히 찍기만.
    python3 rtcore/tools/shadow_supply.py eth 300        # 코인 · 몇 초
    python3 rtcore/tools/shadow_supply.py --selftest     # 네트워크 없이 비교 함수만
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

CORE = "ws://127.0.0.1:8788"
PY = "http://127.0.0.1:8787/api/supply-1s"
LANES = ("bn", "liq", "spot", "okx", "okxLiq", "okxOi", "bybitLiq")
KEY = {"bn": "seconds", "liq": "liq", "spot": "spot", "okx": "okx", "okxLiq": "okxLiq", "okxOi": "okxOi", "bybitLiq": "bybitLiq"}


def put(view: dict, payload: dict) -> None:
    for lane in LANES:
        for r in payload.get(KEY[lane]) or []:
            view.setdefault(lane, {})[int(r[0])] = list(r[1:])


def same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x == y or abs(x - y) <= 1e-9 * max(abs(x), abs(y)) for x, y in zip(a, b))


def compare(rt: dict, py: dict, rt_meta: dict, py_meta: dict, py_at: float) -> list[str]:
    """rt/py: 칸 -> {초: 값}. *_meta: 마지막 now·spotNow·okxNow. py_at: 파이썬 마지막 응답 시각(초)."""
    def first(lane):
        return min(rt.get(lane) or {0: 0})
    lo = {"bn": first("bn"), "liq": first("bn"), "bybitLiq": first("bn"), "spot": first("spot"),
          "okx": first("okx"), "okxLiq": first("okx"), "okxOi": first("okxOi")}
    m = lambda k: min(rt_meta.get(k) or 0, py_meta.get(k) or 0)  # noqa: E731
    top_oi = min(max(rt.get("okxOi") or {0: 0}), max(py.get("okxOi") or {0: 0}))
    hi = {"bn": m("now"), "liq": m("now"), "spot": m("spotNow"), "okx": m("okxNow"), "okxLiq": m("okxNow"),
          "okxOi": top_oi, "bybitLiq": int(py_at) - 10 + 1}
    out = []
    for lane in LANES:
        a, b = rt.get(lane) or {}, py.get(lane) or {}
        keys = sorted(s for s in set(a) | set(b) if lo[lane] <= s < hi[lane])
        both = [s for s in keys if s in a and s in b]
        bad = [s for s in both if not same(a[s], b[s])]
        only_py = [s for s in keys if s not in a]
        only_rt = [s for s in keys if s not in b]
        eg = "; ".join(f"{s} rt={a[s]} py={b[s]}" for s in bad[:3])
        out.append(f"{lane}: 비교 초 {len(both)} · 불일치 {len(bad)} · 파이썬만 {len(only_py)}{only_py[:3]} · "
                   f"rtcore만 {len(only_rt)}{only_rt[:3]} · 범위 [{lo[lane]}, {hi[lane]})" + (f" · 첫 불일치 {eg}" if eg else ""))
    return out


async def main(coin: str, secs: float) -> None:
    import aiohttp
    sym = f"{coin.lower()}usdt"
    rt: dict = {}
    py: dict = {}
    rt_last: dict = {}
    py_last: dict = {}
    py_at = 0.0
    n_push = 0
    end = time.time() + secs
    async with aiohttp.ClientSession() as http:
        async def poll_py() -> None:
            nonlocal py_at
            while time.time() < end:
                async with http.get(PY, params={"asset": coin.lower(), "since": "0"}, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    p = await r.json()
                put(py, p)
                py_last.update(p)
                py_at = time.time()
                await asyncio.sleep(min(10.0, max(0.0, end - time.time())))

        async def read_core() -> None:
            nonlocal n_push
            async with http.ws_connect(CORE) as ws:
                await ws.send_str(json.dumps({"subscribe": [sym]}))
                while time.time() < end:
                    try:
                        msg = await ws.receive(timeout=max(0.1, end - time.time()))
                    except asyncio.TimeoutError:
                        break
                    if msg.type is not aiohttp.WSMsgType.TEXT:
                        raise RuntimeError(f"rtcore 연결 끊김 {msg.type!r}")
                    m = json.loads(msg.data)
                    sup = m.get("supply") if m.get("type") == "snapshot" else (m if m.get("type") == "supply1s" else None)
                    if sup is not None:
                        n_push += m.get("type") == "supply1s"
                        put(rt, sup)
                        rt_last.update(sup)

        await asyncio.gather(read_core(), poll_py())
        await poll_once(http, coin, py, py_last)
        py_at = time.time()
    print(f"그림자 대조 {coin} {secs:.0f}초 · rtcore 밀기 {n_push} · now rt {rt_last.get('now')} py {py_last.get('now')}")
    for line in compare(rt, py, rt_last, py_last, py_at):
        print(line)
    for k in ("spotMeta", "okxMeta"):
        print(f"{k}: rt {rt_last.get(k)} · py {py_last.get(k)}")


async def poll_once(http, coin: str, py: dict, py_last: dict) -> None:
    async with http.get(PY, params={"asset": coin.lower(), "since": "0"}) as r:
        p = await r.json()
    put(py, p)
    py_last.update(p)


def selftest() -> None:
    rt, py = {}, {}
    put(rt, {"seconds": [[10, 1, 0, 0, 0, 1, 0, 3.0], [11, 2, 0, 0, 0, 2, 0, 3.0]], "liq": [[10, 1.0, 0, 3, 0]],
             "bybitLiq": [[50, 1, 0, 2, 0]]})
    put(py, {"seconds": [[9, 9, 0, 0, 0, 9, 0, 3.0], [10, 1, 0, 0, 0, 1, 0, 3.0], [11, 2, 0, 0, 0, 2.5, 0, 3.0]],
             "liq": [[10, 1.0, 0, 3, 0]], "bybitLiq": [[50, 1, 0, 2, 0]]})
    out = dict(line.split(":", 1) for line in compare(rt, py, {"now": 12}, {"now": 12}, 100))
    assert "비교 초 2 · 불일치 1" in out["bn"] and "파이썬만 0" in out["bn"], out["bn"]     # 9 는 rtcore 시작 전이라 범위 밖
    assert "비교 초 1 · 불일치 0" in out["liq"], out["liq"]
    assert "비교 초 1 · 불일치 0" in out["bybitLiq"], out["bybitLiq"]
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        asyncio.run(main(sys.argv[1], float(sys.argv[2])))

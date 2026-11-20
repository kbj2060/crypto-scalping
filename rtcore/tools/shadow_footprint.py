#!/usr/bin/env python3
"""실서버 그림자 대조: rtcore(ws://127.0.0.1:8788/rt) 풋프린트 vs 파이썬 /api/footprint?bars=144&since=0 (2026-11-20). 읽기 전용 · aiohttp 만.

rtcore 쪽 화면 = 스냅숏 1회 + 밀기를 proto.rs `Footprint` 주석대로 합친 것: full 이면 통째로 갈고, 아니면 실린 봉마다 칸을 **가격으로
덮어쓰고**(안 실린 칸은 그대로) dhl 은 통째로 갈고(키가 없으면 지움) → drop 의 봉을 지우고 → barsExpected 넘으면 오래된 봉부터 버린다.
파이썬은 10초마다 since=0 전량을 받아 갈아 끼운다. 받을 때마다 닫힌 봉을 한 줄, 끝에 전체를 찍는다.
  비교 봉: rtcore 가 **통째로 본** 봉 = rtcore 첫 스냅숏의 가장 이른 봉(없으면 처음 밀려온 봉)의 **다음 봉**부터
     -- rtcore 는 재시작 복원이 없어 첫 봉은 중간부터 센 반쪽이고, 파이썬은 스냅샷·백필로 되살린 봉이다.
  닫힌 봉(엄격): 봉 끝 + SLACK 초 ≤ min(rtcore 마지막 메시지 시각, 파이썬 마지막 응답 시각) -- 두 쪽이 본 마지막 체결 시각이 달라
     봉 경계 직후 몇 초는 한쪽에만 늦은 체결이 있을 수 있다. 그보다 새 봉(형성 중)은 «참고» 로만 센다.
  칸: 가격 키 = round(가격 / 칸 폭) · 필드 6개(매수·매도·고래매수·고래매도·리테일매수·리테일매도) 상대 1e-9.
  dhl(tests/footprint_parity.rs 머리 주석의 가지): 둘 다 같음(없음 포함) → 같음 · 파이썬만 있음 → 몸통만(코어는 테이프가 없다)
     · 코어만 있음 → 코어가 먼저 냄(파이썬은 봉 끝 6분 뒤 테이프로 낸다) · 둘 다 있는데 다름 → 불일치.
    python3 rtcore/tools/shadow_footprint.py eth 900      # 코인 · 몇 초(5분봉이라 닫힌 봉을 보려면 600초 넘게)
    python3 rtcore/tools/shadow_footprint.py --selftest
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

CORE = "ws://127.0.0.1:8788/rt"
PY = "http://127.0.0.1:8787/api/footprint"
BS = 300
SLACK = 5
FIELDS = ("매수", "매도", "고래매수", "고래매도", "리테일매수", "리테일매도")


def put(view: dict, fp: dict) -> None:
    """proto.rs Footprint 받는 규칙. view = {봉: {"cells": {가격키: [6]}, "dhl": [3] | None}}."""
    if fp.get("full"):
        view.clear()
    bucket = fp["bucket"]
    for b in fp.get("bars") or []:
        bar = view.setdefault(int(b["time"]), {"cells": {}, "dhl": None})
        for lv in b["levels"]:
            bar["cells"][round(lv[0] / bucket)] = list(lv[1:])
        bar["dhl"] = b.get("dhl")
    for t in fp.get("drop") or []:
        view.pop(int(t), None)
    while len(view) > fp["barsExpected"]:
        view.pop(min(view))


def same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x == y or abs(x - y) <= 1e-9 * max(abs(x), abs(y)) for x, y in zip(a, b))


def compare(rt: dict, py: dict, lo: int, seen_until: float) -> dict:
    """rt/py: put() 의 view. lo = 비교 첫 봉. seen_until = min(양쪽 마지막 시각)(초). 닫힌 봉·형성 중 봉 따로 센다."""
    out = {}
    for kind in ("닫힌", "형성"):
        keys = sorted(b for b in set(rt) | set(py) if b >= lo and ((b + BS + SLACK <= seen_until) == (kind == "닫힌")))
        t = {"봉": 0, "칸": 0, "불일치": 0, "파이썬만칸": 0, "rtcore만칸": 0, "파이썬만봉": [], "rtcore만봉": [],
             "필드": [0] * 6, "예": [], "dhl": {"같음": 0, "몸통만": 0, "코어가먼저": 0, "불일치": 0}}
        for b in keys:
            if b not in rt or b not in py:
                t["rtcore만봉" if b in rt else "파이썬만봉"].append(b)
                continue
            t["봉"] += 1
            a, p = rt[b]["cells"], py[b]["cells"]
            for k in sorted(set(a) | set(p)):
                if k not in a or k not in p:
                    t["rtcore만칸" if k in a else "파이썬만칸"] += 1
                    if len(t["예"]) < 3:
                        t["예"].append(f"{b}/{k} rt={a.get(k)} py={p.get(k)}")
                    continue
                t["칸"] += 1
                if not same(a[k], p[k]):
                    t["불일치"] += 1
                    for j, (x, y) in enumerate(zip(a[k], p[k])):
                        t["필드"][j] += not same([x], [y])
                    if len(t["예"]) < 3:
                        t["예"].append(f"{b}/{k} rt={a[k]} py={p[k]}")
            da, dp = rt[b]["dhl"], py[b]["dhl"]
            d = t["dhl"]
            if da == dp or (da and dp and same(da, dp)):
                d["같음"] += 1
            elif da is None:
                d["몸통만"] += 1
            elif dp is None:
                d["코어가먼저"] += 1
            else:
                d["불일치"] += 1
                if len(t["예"]) < 3:
                    t["예"].append(f"{b}/dhl rt={da} py={dp}")
        out[kind] = t
    return out


def line(name: str, t: dict) -> str:
    f = " ".join(f"{n}{c}" for n, c in zip(FIELDS, t["필드"]) if c)
    return (f"{name}: 봉 {t['봉']} · 칸 {t['칸']} · 불일치 칸 {t['불일치']}{' (' + f + ')' if f else ''} · "
            f"파이썬만 칸 {t['파이썬만칸']} · rtcore만 칸 {t['rtcore만칸']} · 파이썬만 봉 {t['파이썬만봉'][:3]} · rtcore만 봉 {t['rtcore만봉'][:3]} · "
            f"dhl {t['dhl']}" + (f" · 첫 불일치 {'; '.join(t['예'])}" if t["예"] else ""))


async def main(coin: str, secs: float) -> None:
    import aiohttp
    sym = f"{coin.lower()}usdt"
    rt: dict = {}
    py: dict = {}
    st = {"lo": None, "rt_at": 0.0, "py_at": 0.0, "push": 0}
    end = time.time() + secs

    def report(final: bool) -> None:
        if st["lo"] is None:
            print("rtcore 에서 봉을 아직 못 받았다")
            return
        r = compare(rt, py, st["lo"], min(st["rt_at"], st["py_at"]))
        print(line(f"닫힌 봉 [{st['lo']}~) 밀기 {st['push']}", r["닫힌"]), flush=True)
        if final:
            print(line("형성 중(참고)", r["형성"]))

    async with aiohttp.ClientSession() as http:
        async def poll_once() -> None:
            async with http.get(PY, params={"asset": coin.lower(), "bars": "144", "since": "0"},
                                timeout=aiohttp.ClientTimeout(total=10)) as r:
                put(py, await r.json())
            st["py_at"] = time.time()

        async def poll_py() -> None:
            while time.time() < end:
                await poll_once()
                report(False)
                await asyncio.sleep(min(10.0, max(0.0, end - time.time())))

        async def read_core() -> None:
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
                    fp = m.get("footprint") if m.get("type") == "snapshot" else (m if m.get("type") == "footprint" else None)
                    if fp is None:
                        continue
                    st["push"] += m["type"] == "footprint"
                    put(rt, fp)
                    st["rt_at"] = m["tx_us"] / 1e6 if m.get("tx_us") else time.time()
                    if st["lo"] is None and rt:
                        st["lo"] = min(rt) + BS   # 첫 봉은 반쪽 -- 머리 주석

        await asyncio.gather(read_core(), poll_py())
        await poll_once()
    print(f"그림자 대조 풋프린트 {coin} {secs:.0f}초 · rtcore 봉 {len(rt)} · 파이썬 봉 {len(py)}")
    report(True)


def selftest() -> None:
    lv = lambda px, *v: [px, *v]  # noqa: E731
    head = {"bucket": 0.5, "barsExpected": 4}
    # 파이썬(전량): 봉 0·300·600·900 -- 0 은 rtcore 가 중간에 붙은 반쪽 봉, 900 은 형성 중
    py_resp = {**head, "full": True, "bars": [
        {"time": 0, "levels": [lv(100.0, 9, 9, 0, 0, 0, 0)]},
        {"time": 300, "levels": [lv(100.0, 1, 2, 0, 0, 1, 0), lv(100.5, 3, 0, 3, 0, 0, 0)], "dhl": [-1.0, 2.0, 2.0]},
        {"time": 600, "levels": [lv(101.0, 1, 1, 0, 0, 0, 0)]},
        {"time": 900, "levels": [lv(101.0, 5, 0, 0, 0, 0, 0)]}]}
    core = [
        {**head, "full": True, "bars": [{"time": 0, "levels": [lv(100.0, 1, 1, 0, 0, 0, 0)]},
                                        {"time": 300, "levels": [lv(100.0, 1, 2, 0, 0, 1, 0)], "dhl": [0.0, 1.0, 0.5]}]},
        # 밀기 = 바뀐 칸만: 100.0 은 안 실렸다 -- 덮어쓰기면 남고, 봉을 통째로 갈면 사라진다(음성 대조)
        {**head, "full": False, "bars": [{"time": 300, "levels": [lv(100.5, 3, 0, 3, 0, 0, 0)], "dhl": [-1.0, 2.0, 2.0]}]},
        {**head, "full": False, "bars": [{"time": 600, "levels": [lv(101.0, 1, 1, 0, 0, 0, 0)], "dhl": [0.0, 0.0, 0.0]},
                                         {"time": 900, "levels": [lv(101.0, 4, 0, 0, 0, 0, 0)]}]},   # 형성 중: 파이썬보다 체결 1건 덜 봄
        {**head, "full": False, "bars": [{"time": 600, "levels": []}]},   # dhl 키 없음 = 지움
    ]
    rt, py = {}, {}
    for m in core:
        put(rt, m)
    put(py, py_resp)
    assert sorted(rt) == [0, 300, 600, 900] and rt[600]["dhl"] is None and sorted(rt[300]["cells"]) == [200, 201]
    lo = 0 + BS                                   # 첫 스냅숏의 가장 이른 봉(0)의 다음 봉
    r = compare(rt, py, lo, 900 + SLACK)          # 600 까지 닫힘 · 900 형성 중
    c, f = r["닫힌"], r["형성"]
    # 일치(양성 없음): 닫힌 두 봉 칸 3개 전부 같다 · 반쪽 첫 봉(0)은 안 본다 · 형성 중 봉 차이는 엄격 쪽에 안 섞인다
    assert (c["봉"], c["칸"], c["불일치"], c["파이썬만칸"], c["rtcore만칸"], c["파이썬만봉"], c["rtcore만봉"]) == (2, 3, 0, 0, 0, [], []), c
    assert c["dhl"] == {"같음": 2, "몸통만": 0, "코어가먼저": 0, "불일치": 0}, c["dhl"]
    assert (f["봉"], f["불일치"], f["필드"][0]) == (1, 1, 1), f
    # 반쪽 첫 봉을 넣으면(lo=0) 불일치가 생긴다 -- 시작 봉을 빼는 규칙이 실제로 일한다
    assert compare(rt, py, 0, 900 + SLACK)["닫힌"]["불일치"] == 1
    # 음성 대조(대조기 결함 흉내): 밀기를 «봉 통째로 갈기»로 합치면 안 실린 칸이 사라져 가짜 «파이썬만 칸»이 난다
    naive: dict = {}
    for m in core:
        if m["full"]:
            naive.clear()
        for b in m["bars"]:
            naive[b["time"]] = {"cells": {round(x[0] / 0.5): x[1:] for x in b["levels"]}, "dhl": b.get("dhl")}
    assert compare(naive, py, lo, 900 + SLACK)["닫힌"]["파이썬만칸"] == 2, "이 사례가 봉 통째 갈기 결함을 못 잡는다"
    # 양성: 닫힌 봉 값 하나·dhl 하나를 틀리게 → 잡는다 · 파이썬만 dhl → 몸통만 · 코어만 dhl → 코어가 먼저
    rt[300]["cells"][201][2] = 2.999
    rt[300]["dhl"] = [-1.0, 2.0, 2.5]
    py[600]["dhl"] = [0.0, 0.0, 0.0]
    del py[600]["cells"][202]
    c = compare(rt, py, lo, 900 + SLACK)["닫힌"]
    assert (c["불일치"], c["필드"], c["rtcore만칸"]) == (1, [0, 0, 1, 0, 0, 0], 1), c
    assert c["dhl"] == {"같음": 0, "몸통만": 1, "코어가먼저": 0, "불일치": 1}, c["dhl"]
    py[600]["dhl"], rt[600]["dhl"] = None, [0.0, 0.0, 0.0]
    assert compare(rt, py, lo, 900 + SLACK)["닫힌"]["dhl"]["코어가먼저"] == 1
    # 한쪽에만 있는 닫힌 봉
    del py[600]
    assert compare(rt, py, lo, 900 + SLACK)["닫힌"]["rtcore만봉"] == [600]
    put(rt, {**head, "full": False, "bars": [{"time": 1200, "levels": []}]})
    assert sorted(rt) == [300, 600, 900, 1200], "창(barsExpected) 넘친 오래된 봉 버리기"
    put(rt, {**head, "full": False, "bars": [], "drop": [900]})
    assert sorted(rt) == [300, 600, 1200], "drop"
    print(line("selftest 예시 줄", c))
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        asyncio.run(main(sys.argv[1], float(sys.argv[2])))

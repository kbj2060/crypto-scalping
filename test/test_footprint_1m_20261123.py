"""/api/footprint?tf=1m (2026-11-23 «풋프린트에 1m 도») -- 오프라인 앱으로 1분봉 응답 모양·m1_from 자르기·since 증분을 본다.
실행: python test/test_footprint_1m_20261123.py"""
import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "test")]
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from dashboard import server  # noqa: E402
from offline_app import offline_app  # noqa: E402


def find_cell(fn, name, seen=None):
    seen = seen if seen is not None else set()
    if id(fn) in seen or not callable(fn):
        return None
    seen.add(id(fn))
    for c in getattr(fn, "__closure__", None) or ():
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if fn.__code__.co_freevars[list(fn.__closure__).index(c)] == name:
            return v
        if callable(v) and (hit := find_cell(v, name, seen)) is not None:
            return hit
    return None


async def main():
    app = offline_app(server)
    h = next(r.handler for r in app.router.routes() if getattr(r.resource, "canonical", "") == "/api/footprint")
    flows = find_cell(h, "flows")
    st = flows["eth"].footprint_state
    t0 = 1_800_000_000 // 60 * 60
    st.update(ready=True, m1_from=t0 + 60,
              bars1m={t0: {4000: [1.0, 2, 0, 0, 0, 0]}, t0 + 60: {4000: [3.0, 1, 0, 0, 0, 0], 4001: [0.5, 0, 0, 0, 0.5, 0]}, t0 + 120: {4002: [1.0, 0, 0, 0, 0, 0]}},
              ohlc1m={t0: [2000, 2000, 2000, 2000, 0, 0], t0 + 60: [2000.0, 2000.5, 2000.0, 2000.5, 0, 0], t0 + 120: [2001, 2001, 2001, 2001, 0, 0]})
    c = TestClient(TestServer(app))
    await c.start_server()
    try:
        j = await (await c.get("/api/footprint?asset=eth&tf=1m")).json()
        assert j["barSeconds"] == 60 and j["full"] is True, j
        assert [b["time"] for b in j["bars"]] == [t0 + 60, t0 + 120], "m1_from 앞(덜 센) 봉은 뺀다"
        b = j["bars"][0]
        assert b["ohlc"] == [2000.0, 2000.5, 2000.0, 2000.5], b
        assert b["levels"][0][:3] == [2000.0, 3.0, 1.0] and len(b["levels"][0]) == 7, b["levels"]
        j2 = await (await c.get(f"/api/footprint?asset=eth&tf=1m&since={t0 + 120}")).json()
        assert j2["full"] is False and [x["time"] for x in j2["bars"]] == [t0 + 120], j2
        j5 = await (await c.get("/api/footprint?asset=eth")).json()
        assert j5["barSeconds"] == 300, "tf 없으면 5분봉 그대로"
    finally:
        await c.close()
    print("OK footprint 1m: 모양·m1_from·since·5분봉 불변")


asyncio.run(main())

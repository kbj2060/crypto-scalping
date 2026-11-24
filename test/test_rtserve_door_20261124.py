"""파이썬 → rtserve 넘기기(2026-11-24 · server.py rtserve_door): /api/tick-bars · /api/footprint?tf=1m 은 쿼리 원문 그대로 rtserve 로,
나머지(/api/footprint 5분봉 등)는 파이썬이 그대로 답하고, rtserve 가 닫혀 있으면 502 rtserve_unavailable. 실행: python test/test_rtserve_door_20261124.py"""
import asyncio
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "test")]
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from dashboard import server  # noqa: E402
from offline_app import offline_app  # noqa: E402


async def main():
    got = []

    async def any_(request):
        got.append((request.path, request.rel_url.raw_query_string))
        return web.Response(body=b'{"door":"rtserve"}', content_type="application/json", headers={"Cache-Control": "no-cache"})
    fake = web.Application()
    fake.router.add_route("GET", "/{tail:.*}", any_)
    rs = TestServer(fake, host="127.0.0.1")
    await rs.start_server()
    os.environ["RTSERVE_DOOR"] = f"127.0.0.1:{rs.port}"
    c = TestClient(TestServer(offline_app(server)))
    await c.start_server()
    try:
        r = await c.get("/api/tick-bars?asset=eth&since=1,2,3,4,5")
        assert r.status == 200 and (await r.json()) == {"door": "rtserve"} and r.headers.get("Cache-Control") == "no-cache", r
        r = await c.get("/api/footprint?asset=eth&tf=1m&since=0")
        assert (await r.json()) == {"door": "rtserve"}
        assert got == [("/api/tick-bars", "asset=eth&since=1,2,3,4,5"), ("/api/footprint", "asset=eth&tf=1m&since=0")], got
        r = await c.get("/api/footprint?asset=eth")
        assert (await r.json()).get("barSeconds") == 300 and len(got) == 2, "5분봉은 파이썬이 답한다"
        await rs.close()
        r = await c.get("/api/tick-bars?asset=eth")
        assert r.status == 502 and (await r.json())["error"] == "rtserve_unavailable"
    finally:
        await c.close()
    print("OK rtserve_door: 두 주소 넘김(쿼리 원문) · 5분봉 그대로 · 닫히면 502")


asyncio.run(main())

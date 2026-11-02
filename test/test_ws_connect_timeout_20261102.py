"""WS 접속 시한 (2026-11-02).   python -m pytest -q test/test_ws_connect_timeout_20261102.py
TCP 는 받고 HTTP 응답은 안 주는 서버에서 ws_open 은 시한 안에 TimeoutError 로 끝나야 하고(전엔 영원히 대기),
정상 WS 서버에는 붙어야 한다(대조군)."""
import asyncio
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_trade_tape_collector_20260916 as m  # noqa: E402


def test_half_open_endpoint_times_out_and_healthy_connects():
    async def go():
        m.WS_CONNECT_S = 1.0
        hung = await asyncio.start_server(lambda r, w: asyncio.sleep(3600), "127.0.0.1", 0)
        app = web.Application()
        app.router.add_get("/ws", lambda req: _echo(req))
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        ok_port = site._server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
            t = time.time()
            try:
                await m.ws_open(s, f"ws://127.0.0.1:{hung.sockets[0].getsockname()[1]}/", heartbeat=20)
                raise AssertionError("반쯤 열린 접속이 붙으면 안 된다")
            except asyncio.TimeoutError:
                assert time.time() - t < 3, "시한 안에 끝나야 한다"
            async with await m.ws_open(s, f"ws://127.0.0.1:{ok_port}/ws") as ws:      # 대조군: 정상 서버
                await ws.send_str("x")
                assert (await ws.receive(timeout=2)).data == "x"
        hung.close()
        await runner.cleanup()
    asyncio.run(go())


async def _echo(req):
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    async for msg in ws:
        await ws.send_str(msg.data)
    return ws

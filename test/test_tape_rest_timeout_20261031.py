"""2026-10-31: 응답 없는 REST 가 WS 수신 루프를 영원히 세우지 않는다 -- 시한(REST_TIMEOUT_S) 안에 예외로 돌아온다."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.live_trade_tape_collector_20260916 as T  # noqa: E402


def test_hung_rest_times_out(monkeypatch):
    from aiohttp import ClientSession, ClientTimeout, web

    async def hang(request):
        await asyncio.sleep(4)

    async def main():
        app = web.Application()
        app.router.add_get("/k", hang)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/k"
        monkeypatch.setattr(T, "REST_TIMEOUT_S", 0.3)
        async with ClientSession(timeout=ClientTimeout(total=None)) as session:   # 수집기 세션과 같은 «시한 없음»
            fetch = T.binance_minute_fetcher(session, "ethusdt", 0.1, url, url)
            t0 = time.monotonic()
            try:
                await asyncio.wait_for(fetch(60), 10)
            except asyncio.TimeoutError:
                pass
            else:
                raise AssertionError("시한 없이 끝났다")
            assert time.monotonic() - t0 < 2.0, "요청별 시한(0.3초)이 아니라 바깥 시한에서 끝났다"
        await runner.cleanup()

    asyncio.run(main())

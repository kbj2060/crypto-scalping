"""/api/stream: 서버가 든 커서로 닫힌 초를 **정확히 한 번씩** 보내고, 상황은 바뀔 때만. 2026-09-24.

`python3 test/test_stream_supply_cursor_20260924.py` -- 클로저 안이라 소스를 떼어 내 실제 aiohttp
서버로 돌린다. 커서 갱신을 빼면 매 틱 5분치를 다시 보내 «두 번»에서 실패한다.
"""
import asyncio, json, re, time
from collections import deque
from pathlib import Path
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

src = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")
def cut(a, b):
    i = src.index(a); return re.sub(r"^    ", "", src[i:src.index(b, i)], flags=re.M)
chunk = (cut("    def situation_payload()", "    async def load_chart_klines_frames(")
         + cut("    def supply_1s_payload(", "    async def api_supply_1s("))
by_sec: dict = {}
ns = {"Any": object, "web": web, "json": json, "asyncio": asyncio, "time": time, "STREAM_TICK_S": 0.02,
      "footprint_state": {"sec": by_sec}, "okx_sec": {}, "spot_sec": {}, "okx_oi_1s": {}, "oi_1s": {},
      "okx_liq_events": deque(), "liq_events": deque(),
      "okx_state": {"connected": False, "last_trade_ms": 0, "last_oi_ms": 0, "errors": 0},
      "spot_state": {"connected": False, "last_trade_ms": 0, "errors": 0},
      "SUPPLY_1S_SECONDS": 660, "RETAIL_MAX_USD": 1e4, "WHALE_MIN_USD": 1e5, "FOOTPRINT_SYMBOL": "ETHUSDT",
      "OKX_INST": "ETH-USDT-SWAP", "SITUATION_HORIZON_S": 1800, "fo_state": {}, "mp_state": {},
      "situation_state": {"log": [], "now": {"ok": True}, "computed_at": 1.0},
      "sit": type("S", (), {"calibration": staticmethod(lambda log, h: {})})}
exec(compile(chunk, "stream", "exec"), ns)

async def main():
    app = web.Application(); app.router.add_get("/api/stream", ns["api_stream"])
    for s in range(1000, 1010):
        by_sec[s] = [1.0] * 7
    async with TestClient(TestServer(app)) as cl:
        resp = await cl.get("/api/stream?supply=1&since=0")
        got, sit, buf, t_end, added = [], 0, "", time.monotonic() + 1.5, 0
        while time.monotonic() < t_end:
            buf += (await resp.content.readany()).decode()
            while "\n\n" in buf:
                msg, buf = buf.split("\n\n", 1)
                ev = re.search(r"event: (\w+)", msg).group(1); data = json.loads(msg.split("data: ", 1)[1])
                if ev == "supply":
                    got += [r[0] for r in data["seconds"]]
                    assert data["partial"]["bn"][0] not in got           # 진행 중인 초는 목록 밖으로 따로 온다
                else:
                    sit += 1
            if added < 20:                                              # 초가 계속 닫힌다
                by_sec[1010 + added] = [1.0] * 7; added += 1
            if added == 10:
                ns["situation_state"]["computed_at"] = 2.0
            await asyncio.sleep(0.03)
        resp.close()
    closed = [s for s in sorted(by_sec) if s < max(by_sec)]
    assert sorted(got) == closed, (len(got), len(closed))              # 빠짐도 두 번도 없다
    assert sit == 2, sit                                                # 처음 한 번 + 바뀐 한 번
    print("ok", len(got), "closed seconds, situation", sit)

asyncio.run(main())

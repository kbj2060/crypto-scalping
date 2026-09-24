"""OKX 합산: /api/oi-5m 봉별 Δ 에 OKX 를 더하고(캐시는 안 고친다), /api/footprint 는 진행 중 봉의
OKX 몫(okxLive)을 따로 준다. 2026-09-24. `python3 test/test_okx_sum_oi_footprint_20260924.py`
"""
import asyncio, json, re
from pathlib import Path
from aiohttp import web

src = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")
def cut(a, b):
    i = src.index(a); return re.sub(r"^    ", "", src[i:src.index(b, i)], flags=re.M)
chunk = cut("    async def api_oi_5m(", "    async def api_supply_profile(") \
    + cut("    async def api_footprint(", "    def supply_1s_payload(")
cached = [[600, 10.0, 1000.0, 50, 0], [900, -4.0, 996.0, 50, 0], [1200, 2.0, 998.0, 50, 0]]
async def swr_cached(key, ttl, produce, **kw): return cached
ns = {"web": web, "asyncio": asyncio, "swr_cached": swr_cached, "oi_1s": {}, "OI_5M_WINDOW_BARS": 48,
      "OI_5M_BAR_SECONDS": 300, "FOOTPRINT_SYMBOL": "ETHUSDT", "NOCACHE": {}, "oi_5m_buckets": None,
      # OKX 는 900 봉부터(재기동). 900 은 봉 안 Δ, 1200 은 직전 봉 끝 기준 Δ
      "okx_oi_5m": {900: [500.0, 503.0], 1200: [503.5, 501.0]},
      "okx_fp": {"first_bar": 600}, "okx_bars": {1200: {5480: [1.0, 2.0, 0.0, 0.0, 1.0, 0.0]}},
      "footprint_state": {"bars": {900: {5480: [3.0] * 6}, 1200: {5481: [4.0] * 6}}, "agg_bars": set(), "ready": True},
      "footprint_window_bars": lambda r: 12, "FOOTPRINT_BUCKET": 0.5, "FOOTPRINT_BAR_SECONDS": 300,
      "RETAIL_MAX_USD": 1e4, "WHALE_MIN_USD": 1e5, "Any": object}
exec(compile(chunk, "okxsum", "exec"), ns)

class Req:
    def __init__(self, q): self.query = q

async def main():
    oi = json.loads((await ns["api_oi_5m"](Req({"bars": "3"}))).body)
    assert oi["venues"] == ["binance-perp", "okx-swap"]
    assert oi["bars"][0] == [600, 10.0, 1000.0, 50, 0]                 # OKX 기록 없는 봉 = 바이낸스만
    assert oi["bars"][1] == [900, -1.0, 1499.0, 50, 0]                 # -4 + (503-500)
    assert oi["bars"][2] == [1200, 0.0, 1499.0, 50, 0]                 # 2 + (501-503)
    assert cached[1] == [900, -4.0, 996.0, 50, 0], "swr 캐시를 고쳤다 -- 다음 요청에 두 번 더해진다"
    fp = json.loads((await ns["api_footprint"](Req({"since": "0"}))).body)
    assert [b["time"] for b in fp["bars"]] == [900, 1200]
    assert fp["okxLive"] == {"time": 1200, "levels": [[2740.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0]]}
    ns["okx_fp"]["first_bar"] = 0                                         # OKX 없음 -> okxLive 없음
    assert json.loads((await ns["api_footprint"](Req({"since": "0"}))).body)["okxLive"] is None
    print("ok")

asyncio.run(main())

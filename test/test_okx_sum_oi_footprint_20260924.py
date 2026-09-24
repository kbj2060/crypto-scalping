"""OKX 합산: /api/oi-5m 봉별 Δ 에 OKX 를 더하고(캐시는 안 고친다), /api/footprint 는 진행 중 봉의
OKX 몫(okxLive)을 따로 준다. 2026-09-24. `python3 test/test_okx_sum_oi_footprint_20260924.py`
"""
import asyncio, json, re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from aiohttp import web

src = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")
def cut(a, b):
    i = src.index(a); return re.sub(r"^    ", "", src[i:src.index(b, i)], flags=re.M)
chunk = cut("    async def api_oi_5m(", "    async def api_supply_profile(") \
    + cut("    async def api_footprint(", "    def supply_1s_payload(") \
    + cut("    async def api_supply_profile(", "    # 🔴이 응답은 **페이지 로드를 막는다**") \
    + cut("    async def api_liquidation_5m_history(", "    async def api_position_sizing(")
cached = [[600, 10.0, 1000.0, 50, 0], [900, -4.0, 996.0, 50, 0], [1200, 2.0, 998.0, 50, 0]]
async def swr_cached(key, ttl, produce, **kw): return cached
ns = {"web": web, "asyncio": asyncio, "swr_cached": swr_cached, "oi_1s": {}, "OI_5M_WINDOW_BARS": 48,
      "OI_5M_BAR_SECONDS": 300, "FOOTPRINT_SYMBOL": "ETHUSDT", "NOCACHE": {}, "oi_5m_buckets": None,
      # OKX 는 900 봉부터(재기동). 900 은 봉 안 Δ, 1200 은 직전 봉 끝 기준 Δ
      "okx_oi_5m": {900: [500.0, 503.0], 1200: [503.5, 501.0]},
      "okx_fp": {"first_bar": 600}, "okx_bars": {1200: {5480: [1.0, 2.0, 0.0, 0.0, 1.0, 0.0]}},
      "footprint_state": {"bars": {900: {5480: [3.0] * 6}, 1200: {5481: [4.0] * 6}}, "agg_bars": set(), "ready": True},
      "footprint_window_bars": lambda r: 12, "FOOTPRINT_BUCKET": 0.5, "FOOTPRINT_BAR_SECONDS": 300,
      "RETAIL_MAX_USD": 1e4, "WHALE_MIN_USD": 1e5, "Any": object,
      # 청산: 바이낸스 캐시 3봉(600·900·1200) + OKX 이벤트(600 은 재기동 봉이라 제외돼야 한다)
      "datetime": datetime, "STALE_GRACE_SECONDS": 600, "_query_coin_asset": lambda r: r.query.get("asset", "eth"),
      "compute_liquidation_5m_history": None,
      "okx_liq_events": deque([{"ts_ms": 610_000, "side": "long", "usd": 999.0},
                               {"ts_ms": 950_000, "side": "long", "usd": 100.0},
                               {"ts_ms": 960_000, "side": "short", "usd": 40.0}])}
exec(compile(chunk, "okxsum", "exec"), ns)
liq_cached = {"warmed_up": True, "bars": [
    {"ts": datetime.fromtimestamp(t, timezone.utc).isoformat(), "long_usd": 10.0, "short_usd": 5.0,
     "events": 1, "partial": t == 1200} for t in (600, 900, 1200)]}
async def swr_any(key, ttl, produce, **kw):
    return liq_cached if key.startswith("liq5m") else cached
ns["swr_cached"] = swr_any

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
    # ── 수급 프로파일 = 풋프린트와 같은 봉·같은 합 (한 카드의 두 그림) ──
    prof = json.loads((await ns["api_supply_profile"](Req({}))).body)
    tot = lambda levels: [round(sum(l[i] for l in levels), 6) for i in range(1, 7)]
    fp_levels = [l for b in fp["bars"] for l in b["levels"]]
    assert prof["barCount"] == 2 and prof["venues"] == ["binance-perp", "okx-swap"]
    assert tot(prof["levels"]) == tot(fp_levels), (tot(prof["levels"]), tot(fp_levels))
    # ── 5분봉 청산 원: OKX 가 온전히 본 봉(600 초과)만 더한다 · 캐시 불변 · ETH 만 ──
    lq = json.loads((await ns["api_liquidation_5m_history"](Req({"asset": "eth"}))).body)
    b600, b900, b1200 = lq["bars"]
    assert "okx" not in b600 and b600["long_usd"] == 10.0, "재기동 봉(OKX 가 반쪽만 봄)에 더했다"
    assert b900["okx"] and b900["long_usd"] == 110.0 and b900["short_usd"] == 45.0 and b900["events"] == 3
    assert b1200["okx"] and b1200["long_usd"] == 10.0                      # OKX 이벤트 없는 봉도 표시는 합산판
    assert liq_cached["bars"][1]["long_usd"] == 10.0, "swr 캐시를 고쳤다 -- 30초 동안 요청마다 또 더해진다"
    assert json.loads((await ns["api_liquidation_5m_history"](Req({"asset": "btc"}))).body) == liq_cached
    ns["okx_fp"]["first_bar"] = 0                                         # OKX 없음 -> okxLive 없음
    assert json.loads((await ns["api_footprint"](Req({"since": "0"}))).body)["okxLive"] is None
    print("ok")

asyncio.run(main())

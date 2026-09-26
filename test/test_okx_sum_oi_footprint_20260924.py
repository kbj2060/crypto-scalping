"""OKX 합산: /api/oi-5m 봉별 Δ 에 OKX 를 더하고(캐시는 안 고친다), /api/footprint 는 진행 중 봉의
OKX 몫(okxLive)을 따로 준다. 2026-09-24. `python3 test/test_okx_sum_oi_footprint_20260924.py`
"""
import asyncio, json, re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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
# 2026-09-26 코인별 흐름 엔진 -- 핸들러는 `?asset=` 로 엔진을 고른다. ETH 엔진은 위 값들을 그대로 묶는다.
def _flow(asset, symbol, bucket, dp, **state):
    return SimpleNamespace(spec=SimpleNamespace(asset=asset, symbol=symbol, bucket=bucket, price_dp=dp,
                                                okx_inst=f"{asset.upper()}-USDT-SWAP"), **state)
flows = {"eth": _flow("eth", "ETHUSDT", 0.5, 2, footprint_state=ns["footprint_state"], oi_1s=ns["oi_1s"],
                      okx_oi_5m=ns["okx_oi_5m"], okx_fp=ns["okx_fp"], okx_bars=ns["okx_bars"],
                      okx_liq_events=ns["okx_liq_events"]),
         # XRP: 칸 0.0003 -- 소수 둘째 자리로 반올림하면 이웃 칸이 한 값으로 뭉친다(가격 1.5363 -> 1.54)
         "xrp": _flow("xrp", "XRPUSDT", 0.0003, 4,
                      footprint_state={"bars": {1200: {5121: [1.0] * 6, 5122: [2.0] * 6}}, "agg_bars": set(), "ready": True},
                      oi_1s={}, okx_oi_5m={}, okx_fp={"first_bar": 0}, okx_bars={}, okx_liq_events=deque())}
def flow_for(q):
    f = flows.get(str(q.get("asset") or "eth").lower())
    if f is None:
        raise web.HTTPNotFound(reason="flow_off")
    return f
ns.update(flows=flows, flow_for=flow_for, SimpleNamespace=SimpleNamespace,
          liq_5m_state={"from_s": {}},   # 실시간 누적 없음 -> 캐시(liq_cached) 경로를 탄다
          # HL 고래 합산(09-24)은 이 시험 대상이 아니다 -- 있는 그대로 통과시킨다
          time=__import__("time"), FOOTPRINT_KEEP_BARS=288, hl_whale_liq_events=None,
          merge_hl_liq=lambda bars, ev, bar_s: bars,
          HL_LIQ_BY_ASSET={"eth": (None, 5.0, 2)})   # 코인별 HL DB 표(09-26) -- 이 시험은 ETH 만 본다
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
    # 09-25 커버리지 가드(test_oi5m_okx_coverage_guard): OKX 가 온전히 못 본 봉(600 = 재기동 봉)은 뺀다
    #   -- 한 창에 1거래소 봉·2거래소 봉이 섞이지 않게. 이 시험은 그 가드 전 기대값에 멈춰 있었다.
    assert [b[0] for b in oi["bars"]] == [900, 1200]
    assert oi["bars"][0] == [900, -1.0, 1499.0, 50, 0]                 # -4 + (503-500)
    assert oi["bars"][1] == [1200, 0.0, 1499.0, 50, 0]                 # 2 + (501-503)
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
    # ── 코인별(2026-09-26): XRP 칸 가격은 4자리 · 꺼진 코인은 ETH 대신 404 ──
    xfp = json.loads((await ns["api_footprint"](Req({"since": "0", "asset": "xrp"}))).body)
    assert xfp["symbol"] == "XRPUSDT" and xfp["bucket"] == 0.0003
    assert [l[0] for l in xfp["bars"][0]["levels"]] == [1.5363, 1.5366], xfp["bars"][0]["levels"]
    try:
        await ns["api_footprint"](Req({"since": "0", "asset": "btc"}))
        raise AssertionError("꺼진 코인에 ETH 풋프린트를 줬다")
    except web.HTTPNotFound:
        pass
    ns["okx_fp"]["first_bar"] = 0                                         # OKX 없음 -> okxLive 없음
    assert json.loads((await ns["api_footprint"](Req({"since": "0"}))).body)["okxLive"] is None
    print("ok")

asyncio.run(main())

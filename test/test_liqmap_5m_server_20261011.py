"""/api/liquidation-map 이 5분 경계 이력·같은 창 막대를 내는가 (2026-10-11 사용자 «5분봉이 닫힐 때마다» · «둘 다 5분봉 기준»).

진짜 load_liquidation_map 을 돌리고 거래소 HTTP 세션만 가짜로 -- klines 는 서버 실캡처 사본(test/fixtures/liqmap_eth_5m_20261011.json)의
테이프 5분봉과 그 1시간 묶음을, 지금 시각에 맞춰 민 것. 🔴바깥 요청 0(가짜 세션이 아닌 URL 이면 실패).
검사: 이력 144개 · 5분 간격 · 마지막 = 형성 5분봉 시작(= 마지막 닫힌 5분봉 끝) · 막대 known_until = 그 경계 ·
막대 칸 = 마지막 열 칸(같은 격자) · 값 비례 · 막대 current_price = 창 끝 종가 · 레벨은 닫힌 1시간봉 그대로.
실행: python -m pytest -q test/test_liqmap_5m_server_20261011.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app, patch_live_dir  # noqa: E402

from dashboard import server  # noqa: E402

FIX = json.loads((Path(__file__).resolve().parent / "fixtures/liqmap_eth_5m_20261011.json").read_text("utf-8"))


def klines(now: float) -> dict[str, list]:
    """사본 5분봉(마지막 = 형성 봉)을 지금 형성 봉에 맞춰 민다. 1시간봉 = 그 묶음(마지막 = 형성 1시간봉)."""
    shift = int(now) // 300 * 300 - FIX["bars5"][-1][0]
    rows5 = [[(t + shift) * 1000, cl * .1, hi * .1, lo * .1, cl * .1, v, (t + shift + 300) * 1000 - 1] + [0] * 5
             for t, lo, hi, v, cl in FIX["bars5"]]
    hours: dict[int, list] = {}
    for r in rows5:
        h = r[0] // 3_600_000 * 3_600_000
        if h not in hours:
            hours[h] = [h, r[1], r[2], r[3], r[4], 0.0, h + 3_600_000 - 1] + [0] * 5
        x = hours[h]; x[2] = max(x[2], r[2]); x[3] = min(x[3], r[3]); x[4] = r[4]; x[5] += r[5]
    return {"5m": rows5, "1h": list(hours.values())}


class FakeResp:
    def __init__(self, body): self.status, self._b = 200, body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self): return self._b


class FakeSession:
    calls: list = []

    def __init__(self, *a, **k): self.closed = False
    def get(self, url, params=None, timeout=None):
        assert url == "https://fapi.binance.com/fapi/v1/klines", url      # 그 밖의 바깥 요청은 실패
        FakeSession.calls.append(dict(params))
        return FakeResp(klines(time.time())[params["interval"]][-int(params["limit"]):])
    async def close(self): self.closed = True


class LiqMap5mTest(unittest.TestCase):
    def test_history_and_bars_share_5m_boundary(self):
        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                live, dash = Path(tmp) / "live", Path(tmp) / "dashboard"
                live.mkdir(); dash.mkdir()
                with mock.patch.object(server, "ClientSession", FakeSession), patch_live_dir(server, live), \
                     mock.patch.object(server, "DASHBOARD_DIR", dash):
                    async with TestClient(TestServer(offline_app(server))) as c:
                        t0 = time.perf_counter()
                        r = await c.get("/api/liquidation-map?asset=eth")
                        return r.status, await r.json(), time.perf_counter() - t0
        st, p, dt = asyncio.run(go())
        self.assertEqual(st, 200, p)
        h = p["heatmap_history"]
        ts = [int(np.datetime64(s["ts_utc"][:19]).astype("datetime64[s]").astype(int)) for s in h]
        now5 = int(time.time()) // 300 * 300
        self.assertEqual(len(h), 144)
        self.assertTrue(all(b - a == 300 for a, b in zip(ts, ts[1:])))
        self.assertEqual(ts[-1], now5, "마지막 열 = 형성 5분봉 시작(= 마지막 닫힌 5분봉 끝)")
        tp = p["tier_profile"]
        self.assertEqual(int(np.datetime64(tp["known_until"][:19]).astype("datetime64[s]").astype(int)), now5)
        bw = p["bin_width"]
        self.assertAlmostEqual(tp["bin_width"], bw, places=3)                 # 같은 격자(같은 기준가)
        col = {round(b["price"] / bw): b["weight_pct"] for b in h[-1]["bins"]}
        tot = np.sum([t["values"] for t in tp["tiers"]], axis=0)
        bar = {tp["lo"] + i: v for i, v in enumerate(tot) if v > 0}
        self.assertEqual(set(bar), set(col))
        sup = [k for k in col if k * bw < p["current_price"]]               # 열은 쪽별 정규화 -- 쪽마다 비례
        for side in (sup, [k for k in col if k not in sup]):
            if len(side) > 2:
                self.assertGreater(np.corrcoef([bar[k] for k in side], [col[k] for k in side])[0, 1], 0.99)
        self.assertAlmostEqual(tp["current_price"], round(FIX["bars5"][-2][4] * .1, 2), places=2)   # 창 끝 = 마지막 닫힌 5분봉 종가
        # 레벨 = 닫힌 1시간봉: 기준가 = 마지막 닫힌 1시간봉 종가
        last_closed_hour = (now5 // 3600 - 1) * 3600
        closes = {(t + now5 - FIX["bars5"][-1][0]): cl * .1 for t, _, _, _, cl in FIX["bars5"]}
        self.assertAlmostEqual(p["current_price"], closes[last_closed_hour + 3300], places=6)
        self.assertEqual(sorted(c["interval"] for c in FakeSession.calls), ["1h", "5m"])
        print(f"응답 {len(json.dumps(p)) // 1000} KB · 요청~응답 {dt * 1e3:.0f} ms · klines 요청 {FakeSession.calls}")


if __name__ == "__main__":
    unittest.main()

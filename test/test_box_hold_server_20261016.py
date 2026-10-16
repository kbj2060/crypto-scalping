"""/api/box-hold 가 닫힌 1분봉으로 박스 유지 확률을 내는가 (2026-10-16 사용자 «B로 배포»).

진짜 api_box_hold 를 돌리고 거래소 HTTP 세션만 가짜로(합성 1분봉 1500개 = 닫힌 1499 + 형성 1). 🔴바깥 요청 0(klines 가 아니면 실패).
검사: 200 · ok · 30분·1h·4h 셋 · 값 = dashboard/box_hold.payload(닫힌 봉만) 와 같음 · 형성 봉 제외 · 요청 = ETHUSDT 1m 1500 · 1분 캐시(두 번째 호출은 바깥 요청 없음).
실행: python -m pytest -q test/test_box_hold_server_20261016.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
from dashboard.box_hold import payload  # noqa: E402


def rows(now: float) -> list:
    rng = np.random.default_rng(5)
    t_form = int(now) // 60 * 60 * 1000                       # 형성 1분봉 시작
    t = t_form - np.arange(1499, -1, -1) * 60_000
    c = 2400 * np.exp(np.cumsum(rng.normal(0, 6e-4, 1500)))
    h, l = c * (1 + rng.uniform(0, 4e-4, 1500)), c * (1 - rng.uniform(0, 4e-4, 1500))
    return [[int(t[i]), c[i], h[i], l[i], c[i], 10.0, int(t[i]) + 59_999] + [0] * 5 for i in range(1500)]


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
        return FakeResp(rows(time.time())[-int(params["limit"]):])
    async def close(self): self.closed = True


class BoxHoldServerTest(unittest.TestCase):
    def test_box_hold_endpoint(self):
        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                live, dash = Path(tmp) / "live", Path(tmp) / "dashboard"
                live.mkdir(); dash.mkdir()
                with mock.patch.object(server, "ClientSession", FakeSession), mock.patch.object(server, "LIVE_DIR", live), \
                     mock.patch.object(server, "DASHBOARD_DIR", dash):
                    async with TestClient(TestServer(offline_app(server))) as c:
                        r1 = await c.get("/api/box-hold")
                        n1 = len(FakeSession.calls)
                        r2 = await c.get("/api/box-hold")
                        return r1.status, await r1.json(), n1, len(FakeSession.calls), await r2.json()
        FakeSession.calls = []
        st, p, n1, n2, p2 = asyncio.run(go())
        self.assertEqual(st, 200, p)
        self.assertTrue(p["ok"], p)
        self.assertEqual(set(p["scales"]), {"30m", "1h", "4h"})
        self.assertEqual(FakeSession.calls[0], {"symbol": "ETHUSDT", "interval": "1m", "limit": 1500})
        self.assertEqual((n1, n2), (1, 1), "두 번째 호출은 1분 캐시")
        self.assertEqual(p2, p)
        raw = rows(time.time())[:-1]                                   # 형성 봉 제외 = 서버가 본 봉
        d = pd.DataFrame({"timestamp": pd.to_datetime([r[0] for r in raw], unit="ms", utc=True),
                          "high": [r[2] for r in raw], "low": [r[3] for r in raw], "close": [r[4] for r in raw]})
        want = payload(d)
        self.assertEqual(p["asof_ms"], want["asof_ms"])
        for k, s in p["scales"].items():
            self.assertAlmostEqual(s["p"], want["scales"][k]["p"], places=4)
            self.assertTrue(0 < s["p"] < 1 and s["dn"] < p["px"] < s["up"])


if __name__ == "__main__":
    unittest.main()

"""2026-11-12 주문 요청 기록(order_requests.jsonl): POST /api/manual-* 마다 한 줄 · GET 은 안 남긴다. 실주문 0(confirm 없는 요청 = 서버가 400 으로 거절).
python -m pytest -q test/test_order_request_log_20261112.py"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402


class OrderRequestLogTest(unittest.TestCase):
    def test_post_is_logged_get_is_not(self) -> None:
        async def go():
            client = TestClient(TestServer(offline_app(server)))
            await client.start_server()
            try:
                r = await client.post("/api/manual-exit/submit?side=SHORT&asset=soxl&pct=50&src=ep-touch", headers={"User-Agent": "TestPhone", "CF-Connecting-IP": "1.2.3.4"})
                await client.get("/api/manual-entry/status")
                return r.status
            finally:
                await client.close()
        with _isolated_dirs():
            status = asyncio.run(go())
            rows = [json.loads(x) for x in (server.LIVE_DIR / "order_requests.jsonl").read_text("utf-8").splitlines()]
        self.assertEqual(len(rows), 1, rows)
        r = rows[0]
        self.assertEqual((r["path"], r["query"]["pct"], r["query"]["src"], r["status"], r["ua"], r["ip"]),
                         ("/api/manual-exit/submit", "50", "ep-touch", status, "TestPhone", "1.2.3.4"), r)
        self.assertTrue(r["error"], r)   # 거절 사유가 남는다
        self.assertTrue(r["at"].endswith("+09:00"))


if __name__ == "__main__":
    unittest.main()

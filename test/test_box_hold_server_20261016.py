"""2026-11-21 사용자 «서빙을 끈다»: /api/box-hold 는 없다(404 · 바깥 요청 0) · LLM 브리핑 snapshot 은 box_hold 없이 만든다.

옛 시험(10-16 «B로 배포» · 값 = dashboard/box_hold.payload 대조)은 라우트와 함께 뺐다 -- 되살리려면 git log.
실행: python -m pytest -q test/test_box_hold_server_20261016.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app, patch_live_dir  # noqa: E402

from dashboard import server  # noqa: E402
from scripts import live_llm_brief_20261017 as brief  # noqa: E402


class NoNet:
    def __init__(self, *a, **k): self.closed = False
    def get(self, *a, **k): raise AssertionError(f"바깥 요청 {a}")
    async def close(self): self.closed = True


class BoxHoldOffTest(unittest.TestCase):
    def test_route_gone(self):
        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                live, dash = Path(tmp) / "live", Path(tmp) / "dashboard"
                live.mkdir(); dash.mkdir()
                with mock.patch.object(server, "ClientSession", NoNet), patch_live_dir(server, live), \
                     mock.patch.object(server, "DASHBOARD_DIR", dash):
                    async with TestClient(TestServer(offline_app(server))) as c:
                        return (await c.get("/api/box-hold")).status
        self.assertEqual(asyncio.run(go()), 404)

    def test_brief_snapshot_without_box_hold(self):
        s = brief.snapshot({}, {}, {}, {}, datetime.now(timezone.utc))
        self.assertNotIn("box_hold", s)
        self.assertIn("scheduled_events_24h", s)


if __name__ == "__main__":
    unittest.main()

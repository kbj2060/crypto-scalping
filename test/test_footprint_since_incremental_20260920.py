"""`/api/footprint?since=` 증분 계약, 2026-09-20.

왜 증분인가: 400ms 폴링인데 48봉을 통째로 보내고 있었다. 서버 실측(400ms 간격 19번) 내용이
실제로 바뀐 건 12번이고 바뀌는 건 **맨 오른쪽 봉 하나**다 -- 닫힌 봉은 5분에 한 번만 바뀐다.
폴링 주기도 렌더 주기도 안 건드리므로 **지연은 1ms 도 안 늘어난다**. 줄어드는 건 바이트뿐이다.

지키려는 계약 다섯 -- 하나라도 깨지면 화면에 **없는 봉이 생기거나 옛 봉이 굳는다**:
  ① since 는 `>=` 다. 봉 경계에서 늦게 온 체결이 직전 봉에 들어가므로 클라가
     `since = 최신봉 - 1봉` 으로 물어 **두 봉**을 받는다. `>` 로 바꾸면 그 체결이 영영 안 보인다.
  ② `ready` 가 False 면 **무조건 전량**이다. 백필은 REST 로 몇 분에 걸쳐 **과거 봉을** 채우는데,
     증분을 주면 클라는 그 갱신을 영영 못 받는다.
  ③ 응답에 `full` 이 있어야 한다. 클라가 쿼리로 추측하면 ②의 되돌림을 놓친다.
  ④ 창(`?bars=`)이 **먼저** 자르고 그 안에서 since 가 걸러야 한다.
  ⑤ 클라도 실제로 그 꼬리를 물어야 한다 -- 서버만 고치면 바이트가 하나도 안 준다.

⚠️`footprint_state` 는 make_app() 클로저 안이라 밖에서 못 갈아끼운다. 그래서 슬라이싱 규칙은
  **server.py 원문에 그 줄이 그대로 있는지 확인한 뒤**(test_source_still_matches_this_rule)
  같은 규칙으로 검사한다 -- 사본이 조용히 갈라지는 걸 그 시험이 막는다.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiohttp.test_utils import TestClient, TestServer

from dashboard import server

BAR = server.FOOTPRINT_BAR_SECONDS
SERVER_SRC = Path(server.__file__)
APP_JS = SERVER_SRC.parent / "live" / "app.js"

# server.py 의 증분 세 줄. 아래 첫 시험이 «원문에 그대로 있는가»를 지킨다.
RULE_LINES = (
    'full = since <= 0 or not footprint_state["ready"]',
    "sent = recent if full else [(b, c) for b, c in recent if b >= since]",
    '"full": full,',
)


def seeded_bars(n: int) -> dict[int, dict]:
    """n 개의 연속 봉. 반환 키가 곧 봉시각(오래된 것부터)."""
    base = (1_700_000_000 // BAR) * BAR - (n - 1) * BAR
    return {base + i * BAR: {5000 + i: [1.0, 2.0, 0.0, 0.0, 1.0, 2.0]} for i in range(n)}


def slice_like_server(bars: dict, ready: bool, want: int, since: int) -> tuple[bool, list[int]]:
    recent = sorted(bars.items())[-want:]
    full = since <= 0 or not ready
    sent = recent if full else [(b, c) for b, c in recent if b >= since]
    return full, [b for b, _ in sent]


class FootprintSinceRuleTest(unittest.TestCase):
    def test_source_still_matches_this_rule(self) -> None:
        src = SERVER_SRC.read_text(encoding="utf-8")
        for line in RULE_LINES:
            self.assertIn(line, src, f"server.py 의 증분 규칙이 바뀌었다 -- 이 시험도 같이 고쳐라: {line}")

    def test_since_is_inclusive_so_the_boundary_bar_comes_back(self) -> None:
        bars = seeded_bars(5)
        times = sorted(bars)
        full, sent = slice_like_server(bars, True, 48, times[-1] - BAR)
        self.assertFalse(full)
        self.assertEqual(sent, times[-2:], "꼬리 두 봉이 와야 한다 -- `>` 면 경계 체결을 잃는다")

    def test_not_ready_forces_full_even_with_since(self) -> None:
        bars = seeded_bars(5)
        times = sorted(bars)
        full, sent = slice_like_server(bars, False, 48, times[-1] - BAR)
        self.assertTrue(full)
        self.assertEqual(sent, times, "백필 중(ready=False)에는 창 전체여야 한다")

    def test_since_zero_is_full(self) -> None:
        bars = seeded_bars(5)
        full, sent = slice_like_server(bars, True, 48, 0)
        self.assertTrue(full)
        self.assertEqual(sent, sorted(bars))

    def test_window_caps_before_since(self) -> None:
        bars = seeded_bars(50)
        full, sent = slice_like_server(bars, True, 12, 0)
        self.assertTrue(full)
        self.assertEqual(sent, sorted(bars)[-12:], "창 밖 봉이 새어 나왔다")

    def test_client_actually_asks_for_the_tail(self) -> None:
        """⑤서버만 고치면 바이트가 하나도 안 준다."""
        app_js = APP_JS.read_text(encoding="utf-8")
        for needle in ("&since=${since}",
                       "const since = newest ? newest - barSec : 0;",
                       "if (payload.full) footprintBars = new Map();"):
            self.assertIn(needle, app_js, f"app.js 가 증분을 안 쓴다: {needle}")


class FootprintPayloadShapeTest(unittest.TestCase):
    """라우팅·직렬화까지 진짜로 도는가. (봉이 비어 있어도 계약 필드는 있어야 한다.)"""

    def _exercise(self, queries: list[str]) -> None:
        async def run() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                for q in queries:
                    res = await client.get(f"/api/footprint{q}")
                    self.assertEqual(res.status, 200, q)
                    body = json.loads(await res.text())
                    self.assertIn("full", body, q)
                    self.assertIn("bars", body, q)
                    self.assertNotIn(
                        "updated", body,
                        "`updated` 는 체결마다 바뀌어 ETag 를 깨뜨린다(읽는 곳도 없다) -- 뺀 채로 둔다")
            finally:
                await client.close()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(server, "LIVE_DIR", Path(tmp)):
                asyncio.run(run())

    def test_endpoint_serves_and_carries_full_flag(self) -> None:
        self._exercise(["", "?since=0", "?bars=48&since=1700000000"])

    def test_bad_since_falls_back_to_full_not_500(self) -> None:
        """신뢰경계 입력 -- 쓰레기가 와도 전량으로 떨어질 뿐 터지지 않는다."""
        self._exercise([f"?since={bad}" for bad in ("abc", "", "1e9", "-5", "9" * 40, "1.5")])


if __name__ == "__main__":
    unittest.main()

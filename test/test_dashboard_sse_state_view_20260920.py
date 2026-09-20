"""SSE 로 나가는 상태 페이로드가 «화면이 읽는 것»과 정확히 같은지 본다 (2026-09-20).

배경: publish_dashboard_events 가 dashboard_state.json 전체(실측 30,109B)를 상태가 바뀔
때마다 SSE 로 밀고 있었는데, app.js 의 render() 가 읽는 것은 session / microstructure /
tail_risk 세 블록뿐이고 compactState 는 인자로 넘어가기만 하고 본문에서 한 번도 안 읽힌다.
SSE 는 gzip 도 안 걸린다(json_compress_etag 는 StreamResponse 를 건드리지 않는다).

🔴이 검사가 지키는 계약은 둘이다:
  ① SSE 는 세 블록만 보낸다(그 이상 늘면 이 검사가 깨진다 -- 늘릴 거면 app.js 가 실제로
     읽는지부터 확인하라는 뜻이다).
  ② 그 세 이름이 app.js 가 실제로 읽는 이름과 **같다**. 서버만 고치고 클라를 안 고치면
     화면이 통째로 빈다.
`/api/state` 는 계약이 달라 여전히 전체를 준다 -- test_dashboard_server.py 가 그쪽을 본다.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dashboard.server import SSE_STATE_KEYS, sse_state_view  # noqa: E402


class SseStateViewTest(unittest.TestCase):
    def test_keeps_only_what_the_client_reads(self) -> None:
        payload = {
            "state": {
                "session": {"session_us": 1},
                "microstructure": {"nif_whale": 0.3},
                "tail_risk": {"hawkes_active": False},
                "signal": {"big": "x" * 3000},
                "agents": {"more": "y" * 2000},
            },
            "compactState": {"huge": "z" * 14000},
        }
        view = sse_state_view(payload)
        self.assertEqual(set(view["state"]), set(SSE_STATE_KEYS))
        self.assertEqual(view["state"]["microstructure"], {"nif_whale": 0.3})
        self.assertNotIn("compactState", view, "compactState 는 화면이 안 읽는다")
        self.assertLess(len(repr(view)), len(repr(payload)) / 5,
                        "추리는 의미가 있어야 한다 -- 5배는 줄어야 한다")

    def test_missing_block_becomes_none_not_keyerror(self) -> None:
        """봇이 아직 안 쓴 블록은 None 으로 간다. app.js 는 `state.session || {}` 로 받는다."""
        view = sse_state_view({"state": {"session": {"a": 1}}})
        self.assertEqual(view["state"]["microstructure"], None)
        self.assertEqual(view["state"]["tail_risk"], None)

    def test_empty_state_passes_through_unchanged(self) -> None:
        """🔴상태 파일이 없을 때 `{"state": {...}}` 를 만들어 보내면 클라의
        `if (payload?.state?.state)` 가 참이 되어 **빈 화면을 그린다**. 원래 모양 그대로 보낸다."""
        for empty in (None, {}, {"state": None}, {"state": {}}):
            self.assertIs(sse_state_view(empty), empty)

    def test_key_names_match_what_app_js_reads(self) -> None:
        body = (REPO_ROOT / "dashboard" / "live" / "app.js").read_text(encoding="utf-8")
        render = re.search(r"\nfunction render\(state, compactState[\s\S]*?\n  const micro = [^\n]*\n",
                           body)
        self.assertIsNotNone(render, "render() 머리를 못 찾았다 -- 이름이 바뀌었나?")
        for key in SSE_STATE_KEYS:
            self.assertIn(f"state.{key}", render.group(0),
                          f"app.js render() 가 state.{key} 를 안 읽는다")
        # compactState 는 인자로만 있고 본문에서 안 쓰인다 -- 그게 SSE 에서 뺀 근거다.
        fn = re.search(r"\nfunction render\(state, compactState[\s\S]*?\n\}\n\nasync function tick",
                       body)
        self.assertIsNotNone(fn, "render() 본문을 못 찾았다")
        # 인자 목록에 한 번 나오는 것이 전부여야 한다 -- 본문에서 쓰기 시작했다면 2회 이상이다.
        self.assertEqual(fn.group(0).count("compactState"), 1,
                         "compactState 를 render() 본문에서 쓰기 시작했다면 SSE 에 다시 실어야 한다")


if __name__ == "__main__":
    unittest.main()

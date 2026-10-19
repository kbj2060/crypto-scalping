from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiohttp.test_utils import TestClient, TestServer
from offline_app import offline_app  # noqa: E402 -- 수집기·주문 감시기를 안 띄운다(같은 공인 IP 한도)

from dashboard import server


class DashboardServerTest(unittest.TestCase):
    def test_dashboard_api_compresses_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            live_dir = tmp_path / "live"
            dashboard_dir = tmp_path / "dashboard"
            live_dir.mkdir()
            dashboard_dir.mkdir()
            (dashboard_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            (live_dir / "dashboard_state.json").write_text(
                json.dumps({"updated_at": "2026-07-17T12:00:00", "price": 100}),
                encoding="utf-8",
            )

            async def exercise_api() -> None:
                client = TestClient(TestServer(offline_app(server)))
                await client.start_server()
                try:
                    state_response = await client.get(
                        "/api/state",
                        headers={"Accept-Encoding": "gzip"},
                    )
                    self.assertEqual(state_response.status, 200)
                    self.assertEqual(state_response.headers["Cache-Control"], "no-cache")
                    self.assertEqual(state_response.headers["Content-Encoding"], "gzip")
                    state_etag = state_response.headers["ETag"]
                    self.assertEqual((await state_response.json())["state"]["price"], 100)

                    unchanged_state = await client.get(
                        "/api/state",
                        headers={"If-None-Match": state_etag},
                    )
                    self.assertEqual(unchanged_state.status, 304)
                    self.assertEqual(await unchanged_state.read(), b"")

                    (live_dir / "dashboard_state.json").write_text(
                        json.dumps({"updated_at": "2026-07-17T12:00:01", "price": 101}),
                        encoding="utf-8",
                    )
                    changed_state = await client.get(
                        "/api/state",
                        headers={"If-None-Match": state_etag},
                    )
                    self.assertEqual(changed_state.status, 200)
                    self.assertNotEqual(changed_state.headers["ETag"], state_etag)
                    self.assertEqual((await changed_state.json())["state"]["price"], 101)
                finally:
                    await client.close()

            with (
                mock.patch.object(server, "LIVE_DIR", live_dir),
                mock.patch.object(server, "DASHBOARD_DIR", dashboard_dir),
            ):
                asyncio.run(exercise_api())


if __name__ == "__main__":
    unittest.main()


class StaticAssetCacheHeaderTest(unittest.TestCase):
    """배포마다 바뀌는 정적 파일은 재검증돼야 한다.

    2026-09-13: app.js 가 immutable(1년)인데 캐시버스터는 날짜라, 같은 날 두 번째
    배포가 사용자 브라우저에 영영 안 갔다. index.html 은 헤더가 아예 없어 함께 묵었다.
    """

    def test_mutable_assets_revalidate_and_fonts_stay_immutable(self) -> None:
        # 🔴DASHBOARD_DIR 은 **진짜**여야 한다(실제 app.js/styles.css 의 헤더를 보는 시험이다).
        #   LIVE_DIR 은 아니다 -- 이 시험은 /data/live/ 를 건드리지 않는데, make_app() 의
        #   add_static 이 그 디렉터리가 없으면 FileNotFoundError 로 죽는다. data/live 는
        #   gitignore 라 서버 말고는 어디에도 없어서, 이 시험은 개발 체크아웃에서 **늘 실패**했다
        #   (2026-09-20 확인). 아무 데서도 못 도는 시험은 회귀를 못 잡는다. 같은 파일의 다른
        #   시험들이 이미 쓰는 방식(임시 디렉터리로 패치)을 여기에도 쓴다.
        async def exercise() -> None:
            client = TestClient(TestServer(offline_app(server)))
            await client.start_server()
            try:
                for path in ("app.js", "styles.css", "index.html"):
                    response = await client.get(f"/dashboard/live/{path}")
                    self.assertEqual(response.status, 200, path)
                    self.assertEqual(
                        response.headers.get("Cache-Control"), "no-cache", path
                    )
            finally:
                await client.close()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(server, "LIVE_DIR", Path(tmp)):
                asyncio.run(exercise())

    def test_index_buster_points_at_a_file_that_exists(self) -> None:
        index = (server.DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
        referenced = re.findall(r'(?:src|href)="\.?/?([\w.-]+\.(?:js|css))\?v=', index)
        self.assertTrue(referenced)
        for name in referenced:
            self.assertTrue((server.DASHBOARD_DIR / name).is_file(), name)

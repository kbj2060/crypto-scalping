"""극점 탐지기 채점을 워커로 옮긴 뒤의 대시보드 읽기 경로 (2026-09-10).

전에는 대시보드가 모델을 인라인으로 돌렸다. TabPFN v3 로 올리면 0.49초 -> 5.14초(서버 실측)가
되고 그 GPU 를 V자 TabPFN·증거신호가 공유한다. 이제 워커가 쓴 상태 파일을 읽기만 한다.
⭐인라인 폴백은 **일부러** 두지 않았다 -- 폴백이 있으면 워커가 죽어도 화면은 정상이고
  대신 응답이 5초씩 느려진다. 그게 이 구조를 만든 이유이므로, 죽으면 죽었다고 보여야 한다.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from dashboard import server

LIVE = {"available": True, "tone": "good", "subText": "바닥 발동", "grade": "강",
        "proba": 0.61, "history": ["neutral"], "times": ["2026-09-10T00:00:00+00:00"]}


def _state(age_min: float) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    return {**LIVE, "updated_utc": ts.isoformat()}


class ExtremeDetectorWorkerRead(unittest.TestCase):
    def _payload(self, state: dict | None):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.json"
            if state is not None:
                p.write_text(json.dumps(state, ensure_ascii=False))
            with mock.patch.object(server, "EXTREME_DETECTOR_STATE_PATH", p):
                return server.extreme_detector_payload()

    def test_missing_state_is_data_absent_not_a_stale_reading(self) -> None:
        out = self._payload(None)
        self.assertFalse(out["available"])
        self.assertEqual(out["subText"], "데이터 없음")
        self.assertEqual(out["error"], "worker_state_missing")

    def test_fresh_state_passes_through_with_age(self) -> None:
        out = self._payload(_state(1.0))
        self.assertTrue(out["available"])
        self.assertEqual(out["subText"], "바닥 발동")
        self.assertEqual(out["grade"], "강")
        self.assertLess(out["stale_min"], 3.0)

    def test_stale_state_falls_back_to_data_absent(self) -> None:
        """워커가 멈춘 채로 옛 판정을 계속 보여주면 그게 지금 값인 줄 안다."""
        out = self._payload(_state(server.EXTREME_DETECTOR_MAX_AGE_MIN + 5))
        self.assertFalse(out["available"])
        self.assertEqual(out["subText"], "데이터 없음")
        self.assertEqual(out["error"], "worker_stale")
        self.assertGreater(out["stale_min"], server.EXTREME_DETECTOR_MAX_AGE_MIN)

    def test_max_age_covers_three_five_minute_bars(self) -> None:
        self.assertGreaterEqual(server.EXTREME_DETECTOR_MAX_AGE_MIN, 15.0)

    def test_dashboard_no_longer_imports_the_scorer(self) -> None:
        """인라인 채점 경로가 남아 있으면 언젠가 다시 그리로 샌다."""
        src = Path(server.__file__).read_text(encoding="utf-8")
        self.assertNotIn("compute_eth_extreme_detector", src)


if __name__ == "__main__":
    unittest.main()

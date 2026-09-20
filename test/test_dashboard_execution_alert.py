"""트레이딩봇 치명적 오류 통보 계약.

🔴2026-09-20 «화면 배너»를 공식 은퇴시켰다(사용자 결정). 통보 채널은 **텔레그램 하나**다.

경위: 배너(`#executionAlertBanner`)는 `liveTabPanel` **안**에 있었고, 2026-08-31 라이브 탭
제거(60ab72b7)와 함께 사라졌다 -- 커밋 메시지에 언급이 없어 3주 넘게 이 시험 두 개가 조용히
빨간 채로 남아 있었다. 되살릴지 물었고 «텔레그램 단일 채널»로 결정됐다. 함께 정리한 것:
`dashboard/live/styles.css` 의 `.execution-alert-*` 47줄(아무도 안 쓰는 고아 CSS).

⚠️그래서 **봇이 멈춰도 대시보드 화면에는 안 뜬다.** 그건 알려진 상태이지 사고가 아니다.
  아래 시험이 지키는 것은 «텔레그램으로는 반드시 나간다» 하나다 -- 이게 깨지면 통보가
  **어디로도** 안 가므로, 화면이 없는 지금은 이 시험 하나가 유일한 안전망이다.
  배너를 다시 만들 거라면 `state.execution_alert` 를 SSE 로도 실어야 한다
  (dashboard/server.py 의 `SSE_STATE_KEYS` -- 지금은 화면이 안 읽으므로 빠져 있다).
"""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardExecutionAlertTests(unittest.TestCase):
    def test_trading_bot_publishes_alert_to_dashboard_and_telegram(self) -> None:
        source = (ROOT / "trading_bot.py").read_text(encoding="utf-8")

        self.assertIn('"execution_alert": dict(_execution_alert)', source)
        self.assertIn("telegram-execution-alert", source)
        self.assertIn("execution alert dashboard write failed", source)
        self.assertIn("[트레이딩봇 치명적 오류]", source)

    def test_no_orphan_execution_alert_surface_left_behind(self) -> None:
        """은퇴는 «지웠다»로 끝나야 한다 -- 반쯤 남은 표면이 다음 사람을 헷갈리게 한다.

        배너를 되살리기로 마음이 바뀌면 이 시험을 지우고 위 docstring 의 SSE 주의사항부터 읽을 것.
        """
        for name in ("dashboard/live/index.html", "dashboard/live/app.js",
                     "dashboard/live/styles.css"):
            text = (ROOT / name).read_text(encoding="utf-8")
            for token in ("executionAlert", "execution-alert"):
                with self.subTest(file=name, token=token):
                    self.assertNotIn(token, text,
                                     f"{name} 에 은퇴한 실행경보 표면이 남아 있다: {token}")


if __name__ == "__main__":
    unittest.main()

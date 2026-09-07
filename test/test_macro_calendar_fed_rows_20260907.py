"""연준 월별 캘린더 스크레이퍼가 어떤 행을 잡는가, 2026-09-07.

이건 **남의 HTML을 긁는 코드**라 조용히 죽는다. 실제로 그랬다 -- 필터가
`Speech|Testimony|Discussion - Chair|Chairman` 이었는데 연준 2026-09 페이지에는 "Chair"라는
단어가 **한 번도** 안 나온다(행은 전부 "Speech - Governor <이름>"). 그래서 이 소스는 배포된
내내 0건을 돌려주고 있었고, 아무도 예외를 못 봤다(예외가 안 나니까).

그래서 여기서는 네트워크를 타지 않고 **실제 페이지에서 확인한 마크업 구조 그대로**의 고정
픽스처로 파싱 계약을 못박는다: 바깥 row > panel > panel-body > 안쪽 row > col-xs-2(시각) /
col-xs-7(내용) / col-xs-3(날짜).
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.live_macro_calendar_20260826 import _parse_fed_calendar_events  # noqa: E402

TODAY, HORIZON = date(2026, 9, 1), date(2026, 9, 30)


def row(time_txt: str, content: str, day: str) -> str:
    """연준 페이지의 실제 행 구조(2026-09-07 원문 확인)."""
    return f'''<div class="row"><div class="panel panel-unstyled col-xs-12 col-sm-12 col-md-12">
      <div class="panel-body"><div class="row">
        <div class="col-xs-2"><p>{time_txt}</p></div>
        <div class="col-xs-7"><p>{content}</p><p><a class='watchLive'>Watch Live</a></p></div>
        <div class="col-xs-3"><p>{day}</p></div>
      </div></div></div></div>'''


def parse(html: str, today: date = TODAY, horizon: date = HORIZON) -> list[dict]:
    return _parse_fed_calendar_events(html, 2026, 9, "https://x/2026-september.htm", today, horizon)


class FedCalendarRowTests(unittest.TestCase):
    def test_governor_speech_is_captured(self) -> None:
        """이게 회귀의 핵심 -- 배포본은 이 행을 하나도 못 잡았다."""
        (ev,) = parse(row("8:30 a.m.", "Speech - Governor Christopher J. Waller", "3"))
        self.assertEqual(ev["title_ko"], "연준 이사 연설")
        self.assertEqual(ev["importance"], "medium")   # 캘린더엔 보이되 +-30분 푸시는 안 울린다
        self.assertEqual(ev["category"], "fed_speech")
        self.assertTrue(ev["time_utc"].startswith("2026-09-03T12:30"))  # 8:30 ET = 12:30 UTC

    def test_chair_and_vice_chair_roles(self) -> None:
        """'Vice Chair'가 'Chair'보다 먼저 매치돼야 한다 -- 순서가 뒤집히면 부의장이 의장이 된다."""
        (chair,) = parse(row("10:00 a.m.", "Testimony - Chair Jerome H. Powell", "9"))
        self.assertEqual(chair["title_ko"], "연준 의장 증언")
        self.assertEqual(chair["importance"], "high")
        (vice,) = parse(row("10:00 a.m.", "Speech - Vice Chair for Supervision Michelle W. Bowman", "9"))
        self.assertEqual(vice["title_ko"], "연준 부의장 연설")
        self.assertEqual(vice["importance"], "high")

    def test_fomc_press_conference_is_captured(self) -> None:
        (ev,) = parse(row("2:30 p.m.", "FOMC Press Conference", "16"))
        self.assertEqual(ev["title_ko"], "FOMC 기자회견")
        self.assertEqual(ev["importance"], "high")
        self.assertTrue(ev["time_utc"].startswith("2026-09-16T18:30"))  # 2:30pm ET = 18:30 UTC

    def test_routine_statistical_rows_are_ignored(self) -> None:
        """페이지의 대부분은 H.4.1/G.19 같은 정기 통계 발표다. 이걸 다 넣으면 캘린더가 잠긴다."""
        html = "".join(row("4:30 p.m.", c, "10") for c in (
            "H.4.1 - Factors Affecting Reserve Balances", "G.19 - Consumer Credit",
            "Beige Book", "FOMC Minutes Meeting of September 15-16",
            "Holiday - Labor Day"))
        self.assertEqual(parse(html), [])

    def test_rows_outside_the_window_are_dropped(self) -> None:
        html = row("8:30 a.m.", "Speech - Governor Michael S. Barr", "3")
        self.assertEqual(parse(html, today=date(2026, 9, 5)), [])   # 이미 지난 일정
        self.assertEqual(parse(html, horizon=date(2026, 9, 2)), [])  # 지평 밖

    def test_row_without_a_parseable_time_is_dropped(self) -> None:
        """시각이 없으면 UTC 변환이 불가능하다 -- 조용히 자정으로 찍지 말고 버려야 한다."""
        self.assertEqual(parse(row("", "Speech - Governor Lisa D. Cook", "8")), [])


if __name__ == "__main__":
    unittest.main()

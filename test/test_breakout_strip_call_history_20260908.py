"""돌파/되돌림 띠(타임 게이지)의 봉별 판정 단어.

2026-09-08 사용자 신고: 배지는 "직전 되돌림↓"인데 띠 캡션은 "돌파 ↓"였다. 톤은 **매매 방향**만
담아서(good=↑ · bad=↓) 톤→라벨 사전이 양쪽 다 "돌파"라고 못박고 있었다. 서버가 봉별 판정
단어를 함께 주고, 화면은 그 단어로 캡션을 만든다.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from dashboard import server


def _row(minute: int, call: str, dir_up: bool, span: int = 5) -> dict:
    t0 = datetime(2026, 9, 8, 13, minute, tzinfo=timezone.utc)
    return {"trigger_utc": t0.isoformat(), "call": call, "dir_up": dir_up,
            "exit_utc": (t0 + timedelta(minutes=span)).isoformat()}


class BreakoutStripCallHistory(unittest.TestCase):
    END = datetime(2026, 9, 8, 13, 40, tzinfo=timezone.utc)

    def _hist(self, rows, bars=4):
        return server._breakout_tone_history(rows, self.END, bars=bars)

    def test_reversal_call_keeps_its_own_word(self):
        """하락 발현 + 되돌림 = ↑(롱). 단어는 '돌파'가 아니라 '되돌림'이어야 한다."""
        tones, calls = self._hist([_row(30, "되돌림", dir_up=False)])
        self.assertEqual(tones[-3], "good")
        self.assertEqual(calls[-3], "되돌림")

    def test_breakout_call_keeps_its_own_word(self):
        tones, calls = self._hist([_row(30, "돌파", dir_up=False)])
        self.assertEqual(tones[-3], "bad")
        self.assertEqual(calls[-3], "돌파")

    def test_same_direction_opposite_calls_are_distinguishable(self):
        """상승·돌파와 하락·되돌림은 둘 다 ↑(good)이다 -- 단어가 유일한 구분점이다."""
        _, calls = self._hist([_row(25, "돌파", dir_up=True), _row(35, "되돌림", dir_up=False)])
        self.assertEqual(calls[-4], "돌파")
        self.assertEqual(calls[-2], "되돌림")

    def test_overlapping_calls_report_혼재(self):
        _, calls = self._hist([_row(30, "돌파", dir_up=True), _row(30, "되돌림", dir_up=True)])
        self.assertEqual(calls[-3], "혼재")

    def test_idle_bars_have_no_word(self):
        tones, calls = self._hist([])
        self.assertEqual(set(tones), {"neutral"})
        self.assertEqual(set(calls), {""})

    def test_payload_exposes_both_arrays(self):
        """캡션이 톤과 단어를 같은 인덱스로 읽으므로 길이가 같아야 한다."""
        tones, calls = self._hist([_row(30, "되돌림", dir_up=False)], bars=48)
        self.assertEqual(len(tones), len(calls))


if __name__ == "__main__":
    unittest.main()

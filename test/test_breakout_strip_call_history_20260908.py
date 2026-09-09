"""돌파/되돌림 띠(타임 게이지)의 봉별 판정 단어.

2026-09-08 사용자 신고: 배지는 "직전 되돌림↓"인데 띠 캡션은 "돌파 ↓"였다. 톤은 **매매 방향**만
담아서(good=↑ · bad=↓) 톤→라벨 사전이 양쪽 다 "돌파"라고 못박고 있었다. 서버가 봉별 판정
단어를 함께 주고, 화면은 그 단어로 캡션을 만든다.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


class BreakoutLedgerLabelRegime(unittest.TestCase):
    """집계는 **현행 라벨 정의 행만** 센다.

    2026-09-08 배리어 개정(절대 ±0.25% → ±0.8×ATR) 전후 행이 한 원장에 섞였다. rule_id 는
    러너 상수가 안 바뀌어 둘 다 `..._p025_...` 라 못 가른다 -- 행이 쓴 배리어로 가른다.
    """
    ATR = {"barrier_mode": "atr_relative", "barrier_k_atr": 0.8}
    ABS = {"barrier_pct": 0.25}

    def test_rule_id_wins_over_barrier_when_the_artifact_has_one(self):
        """2026-09-09: 발현창 15분/60분은 **배리어가 같고 모집단만 다르다** -- 배리어로는 못 가른다.
        러너가 아티팩트의 rule_id 를 찍으므로 그게 유일하게 정확한 기준이다."""
        meta = {"rule_id": "br_w60", "barrier_mode": "atr_relative", "barrier_k_atr": 0.8}
        cur = {"rule_id": "br_w60", "atr_pct": 0.002, "barrier_pct": 0.16}
        old = {"rule_id": "br_w15", "atr_pct": 0.002, "barrier_pct": 0.16}   # 배리어는 같다
        self.assertTrue(server._br_current_label(cur, meta))
        self.assertFalse(server._br_current_label(old, meta))
        self.assertFalse(server._br_current_label({"atr_pct": 0.002}, meta))  # rule_id 없는 옛 행

    def test_atr_row_matching_its_own_atr_is_current(self):
        r = {"atr_pct": 0.0022577, "barrier_pct": 0.0022577 * 0.8 * 100}
        self.assertTrue(server._br_current_label(r, self.ATR))

    def test_old_row_without_barrier_pct_is_not_current(self):
        """옛 러너는 barrier_pct 를 아예 안 남겼다."""
        self.assertFalse(server._br_current_label({"atr_pct": 0.0022}, self.ATR))

    def test_row_scored_with_a_different_barrier_is_not_current(self):
        self.assertFalse(server._br_current_label({"atr_pct": 0.0022, "barrier_pct": 0.25}, self.ATR))

    def test_absolute_mode_accepts_its_own_rows(self):
        self.assertTrue(server._br_current_label({"barrier_pct": 0.25}, self.ABS))
        self.assertTrue(server._br_current_label({}, self.ABS))          # 옛 러너 = 절대 시절
        self.assertFalse(server._br_current_label({"barrier_pct": 0.19}, self.ABS))

    def test_payload_counts_only_current_rows_and_reports_the_rest(self):
        import json, tempfile
        from unittest import mock
        old = {"trigger_utc": "2026-09-08 07:50:00", "exit_utc": "2026-09-08 07:55:00",
               "dir_up": False, "call": "되돌림", "p_breakout": 0.46, "outcome": "fade",
               "correct": True, "atr_pct": 0.0015, "tier": "약"}          # barrier_pct 없음
        new = {**old, "trigger_utc": "2026-09-08 18:50:00", "exit_utc": "2026-09-08 18:52:00",
               "outcome": "cont", "correct": False, "atr_pct": 0.0020,
               "barrier_pct": 0.0020 * 0.8 * 100}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "meta.json").write_text(json.dumps(
                {"barrier_mode": "atr_relative", "barrier_k_atr": 0.8, "horizon_bars": 12}))
            (d / "state.json").write_text(json.dumps({"positions": [], "watching": []}))
            (d / "led.jsonl").write_text("\n".join(json.dumps(r) for r in (old, new)))
            with mock.patch.object(server, "BREAKOUT_REV_ARTIFACT_PATH", d / "meta.json"), \
                 mock.patch.object(server, "BREAKOUT_REV_STATE_PATH", d / "state.json"), \
                 mock.patch.object(server, "BREAKOUT_REV_LEDGER_PATH", d / "led.jsonl"):
                out = server.breakout_reversal_shadow_payload()
        self.assertEqual(out["closed"], 1)
        self.assertEqual(out["stale_closed"], 1)
        self.assertEqual(out["accuracy"], 0.0)               # 현행 행 1건은 빗나감
        self.assertEqual(out["outcomes"], {"cont": 1, "fade": 0, "timeout": 0})
        self.assertEqual(len(out["call_history"]), len(out["tone_history"]))
        # 직전 판정은 집계에서 뺀 행도 포함해 고른다(가장 최근 판정이므로).
        self.assertEqual(out["last"]["trigger_utc"], "2026-09-08 18:50:00")

    def test_strip_still_paints_rows_excluded_from_the_tally(self):
        """띠는 "언제 무슨 판정이 있었나"의 기록이다 -- 집계에서 뺀 행도 색이 남아야 한다.

        (벽시계에 의존하지 않도록 end 를 직접 준다.)
        """
        end = datetime(2026, 9, 8, 8, 5, tzinfo=timezone.utc)
        stale = {"trigger_utc": "2026-09-08 07:50:00", "exit_utc": "2026-09-08 07:55:00",
                 "dir_up": False, "call": "되돌림"}                      # barrier_pct 없음 = 옛 배리어
        self.assertFalse(server._br_current_label(stale, self.ATR))
        _, calls = server._breakout_tone_history([stale], end, bars=4)
        self.assertIn("되돌림", calls)


if __name__ == "__main__":
    unittest.main()

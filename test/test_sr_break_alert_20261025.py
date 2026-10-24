"""지지·저항 뚫림 판정(server.sr_breaks, 2026-10-25). 실행: python -m pytest -q test/test_sr_break_alert_20261025.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.server import sr_breaks  # noqa: E402

SUP, RES = [99.0, 98.0, 97.0, 96.0], [101.0, 102.0, 103.0]


def test_no_touch_no_alert() -> None:
    assert sr_breaks(SUP, RES, 100.0, 99.5, 100.5, set()) == []


def test_support1_and_2_in_one_tick_then_once_only() -> None:
    broken: set = set()
    assert sr_breaks(SUP, RES, 100.0, 97.5, 100.0, broken) == [("지지1", 99.0, 97.0), ("지지2", 98.0, 97.0)]
    assert sr_breaks(SUP, RES, 97.6, 97.5, 99.5, broken) == []          # 되돌아와 다시 닿아도 같은 지도에선 한 번만
    assert sr_breaks(SUP, RES, 97.6, 96.9, 97.6, broken) == [("지지1", 97.0, 96.0)]   # 남은 것 중 가장 가까운 = 지지1


def test_resistance_touch_counts_and_labels() -> None:
    assert sr_breaks(SUP, RES, 100.0, 100.0, 102.0, set()) == [("저항1", 101.0, 103.0), ("저항2", 102.0, 103.0)]


def test_beyond_shown_marked_but_not_sent() -> None:
    broken: set = set()
    assert [h[0] for h in sr_breaks(SUP, RES, 100.0, 95.0, 100.0, broken)] == ["지지1", "지지2", "지지3"]
    assert 96.0 in broken                                               # 4번째도 쓸림 처리 -- 화면에서도 안 보인다

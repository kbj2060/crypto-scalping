"""봉 안 누적 델타 [최저, 최고, 마감](server.bar_delta_hl, 2026-11-10). 실행: python -m pytest -q test/test_bar_delta_hl_20261110.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.server import bar_delta_hl  # noqa: E402


def test_sold_down_then_bought_back() -> None:
    # −1,500 까지 팔리다 +200 마감 = 아래 꼬리 길고 몸통 양
    assert bar_delta_hl([-500, -1000, 900, 800]) == [-1500.0, 200.0, 200.0]


def test_start_zero_counts_and_empty_bar() -> None:
    assert bar_delta_hl([10, 5]) == [0.0, 15.0, 15.0]   # 한 번도 0 밑으로 안 가면 최저 = 0
    assert bar_delta_hl([]) == [0.0, 0.0, 0.0]

#!/usr/bin/env python3
"""창 토글(1h/2h/4h)이 서버에서 어떻게 잘리는지 자체점검 (2026-09-19).

`?bars=` 는 **신뢰경계 입력**이다. 화면이 보내지만 누구든 보낼 수 있다. 파싱 실패와 범위를
한 함수에서 닫아 두 엔드포인트(/api/footprint · /api/supply-profile)가 같은 답을 내게 한다 --
둘이 갈리면 한 카드 안의 위아래 두 그림이 다른 구간을 말하게 되고, 그건 읽는 사람을 속인다.

python test/test_footprint_window_bars_20260919.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import server as dash  # noqa: E402


class FakeRequest:
    """aiohttp Request 에서 이 함수가 쓰는 건 `.query` 하나다."""

    def __init__(self, **query: str) -> None:
        self.query = query


def test_toggle_values_pass_through() -> None:
    """화면이 실제로 보내는 세 값(1h/2h/4h)은 그대로 통과한다."""
    for bars in (12, 24, 48):
        assert dash.footprint_window_bars(FakeRequest(bars=str(bars))) == bars


def test_default_is_one_hour() -> None:
    """파라미터가 없으면 옛 동작(1시간) 그대로 -- 북마크·외부 호출이 안 깨진다."""
    assert dash.footprint_window_bars(FakeRequest()) == dash.FOOTPRINT_BARS
    assert dash.FOOTPRINT_BARS * dash.FOOTPRINT_BAR_SECONDS == 3600


def test_garbage_falls_back_instead_of_raising() -> None:
    """숫자가 아니면 500 이 아니라 기본값이다. 화면 하나 때문에 엔드포인트가 죽으면 안 된다."""
    for bad in ("", "abc", "12.5", "1e3", "NaN", "-"):
        assert dash.footprint_window_bars(FakeRequest(bars=bad)) == dash.FOOTPRINT_BARS, bad


def test_clamped_to_the_ring() -> None:
    """링보다 길게는 못 준다. 0·음수도 최소 1봉으로 막는다(빈 슬라이스는 `[-0:]` = 전체다)."""
    assert dash.footprint_window_bars(FakeRequest(bars="99999")) == dash.FOOTPRINT_KEEP_BARS
    assert dash.footprint_window_bars(FakeRequest(bars="0")) == 1
    assert dash.footprint_window_bars(FakeRequest(bars="-5")) == 1


def test_snapshot_covers_the_longest_window() -> None:
    """🔴저장 창이 토글의 최대치보다 짧으면, 재시작 직후 4h 를 골라도 그만큼밖에 안 보인다.
    배포가 하루에도 여러 번이라 이게 곧 «토글이 대부분의 시간 동안 거짓말»이 된다는 뜻이다.
    ⭐이 스냅샷이 **유일한** 복원 경로다(2026-09-19). 체결 테이프 duckdb 에서 읽는 길도
      만들어 봤지만 250ms vs 0.8ms 로 300배 느렸고, 같은 4시간이 이 json 안에 있다."""
    assert dash.FOOTPRINT_MAX_WINDOW_BARS >= 48
    assert dash.FOOTPRINT_MAX_WINDOW_BARS <= dash.FOOTPRINT_KEEP_BARS


if __name__ == "__main__":
    test_toggle_values_pass_through()
    test_default_is_one_hour()
    test_garbage_falls_back_instead_of_raising()
    test_clamped_to_the_ring()
    test_snapshot_covers_the_longest_window()
    print("ok — 토글값 · 기본값 · 잘못된 입력 · 범위 · 저장창 5건 통과")

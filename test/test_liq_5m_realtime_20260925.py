"""풋프린트 차트 청산 원 — 실시간 누적(2026-09-25).   python -m pytest -q test/test_liq_5m_realtime_20260925.py

계약:
  ① 수급 1초 차트와 **같은 원천·같은 값** -- 이벤트의 usd(z x ap)를 그대로 더한다.
  ② 봇 DB 판과 **같은 모양**(ts·long_usd·short_usd·events·partial) -- 화면·OKX·HL 합산 코드가 그대로 돈다.
  ③ 기록 시작 전 봉은 싣지 않는다(= 모름). 기록이 이어진 구간의 0 은 «청산 없음»이라 싣는다.
  ④ 방금 들어온 이벤트가 **다음 호출에 바로** 보인다(캐시 없음).
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.server import FOOTPRINT_KEEP_BARS, liq_5m_add, liq_5m_payload  # noqa: E402

T0 = 1_790_000_100          # 봉 경계(300 의 배수)
BAR = 300


def ev(t_s, side, usd):
    return {"ts_ms": int(t_s * 1000), "side": side, "usd": usd}


def ts_of(p):
    return [int(datetime.fromisoformat(b["ts"]).timestamp()) for b in p["bars"]]


def test_sums_usd_per_bar_and_splits_sides():
    st = {}
    for e in (ev(T0 + 5, "long", 100.0), ev(T0 + 250, "long", 50.0), ev(T0 + 299, "short", 7.0),
              ev(T0 + 300, "short", 1.0)):                     # 마지막은 다음 봉
        liq_5m_add(st, e)
    assert st[T0] == [150.0, 7.0, 3] and st[T0 + BAR] == [0.0, 1.0, 1]


def test_payload_shape_matches_bot_db_version_and_marks_partial():
    st = {}
    liq_5m_add(st, ev(T0 + 10, "long", 1327307.0))
    p = liq_5m_payload(st, from_s=T0 - 10 * BAR, bars=3, now_s=T0 + BAR + 30)
    assert p["warmed_up"] and p["error"] is None and p["source"] == "dashboard-forceorder"
    assert ts_of(p) == [T0 - BAR, T0, T0 + BAR]
    b = p["bars"][1]
    assert set(b) == {"ts", "long_usd", "short_usd", "events", "partial"}
    assert (b["long_usd"], b["short_usd"], b["events"], b["partial"]) == (1327307.0, 0.0, 1, False)
    assert p["bars"][0]["events"] == 0, "기록이 이어진 구간의 빈 봉은 0 으로 싣는다(= 청산 없음)"
    assert p["bars"][2]["partial"] is True and p["bars"][1]["partial"] is False


def test_bars_before_recording_started_are_omitted_not_zeroed():
    """🔴기록 시작 전 봉을 0 으로 내보내면 «청산이 없었다»라는 거짓이 된다 -- 싣지 않는다(= 모름)."""
    p = liq_5m_payload({}, from_s=T0 + 40, bars=5, now_s=T0 + 3 * BAR + 1)
    assert ts_of(p) == [T0 + BAR, T0 + 2 * BAR, T0 + 3 * BAR], "시작 봉(T0)은 반쪽이라 빠져야 한다"


def test_new_event_is_visible_on_the_very_next_call():
    st = {}
    now = T0 + 100
    before = liq_5m_payload(st, T0 - 5 * BAR, 2, now)["bars"][-1]
    liq_5m_add(st, ev(T0 + 90, "short", 2500.0))
    after = liq_5m_payload(st, T0 - 5 * BAR, 2, now)["bars"][-1]
    assert (before["short_usd"], after["short_usd"]) == (0.0, 2500.0)


def test_old_bars_are_pruned_to_the_keep_window():
    st = {}
    liq_5m_add(st, ev(T0, "long", 1.0))
    liq_5m_add(st, ev(T0 + (FOOTPRINT_KEEP_BARS + 1) * BAR, "long", 1.0))
    assert T0 not in st and len(st) == 1


def test_gauge_is_derived_from_the_chart_circles():
    """게이지 = 현재 30분 봉에 든 청산 원의 합 -- 원과 **정의상** 같은 값이어야 한다(2026-09-25).
    전에는 봇 DB 30분 합(l 기반, ~6배 작고 1~3분 늦음)을 따로 받아 원과 다른 크기를 말했다."""
    import re
    app = (Path(__file__).resolve().parents[1] / "dashboard/live/app.js").read_text(encoding="utf-8")
    body = app[app.index("function renderLiquidationVolumeGauge()"):]
    body = re.sub(r"(?m)^\s*//.*$", "", body[:body.index("\nfunction ", 10)])   # 주석에 속지 않는다
    assert "latestLiquidation5mHist" in body, "게이지가 청산 원 배열을 안 본다"
    assert "Math.floor(Date.now() / 1000 / 1800) * 1800" in body, "30분 경계가 서버 게이지(_bar_start)와 달라졌다"
    # 음성 대조군: 서버 게이지 값이 1순위로 돌아가 있으면 실패
    assert "const longUsd = warmed ? Number(liq5m.long_usd_5m || 0) : 0;" not in body

"""칼시 15분 창 고르기(dashboard/server.py::kalshi_pick) -- 창 경계에서 다음 창이 같이 열려 있어도 지금 창을 고른다."""
from dashboard.server import kalshi_pick

T0 = 1767225600  # 2026-01-01T00:00:00Z


def _m(o, c, **k):
    return dict(open_time=o, close_time=c, ticker=o, **k)


A = _m("2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z", floor_strike=2600.5,
       yes_bid_dollars="0.48", yes_ask_dollars="0.50", volume_fp="10")
B = _m("2026-01-01T00:15:00Z", "2026-01-01T00:30:00Z", floor_strike=None,
       yes_bid_dollars="0", yes_ask_dollars="0")


def test_kalshi_pick():
    r = kalshi_pick([B, A], T0 + 60)
    assert r["ticker"] == A["ticker"] and abs(r["p"] - 0.49) < 1e-9 and r["strike"] == 2600.5
    r = kalshi_pick([B, A], T0 + 900)            # 경계 = 다음 창, 기준가·호가 아직 없음
    assert r["ticker"] == B["ticker"] and r["strike"] is None and r["p"] is None
    assert kalshi_pick([B], T0)["ticker"] == B["ticker"]   # 아직 안 열린 창뿐이면 가장 먼저 닫히는 것
    assert kalshi_pick([], T0)["ok"] is False

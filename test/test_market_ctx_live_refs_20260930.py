"""시장 맥락 기준점 맞추기(market_ctx.ring_at / oi_series_live) (2026-09-30).

    python3 -m pytest -q test/test_market_ctx_live_refs_20260930.py
"""
from dashboard import market_ctx as mctx


def test_ring_at_uses_sample_second_then_falls_back():
    ring = {100: 2600.0, 131: 2610.0}
    assert mctx.ring_at(ring, 100, 9.0) == 2600.0
    assert mctx.ring_at(ring, 102, 9.0) == 2600.0      # 초가 빠져도 ±2초 안이면 그 값
    assert mctx.ring_at(ring, 130, 9.0) == 2610.0      # 가장 가까운 쪽
    assert mctx.ring_at(ring, 115, 9.0) == 9.0         # 링에 없으면 지금 값


def test_oi_series_live_replaces_stale_tail():
    t0 = 1_790_000_000 // 300 * 300
    hist = [(t0 + i * 300, 100.0) for i in range(20)]  # 캐시 격자(끝 칸 = t0+5700)
    live = mctx.oi_series_live(hist, t0 + 5700 + 42, 110.0)
    assert live[-1] == (t0 + 5700, 110.0) and len(live) == 20      # 같은 칸이면 교체
    live = mctx.oi_series_live(hist, t0 + 6000 + 5, 110.0)
    assert live[-1] == (t0 + 6000, 110.0) and len(live) == 21      # 새 칸이면 붙임
    st = mctx.oi_stats(live)
    assert st["oi"] == 110.0 and abs(st["d1h_pct"] - 10.0) < 1e-9  # 1시간 전 칸은 격자 그대로

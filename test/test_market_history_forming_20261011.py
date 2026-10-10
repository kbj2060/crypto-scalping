"""server.forming_kline -- /api/market-history 가 덧붙이는 형성 봉(partial)을 **시각으로** 가른다 (2026-10-11).

사용자 «새로고침하면 최신 5분봉이 사라져». 가짜 klines 행만 쓴다(거래소 호출 없음).
    python3 -m pytest -q test/test_market_history_forming_20261011.py
"""
from dashboard.server import forming_kline

B = 1_791_659_700                     # 봉 시작(초)


def k(t, o=2511.4, h=2512.0, lo=2511.3, c=2511.9):
    return [t * 1000, str(o), str(h), str(lo), str(c), "1.0", t * 1000 + 299_999]


def test_forming_row_becomes_partial():
    f = forming_kline([k(B - 300), k(B)], (B + 150) * 1000)
    assert f == {"time": B, "open": 2511.4, "high": 2512.0, "low": 2511.3, "close": 2511.9, "partial": True}


def test_new_bar_row_not_yet_appended_gives_none():
    # 경계 뒤 ~5초: REST 마지막 행이 방금 닫힌 봉 -- 행 수로 «마지막 = 형성»이라 하면 닫힌 봉을 형성 봉으로 오인한다
    assert forming_kline([k(B - 600), k(B - 300)], (B + 3) * 1000) is None
    assert forming_kline([k(B - 300)], B * 1000) is None          # 정확히 경계 = 닫힘
    assert forming_kline([], B * 1000) is None
